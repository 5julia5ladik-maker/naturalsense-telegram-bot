import os
import re
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal, Any

import httpx
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    Boolean,
    BigInteger,
    select,
    text as sql_text,
    update,
    func,
    Index,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


# -----------------------------------------------------------------------------
# CONFIG (ENV)
# -----------------------------------------------------------------------------
def env_get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None else default


BOT_TOKEN = env_get("BOT_TOKEN")
BOT_USERNAME = (env_get("BOT_USERNAME", "") or "").strip().lstrip("@")
PUBLIC_BASE_URL = (env_get("PUBLIC_BASE_URL", "") or "").rstrip("/")
CHANNEL_USERNAME = env_get("CHANNEL_USERNAME", "NaturalSense") or "NaturalSense"
DATABASE_URL = env_get("DATABASE_URL", "sqlite+aiosqlite:///./ns.db") or "sqlite+aiosqlite:///./ns.db"
ADMIN_CHAT_ID = int(env_get("ADMIN_CHAT_ID", "5443870760") or "5443870760")

# Fix Railway postgres schemes for async SQLAlchemy
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

tok = BOT_TOKEN or ""
logger.info(
    "ENV CHECK: BOT_TOKEN_present=%s BOT_TOKEN_len=%s PUBLIC_BASE_URL_present=%s DATABASE_URL_present=%s CHANNEL=%s ADMIN=%s BOT_USERNAME=%s",
    bool(BOT_TOKEN),
    len(tok),
    bool(PUBLIC_BASE_URL),
    bool(DATABASE_URL),
    CHANNEL_USERNAME,
    ADMIN_CHAT_ID,
    BOT_USERNAME or "-",
)


# -----------------------------------------------------------------------------
# BLOCKED TAGS (не отдаём наружу)
# -----------------------------------------------------------------------------
BLOCKED_TAGS = {"SephoraTR", "SephoraGuide"}


# -----------------------------------------------------------------------------
# GAMIFICATION CONFIG
# -----------------------------------------------------------------------------
DAILY_BONUS_POINTS = 5
REGISTER_BONUS_POINTS = 10
REFERRAL_BONUS_POINTS = 20

STREAK_MILESTONES = {
    3: 10,
    7: 30,
    14: 80,
    30: 250,
}

RAFFLE_TICKET_COST = 500
ROULETTE_SPIN_COST = 2000
ROULETTE_LIMIT_WINDOW = timedelta(seconds=5)  # TEST: 5s cooldown
DEFAULT_RAFFLE_ID = 1

# -----------------------------------------------------------------------------
# INVENTORY / CONVERSION (FIXED RATES)
# -----------------------------------------------------------------------------
TICKET_CONVERT_RATE = 300          # 1 raffle ticket -> 300 points
DIOR_PALETTE_CONVERT_VALUE = 50_000  # 1 Dior palette -> 50_000 points (fixed)

PrizeType = Literal["points", "raffle_ticket", "physical_dior_palette"]

# per 1_000_000
ROULETTE_DISTRIBUTION: list[dict[str, Any]] = [
    {"weight": 416_667, "type": "points", "value": 500, "label": "+500 баллов"},
    {"weight": 291_667, "type": "points", "value": 1000, "label": "+1000 баллов"},
    {"weight": 125_000, "type": "points", "value": 1500, "label": "+1500 баллов"},
    {"weight": 83_333, "type": "points", "value": 2000, "label": "+2000 баллов"},
    {"weight": 41_667, "type": "raffle_ticket", "value": 1, "label": "🎟 +1 билет"},
    {"weight": 29_166, "type": "points", "value": 3000, "label": "+3000 баллов"},
    {"weight": 12_500, "type": "physical_dior_palette", "value": 1, "label": "💎 главный приз"},
]
ROULETTE_TOTAL = sum(x["weight"] for x in ROULETTE_DISTRIBUTION)
if ROULETTE_TOTAL != 1_000_000:
    raise RuntimeError("ROULETTE_DISTRIBUTION must sum to 1_000_000")


# -----------------------------------------------------------------------------
# DATABASE MODELS
# -----------------------------------------------------------------------------
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, index=True, nullable=False)

    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)

    tier = Column(String, default="free")  # free / premium / vip
    points = Column(Integer, default=10)
    favorites = Column(JSON, default=list)
    joined_at = Column(DateTime, default=lambda: datetime.utcnow())  # naive UTC

    last_daily_bonus_at = Column(DateTime, nullable=True)  # naive UTC
    daily_streak = Column(Integer, default=0)
    best_streak = Column(Integer, default=0)

    referred_by = Column(BigInteger, nullable=True)
    referral_count = Column(Integer, default=0)
    ref_bonus_paid = Column(Boolean, default=False, nullable=False)


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, unique=True, index=True, nullable=False)

    date = Column(DateTime, nullable=True)  # naive UTC
    text = Column(String, nullable=True)
    media_type = Column(String, nullable=True)
    media_file_id = Column(String, nullable=True)
    permalink = Column(String, nullable=True)

    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.utcnow())  # naive UTC

    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)


class PointTransaction(Base):
    __tablename__ = "point_transactions"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    type = Column(String, nullable=False)  # daily/referral/raffle_ticket/roulette_spin/roulette_prize
    delta = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)
    meta = Column(JSON, default=dict)


Index("ix_point_transactions_tid_created", PointTransaction.telegram_id, PointTransaction.created_at)


class Raffle(Base):
    __tablename__ = "raffles"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False, default="NS Raffle")
    is_active = Column(Boolean, default=True, nullable=False)
    ends_at = Column(DateTime, nullable=True)  # naive UTC
    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)


class RaffleTicket(Base):
    __tablename__ = "raffle_tickets"

    id = Column(Integer, primary_key=True)
    raffle_id = Column(Integer, index=True, nullable=False)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    count = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)


Index("ix_raffle_tickets_unique", RaffleTicket.raffle_id, RaffleTicket.telegram_id, unique=True)


class RouletteSpin(Base):
    __tablename__ = "roulette_spins"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)

    cost_points = Column(Integer, default=ROULETTE_SPIN_COST, nullable=False)

    roll = Column(Integer, nullable=False)  # 0..999999
    prize_type = Column(String, nullable=False)
    prize_value = Column(Integer, nullable=False)
    prize_label = Column(String, nullable=False)


Index("ix_roulette_spins_tid_created", RouletteSpin.telegram_id, RouletteSpin.created_at)


class RouletteClaim(Base):
    __tablename__ = "roulette_claims"

    id = Column(Integer, primary_key=True)
    claim_code = Column(String, unique=True, index=True, nullable=False)  # e.g. "NS-AB12CD34"
    telegram_id = Column(BigInteger, index=True, nullable=False)
    spin_id = Column(Integer, nullable=True)  # optional link to roulette_spins.id

    prize_type = Column(String, nullable=False)
    prize_label = Column(String, nullable=False)

    status = Column(String, default="awaiting_choice", nullable=False)  # awaiting_choice|awaiting_contact|submitted|closed
    contact_text = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)

Index("ix_roulette_claims_tid_created", RouletteClaim.telegram_id, RouletteClaim.created_at)


# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _safe_exec(conn, sql: str):
    try:
        await conn.execute(sql_text(sql))
    except Exception as e:
        logger.info("DB migration skipped/failed (ok in some DBs): %s | %s", sql, e)


async def ensure_default_raffle(session: AsyncSession) -> None:
    existing = (await session.execute(select(Raffle).where(Raffle.id == DEFAULT_RAFFLE_ID))).scalar_one_or_none()
    if existing:
        return
    session.add(
        Raffle(
            id=DEFAULT_RAFFLE_ID,
            title="NS · Розыгрыш",
            is_active=True,
            ends_at=None,
            created_at=datetime.utcnow(),
        )
    )


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # posts
        await _safe_exec(conn, "ALTER TABLE posts ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;")
        await _safe_exec(conn, "ALTER TABLE posts ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP NULL;")

        # users (для старой базы)
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_daily_bonus_at TIMESTAMP NULL;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_streak INTEGER NOT NULL DEFAULT 0;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS best_streak INTEGER NOT NULL DEFAULT 0;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS referred_by BIGINT NULL;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS referral_count INTEGER NOT NULL DEFAULT 0;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS ref_bonus_paid BOOLEAN NOT NULL DEFAULT FALSE;")

        # ✅ Postgres: int32 -> bigint
        await _safe_exec(conn, "ALTER TABLE users ALTER COLUMN telegram_id TYPE BIGINT;")
        await _safe_exec(conn, "ALTER TABLE users ALTER COLUMN referred_by TYPE BIGINT;")
        await _safe_exec(conn, "ALTER TABLE posts ALTER COLUMN message_id TYPE BIGINT;")

    async with async_session_maker() as session:
        await ensure_default_raffle(session)
        await session.commit()

    logger.info("✅ Database initialized")


# -----------------------------------------------------------------------------
# USER / POINTS / STREAK / REFERRAL
# -----------------------------------------------------------------------------
def _recalc_tier(user: User):
    if (user.points or 0) >= 500:
        user.tier = "vip"
    elif (user.points or 0) >= 100:
        user.tier = "premium"
    else:
        user.tier = "free"


async def get_user(telegram_id: int) -> Optional[User]:
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        return result.scalar_one_or_none()


async def find_user_by_username(username: str) -> Optional[User]:
    u = (username or "").strip()
    if not u:
        return None
    if u.startswith("@"):
        u = u[1:]
    u = u.lower()
    async with async_session_maker() as session:
        res = await session.execute(select(User).where(func.lower(User.username) == u))
        return res.scalar_one_or_none()


async def create_user_with_referral(
    telegram_id: int,
    username: str | None,
    first_name: str | None,
    referred_by: int | None,
) -> tuple[User, bool]:
    now = datetime.utcnow()
    referral_paid = False

    async with async_session_maker() as session:
        existing = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalar_one_or_none()
        if existing:
            return existing, False

        inviter: User | None = None
        if referred_by and referred_by != telegram_id:
            inviter = (await session.execute(select(User).where(User.telegram_id == referred_by))).scalar_one_or_none()

        user = User(
            telegram_id=telegram_id,
            username=(username.lower() if username else None),
            first_name=first_name,
            points=REGISTER_BONUS_POINTS,
            joined_at=now,
            last_daily_bonus_at=now,
            daily_streak=1,
            best_streak=1,
            referred_by=(referred_by if inviter else None),
            referral_count=0,
            ref_bonus_paid=False,
        )
        _recalc_tier(user)
        session.add(user)
        await session.flush()

        if inviter and not user.ref_bonus_paid:
            inviter.points = (inviter.points or 0) + REFERRAL_BONUS_POINTS
            inviter.referral_count = (inviter.referral_count or 0) + 1
            _recalc_tier(inviter)
            user.ref_bonus_paid = True
            referral_paid = True

            session.add(
                PointTransaction(
                    telegram_id=int(inviter.telegram_id),
                    type="referral",
                    delta=REFERRAL_BONUS_POINTS,
                    meta={"invited": telegram_id},
                )
            )

        session.add(
            PointTransaction(
                telegram_id=telegram_id,
                type="register",
                delta=REGISTER_BONUS_POINTS,
                meta={},
            )
        )

        await session.commit()
        await session.refresh(user)
        logger.info("✅ New user created: %s", telegram_id)
        return user, referral_paid


async def add_points(telegram_id: int, points: int) -> Optional[User]:
    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalar_one_or_none()
        if not user:
            return None
        user.points = (user.points or 0) + points
        _recalc_tier(user)
        session.add(PointTransaction(telegram_id=telegram_id, type="admin", delta=points, meta={}))
        await session.commit()
        await session.refresh(user)
        return user


async def add_daily_bonus_and_update_streak(telegram_id: int) -> tuple[Optional[User], bool, int, int]:
    async with async_session_maker() as session:
        user: User | None = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalar_one_or_none()
        if not user:
            return None, False, 0, 0

        now = datetime.utcnow()
        last = user.last_daily_bonus_at

        if last is not None and (now - last) < timedelta(days=1):
            delta = timedelta(days=1) - (now - last)
            hours_left = max(
                0,
                int(delta.total_seconds() // 3600) + (1 if (delta.total_seconds() % 3600) > 0 else 0),
            )
            return user, False, hours_left, 0

        user.points = (user.points or 0) + DAILY_BONUS_POINTS

        if last is None:
            user.daily_streak = 1
        else:
            if (now - last) <= timedelta(days=2):
                user.daily_streak = (user.daily_streak or 0) + 1
            else:
                user.daily_streak = 1

        user.best_streak = max(user.best_streak or 0, user.daily_streak or 0)
        user.last_daily_bonus_at = now

        streak_bonus = 0
        if user.daily_streak in STREAK_MILESTONES:
            streak_bonus = STREAK_MILESTONES[user.daily_streak]
            user.points = (user.points or 0) + streak_bonus

        _recalc_tier(user)

        session.add(PointTransaction(telegram_id=telegram_id, type="daily", delta=DAILY_BONUS_POINTS, meta={}))
        if streak_bonus:
            session.add(PointTransaction(telegram_id=telegram_id, type="streak_bonus", delta=streak_bonus, meta={"streak": user.daily_streak}))

        await session.commit()
        await session.refresh(user)
        return user, True, 0, streak_bonus


# -----------------------------------------------------------------------------
# POSTS INDEX (TAGS)
# -----------------------------------------------------------------------------
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")


def extract_tags(text_: str | None) -> list[str]:
    if not text_:
        return []
    tags = [m.group(1) for m in TAG_RE.finditer(text_)]
    out, seen = [], set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def preview_text(text_: str | None, limit: int = 180) -> str:
    if not text_:
        return ""
    s = re.sub(r"\s+", " ", text_.strip())
    return (s[:limit] + "…") if len(s) > limit else s


def make_permalink(message_id: int) -> str:
    return f"https://t.me/{CHANNEL_USERNAME}/{message_id}"


def to_naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


async def upsert_post_from_channel(
    message_id: int,
    date: datetime | None,
    text_: str | None,
    media_type: str | None = None,
    media_file_id: str | None = None,
):
    tags = extract_tags(text_)
    permalink = make_permalink(int(message_id))
    date_naive = to_naive_utc(date)

    async with async_session_maker() as session:
        p = (await session.execute(select(Post).where(Post.message_id == message_id))).scalar_one_or_none()

        if p:
            p.date = date_naive
            p.text = text_
            p.media_type = media_type
            p.media_file_id = media_file_id
            p.permalink = permalink
            p.tags = tags
            p.is_deleted = False
            p.deleted_at = None
            await session.commit()
            return p

        p = Post(
            message_id=message_id,
            date=date_naive,
            text=text_,
            media_type=media_type,
            media_file_id=media_file_id,
            permalink=permalink,
            tags=tags,
            created_at=datetime.utcnow(),
            is_deleted=False,
            deleted_at=None,
        )
        session.add(p)
        await session.commit()
        await session.refresh(p)
        logger.info("✅ Indexed post %s tags=%s", message_id, tags)
        return p


async def list_posts(tag: str | None, limit: int = 50, offset: int = 0):
    if tag and tag in BLOCKED_TAGS:
        return []

    async with async_session_maker() as session:
        q = (
            select(Post)
            .where(Post.is_deleted == False)  # noqa: E712
            .order_by(Post.message_id.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await session.execute(q)).scalars().all()

    if tag:
        rows = [p for p in rows if tag in (p.tags or [])]
    return rows


# -----------------------------------------------------------------------------
# DELETE SWEEPER (AUTO CHECK)
# -----------------------------------------------------------------------------
async def message_exists_public(message_id: int) -> bool:
    url = f"https://t.me/{CHANNEL_USERNAME}/{message_id}?embed=1"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404:
                return False
            if r.status_code != 200:
                return True

            html = (r.text or "").lower()
            if "message not found" in html or "post not found" in html:
                return False
            if "join channel" in html or "this channel is private" in html:
                return True
            return True
    except Exception as e:
        logger.warning("Sweeper check failed for %s: %s", message_id, e)
        return True


async def sweep_deleted_posts(batch: int = 80):
    async with async_session_maker() as session:
        posts = (
            await session.execute(
                select(Post)
                .where(Post.is_deleted == False)  # noqa: E712
                .order_by(Post.message_id.desc())
                .limit(batch)
            )
        ).scalars().all()

    if not posts:
        return []

    to_mark: list[int] = []
    for p in posts:
        ok = await message_exists_public(int(p.message_id))
        if not ok:
            to_mark.append(int(p.message_id))

    if not to_mark:
        return []

    async with async_session_maker() as session:
        now = datetime.utcnow()
        await session.execute(
            update(Post)
            .where(Post.message_id.in_(to_mark))
            .values(is_deleted=True, deleted_at=now)
        )
        await session.commit()

    logger.info("🧹 Marked deleted posts: %s", to_mark)
    return to_mark


async def sweeper_loop():
    while True:
        try:
            await sweep_deleted_posts(batch=80)
        except Exception as e:
            logger.error("Sweeper error: %s", e)
        await asyncio.sleep(300)  # 5 минут


# -----------------------------------------------------------------------------
# TELEGRAM BOT
# -----------------------------------------------------------------------------
tg_app: Application | None = None
tg_task: asyncio.Task | None = None
sweeper_task: asyncio.Task | None = None


def is_admin(user_id: int) -> bool:
    return int(user_id) == int(ADMIN_CHAT_ID)


def get_main_keyboard():
    # ✅ СНИЗУ ТОЛЬКО: Профиль + Помощь
    return ReplyKeyboardMarkup(
        [[KeyboardButton("👤 Профиль"), KeyboardButton("ℹ️ Помощь")]],
        resize_keyboard=True,
    )


def build_start_inline_kb() -> InlineKeyboardMarkup:
    # ✅ “В канал” прикреплена к сообщению /start
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("↩️ В канал", url=f"https://t.me/{CHANNEL_USERNAME}")]]
    )


async def set_keyboard_silent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Telegram показывает ReplyKeyboard только через сообщение.
    # Делаем невидимое сообщение и удаляем -> в чате ничего не видно, кнопки остаются.
    chat = update.effective_chat
    if not chat:
        return
    try:
        m = await context.bot.send_message(chat_id=chat.id, text="\u200b", reply_markup=get_main_keyboard())
        await asyncio.sleep(0.8)
        try:
            await context.bot.delete_message(chat_id=chat.id, message_id=m.message_id)
        except Exception:
            pass
    except Exception:
        pass


def build_help_text() -> str:
    return """\
ℹ️ *Помощь / Как пользоваться*

1) Нажми *📲 Открыть журнал* — откроется Mini App внутри Telegram.
2) Выбирай категории/бренды и открывай посты.
3) *👤 Профиль* — баллы, уровень, стрик.
4) *↩️ В канал* — кнопка под сообщением /start.

💎 *Баллы и антифарм*
• Первый /start: +10 за регистрацию
• Далее: +5 за визит, строго 1 раз в 24 часа

🔥 *Стрик (серия дней)*
• 3 дня: +10
• 7 дней: +30
• 14 дней: +80
• 30 дней: +250

🎟 *Рефералка*
Команда /invite даёт твою ссылку.
За каждого нового пользователя по ссылке: +20 (1 раз за каждого).
"""


async def tg_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram handler error: %s", context.error)
    try:
        if ADMIN_CHAT_ID:
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=f"❌ Ошибка в боте:\n{repr(context.error)}"
            )
    except Exception:
        pass


def build_welcome_text(
    first_name: str | None,
    is_new: bool,
    daily_granted: bool,
    hours_left: int,
    streak: int,
    streak_bonus: int,
    referral_paid: bool,
) -> str:
    name = first_name or "друг"

    if is_new:
        bonus_line = f"✅ +{REGISTER_BONUS_POINTS} баллов за регистрацию ✨"
    else:
        if daily_granted:
            bonus_line = f"✅ +{DAILY_BONUS_POINTS} баллов за визит ✨ (раз в 24 часа)"
        else:
            bonus_line = f"ℹ️ Бонус за визит уже получен. Следующий — примерно через {hours_left} ч."

    streak_line = f"🔥 Стрик: {streak} день(дней) подряд"
    if streak_bonus > 0:
        streak_line += f"\n🎉 Бонус за стрик: +{streak_bonus}"

    ref_line = ""
    if referral_paid:
        ref_line = f"\n🎁 Тебя пригласили — твой друг получил +{REFERRAL_BONUS_POINTS} баллов."

    return f"""\
Привет, {name}! 🖤

{bonus_line}
{streak_line}{ref_line}
"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not update.message:
        return

    referred_by: int | None = None
    if context.args:
        arg0 = (context.args[0] or "").strip()

        # ✅ claim flow via deep-link: /start claim_<CODE>
        if arg0.startswith("claim_"):
            code = arg0.replace("claim_", "", 1).strip()
            await claim_start_flow(update, context, code)
            return
        if arg0.isdigit():
            referred_by = int(arg0)

    db_user = await get_user(user.id)

    if not db_user:
        created_user, referral_paid = await create_user_with_referral(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            referred_by=referred_by,
        )
        text_ = build_welcome_text(
            first_name=user.first_name,
            is_new=True,
            daily_granted=True,
            hours_left=0,
            streak=created_user.daily_streak or 1,
            streak_bonus=0,
            referral_paid=referral_paid,
        )
        await update.message.reply_text(text_, reply_markup=build_start_inline_kb())
        await set_keyboard_silent(update, context)
        return

    user2, granted, hours_left, streak_bonus = await add_daily_bonus_and_update_streak(user.id)
    if not user2:
        await update.message.reply_text("Ошибка пользователя. Нажми /start ещё раз.", reply_markup=build_start_inline_kb())
        await set_keyboard_silent(update, context)
        return

    text_ = build_welcome_text(
        first_name=user.first_name,
        is_new=False,
        daily_granted=granted,
        hours_left=hours_left,
        streak=user2.daily_streak or 0,
        streak_bonus=streak_bonus,
        referral_paid=False,
    )
    await update.message.reply_text(text_, reply_markup=build_start_inline_kb())
    await set_keyboard_silent(update, context)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await update.message.reply_text(build_help_text(), parse_mode="Markdown", reply_markup=get_main_keyboard())

async def claim_start_flow(update: Update, context: ContextTypes.DEFAULT_TYPE, code: str):
    """Логика получения физического приза (deep-link /start claim_CODE или /claim CODE)

    Статусы:
      awaiting_choice   — приз получен, но выбор (забрать/в бонусы) ещё не сделан
      awaiting_contact  — пользователь нажал "Забрать", ждём контакты/адрес
      submitted         — контакты получены, заявка отправлена админу
      closed            — заявка закрыта (конвертировано или выдано)
    """
    if not update.message:
        return
    code = (code or "").strip().upper()
    if not code:
        await update.message.reply_text("Использование: /claim NS-XXXXXXXX")
        return

    async with async_session_maker() as session:
        claim = (
            await session.execute(
                select(RouletteClaim).where(RouletteClaim.claim_code == code)
            )
        ).scalar_one_or_none()

        if not claim:
            await update.message.reply_text("❌ Код не найден. Проверьте, пожалуйста.")
            return

        if int(claim.telegram_id) != int(update.effective_user.id):
            await update.message.reply_text("⛔️ Этот код принадлежит другому пользователю.")
            return

        st = (claim.status or "").strip()

        if st == "closed":
            await update.message.reply_text("✅ Эта заявка уже закрыта.")
            return

        if st == "submitted":
            await update.message.reply_text("✅ Заявка уже отправлена. Мы скоро свяжемся.")
            return

        # если это первый заход по кнопке "Забрать" — фиксируем выбор и блокируем конвертацию в бонусы
        if st == "awaiting_choice":
            claim.status = "awaiting_contact"
            claim.updated_at = datetime.utcnow()
            await session.commit()

        # st == awaiting_contact (или только что перевели) — просим контакты/адрес
        await update.message.reply_text(
            "🎁 Вы выбрали получение приза.\n\n"
            "Напишите одним сообщением удобный способ связи (Telegram/WhatsApp) и адрес/город доставки.\n"
            f"Код заявки: {code}\n\n"
            "После отправки данных обмен на бонусы будет недоступен."
        )

async def cmd_claim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    code = (context.args[0] if context.args else "").strip()
    await claim_start_flow(update, context, code)


async def cmd_invite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    user = update.effective_user
    me = await context.bot.get_me()
    bot_username = me.username or ""
    if not bot_username:
        await update.message.reply_text("Не удалось получить username бота. Проверь настройки.", reply_markup=get_main_keyboard())
        return

    link = f"https://t.me/{bot_username}?start={user.id}"
    text = f"""\
🎟 Твоя реферальная ссылка:

{link}

За каждого нового пользователя по этой ссылке: +{REFERRAL_BONUS_POINTS} баллов (1 раз за каждого).
"""
    await update.message.reply_text(text, reply_markup=get_main_keyboard())


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    user = update.effective_user
    db_user = await get_user(user.id)

    if not db_user:
        await update.message.reply_text("Нажми /start для регистрации", reply_markup=get_main_keyboard())
        return

    tier_emoji = {"free": "🥉", "premium": "🥈", "vip": "🥇"}
    tier_name = {"free": "Bronze", "premium": "Silver", "vip": "Gold VIP"}

    next_tier_points = {
        "free": (100, "Silver"),
        "premium": (500, "Gold VIP"),
        "vip": (1000, "Platinum"),
    }

    next_points, next_name = next_tier_points.get(db_user.tier, (0, "Max"))
    remaining = max(0, next_points - (db_user.points or 0))

    streak = db_user.daily_streak or 0
    best = db_user.best_streak or 0
    refs = db_user.referral_count or 0

    last_bonus = db_user.last_daily_bonus_at
    if last_bonus:
        now = datetime.utcnow()
        if (now - last_bonus) >= timedelta(days=1):
            bonus_hint = "✅ Доступен ежедневный бонус — нажми /start"
        else:
            delta = timedelta(days=1) - (now - last_bonus)
            hours_left = max(
                0,
                int(delta.total_seconds() // 3600) + (1 if (delta.total_seconds() % 3600) > 0 else 0),
            )
            bonus_hint = f"ℹ️ Ежедневный бонус через ~{hours_left} ч"
    else:
        bonus_hint = "ℹ️ Нажми /start для бонуса"

    joined = db_user.joined_at.strftime("%d.%m.%Y") if db_user.joined_at else "-"

    text_ = f"""\
👤 **Твой профиль**

{tier_emoji.get(db_user.tier, "🥉")} Уровень: {tier_name.get(db_user.tier, "Bronze")}
💎 Баллы: **{db_user.points}**

🔥 Стрик: **{streak}** • Лучший: **{best}**
🎟 Приглашено: **{refs}**

📊 До {next_name}: {remaining} баллов
📅 Регистрация: {joined}

{bonus_hint}
"""
    await update.message.reply_text(text_, parse_mode="Markdown", reply_markup=get_main_keyboard())


async def cmd_myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    u = update.effective_user
    await update.message.reply_text(f"Твой telegram_id: {u.id}", reply_markup=get_main_keyboard())


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Ответь на сообщение человека и напиши /id", reply_markup=get_main_keyboard())
        return
    target = update.message.reply_to_message.from_user
    await update.message.reply_text(
        f"ID пользователя: {target.id}\nusername: @{target.username or '-'}\nname: {target.first_name or '-'}",
        reply_markup=get_main_keyboard()
    )


# --- admin ---
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Статистика", callback_data="admin_stats")],
            [InlineKeyboardButton("🧹 Sweep (проверка удалённых постов)", callback_data="admin_sweep")],
        ]
    )
    await update.message.reply_text("👑 Админ-панель:", reply_markup=kb)


async def admin_stats_text() -> str:
    async with async_session_maker() as session:
        total_users = (await session.execute(select(func.count(User.id)))).scalar() or 0
        total_posts = (await session.execute(select(func.count(Post.id)))).scalar() or 0
        deleted_posts = (
            (await session.execute(select(func.count(Post.id)).where(Post.is_deleted == True)))  # noqa: E712
        ).scalar() or 0

        since = datetime.utcnow() - timedelta(days=1)
        users_24h = (await session.execute(select(func.count(User.id)).where(User.joined_at >= since))).scalar() or 0

    return f"""\
📊 *Статистика*

👥 Пользователей всего: *{total_users}*
👥 Новых за 24ч: *{users_24h}*

📝 Постов в базе: *{total_posts}*
🗑 Помечено удалённых: *{deleted_posts}*
"""


async def cmd_admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return
    await update.message.reply_text(await admin_stats_text(), parse_mode="Markdown", reply_markup=get_main_keyboard())


async def cmd_admin_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return
    marked = await sweep_deleted_posts(batch=120)
    if not marked:
        await update.message.reply_text("🧹 Sweep: ничего не найдено.", reply_markup=get_main_keyboard())
    else:
        await update.message.reply_text(f"🧹 Sweep: помечены удалёнными: {marked}", reply_markup=get_main_keyboard())


async def cmd_admin_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return

    if not context.args or not (context.args[0] or "").isdigit():
        await update.message.reply_text("Используй: /admin_user <telegram_id>", reply_markup=get_main_keyboard())
        return

    tid = int(context.args[0])
    u = await get_user(tid)
    if not u:
        await update.message.reply_text("Юзер не найден.", reply_markup=get_main_keyboard())
        return

    text = f"""\
👤 Пользователь: {u.telegram_id}
Имя: {u.first_name or "-"} @{u.username or "-"}

Tier: {u.tier}
Баллы: {u.points}

Стрик: {u.daily_streak} (best {u.best_streak})
Last bonus: {u.last_daily_bonus_at}

Referred_by: {u.referred_by}
Referral_count: {u.referral_count}
Ref_paid: {u.ref_bonus_paid}
"""
    await update.message.reply_text(text, reply_markup=get_main_keyboard())


async def cmd_admin_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return

    if len(context.args) < 2 or not context.args[0].isdigit() or not re.match(r"^-?\d+$", context.args[1]):
        await update.message.reply_text("Используй: /admin_add <telegram_id> <баллы>", reply_markup=get_main_keyboard())
        return

    tid = int(context.args[0])
    pts = int(context.args[1])

    u = await add_points(tid, pts)
    if not u:
        await update.message.reply_text("Юзер не найден.", reply_markup=get_main_keyboard())
        return

    await update.message.reply_text(f"✅ Начислено {pts}. Теперь у юзера {u.points} баллов.", reply_markup=get_main_keyboard())


async def cmd_admin_find(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return

    if not context.args:
        await update.message.reply_text("Используй: /find @username", reply_markup=get_main_keyboard())
        return

    username = context.args[0]
    u = await find_user_by_username(username)
    if not u:
        await update.message.reply_text("Не найдено. Этот человек ещё не писал боту (/start).", reply_markup=get_main_keyboard())
        return

    await update.message.reply_text(
        f"✅ Найден:\n"
        f"telegram_id: {u.telegram_id}\n"
        f"username: @{u.username or '-'}\n"
        f"name: {u.first_name or '-'}\n"
        f"points: {u.points}\n"
        f"tier: {u.tier}",
        reply_markup=get_main_keyboard()
    )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    await q.answer()

    uid = q.from_user.id
    if not is_admin(uid):
        await q.edit_message_text("⛔️ Нет доступа.")
        return

    data = q.data or ""
    if data == "admin_stats":
        await q.edit_message_text((await admin_stats_text()), parse_mode="Markdown")
        return

    if data == "admin_sweep":
        marked = await sweep_deleted_posts(batch=120)
        if not marked:
            await q.edit_message_text("🧹 Sweep: ничего не найдено.")
        else:
            await q.edit_message_text(f"🧹 Sweep: помечены удалёнными: {marked}")
        return


async def on_text_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    txt = update.message.text.strip()

    # ✅ если ожидаем контакты/адрес по claim — принимаем любое сообщение
    async with async_session_maker() as session:
        pending = (
            await session.execute(
                select(RouletteClaim)
                .where(RouletteClaim.telegram_id == update.effective_user.id)
                .where(RouletteClaim.status == "awaiting_contact")
                .order_by(RouletteClaim.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if pending and txt not in ("👤 Профиль", "ℹ️ Помощь", "↩️ В канал"):
            pending.contact_text = txt
            pending.status = "submitted"
            pending.updated_at = datetime.utcnow()
            await session.commit()

            uname = (update.effective_user.username or "").strip()
            mention = f"@{uname}" if uname else "(без username)"
            await notify_admin(
                "✅ CLAIM: пользователь отправил контакты\n"
                f"user: {mention} | {update.effective_user.first_name or '-'}\n"
                f"telegram_id: {update.effective_user.id}\n"
                f"link: {tg_user_link(update.effective_user.id)}\n"
                f"claim: {pending.claim_code}\n"
                f"prize: {pending.prize_label}\n"
                f"contacts: {txt}"
            )
            await update.message.reply_text("✅ Спасибо! Заявка отправлена. Мы скоро свяжемся.")
            return

    if txt == "👤 Профиль":
        await cmd_profile(update, context)
        return

    if txt == "ℹ️ Помощь":
        await cmd_help(update, context)
        return


# -----------------------------------------------------------------------------
# CHANNEL INDEXING
# -----------------------------------------------------------------------------
async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return
    text_ = msg.text or msg.caption or ""
    await upsert_post_from_channel(
        message_id=msg.message_id,
        date=msg.date,
        text_=text_,
        media_type=None,
        media_file_id=None,
    )


async def on_edited_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.edited_channel_post
    if not msg:
        return
    text_ = msg.text or msg.caption or ""
    await upsert_post_from_channel(
        message_id=msg.message_id,
        date=msg.date,
        text_=text_,
        media_type=None,
        media_file_id=None,
    )


# -----------------------------------------------------------------------------
# TELEGRAM RUNNER (polling)
# -----------------------------------------------------------------------------
async def _telegram_runner():
    global tg_app
    try:
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        logger.info("✅ Telegram bot started (polling)")
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Telegram runner crashed: %s", e)
    finally:
        try:
            if tg_app:
                try:
                    await tg_app.updater.stop()
                except Exception:
                    pass
                await tg_app.stop()
                await tg_app.shutdown()
        except Exception:
            pass


async def start_telegram_bot():
    global tg_app, tg_task

    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN not set; starting API WITHOUT Telegram bot")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_error_handler(tg_error_handler)

    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("claim", cmd_claim))
    tg_app.add_handler(CommandHandler("invite", cmd_invite))
    tg_app.add_handler(CommandHandler("profile", cmd_profile))
    tg_app.add_handler(CommandHandler("myid", cmd_myid))
    tg_app.add_handler(CommandHandler("id", cmd_id))

    tg_app.add_handler(CommandHandler("admin", cmd_admin))
    tg_app.add_handler(CommandHandler("admin_stats", cmd_admin_stats))
    tg_app.add_handler(CommandHandler("admin_sweep", cmd_admin_sweep))
    tg_app.add_handler(CommandHandler("admin_user", cmd_admin_user))
    tg_app.add_handler(CommandHandler("admin_add", cmd_admin_add))
    tg_app.add_handler(CommandHandler("find", cmd_admin_find))

    tg_app.add_handler(CallbackQueryHandler(on_callback))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_button))

    tg_app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST, on_channel_post))
    tg_app.add_handler(MessageHandler(filters.UpdateType.EDITED_CHANNEL_POST, on_edited_channel_post))

    tg_task = asyncio.create_task(_telegram_runner())


async def stop_telegram_bot():
    global tg_task
    if tg_task:
        tg_task.cancel()
        try:
            await tg_task
        except Exception:
            pass
        tg_task = None



def generate_claim_code() -> str:
    # короткий читаемый код
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    rnd = "".join(secrets.choice(alphabet) for _ in range(8))
    return f"NS-{rnd}"


def tg_user_link(user_id: int) -> str:
    return f"tg://user?id={int(user_id)}"


async def notify_user_top_prize(user_id: int, prize_label: str, claim_code: str) -> None:
    """Отправляем человеку сообщение в личный чат, чтобы он не потерял выигрыш."""
    if not tg_app or not BOT_TOKEN:
        return
    try:
        await tg_app.bot.send_message(
            chat_id=int(user_id),
            text=(
                "🎉 Поздравляем! Вы выиграли: " + str(prize_label) + "\n\n"
                "Чтобы забрать приз:\n"
                f"/claim {claim_code}\n\n"
                "После этого отправьте одним сообщением удобный способ связи (Telegram/WhatsApp) и адрес/город доставки."
            ),
        )
    except Exception as e:
        logger.warning("Failed to notify winner: %s", e)


async def notify_admin(text: str) -> None:
    if not tg_app or not BOT_TOKEN or not ADMIN_CHAT_ID:
        logger.info("ADMIN ALERT (no bot): %s", text)
        return
    try:
        await tg_app.bot.send_message(chat_id=ADMIN_CHAT_ID, text=text)
    except Exception as e:
        logger.warning("Failed to notify admin: %s", e)


# -----------------------------------------------------------------------------
# MINI APP (WEBAPP HTML)
# -----------------------------------------------------------------------------
def get_webapp_html() -> str:
    html = r"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <title>NS · Natural Sense</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    *{box-sizing:border-box}
    :root{
      --bg:#0c0f14;
      --card:rgba(255,255,255,0.08);
      --text:rgba(255,255,255,0.92);
      --muted:rgba(255,255,255,0.60);
      --stroke:rgba(255,255,255,0.12);
      --gold:rgba(230,193,128,0.9);
      --accent:rgba(230,193,128,0.14);
    }
    body{
      margin:0;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      background: radial-gradient(1200px 800px at 20% 10%, rgba(230,193,128,0.18), transparent 60%), var(--bg);
      color:var(--text);
      overflow-x:hidden;
    }
    .wrap{padding:18px 16px 26px; max-width:520px; margin:0 auto;}
    .card{
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      border-radius:22px;
      padding:14px;
      box-shadow:0 10px 30px rgba(0,0,0,0.35);
    }
    .hero{background:linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05)); position:relative; overflow:hidden;}
    .hero::before{
      content:"";
      position:absolute; inset:-2px;
      background:radial-gradient(600px 300px at 10% 0%, rgba(230,193,128,0.26), transparent 60%);
      pointer-events:none;
    }
    .row{display:flex; justify-content:space-between; align-items:center; gap:10px;}
    .title{font-size:20px; font-weight:650; letter-spacing:0.2px;}
    .sub{margin-top:6px; font-size:13px; color:var(--muted);}
    .pill{
      display:inline-flex; align-items:center; gap:6px;
      padding:6px 10px; border-radius:999px;
      border:1px solid rgba(230,193,128,0.25);
      background:rgba(230,193,128,0.10);
      font-size:13px; font-weight:700;
    }
    .tabs{display:flex; gap:8px; margin-top:14px;}
    .tab{
      flex:1;
      padding:10px;
      border-radius:14px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      text-align:center;
      font-size:13px;
      user-select:none;
      cursor:pointer;
      transition:all .2s;
    }
    .tab.active{
      border:1px solid rgba(230,193,128,0.40);
      background:rgba(230,193,128,0.12);
    }
    .btn{
      width:100%;
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:14px;
      border-radius:18px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      margin:10px 0;
      cursor:pointer;
      user-select:none;
    }
    .btn.disabled{opacity:.5; cursor:not-allowed;}
    .btn .left{display:flex; flex-direction:column; gap:4px;}
    .btn .label{font-size:15px;}
    .btn .hint{font-size:12px; color:var(--muted);}
    .section{margin-top:14px;}
    .muted{color:var(--muted); font-size:13px;}
    .post{
      margin-top:10px;
      padding:12px;
      border-radius:18px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      cursor:pointer;
    }
    .tagpill{
      font-size:12px;
      padding:5px 8px;
      border-radius:999px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.08);
      display:inline-block;
      margin-right:6px;
      margin-top:6px;
    }
    .grid{display:grid; gap:10px;}
    .note{
      margin-top:14px;
      padding:10px;
      border-radius:14px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.08);
      font-size:13px;
    }
    .smallbtn{
      flex:1;
      padding:12px;
      text-align:center;
      border-radius:14px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      cursor:pointer;
      user-select:none;
      font-weight:750;
    }
    .smallbtn.gold{
      border:1px solid rgba(230,193,128,0.35);
      background:rgba(230,193,128,0.14);
      font-weight:850;
    }
    .hr{height:1px; background:var(--stroke); margin:14px 0 8px;}
    .center{ text-align:center; }
  </style>
</head>
<body>
  <div id="app" class="wrap"></div>

<script>
(function(){
  const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;

  const CHANNEL = "__CHANNEL__";
  const BOT_USERNAME = "__BOT_USERNAME__"; // может быть пустым
  const DEFAULT_BG = "#0c0f14";

  function hexToRgba(hex, a){
    if(!hex) return "rgba(12,15,20,"+a+")";
    let h = String(hex).trim();
    if(h[0]==="#") h=h.slice(1);
    if(h.length===3) h=h.split("").map(c=>c+c).join("");
    if(h.length!==6) return "rgba(12,15,20,"+a+")";
    const r=parseInt(h.slice(0,2),16), g=parseInt(h.slice(2,4),16), b=parseInt(h.slice(4,6),16);
    return "rgba("+r+","+g+","+b+","+a+")";
  }
  function setVar(k,v){ document.documentElement.style.setProperty(k,v); }

  function applyTelegramTheme(){
    const scheme = tg && tg.colorScheme ? tg.colorScheme : "dark";
    const p = tg && tg.themeParams ? tg.themeParams : {};
    const bg = p.bg_color || DEFAULT_BG;
    const text = p.text_color || (scheme==="dark" ? "rgba(255,255,255,0.92)" : "rgba(17,17,17,0.92)");
    const muted = p.hint_color || (scheme==="dark" ? "rgba(255,255,255,0.60)" : "rgba(0,0,0,0.55)");
    setVar("--bg", bg);
    setVar("--text", text);
    setVar("--muted", muted);
    setVar("--stroke", scheme==="dark" ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.10)");
    setVar("--card", scheme==="dark" ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.72)");
    if(tg){
      try{ tg.setHeaderColor(bg); tg.setBackgroundColor(bg); }catch(e){}
    }
  }

  if(tg){
    tg.expand();
    applyTelegramTheme();
    tg.onEvent("themeChanged", applyTelegramTheme);
  }

  const el = (tag, attrs={}, children=[]) => {
    const n = document.createElement(tag);
    Object.entries(attrs||{}).forEach(([k,v])=>{
      if(k==="class") n.className = v;
      else if(k==="html") n.innerHTML = v;
      else if(k==="onclick") n.addEventListener("click", v);
      else n.setAttribute(k, v);
    });
    (children||[]).forEach(c=>{
      if(c==null) return;
      if(typeof c==="string") n.appendChild(document.createTextNode(c));
      else n.appendChild(c);
    });
    return n;
  };

  const state = {
    activeTab: "home",
    view: "main", // main|posts|inventory|profile
    selectedTag: null,
    user: null,
    raffle: null,
    history: [],
    inventory: null,
    loading: false,
    msg: "",
    invMsg: "",
    busy: false,
    botUsername: (BOT_USERNAME || "").trim().replace(/^@/,"")
  };

  const tgUserId = tg && tg.initDataUnsafe && tg.initDataUnsafe.user ? tg.initDataUnsafe.user.id : null;

  function openLink(url){
    if(tg && tg.openTelegramLink) tg.openTelegramLink(url);
    else window.open(url, "_blank");
  }

  async function apiGet(url){
    const r = await fetch(url);
    if(!r.ok) throw new Error("HTTP "+r.status);
    return await r.json();
  }
  async function apiPost(url, body){
    const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})});
    const data = await r.json().catch(()=> ({}));
    if(!r.ok) throw new Error(data.detail || ("HTTP "+r.status));
    return data;
  }

  function tierLabel(t){
    return ({free:"🥉 Bronze", premium:"🥈 Silver", vip:"🥇 Gold VIP"}[t] || "🥉 Bronze");
  }

  async function refreshUser(){
    if(!tgUserId) return;
    try{ state.user = await apiGet("/api/user/"+encodeURIComponent(tgUserId)); }catch(e){}
  }
  async function loadRaffle(){
    if(!tgUserId) return;
    try{ state.raffle = await apiGet("/api/raffle/status?telegram_id="+encodeURIComponent(tgUserId)); }catch(e){ state.raffle=null; }
  }
  async function loadHistory(){
    if(!tgUserId) return;
    try{ state.history = await apiGet("/api/roulette/history?telegram_id="+encodeURIComponent(tgUserId)+"&limit=5"); }catch(e){ state.history=[]; }
  }
  async function loadInventory(){
    if(!tgUserId) return;
    try{ state.inventory = await apiGet("/api/inventory?telegram_id="+encodeURIComponent(tgUserId)); }catch(e){ state.inventory=null; }
  }
  async function loadBotUsername(){
    try{
      const d = await apiGet("/api/bot/username");
      const u = (d && d.bot_username ? String(d.bot_username) : "").trim().replace(/^@/,"");
      if(u) state.botUsername = u;
    }catch(e){}
  }

  async function openPosts(tag){
    state.view="posts";
    state.selectedTag=tag;
    state.loading=true;
    state.posts=[];
    render();
    try{
      const data = await apiGet("/api/posts?tag="+encodeURIComponent(tag));
      state.posts = Array.isArray(data) ? data : [];
    }catch(e){
      state.posts=[];
    }finally{
      state.loading=false;
      render();
    }
  }

  async function openInventory(){
    state.view="inventory";
    state.invMsg="";
    render();
    await loadInventory();
    render();
  }

  async function openProfile(){
    state.view="profile";
    state.msg="";
    render();
    await Promise.all([loadRaffle(), loadHistory()]);
    render();
  }

  async function buyTicket(){
    if(!tgUserId) return;
    state.busy=true; state.msg="";
    render();
    try{
      const data = await apiPost("/api/raffle/buy_ticket", {telegram_id: tgUserId, qty: 1});
      state.msg = "✅ Билет куплен. Твоих билетов: "+data.ticket_count;
      await refreshUser();
      await loadRaffle();
    }catch(e){
      state.msg = "❌ " + (e.message || "Ошибка");
    }finally{
      state.busy=false;
      render();
    }
  }

  async function spin(){
    if(!tgUserId) return;
    state.busy=true; state.msg="";
    render();
    try{
      const data = await apiPost("/api/roulette/spin", {telegram_id: tgUserId});
      state.msg = "🎡 Выпало: " + data.prize_label;

      if(data.claimable && data.claim_code){
        // сразу показываем кнопку перехода в бота
        const u = state.botUsername;
        const link = u ? ("https://t.me/"+u+"?start=claim_"+data.claim_code) : "";
        if(tg && tg.showPopup){
          tg.showPopup({
            title:"💎 Приз",
            message:"Ваш приз: "+data.prize_label+"\n\nНажмите «Получить» для оформления.",
            buttons:[{id:"claim", type:"default", text:"Получить"}, {type:"cancel"}]
          }, (btnId)=>{
            if(btnId==="claim" && link) openLink(link);
          });
        }else{
          if(link) openLink(link);
        }
      }
      await refreshUser();
      await loadRaffle();
      await loadHistory();
    }catch(e){
      state.msg = "❌ " + (e.message || "Ошибка");
    }finally{
      state.busy=false;
      render();
    }
  }

  async function convertPrize(code){
    if(!tgUserId) return;
    state.busy=true; state.invMsg="";
    render();
    try{
      const data = await apiPost("/api/inventory/convert_prize", {telegram_id: tgUserId, claim_code: code});
      state.invMsg = "✅ Приз превращён в бонусы: +"+data.added_points+" баллов";
      await refreshUser();
      await loadInventory();
    }catch(e){
      state.invMsg = "❌ " + (e.message || "Ошибка");
    }finally{
      state.busy=false;
      render();
    }
  }

  function prizeStatusLabel(s){
    const v = String(s||"");
    if(v==="awaiting_choice") return "🟡 Выбор: забрать или в бонусы";
    if(v==="awaiting_contact") return "⏳ Оформление (обмен недоступен)";
    if(v==="submitted") return "📨 Заявка отправлена";
    if(v==="closed") return "✅ Закрыт";
    return v||"-";
  }

  function buildHero(){
    const u = state.user;
    const hero = el("div",{class:"card hero"},[
      el("div",{class:"row", style:"position:relative;"},[
        el("div",{},[
          el("div",{class:"title"} ,["NS · Natural Sense"]),
          el("div",{class:"sub"},["luxury beauty magazine"])
        ]),
        u ? el("div",{class:"pill"},["💎 ", String(u.points), " • ", tierLabel(u.tier)]) : el("div",{class:"pill"},["NS"])
      ]),
      el("div",{style:"position:relative; margin-top:14px;"},[
        !tgUserId ? el("div",{class:"muted"},["Открой приложение через кнопку «📲 Открыть журнал» в боте."]) :
        (u ? el("div",{class:"muted"},["Привет, ", (u.first_name||"друг"), "! Нажми «Профиль» для бонусов и рулетки."]) :
             el("div",{class:"muted"},["Загрузка профиля…"]))
      ])
    ]);
    return hero;
  }

  function buildTabs(){
    const tabs = [
      ["home","Главное"],
      ["cat","Категории"],
      ["brand","Бренды"],
      ["sephora","Sephora"],
      ["ptype","Продукт"]
    ];
    const wrap = el("div",{class:"tabs"});
    tabs.forEach(([id,label])=>{
      wrap.appendChild(el("div",{class:"tab"+(state.activeTab===id?" active":""), onclick:()=>{state.activeTab=id; state.view="main"; state.selectedTag=null; state.msg=""; state.invMsg=""; render();}},[label]));
    });
    return wrap;
  }

  function button(icon,label, onClick, hint, disabled){
    return el("div",{class:"btn"+(disabled?" disabled":""), onclick: disabled? null : onClick},[
      el("div",{class:"left"},[
        el("div",{class:"label"},[icon+" "+label]),
        hint ? el("div",{class:"hint"},[hint]) : null
      ]),
      el("div",{class:"muted"},["›"])
    ]);
  }

  function buildMainPanel(){
    const panel = el("div",{class:"section card"});
    if(state.view==="posts"){
      panel.appendChild(el("div",{class:"muted"},["Посты ", state.selectedTag ? ("#"+state.selectedTag) : ""]));
      if(state.loading) panel.appendChild(el("div",{class:"muted", style:"margin-top:10px;"},["Загрузка…"]));
      if(!state.loading && (!state.posts || state.posts.length===0)) panel.appendChild(el("div",{class:"muted", style:"margin-top:10px;"},["Постов с этим тегом пока нет."]));
      (state.posts||[]).forEach(p=>{
        const tags = (p.tags||[]).slice(0,6);
        panel.appendChild(el("div",{class:"post", onclick:()=>openLink(p.url)},[
          el("div",{class:"muted", style:"font-size:12px;"},["#", (tags[0]||"post"), " • ID ", String(p.message_id)]),
          el("div",{style:"margin-top:8px; font-size:14px; line-height:1.35;"},[p.preview||"Открыть пост →"]),
          el("div",{style:"margin-top:8px;"}, tags.map(t=>el("span",{class:"tagpill"},["#",t])))
        ]));
      });
      panel.appendChild(button("←","Назад", ()=>{state.view="main"; render();}, null, false));
      return panel;
    }

    if(state.view==="inventory"){
      const inv = state.inventory;
      const diorValue = inv ? Number(inv.dior_convert_value||0) : 0;
      panel.appendChild(el("div",{class:"row"},[
        el("div",{class:"muted"},["👜 Моя косметичка"]),
        el("div",{class:"muted", style:"cursor:pointer;", onclick:()=>{state.view="main"; render();}},["Назад"])
      ]));
      panel.appendChild(el("div",{style:"margin-top:12px;"},[
        el("div",{class:"card", style:"padding:12px; border-radius:18px; box-shadow:none; background:rgba(255,255,255,0.06);"},[
          el("div",{class:"muted"},["Баланс"]),
          el("div",{style:"margin-top:6px; font-size:16px; font-weight:750;"},["💎 ", String(state.user ? (state.user.points||0):0), " баллов"])
        ])
      ]));

      // Tickets convert left as before? Keep minimal text
      panel.appendChild(el("div",{class:"card", style:"margin-top:10px; padding:12px; border-radius:18px; box-shadow:none; background:rgba(255,255,255,0.06);"},[
        el("div",{class:"muted"},["🎁 Призы"]),
        (!inv || !inv.prizes || inv.prizes.length===0) ? el("div",{class:"muted", style:"margin-top:8px;"},["Пока нет призов."]) :
        el("div",{class:"grid", style:"margin-top:10px;"}, inv.prizes.map(p=>{
          const status = String(p.status||"");
          const box = el("div",{class:"card", style:"padding:12px; border-radius:16px; box-shadow:none; border:1px solid rgba(230,193,128,0.22); background:rgba(230,193,128,0.10);"},[
            el("div",{style:"font-size:14px; font-weight:800;"},[p.prize_label || "💎 Главный приз"]),
            el("div",{class:"muted", style:"margin-top:6px; font-size:12px;"},["Статус: ", prizeStatusLabel(status), " • Код: ", p.claim_code])
          ]);

          const actions = el("div",{class:"row", style:"margin-top:12px; gap:10px;"});
          // show two buttons ONLY for awaiting_choice
          if(status==="awaiting_choice"){
            actions.appendChild(el("div",{class:"smallbtn", onclick:()=>{
              const u = state.botUsername;
              if(u && p.claim_code){
                openLink("https://t.me/"+u+"?start=claim_"+p.claim_code);
              }else if(p.claim_code){
                alert("/claim "+p.claim_code);
              }
            }},["🎁 Забрать"]));
            actions.appendChild(el("div",{class:"smallbtn gold", onclick:()=>convertPrize(p.claim_code)},["💎 В бонусы (+", String(diorValue), ")"]));
          } else {
            // no actions after choosing "claim"
            actions.appendChild(el("div",{class:"smallbtn", style:"opacity:.7; cursor:default; text-align:center;"},["Действий нет"]));
          }

          box.appendChild(actions);
          return box;
        }))
      ]));

      if(state.invMsg){
        panel.appendChild(el("div",{class:"note"},[state.invMsg]));
      }
      return panel;
    }

    if(state.view==="profile"){
      const u = state.user;
      panel.appendChild(el("div",{class:"row"},[
        el("div",{class:"muted"},["👤 Профиль"]),
        el("div",{class:"muted", style:"cursor:pointer;", onclick:()=>{state.view="main"; render();}},["Закрыть"])
      ]));
      if(!u){
        panel.appendChild(el("div",{class:"muted", style:"margin-top:10px;"},["Профиль недоступен."]));
        return panel;
      }
      panel.appendChild(el("div",{class:"card", style:"margin-top:12px; padding:12px; border-radius:18px; box-shadow:none; background:rgba(255,255,255,0.08); position:relative;"},[
        el("div",{style:"position:absolute; top:10px; right:10px;"},[
          el("div",{class:"pill"},["💎 ", String(u.points||0)])
        ]),
        el("div",{class:"muted"},["Привет, ", (u.first_name||"друг"), "!"]),
        el("div",{class:"muted", style:"margin-top:6px;"},[tierLabel(u.tier)]),
        el("div",{class:"hr"},[]),
        el("div",{class:"row", style:"margin-top:10px;"},[
          el("div",{class:"muted"},["🔥 Стрик"]),
          el("div",{style:"font-weight:700;"},[String(u.daily_streak||0), " (best ", String(u.best_streak||0), ")"])
        ]),
        el("div",{class:"row", style:"margin-top:10px;"},[
          el("div",{class:"muted"},["🎟 Приглашено"]),
          el("div",{style:"font-weight:700;"},[String(u.referral_count||0)])
        ])
      ]));

      // referral link
      const ref = (state.botUsername && tgUserId) ? ("https://t.me/"+state.botUsername+"?start="+tgUserId) : "";
      panel.appendChild(el("div",{class:"hr"},[]));
      panel.appendChild(el("div",{style:"font-weight:650;"},["🎟 Рефералка"]));
      panel.appendChild(el("div",{class:"muted", style:"margin-top:8px;"},["За нового пользователя: +20 баллов (1 раз за каждого)."]));
      if(ref){
        panel.appendChild(el("div",{class:"note", style:"word-break:break-all;"},[ref]));
      }
      // raffle
      panel.appendChild(el("div",{class:"hr"},[]));
      panel.appendChild(el("div",{style:"font-weight:650;"},["🎁 Розыгрыши"]));
      panel.appendChild(el("div",{class:"muted", style:"margin-top:8px;"},["Билет = 500 баллов."]));
      panel.appendChild(el("div",{class:"muted", style:"margin-top:8px;"},["Твоих билетов: ", String(state.raffle ? (state.raffle.ticket_count||0) : 0)]));
      panel.appendChild(button("🎟","Купить билет (500)", buyTicket, state.busy ? "Подожди…" : "", state.busy || (u.points||0) < 500));

      // roulette
      panel.appendChild(el("div",{class:"hr"},[]));
      panel.appendChild(el("div",{style:"font-weight:650;"},["🎡 Рулетка"]));
      panel.appendChild(el("div",{class:"muted", style:"margin-top:8px;"},["1 спин = 2000 баллов."]));
      panel.appendChild(button("🎡","Крутить (2000)", spin, state.busy ? "Подожди…" : "", state.busy || (u.points||0) < 2000));

      // history
      panel.appendChild(el("div",{class:"hr"},[]));
      panel.appendChild(el("div",{style:"font-weight:650;"},["🧾 История рулетки"]));
      if(!state.history || state.history.length===0){
        panel.appendChild(el("div",{class:"muted", style:"margin-top:8px;"},["Пока пусто."]));
      }else{
        state.history.forEach(x=>{
          panel.appendChild(el("div",{class:"card", style:"margin-top:8px; padding:10px; border-radius:14px; box-shadow:none; background:rgba(255,255,255,0.08);"},[
            el("div",{class:"muted", style:"font-size:12px;"},[String(x.created_at||"")]),
            el("div",{style:"margin-top:4px; font-weight:650;"},[String(x.prize_label||"")])
          ]));
        });
      }

      if(state.msg) panel.appendChild(el("div",{class:"note"},[state.msg]));
      return panel;
    }

    // main content by tab
    const tab = state.activeTab;

    if(tab==="home"){
      panel.appendChild(button("📂","Категории", ()=>{state.activeTab="cat"; render();}));
      panel.appendChild(button("🏷","Бренды", ()=>{state.activeTab="brand"; render();}));
      panel.appendChild(button("💸","Sephora", ()=>{state.activeTab="sephora"; render();}));
      panel.appendChild(button("🧴","Продукт", ()=>{state.activeTab="ptype"; render();}));
      panel.appendChild(button("👜","Моя косметичка", openInventory, null, !tgUserId));
      panel.appendChild(button("👤","Профиль", openProfile, null, !tgUserId));
      panel.appendChild(button("↩️","В канал", ()=>openLink("https://t.me/"+CHANNEL)));
      return panel;
    }

    if(tab==="cat"){
      [
        ["🆕","Новинка","Новинка"],
        ["💎","Кратко о люкс продукте","Люкс"],
        ["🔥","Тренд","Тренд"],
        ["🏛","История бренда","История"],
        ["⭐","Личная оценка","Оценка"],
        ["🧾","Факты","Факты"],
        ["🧪","Составы продуктов","Состав"],
      ].forEach(([ic,lab,tag])=>panel.appendChild(button(ic,lab, ()=>openPosts(tag))));
      return panel;
    }

    if(tab==="brand"){
      const brands = [
        ["The Ordinary","TheOrdinary"],["Dior","Dior"],["Chanel","Chanel"],["Kylie Cosmetics","KylieCosmetics"],
        ["Gisou","Gisou"],["Rare Beauty","RareBeauty"],["Yves Saint Laurent","YSL"],["Givenchy","Givenchy"],
        ["Charlotte Tilbury","CharlotteTilbury"],["NARS","NARS"],["Sol de Janeiro","SolDeJaneiro"],["Huda Beauty","HudaBeauty"],
        ["Rhode","Rhode"],["Tower 28 Beauty","Tower28Beauty"],["Benefit Cosmetics","BenefitCosmetics"],["Estée Lauder","EsteeLauder"],
        ["Sisley","Sisley"],["Kérastase","Kerastase"],["Armani Beauty","ArmaniBeauty"],["Hourglass","Hourglass"],
        ["Shiseido","Shiseido"],["Tom Ford Beauty","TomFordBeauty"],["Tarte","Tarte"],["Sephora Collection","SephoraCollection"],
        ["Clinique","Clinique"],["Dolce & Gabbana","DolceGabbana"],["Kayali","Kayali"],["Guerlain","Guerlain"],
        ["Fenty Beauty","FentyBeauty"],["Too Faced","TooFaced"],["MAKE UP FOR EVER","MakeUpForEver"],["Erborian","Erborian"],
        ["Natasha Denona","NatashaDenona"],["Lancôme","Lancome"],["Kosas","Kosas"],["ONE/SIZE","OneSize"],
        ["Laneige","Laneige"],["Makeup by Mario","MakeupByMario"],["Valentino Beauty","ValentinoBeauty"],["Drunk Elephant","DrunkElephant"],
        ["Olaplex","Olaplex"],["Anastasia Beverly Hills","AnastasiaBeverlyHills"],["Amika","Amika"],["BYOMA","BYOMA"],
        ["Glow Recipe","GlowRecipe"],["Milk Makeup","MilkMakeup"],["Summer Fridays","SummerFridays"],["K18","K18"]
      ];
      brands.forEach(([lab,tag])=>panel.appendChild(button("✨", lab, ()=>openPosts(tag))));
      return panel;
    }

    if(tab==="sephora"){
      panel.appendChild(button("🎁","Подарки / акции", ()=>openPosts("SephoraPromo")));
      return panel;
    }

    if(tab==="ptype"){
      [
        ["Праймер","Праймер"],["Тональная основа","ТональнаяОснова"],["Консилер","Консилер"],["Пудра","Пудра"],
        ["Румяна","Румяна"],["Скульптор","Скульптор"],["Бронзер","Бронзер"],["Продукт для бровей","ПродуктДляБровей"],
        ["Хайлайтер","Хайлайтер"],["Тушь","Тушь"],["Тени","Тени"],["Помада","Помада"],["Карандаш для губ","КарандашДляГуб"],
        ["Палетка","Палетка"],["Фиксатор","Фиксатор"]
      ].forEach(([lab,tag])=>panel.appendChild(button("🧴", lab, ()=>openPosts(tag))));
      return panel;
    }

    return panel;
  }

  function render(){
    const root = document.getElementById("app");
    root.innerHTML="";
    root.appendChild(buildHero());
    root.appendChild(buildTabs());
    root.appendChild(buildMainPanel());
    root.appendChild(el("div",{class:"muted center", style:"margin-top:20px; font-size:12px;"},["Открывается как Mini App внутри Telegram"]));
  }

  async function init(){
    render();
    if(!tgUserId) return;
    await Promise.all([loadBotUsername(), refreshUser()]);
    render();
  }

  init();
})();
</script>
</body>
</html>
"""
    return (
        html.replace("__CHANNEL__", CHANNEL_USERNAME)
        .replace("__BOT_USERNAME__", BOT_USERNAME)
    )


# -----------------------------------------------------------------------------
# FASTAPI LIFESPAN
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app_: FastAPI):
    global sweeper_task
    await init_db()
    await start_telegram_bot()
    sweeper_task = asyncio.create_task(sweeper_loop())
    logger.info("✅ NS · Natural Sense started")
    try:
        yield
    finally:
        if sweeper_task:
            sweeper_task.cancel()
            try:
                await sweeper_task
            except Exception:
                pass
        await stop_telegram_bot()
        logger.info("✅ NS · Natural Sense stopped")


# -----------------------------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="NS · Natural Sense API", version="FINAL", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"app": "NS · Natural Sense", "status": "running", "version": "FINAL"}


@app.get("/webapp", response_class=HTMLResponse)
async def webapp():
    return HTMLResponse(get_webapp_html())


@app.get("/api/user/{telegram_id}")
async def get_user_api(telegram_id: int):
    user = await get_user(int(telegram_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "telegram_id": int(user.telegram_id),
        "username": user.username,
        "first_name": user.first_name,
        "tier": user.tier,
        "points": user.points,
        "favorites": user.favorites,
        "joined_at": user.joined_at.isoformat() if user.joined_at else None,
        "daily_streak": user.daily_streak or 0,
        "best_streak": user.best_streak or 0,
        "referral_count": user.referral_count or 0,
        "last_daily_bonus_at": user.last_daily_bonus_at.isoformat() if user.last_daily_bonus_at else None,
    }


@app.get("/api/posts")
async def api_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    if not tag:
        return []

    tag = (tag or "").strip()
    if tag in BLOCKED_TAGS:
        return []

    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    rows = await list_posts(tag=tag, limit=limit, offset=offset)
    out = []
    for p in rows:
        out.append({
            "message_id": int(p.message_id),
            "url": p.permalink or make_permalink(int(p.message_id)),
            "tags": p.tags or [],
            "preview": preview_text(p.text),
        })
    return out


@app.get("/api/bot/username")
async def api_bot_username():
    """
    Возвращает username бота для формирования реф-ссылки в Mini App.
    Предпочитаем переменную окружения BOT_USERNAME (стабильно).
    Если её нет — пробуем получить через Telegram API (getMe), если бот запущен.
    """
    if BOT_USERNAME:
        return {"bot_username": BOT_USERNAME}
    if tg_app and BOT_TOKEN:
        try:
            me = await tg_app.bot.get_me()
            return {"bot_username": (me.username or "")}
        except Exception:
            return {"bot_username": ""}
    return {"bot_username": ""}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# -----------------------------------------------------------------------------
# RAFFLE + ROULETTE API
# -----------------------------------------------------------------------------
class BuyTicketReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    qty: int = Field(1, ge=1, le=50)


class BuyTicketResp(BaseModel):
    telegram_id: int
    points: int
    ticket_count: int


class RaffleStatusResp(BaseModel):
    raffle_id: int
    title: str
    is_active: bool
    ends_at: str | None
    ticket_count: int
    ticket_cost: int


class SpinReq(BaseModel):
    telegram_id: int = Field(..., ge=1)


class SpinResp(BaseModel):
    telegram_id: int
    points: int
    prize_type: str
    prize_value: int
    prize_label: str
    roll: int
    claimable: bool = False
    claim_code: Optional[str] = None


class InventoryResp(BaseModel):
    telegram_id: int
    points: int
    ticket_count: int
    ticket_convert_rate: int
    dior_convert_value: int
    prizes: list[dict[str, Any]]


class ConvertTicketsReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    qty: int = Field(..., ge=1, le=10_000)


class ConvertTicketsResp(BaseModel):
    telegram_id: int
    points: int
    ticket_count: int
    converted_qty: int
    added_points: int


class ConvertPrizeReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    claim_code: str = Field(..., min_length=3, max_length=64)


class ConvertPrizeResp(BaseModel):
    telegram_id: int
    points: int
    claim_code: str
    added_points: int


def pick_roulette_prize(roll: int) -> dict[str, Any]:
    acc = 0
    for item in ROULETTE_DISTRIBUTION:
        acc += int(item["weight"])
        if roll < acc:
            return item
    return ROULETTE_DISTRIBUTION[-1]


async def get_ticket_row(session: AsyncSession, telegram_id: int, raffle_id: int) -> RaffleTicket:
    row = (
        await session.execute(
            select(RaffleTicket).where(
                RaffleTicket.telegram_id == telegram_id,
                RaffleTicket.raffle_id == raffle_id,
            )
        )
    ).scalar_one_or_none()
    if row:
        return row
    row = RaffleTicket(raffle_id=raffle_id, telegram_id=telegram_id, count=0, updated_at=datetime.utcnow())
    session.add(row)
    await session.flush()
    return row


@app.get("/api/raffle/status", response_model=RaffleStatusResp)
async def raffle_status(telegram_id: int):
    async with async_session_maker() as session:
        raffle = (await session.execute(select(Raffle).where(Raffle.id == DEFAULT_RAFFLE_ID))).scalar_one()
        t = (
            await session.execute(
                select(RaffleTicket.count).where(
                    RaffleTicket.telegram_id == int(telegram_id),
                    RaffleTicket.raffle_id == raffle.id,
                )
            )
        ).scalar_one_or_none()
        return {
            "raffle_id": raffle.id,
            "title": raffle.title,
            "is_active": bool(raffle.is_active),
            "ends_at": raffle.ends_at.isoformat() if raffle.ends_at else None,
            "ticket_count": int(t or 0),
            "ticket_cost": RAFFLE_TICKET_COST,
        }


@app.post("/api/raffle/buy_ticket", response_model=BuyTicketResp)
async def raffle_buy_ticket(req: BuyTicketReq):
    tid = int(req.telegram_id)
    qty = max(1, int(req.qty))
    cost = RAFFLE_TICKET_COST * qty

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(select(User).where(User.telegram_id == tid))
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            if int(user.points or 0) < cost:
                raise HTTPException(status_code=400, detail=f"Недостаточно баллов. Нужно {cost}")

            raffle = (
                await session.execute(select(Raffle).where(Raffle.id == DEFAULT_RAFFLE_ID))
            ).scalar_one()
            if not raffle.is_active:
                raise HTTPException(status_code=400, detail="Розыгрыш сейчас недоступен")

            # списываем баллы
            user.points = int(user.points or 0) - cost
            _recalc_tier(user)

            # увеличиваем билеты
            ticket_row = await get_ticket_row(session, tid, raffle.id)
            ticket_row.count = int(ticket_row.count or 0) + qty
            ticket_row.updated_at = datetime.utcnow()

            session.add(
                PointTransaction(
                    telegram_id=tid,
                    type="raffle_ticket",
                    delta=-cost,
                    meta={"qty": qty, "raffle_id": raffle.id},
                )
            )

            points_now = int(user.points or 0)
            tickets_now = int(ticket_row.count or 0)

    return {"telegram_id": tid, "points": points_now, "ticket_count": tickets_now}


# -----------------------------------------------------------------------------
# INVENTORY API
# -----------------------------------------------------------------------------
@app.get("/api/inventory", response_model=InventoryResp)
async def inventory_api(telegram_id: int):
    tid = int(telegram_id)
    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        raffle = (await session.execute(select(Raffle).where(Raffle.id == DEFAULT_RAFFLE_ID))).scalar_one()
        t = (
            await session.execute(
                select(RaffleTicket.count).where(
                    RaffleTicket.telegram_id == tid,
                    RaffleTicket.raffle_id == raffle.id,
                )
            )
        ).scalar_one_or_none()

        claims = (
            await session.execute(
                select(RouletteClaim)
                .where(RouletteClaim.telegram_id == tid)
                .order_by(RouletteClaim.created_at.desc())
                .limit(50)
            )
        ).scalars().all()

    prizes: list[dict[str, Any]] = []
    for c in claims:
        # показываем только "физические" призы, которые ещё актуальны
        if (c.prize_type or "") != "physical_dior_palette":
            continue
        if (c.status or "") == "closed":
            continue
        prizes.append(
            {
                "claim_code": c.claim_code,
                "prize_type": c.prize_type,
                "prize_label": c.prize_label,
                "status": c.status,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
        )

    return {
        "telegram_id": tid,
        "points": int(user.points or 0),
        "ticket_count": int(t or 0),
        "ticket_convert_rate": int(TICKET_CONVERT_RATE),
        "dior_convert_value": int(DIOR_PALETTE_CONVERT_VALUE),
        "prizes": prizes,
    }


@app.post("/api/inventory/convert_ticket", response_model=ConvertTicketsResp)
async def inventory_convert_ticket(req: ConvertTicketsReq):
    tid = int(req.telegram_id)
    qty = max(1, int(req.qty))
    added = qty * int(TICKET_CONVERT_RATE)

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(
                    select(User).where(User.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            ticket_row = await get_ticket_row(session, tid, DEFAULT_RAFFLE_ID)
            have = int(ticket_row.count or 0)
            if qty > have:
                raise HTTPException(status_code=400, detail=f"Недостаточно билетов. Есть {have}")

            ticket_row.count = have - qty
            ticket_row.updated_at = datetime.utcnow()

            user.points = int(user.points or 0) + added
            _recalc_tier(user)

            session.add(
                PointTransaction(
                    telegram_id=tid,
                    type="ticket_convert",
                    delta=added,
                    meta={"qty": qty, "rate": int(TICKET_CONVERT_RATE)},
                )
            )

            points_now = int(user.points or 0)
            tickets_now = int(ticket_row.count or 0)

    return {
        "telegram_id": tid,
        "points": points_now,
        "ticket_count": tickets_now,
        "converted_qty": qty,
        "added_points": added,
    }


@app.post("/api/inventory/convert_prize", response_model=ConvertPrizeResp)
async def inventory_convert_prize(req: ConvertPrizeReq):
    tid = int(req.telegram_id)
    code = (req.claim_code or "").strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="claim_code required")

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(
                    select(User).where(User.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            claim = (
                await session.execute(
                    select(RouletteClaim)
                    .where(RouletteClaim.claim_code == code)
                    .with_for_update()
                )
            ).scalar_one_or_none()

            if not claim:
                raise HTTPException(status_code=404, detail="Код не найден")
            if int(claim.telegram_id) != tid:
                raise HTTPException(status_code=403, detail="Этот код принадлежит другому пользователю")
            if (claim.prize_type or "") != "physical_dior_palette":
                raise HTTPException(status_code=400, detail="Этот приз нельзя конвертировать")
            st = (claim.status or "").strip()
            if st == "closed":
                raise HTTPException(status_code=400, detail="Этот приз уже закрыт")
            if st != "awaiting_choice":
                raise HTTPException(status_code=400, detail="Нельзя конвертировать: вы уже нажали «Забрать» или заявка отправлена")

            added = int(DIOR_PALETTE_CONVERT_VALUE)

            # Закрываем заявку как "конвертировано"
            claim.status = "closed"
            note = "CONVERTED_TO_POINTS"
            if claim.contact_text:
                if note not in claim.contact_text:
                    claim.contact_text = (claim.contact_text + "\\n" + note).strip()
            else:
                claim.contact_text = note
            claim.updated_at = datetime.utcnow()

            user.points = int(user.points or 0) + added
            _recalc_tier(user)

            session.add(
                PointTransaction(
                    telegram_id=tid,
                    type="prize_convert",
                    delta=added,
                    meta={"claim_code": code, "value": added},
                )
            )

            points_now = int(user.points or 0)

    return {"telegram_id": tid, "points": points_now, "claim_code": code, "added_points": int(DIOR_PALETTE_CONVERT_VALUE)}
@app.get("/api/roulette/history")
async def roulette_history(telegram_id: int, limit: int = 5):
    limit = max(1, min(int(limit), 20))
    tid = int(telegram_id)
    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(RouletteSpin)
                .where(RouletteSpin.telegram_id == tid)
                .order_by(RouletteSpin.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    out = []
    for r in rows:
        out.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "prize_label": r.prize_label,
                "prize_type": r.prize_type,
                "prize_value": r.prize_value,
                "roll": r.roll,
            }
        )
    return out


@app.post("/api/roulette/spin", response_model=SpinResp)
async def roulette_spin(req: SpinReq):
    tid = int(req.telegram_id)
    now = datetime.utcnow()

    claim_code: str | None = None

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(
                    select(User).where(User.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            if (user.points or 0) < ROULETTE_SPIN_COST:
                raise HTTPException(status_code=400, detail=f"Недостаточно баллов. Нужно {ROULETTE_SPIN_COST}")

            last_spin = (
                await session.execute(
                    select(RouletteSpin.created_at)
                    .where(RouletteSpin.telegram_id == tid)
                    .order_by(RouletteSpin.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if last_spin and (now - last_spin) < ROULETTE_LIMIT_WINDOW:
                delta = ROULETTE_LIMIT_WINDOW - (now - last_spin)
                secs_left = max(1, int(delta.total_seconds()) + (1 if (delta.total_seconds() % 1) > 0 else 0))
                raise HTTPException(status_code=400, detail=f"Рулетка доступна через ~{secs_left} сек")

            # списание
            user.points = (user.points or 0) - ROULETTE_SPIN_COST
            session.add(PointTransaction(telegram_id=tid, type="roulette_spin", delta=-ROULETTE_SPIN_COST, meta={}))

            roll = secrets.randbelow(1_000_000)
            prize = pick_roulette_prize(roll)
            prize_type: PrizeType = prize["type"]
            prize_value = int(prize["value"])
            prize_label = str(prize["label"])

            # выдача приза
            if prize_type == "points":
                user.points = (user.points or 0) + prize_value
                session.add(PointTransaction(telegram_id=tid, type="roulette_prize", delta=prize_value, meta={"roll": roll, "prize": prize_label}))
            elif prize_type == "raffle_ticket":
                ticket_row = await get_ticket_row(session, tid, DEFAULT_RAFFLE_ID)
                ticket_row.count = int(ticket_row.count or 0) + prize_value
                ticket_row.updated_at = now
                session.add(PointTransaction(telegram_id=tid, type="roulette_prize", delta=0, meta={"roll": roll, "prize": "raffle_ticket", "qty": prize_value}))
            else:
                # физический приз - только лог + уведомление админу
                session.add(PointTransaction(telegram_id=tid, type="roulette_prize", delta=0, meta={"roll": roll, "prize": "physical_dior_palette"}))

            _recalc_tier(user)

            spin_row = RouletteSpin(
                telegram_id=tid,
                created_at=now,
                cost_points=ROULETTE_SPIN_COST,
                roll=roll,
                prize_type=prize_type,
                prize_value=prize_value,
                prize_label=prize_label,
            )
            if prize_type == "physical_dior_palette":
                # создаём заявку на получение приза
                claim_code = generate_claim_code()
                session.add(
                    RouletteClaim(
                        claim_code=claim_code,
                        telegram_id=tid,
                        spin_id=None,  # id будет после commit, связь не критична
                        prize_type=prize_type,
                        prize_label=prize_label,
                        status="awaiting_choice",
                    )
                )

        await session.refresh(user)

    if prize_type == "physical_dior_palette":
        # ✅ сообщаем победителю в чат сразу (чтобы не потерял выигрыш)
        if claim_code:
            await notify_user_top_prize(tid, prize_label, claim_code)

        uname = (user.username or "").strip()
        mention = f"@{uname}" if uname else "(без username)"
        await notify_admin(
            "💎 ГЛАВНЫЙ ПРИЗ!\n"
            f"user: {mention} | {user.first_name or '-'}\n"
            f"telegram_id: {tid}\n"
            f"link: {tg_user_link(tid)}\n"
            f"claim: {claim_code}\n"
            f"roll: {roll}\n"
            "👉 Пользователю: отправьте /claim <код> и потом сообщение с контактами/адресом."
        )

    return {
        "telegram_id": tid,
        "points": int(user.points or 0),
        "prize_type": prize_type,
        "prize_value": prize_value,
        "prize_label": prize_label,
        "roll": int(roll),
        "claimable": bool(prize_type == "physical_dior_palette"),
        "claim_code": claim_code,
    }
