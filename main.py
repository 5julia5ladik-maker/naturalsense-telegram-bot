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

# Максимум баллов в день за "активности" (просмотры/голоса/челленджи/квесты/оценки/избранное/комменты)
DAILY_ACTIVITY_CAP = 300

STREAK_MILESTONES = {
    3: 10,
    7: 30,
    14: 80,
    30: 250,
}

RAFFLE_TICKET_COST = 500
ROULETTE_SPIN_COST = 2000
ROULETTE_LIMIT_WINDOW = timedelta(hours=3)  # cooldown между спинами
DEFAULT_RAFFLE_ID = 1

PrizeType = Literal["points", "raffle_ticket", "physical_dior_palette"]

# per 1_000_000
ROULETTE_DISTRIBUTION: list[dict[str, Any]] = [
    {"weight": 415_600, "type": "points", "value": 500, "label": "+500 баллов"},
    {"weight": 290_900, "type": "points", "value": 1000, "label": "+1000 баллов"},
    {"weight": 124_700, "type": "points", "value": 1500, "label": "+1500 баллов"},
    {"weight": 83_100, "type": "points", "value": 2000, "label": "+2000 баллов (камбэк)"},
    {"weight": 41_600, "type": "raffle_ticket", "value": 1, "label": "🎟 Билет на розыгрыш"},
    {"weight": 29_100, "type": "points", "value": 3000, "label": "+3000 баллов"},
    {"weight": 15_000, "type": "physical_dior_palette", "value": 1, "label": "💎 Палетка Dior (ТОП приз)"},
]
ROULETTE_TOTAL = sum(x["weight"] for x in ROULETTE_DISTRIBUTION)
if ROULETTE_TOTAL != 1_000_000:
    raise RuntimeError("ROULETTE_DISTRIBUTION must sum to 1_000_000")



class UserDailyStat(Base):
    __tablename__ = "user_daily_stats"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    day = Column(String, index=True, nullable=False)  # YYYY-MM-DD in UTC
    earned_activity_points = Column(Integer, default=0, nullable=False)
    counters = Column(JSON, default=dict)  # flexible per-day counters
    updated_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)

Index("ix_user_daily_stats_unique", UserDailyStat.telegram_id, UserDailyStat.day, unique=True)


class ActivityEvent(Base):
    __tablename__ = "activity_events"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    day = Column(String, index=True, nullable=False)  # YYYY-MM-DD
    kind = Column(String, index=True, nullable=False)  # view/vote/challenge/quest/favorite/rating/comment...
    key = Column(String, nullable=False)  # dedupe key (e.g., post_id/poll_id/brand/product/hash)
    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)

Index("ix_activity_events_unique", ActivityEvent.telegram_id, ActivityEvent.day, ActivityEvent.kind, ActivityEvent.key, unique=True)



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
    ref_active_bonus_paid = Column(Boolean, default=False, nullable=False)


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

    status = Column(String, default="awaiting_contact", nullable=False)  # awaiting_contact|submitted|closed
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
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS ref_active_bonus_paid BOOLEAN NOT NULL DEFAULT FALSE;")

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
3) *👤 Профиль* — баллы, уровень, стрик, реф-ссылка.
4) *↩️ В канал* — кнопка под сообщением /start.

💎 *Как получать баллы*
• Ежедневный визит: +5 (строго 1 раз в 24 часа)
• 3 комментария в день: +10 (текст от 25 символов)
• Просмотр постов из Mini App: +1 (до 15/день)
• Голосования: +3 (до 5/день)
• Challenge дня: +5 (1/день)
• Квест дня: +10 (1/день)
• Избранное: +2 (до 5/день)
• Оценки: +3 / +5 с текстом
• Реферал: +20 +10 за активность реферала

🧱 *Ограничение*
• За активности максимум: 300 баллов в день

🔥 *Стрик (серия дней)*
• 3 дня: +10
• 7 дней: +30
• 14 дней: +80
• 30 дней: +250

🎡 *Рулетка*
• 1 спин = 2000 баллов
• Лимит: 1 раз в 3 часа
• Главный приз: палетка Dior (до 4000 грн)

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
    """Общая логика для /claim и deep-link /start claim_CODE"""
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

        if (claim.status or "") == "submitted":
            await update.message.reply_text("✅ Заявка уже отправлена. Мы скоро свяжемся.")
            return

        # помечаем как ожидание контакта и обновляем время
        claim.status = "awaiting_contact"
        claim.updated_at = datetime.utcnow()
        await session.commit()

    await update.message.reply_text(
        "🎁 Заявка на приз принята!\n\n"
        "Напишите одним сообщением удобный способ связи (Telegram/WhatsApp) и адрес/город доставки.\n"
        f"Код заявки: {code}"
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

async def notify_admin(text: str) -> None:
    if not tg_app or not BOT_TOKEN or not ADMIN_CHAT_ID:
        logger.info("ADMIN ALERT (no bot): %s", text)
        return
    try:
        await tg_app.bot.send_message(chat_id=ADMIN_CHAT_ID, text=text)
    except Exception as e:
        logger.warning("Failed to notify admin: %s", e)

async def notify_user(telegram_id: int, text: str) -> None:
    if not tg_app or not BOT_TOKEN:
        logger.info("USER MSG (no bot) to %s: %s", telegram_id, text)
        return
    try:
        await tg_app.bot.send_message(chat_id=telegram_id, text=text)
    except Exception as e:
        logger.warning("Failed to notify user %s: %s", telegram_id, e)



# -----------------------------------------------------------------------------
# MINI APP (WEBAPP HTML)
# -----------------------------------------------------------------------------
def get_webapp_html() -> str:
    html = r"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>NS · Natural Sense</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    :root {
      --bg: #0c0f14;
      --card: rgba(255,255,255,0.08);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.60);
      --gold: rgba(230, 193, 128, 0.9);
      --stroke: rgba(255,255,255,0.10);
      --sheetOverlay: rgba(12,15,20,0.55);
      --sheetCardBg: rgba(255,255,255,0.10);
      --glassStroke: rgba(255,255,255,0.16);
      --glassShadow: rgba(0,0,0,0.45);
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: radial-gradient(1200px 800px at 20% 10%, rgba(230,193,128,0.18), transparent 60%),
                  var(--bg);
      color: var(--text);
      overflow-x: hidden;
    }
    #root { min-height: 100vh; }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect, useMemo } = React;
    const tg = window.Telegram?.WebApp;

    const DEFAULT_BG = "#0c0f14";

    const hexToRgba = (hex, a) => {
      if (!hex) return `rgba(12,15,20,${a})`;
      let h = String(hex).trim();
      if (h[0] === "#") h = h.slice(1);
      if (h.length === 3) h = h.split("").map((c) => c + c).join("");
      if (h.length !== 6) return `rgba(12,15,20,${a})`;
      const r = parseInt(h.slice(0, 2), 16);
      const g = parseInt(h.slice(2, 4), 16);
      const b = parseInt(h.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${a})`;
    };

    const setVar = (k, v) => {
      document.documentElement.style.setProperty(k, v);
    };

    const applyTelegramTheme = () => {
      const scheme = tg?.colorScheme || "dark";
      const p = tg?.themeParams || {};

      const bg = p.bg_color || DEFAULT_BG;
      const text = p.text_color || (scheme === "dark" ? "rgba(255,255,255,0.92)" : "rgba(17,17,17,0.92)");
      const muted = p.hint_color || (scheme === "dark" ? "rgba(255,255,255,0.60)" : "rgba(0,0,0,0.55)");

      // базовые токены
      setVar("--bg", bg);
      setVar("--text", text);
      setVar("--muted", muted);
      setVar("--stroke", scheme === "dark" ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.10)");
      setVar("--card", scheme === "dark" ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.72)");

      // iOS glass (Sheet)
      setVar("--sheetOverlay", scheme === "dark" ? hexToRgba(bg, 0.55) : hexToRgba(bg, 0.45));
      setVar("--sheetCardBg", scheme === "dark" ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.80)");
      setVar("--glassStroke", scheme === "dark" ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.10)");
      setVar("--glassShadow", scheme === "dark" ? "rgba(0,0,0,0.45)" : "rgba(0,0,0,0.18)");

      // алиасы, которые использует locked-popup
      setVar("--overlayBg", scheme === "dark" ? hexToRgba(bg, 0.55) : hexToRgba(bg, 0.45));
      setVar("--glassBg", scheme === "dark" ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.80)");
      setVar("--accent", p.button_color || (scheme === "dark" ? "#5aa7ff" : "#1b74ff"));

      if (tg) {
        tg.setHeaderColor(bg);
        tg.setBackgroundColor(bg);
      }
    };

    // iOS-style "locked" modal inside WebApp (не закрывается по тапу вокруг)
    const showLockedPopup = ({ title, message, primaryText, onPrimary, okText = "OK" }) => {
      try {
        // не плодим несколько окон
        const existing = document.getElementById("ns_locked_popup");
        if (existing) existing.remove();

        const overlay = document.createElement("div");
        overlay.id = "ns_locked_popup";
        overlay.style.position = "fixed";
        overlay.style.inset = "0";
        overlay.style.zIndex = "99999";
        overlay.style.display = "flex";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
        overlay.style.padding = "20px";
        overlay.style.background = "var(--overlayBg)";
        overlay.style.backdropFilter = "blur(22px) saturate(180%)";
        overlay.style.webkitBackdropFilter = "blur(22px) saturate(180%)";

        // блокируем закрытие кликом по фону (но не ломаем клики по кнопкам)
        overlay.addEventListener("click", (e) => {
          if (e.target === overlay) {
            e.preventDefault();
            e.stopPropagation();
          }
        });

        const card = document.createElement("div");
        card.style.width = "100%";
        card.style.maxWidth = "520px";
        card.style.borderRadius = "18px";
        card.style.padding = "18px 16px 14px";
        card.style.background = "var(--glassBg)";
        card.style.border = "1px solid var(--glassStroke)";
        card.style.boxShadow = `0 18px 60px var(--glassShadow)`;
        card.style.color = "var(--text)";
        card.style.fontFamily = "system-ui, -apple-system, Segoe UI, Roboto, Arial";
        card.addEventListener("click", (e) => { e.stopPropagation(); });

        const h = document.createElement("div");
        h.textContent = title || "";
        h.style.fontSize = "18px";
        h.style.fontWeight = "700";
        h.style.marginBottom = "10px";

        const p = document.createElement("div");
        p.textContent = message || "";
        p.style.whiteSpace = "pre-wrap";
        p.style.lineHeight = "1.4";
        p.style.fontSize = "15px";
        p.style.opacity = "0.95";

        const btnRow = document.createElement("div");
        btnRow.style.display = "flex";
        btnRow.style.gap = "14px";
        btnRow.style.justifyContent = "flex-end";
        btnRow.style.marginTop = "16px";

        const ok = document.createElement("button");
        ok.textContent = okText;
        ok.style.border = "none";
        ok.style.background = "transparent";
        ok.style.color = "var(--muted)";
        ok.style.fontSize = "16px";
        ok.style.padding = "10px 12px";
        ok.style.cursor = "pointer";

        const primary = document.createElement("button");
        primary.textContent = primaryText || "";
        primary.style.border = "none";
        primary.style.background = "transparent";
        primary.style.color = "var(--accent)";
        primary.style.fontSize = "16px";
        primary.style.padding = "10px 12px";
        primary.style.cursor = "pointer";

        const close = () => {
          try { document.body.style.overflow = ""; } catch (e) {}
          overlay.remove();
        };

        ok.onclick = () => close();
        primary.onclick = () => { try { onPrimary && onPrimary(); } catch (e) {} close(); };

        btnRow.appendChild(ok);
        if (primaryText) btnRow.appendChild(primary);

        card.appendChild(h);
        card.appendChild(p);
        card.appendChild(btnRow);
        overlay.appendChild(card);

        document.body.style.overflow = "hidden";
        document.body.appendChild(overlay);
      } catch (e) {
        // fallback
        try { alert(message || ""); } catch (e2) {}
      }
    };



    if (tg) {
      tg.expand();
      applyTelegramTheme();
      tg.onEvent("themeChanged", applyTelegramTheme);
    }

    const CHANNEL = "__CHANNEL__";
    const BOT_USERNAME = "__BOT_USERNAME__"; // может быть пустым, если не задана переменная окружения


    const openLink = (url) => {
      if (tg?.openTelegramLink) tg.openTelegramLink(url);
      else window.open(url, "_blank");
    };

    const tierLabel = (tier) => (
      { free: "🥉 Bronze", premium: "🥈 Silver", vip: "🥇 Gold VIP" }[tier] || "🥉 Bronze"
    );

    const Hero = ({ user, onOpenProfile }) => (
      <div
        onClick={onOpenProfile}
        style={{
          border: "1px solid var(--stroke)",
          background: "linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05))",
          borderRadius: "22px",
          padding: "16px 14px",
          boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
          position: "relative",
          overflow: "hidden",
          cursor: user ? "pointer" : "default"
        }}
      >
        <div style={{
          position: "absolute", inset: "-2px",
          background: "radial-gradient(600px 300px at 10% 0%, rgba(230,193,128,0.26), transparent 60%)",
          pointerEvents: "none"
        }} />
        <div style={{ position: "relative" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <div>
              <div style={{ fontSize: "20px", fontWeight: 650, letterSpacing: "0.2px" }}>NS · Natural Sense</div>
              <div style={{ marginTop: "6px", fontSize: "13px", color: "var(--muted)" }}>luxury beauty magazine</div>
            </div>
            {user && (
              <div style={{ fontSize:"14px", color:"var(--muted)", display:"flex", gap:"6px", alignItems:"center" }}>
                Профиль <span style={{ opacity:0.8 }}>›</span>
              </div>
            )}
          </div>

          {user && (
            <div style={{
              marginTop: "14px",
              padding: "12px",
              background: "rgba(230, 193, 128, 0.1)",
              borderRadius: "14px",
              border: "1px solid rgba(230, 193, 128, 0.2)"
            }}>
              <div style={{ fontSize: "13px", color: "var(--muted)" }}>Привет, {user.first_name}!</div>
              <div style={{ fontSize: "16px", fontWeight: 600, marginTop: "4px" }}>
                💎 {user.points} баллов • {tierLabel(user.tier)}
              </div>
              <div style={{ marginTop:"6px", fontSize:"12px", color:"var(--muted)" }}>
                Нажми, чтобы открыть профиль и бонусы
              </div>
            </div>
          )}
        </div>
      </div>
    );

    const Tabs = ({ active, onChange }) => {
      const tabs = [
        { id: "home", label: "Главное" },
        { id: "cat", label: "Категории" },
        { id: "brand", label: "Бренды" },
        { id: "sephora", label: "Sephora" },
        { id: "ptype", label: "Продукт" },
      ];
      return (
        <div style={{ display: "flex", gap: "8px", marginTop: "14px" }}>
          {tabs.map(tab => (
            <div
              key={tab.id}
              onClick={() => onChange(tab.id)}
              style={{
                flex: 1,
                border: active === tab.id ? "1px solid rgba(230,193,128,0.40)" : "1px solid var(--stroke)",
                background: active === tab.id ? "rgba(230,193,128,0.12)" : "rgba(255,255,255,0.06)",
                color: active === tab.id ? "rgba(255,255,255,0.95)" : "var(--text)",
                padding: "10px",
                borderRadius: "14px",
                fontSize: "13px",
                textAlign: "center",
                cursor: "pointer",
                userSelect: "none",
                transition: "all 0.2s"
              }}
            >
              {tab.label}
            </div>
          ))}
        </div>
      );
    };

    const Button = ({ icon, label, onClick, subtitle, disabled }) => (
      <div
        onClick={disabled ? undefined : onClick}
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "14px",
          borderRadius: "18px",
          border: "1px solid var(--stroke)",
          background: "rgba(255,255,255,0.06)",
          color: "var(--text)",
          fontSize: "15px",
          margin: "10px 0",
          cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? 0.5 : 1
        }}
      >
        <div>
          <div>{icon} {label}</div>
          {subtitle && <div style={{ fontSize:"12px", color:"var(--muted)", marginTop:"4px" }}>{subtitle}</div>}
        </div>
        <span style={{ opacity: 0.8 }}>›</span>
      </div>
    );

    const Panel = ({ children }) => (
      <div style={{
        marginTop: "14px",
        border: "1px solid var(--stroke)",
        background: "rgba(255,255,255,0.05)",
        borderRadius: "22px",
        padding: "12px"
      }}>
        {children}
      </div>
    );

    const PostCard = ({ post }) => (
      <div
        onClick={() => { trackView(post.message_id); openLink(post.url); }}
        style={{
          marginTop: "10px",
          padding: "12px",
          borderRadius: "18px",
          border: "1px solid var(--stroke)",
          background: "rgba(255,255,255,0.06)",
          cursor: "pointer"
        }}
      >
        <div style={{ fontSize:"12px", color:"var(--muted)" }}>
          {"#" + (post.tags?.[0] || "post")} • ID {post.message_id}
        </div>
        <div style={{ marginTop:"8px", fontSize:"14px", lineHeight:"1.35" }}>
          {post.preview || "Открыть пост →"}
        </div>
        <div style={{ marginTop:"8px", display:"flex", gap:"6px", flexWrap:"wrap" }}>
          {(post.tags || []).slice(0,6).map(t => (
            <div key={t} style={{
              fontSize:"12px",
              padding:"5px 8px",
              borderRadius:"999px",
              border:"1px solid var(--stroke)",
              background:"rgba(255,255,255,0.08)"
            }}>#{t}</div>
          ))}
        </div>
      </div>
    );

    const Sheet = ({ open, onClose, children }) => {
      if (!open) return null;
      return (
        <div
          onClick={onClose}
          style={{
            position:"fixed",
            inset:0,
            background:"var(--sheetOverlay)",
            backdropFilter:"blur(22px) saturate(180%)",
            WebkitBackdropFilter:"blur(22px) saturate(180%)",
            zIndex:9999,
            display:"flex",
            justifyContent:"center",
            alignItems:"flex-end",
            padding:"10px"
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width:"100%",
              maxWidth:"520px",
              borderRadius:"22px 22px 18px 18px",
              border:"1px solid var(--glassStroke)",
              background:"var(--sheetCardBg)",
              backdropFilter:"blur(28px) saturate(180%)",
              WebkitBackdropFilter:"blur(28px) saturate(180%)",
              boxShadow:"0 12px 40px var(--glassShadow)",
              padding:"14px 14px 10px",
              maxHeight:"82vh",
              overflow:"auto"
            }}
          >
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
              <div style={{ fontSize:"16px", fontWeight:650 }}>👤 Профиль</div>
              <div
                onClick={onClose}
                style={{ cursor:"pointer", color:"var(--muted)", fontSize:"14px" }}
              >Закрыть</div>
            </div>
            {children}
          </div>
        </div>
      );
    };

    const StatRow = ({ left, right }) => (
      <div style={{ display:"flex", justifyContent:"space-between", marginTop:"10px", fontSize:"14px" }}>
        <div style={{ color:"var(--muted)" }}>{left}</div>
        <div style={{ fontWeight:600 }}>{right}</div>
      </div>
    );

    const Divider = () => (
      <div style={{ marginTop:"14px", marginBottom:"8px", height:"1px", background:"var(--stroke)" }} />
    );

    const PrizeTable = () => (
      <div style={{ marginTop:"10px" }}>
        <div style={{ fontSize:"13px", color:"var(--muted)" }}>Шансы рулетки (честно):</div>
        <div style={{ marginTop:"10px", display:"grid", gap:"8px" }}>
          {[
            ["50%", "+500 баллов"],
            ["25%", "+1000 баллов"],
            ["15%", "🎟 Билет на розыгрыш"],
            ["8%", "+3000 баллов"],
            ["2%", "💎 Dior палетка (ТОП приз)"],
          ].map(([p, t]) => (
            <div key={p+t} style={{
              padding:"10px",
              borderRadius:"14px",
              border:"1px solid var(--stroke)",
              background:"rgba(255,255,255,0.08)",
              display:"flex",
              justifyContent:"space-between",
              fontSize:"14px"
            }}>
              <div style={{ color:"var(--muted)" }}>{p}</div>
              <div style={{ fontWeight:600 }}>{t}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop:"10px", fontSize:"12px", color:"var(--muted)" }}>
          Лимит: 1 спин раз в 3 часа
        </div>
      </div>
    );

    const App = () => {
      const [activeTab, setActiveTab] = useState("home");
      const [user, setUser] = useState(null);
      const [botUsername, setBotUsername] = useState(BOT_USERNAME || "");

      const [postsMode, setPostsMode] = useState(false);
      const [selectedTag, setSelectedTag] = useState(null);
      const [posts, setPosts] = useState([]);
      const [loading, setLoading] = useState(false);

      const [profileOpen, setProfileOpen] = useState(false);
      const [raffle, setRaffle] = useState(null);
      const [rouletteHistory, setRouletteHistory] = useState([]);
      const [busy, setBusy] = useState(false);
      const [msg, setMsg] = useState("");

      const tgUserId = tg?.initDataUnsafe?.user?.id;

      const refreshUser = () => {
        if (!tgUserId) return Promise.resolve();
        return fetch(`/api/user/${tgUserId}`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setUser(data))
          .catch(() => {});
      };


      const apiPost = (path, body) => {
        return fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body || {}),
          keepalive: true,
        })
          .then(r => r.ok ? r.json().catch(() => ({})) : r.json().catch(() => ({})).then(e => Promise.reject(e)))
          .catch(() => ({}));
      };

      const doActivity = async (path, body) => {
        if (!tgUserId) return;
        const res = await apiPost(path, { telegram_id: tgUserId, ...(body || {}) });
        if (res?.message) setMsg(res.message);
        refreshUser();
        return res;
      };

      const trackView = (postId) => {
        if (!postId) return;
        // fire & forget
        doActivity("/api/activity/view", { post_id: postId });
      };

      const loadPosts = (tag) => {
        setLoading(true);
        fetch(`/api/posts?tag=${encodeURIComponent(tag)}`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setPosts(Array.isArray(data) ? data : []))
          .catch(() => setPosts([]))
          .finally(() => setLoading(false));
      };

      const openPosts = (tag) => {
        setSelectedTag(tag);
        setPostsMode(true);
        loadPosts(tag);
      };

      const changeTab = (tabId) => {
        setActiveTab(tabId);
        setPostsMode(false);
        setSelectedTag(null);
        setPosts([]);
        setLoading(false);
      };

      const loadRaffleStatus = () => {
        if (!tgUserId) return Promise.resolve();
        return fetch(`/api/raffle/status?telegram_id=${encodeURIComponent(tgUserId)}`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setRaffle(data))
          .catch(() => setRaffle(null));
      };

      const loadRouletteHistory = () => {
        if (!tgUserId) return Promise.resolve();
        return fetch(`/api/roulette/history?telegram_id=${encodeURIComponent(tgUserId)}&limit=5`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setRouletteHistory(Array.isArray(data) ? data : []))
          .catch(() => setRouletteHistory([]));
      };

      const openProfile = () => {
        if (!user) return;
        setMsg("");
        setProfileOpen(true);
      };

      useEffect(() => {
        if (tgUserId) {
          refreshUser();
        }
      }, []);
useEffect(() => {
        // пробуем подтянуть username бота для реф-ссылки
        fetch(`/api/bot/username`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(d => {
            const u = (d?.bot_username || "").trim().replace(/^@/, "");
            if (u) setBotUsername(u);
          })
          .catch(() => {});
      }, []);


      
      useEffect(() => {
        if (profileOpen) {
          loadRaffleStatus();
          loadRouletteHistory();
        }
      }, [profileOpen]);

      const referralLink = useMemo(() => {
        if (!tgUserId) return "";
        if (!botUsername) return "";
        return `https://t.me/${botUsername}?start=${tgUserId}`;
      }, [tgUserId, botUsername]);

      const copyText = async (t) => {
        if (!t) return;
        try {
          await navigator.clipboard.writeText(t);
          setMsg("✅ Скопировано");
          if (tg?.HapticFeedback?.impactOccurred) tg.HapticFeedback.impactOccurred("light");
          return;
        } catch (e) {
          // fallback для webview/старых браузеров
          try {
            const ta = document.createElement("textarea");
            ta.value = t;
            ta.style.position = "fixed";
            ta.style.left = "-9999px";
            ta.style.top = "-9999px";
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            const ok = document.execCommand("copy");
            document.body.removeChild(ta);
            setMsg(ok ? "✅ Скопировано" : "ℹ️ Не удалось скопировать");
            if (ok && tg?.HapticFeedback?.impactOccurred) tg.HapticFeedback.impactOccurred("light");
          } catch (e2) {
            setMsg("ℹ️ Не удалось скопировать");
          }
        }
      };

      const buyTicket = async () => {
        if (!tgUserId) return;
        setBusy(true);
        setMsg("");
        try {
          const r = await fetch(`/api/raffle/buy_ticket`, {
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify({ telegram_id: tgUserId, qty: 1 })
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка");
          }
          const data = await r.json();
          setMsg(`✅ Билет куплен. Твоих билетов: ${data.ticket_count}`);
          // ✅ обновляем сразу, чтобы счётчик билетов менялся моментально
          setRaffle((prev) => ({ ...(prev || {}), ticket_count: data.ticket_count }));
          await refreshUser();
          await loadRaffleStatus();
        } catch (e) {
          setMsg(`❌ ${e.message || "Ошибка"}`);
        } finally {
          setBusy(false);
        }
      };

      const spinRoulette = async () => {
        if (!tgUserId) return;
        setBusy(true);
        setMsg("");
        try {
          const r = await fetch(`/api/roulette/spin`, {
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify({ telegram_id: tgUserId })
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка");
          }
          const data = await r.json();
          setMsg(`🎡 Выпало: ${data.prize_label}`);
          // ✅ всплывающее окно с призом
          try {
            if (tg?.showPopup) {
              const msg = data.claimable && data.claim_code
                ? `Ваш приз: ${data.prize_label}\n\nЧтобы забрать: откройте чат с ботом и отправьте\n/claim ${data.claim_code}`
                : `Ваш приз: ${data.prize_label}`;

              if (data.claimable && data.claim_code && tg?.openTelegramLink && botUsername) {
                showLockedPopup({
                  title: "🎡 Рулетка",
                  message: msg,
                  primaryText: "Получить приз",
                  onPrimary: () => tg.openTelegramLink(`https://t.me/${botUsername}?start=claim_${data.claim_code}`),
                  okText: "OK"
                });
              } else {
                tg.showPopup({
                  title: "🎡 Рулетка",
                  message: msg,
                  buttons: [{ type: "ok" }]
                });
              }
            } else {
              alert(`Ваш приз: ${data.prize_label}`);
            }
          } catch (e) {}
          await refreshUser();
          await loadRaffleStatus();
          await loadRouletteHistory();
        } catch (e) {
          setMsg(`❌ ${e.message || "Ошибка"}`);
        } finally {
          setBusy(false);
        }
      };

      const PostsScreen = () => (
        <Panel>
          <div style={{ fontSize: "14px", color: "var(--muted)" }}>
            Посты {selectedTag ? ("#" + selectedTag) : ""}
          </div>

          {loading && (
            <div style={{ marginTop: "10px", fontSize: "13px", color: "var(--muted)" }}>
              Загрузка…
            </div>
          )}

          {!loading && posts.length === 0 && (
            <div style={{ marginTop: "10px", fontSize: "13px", color: "var(--muted)" }}>
              Постов с этим тегом пока нет.
            </div>
          )}

          {!loading && posts.map(p => <PostCard key={p.message_id} post={p} />)}
        </Panel>
      );

      const renderContent = () => {
        if (postsMode) return <PostsScreen />;

        switch (activeTab) {
          case "home":
            return (
              <Panel>
                <Button icon="📂" label="Категории" onClick={() => changeTab("cat")} />
                <Button icon="🏷" label="Бренды" onClick={() => changeTab("brand")} />
                <Button icon="💸" label="Sephora" onClick={() => changeTab("sephora")} />
                <Button icon="🧴" label="Продукт" onClick={() => changeTab("ptype")} />
                <Button icon="💎" label="Beauty Challenges" onClick={() => openPosts("Challenge")} />
                <Button icon="↩️" label="В канал" onClick={() => openLink(`https://t.me/${CHANNEL}`)} />
              </Panel>
            );

          case "cat":
            return (
              <Panel>
                <Button icon="🆕" label="Новинка" onClick={() => openPosts("Новинка")} />
                <Button icon="💎" label="Кратко о люкс продукте" onClick={() => openPosts("Люкс")} />
                <Button icon="🔥" label="Тренд" onClick={() => openPosts("Тренд")} />
                <Button icon="🏛" label="История бренда" onClick={() => openPosts("История")} />
                <Button icon="⭐" label="Личная оценка" onClick={() => openPosts("Оценка")} />
                <Button icon="🧾" label="Факты" onClick={() => openPosts("Факты")} />
                <Button icon="🧪" label="Составы продуктов" onClick={() => openPosts("Состав")} />
              </Panel>
            );

          case "brand":
            return (
              <Panel>
                {[
                  ["The Ordinary", "TheOrdinary"],
                  ["Dior", "Dior"],
                  ["Chanel", "Chanel"],
                  ["Kylie Cosmetics", "KylieCosmetics"],
                  ["Gisou", "Gisou"],
                  ["Rare Beauty", "RareBeauty"],
                  ["Yves Saint Laurent", "YSL"],
                  ["Givenchy", "Givenchy"],
                  ["Charlotte Tilbury", "CharlotteTilbury"],
                  ["NARS", "NARS"],
                  ["Sol de Janeiro", "SolDeJaneiro"],
                  ["Huda Beauty", "HudaBeauty"],
                  ["Rhode", "Rhode"],
                  ["Tower 28 Beauty", "Tower28Beauty"],
                  ["Benefit Cosmetics", "BenefitCosmetics"],
                  ["Estée Lauder", "EsteeLauder"],
                  ["Sisley", "Sisley"],
                  ["Kérastase", "Kerastase"],
                  ["Armani Beauty", "ArmaniBeauty"],
                  ["Hourglass", "Hourglass"],
                  ["Shiseido", "Shiseido"],
                  ["Tom Ford Beauty", "TomFordBeauty"],
                  ["Tarte", "Tarte"],
                  ["Sephora Collection", "SephoraCollection"],
                  ["Clinique", "Clinique"],
                  ["Dolce & Gabbana", "DolceGabbana"],
                  ["Kayali", "Kayali"],
                  ["Guerlain", "Guerlain"],
                  ["Fenty Beauty", "FentyBeauty"],
                  ["Too Faced", "TooFaced"],
                  ["MAKE UP FOR EVER", "MakeUpForEver"],
                  ["Erborian", "Erborian"],
                  ["Natasha Denona", "NatashaDenona"],
                  ["Lancôme", "Lancome"],
                  ["Kosas", "Kosas"],
                  ["ONE/SIZE", "OneSize"],
                  ["Laneige", "Laneige"],
                  ["Makeup by Mario", "MakeupByMario"],
                  ["Valentino Beauty", "ValentinoBeauty"],
                  ["Drunk Elephant", "DrunkElephant"],
                  ["Olaplex", "Olaplex"],
                  ["Anastasia Beverly Hills", "AnastasiaBeverlyHills"],
                  ["Amika", "Amika"],
                  ["BYOMA", "BYOMA"],
                  ["Glow Recipe", "GlowRecipe"],
                  ["Milk Makeup", "MilkMakeup"],
                  ["Summer Fridays", "SummerFridays"],
                  ["K18", "K18"],
                ].map(([label, tag]) => (
                  <Button key={tag} icon="✨" label={label} onClick={() => openPosts(tag)} />
                ))}
              </Panel>
            );

          case "sephora":
            return (
              <Panel>
                <Button icon="🎁" label="Подарки / акции" onClick={() => openPosts("SephoraPromo")} />
              </Panel>
            );

          case "ptype":
            return (
              <Panel>
                <Button icon="🧴" label="Праймер" onClick={() => openPosts("Праймер")} />
                <Button icon="🧴" label="Тональная основа" onClick={() => openPosts("ТональнаяОснова")} />
                <Button icon="🧴" label="Консилер" onClick={() => openPosts("Консилер")} />
                <Button icon="🧴" label="Пудра" onClick={() => openPosts("Пудра")} />
                <Button icon="🧴" label="Румяна" onClick={() => openPosts("Румяна")} />
                <Button icon="🧴" label="Скульптор" onClick={() => openPosts("Скульптор")} />
                <Button icon="🧴" label="Бронзер" onClick={() => openPosts("Бронзер")} />
                <Button icon="🧴" label="Продукт для бровей" onClick={() => openPosts("ПродуктДляБровей")} />
                <Button icon="🧴" label="Хайлайтер" onClick={() => openPosts("Хайлайтер")} />
                <Button icon="🧴" label="Тушь" onClick={() => openPosts("Тушь")} />
                <Button icon="🧴" label="Тени" onClick={() => openPosts("Тени")} />
                <Button icon="🧴" label="Помада" onClick={() => openPosts("Помада")} />
                <Button icon="🧴" label="Карандаш для губ" onClick={() => openPosts("КарандашДляГуб")} />
                <Button icon="🧴" label="Палетка" onClick={() => openPosts("Палетка")} />
                <Button icon="🧴" label="Фиксатор" onClick={() => openPosts("Фиксатор")} />
              </Panel>
            );

          default:
            return null;
        }
      };

      return (
        <div style={{ padding:"18px 16px 26px", maxWidth:"520px", margin:"0 auto" }}>
          <Hero user={user} onOpenProfile={openProfile} />
          <Tabs active={activeTab} onChange={changeTab} />
          {renderContent()}

          <Sheet open={profileOpen} onClose={() => setProfileOpen(false)}>
            {!user ? (
              <div style={{ marginTop:"12px", color:"var(--muted)", fontSize:"13px" }}>
                Профиль недоступен.
              </div>
            ) : (
              <div style={{ marginTop:"12px" }}>
                <div style={{
                  padding:"12px",
                  borderRadius:"18px",
                  border:"1px solid var(--stroke)",
                  background:"rgba(255,255,255,0.08)",
                  position:"relative"
                }}>
                  {/* 💎 Баллы — в правом верхнем углу (как просили) */}
                  <div style={{
                    position:"absolute",
                    top:"10px",
                    right:"10px",
                    padding:"6px 10px",
                    borderRadius:"999px",
                    border:"1px solid rgba(230,193,128,0.25)",
                    background:"rgba(230,193,128,0.10)",
                    fontSize:"13px",
                    fontWeight:700
                  }}>
                    💎 {user.points}
                  </div>

                  <div style={{ fontSize:"13px", color:"var(--muted)" }}>Привет, {user.first_name}!</div>
                  <div style={{ fontSize:"13px", color:"var(--muted)", marginTop:"6px" }}>{tierLabel(user.tier)}</div>

                  <StatRow left="🔥 Стрик" right={`${user.daily_streak || 0} (best ${user.best_streak || 0})`} />
                  <StatRow left="🎟 Приглашено" right={`${user.referral_count || 0}`} />
                </div>

                <Divider />

                <div style={{ fontSize:"14px", fontWeight:650 }}>🎟 Рефералка</div>
                <div style={{ marginTop:"8px", fontSize:"13px", color:"var(--muted)" }}>
                  За нового пользователя: +20 баллов (1 раз за каждого).
                </div>
                {botUsername ? (
                  <div style={{
                    marginTop:"10px",
                    padding:"10px",
                    borderRadius:"14px",
                    border:"1px solid var(--stroke)",
                    background:"rgba(255,255,255,0.08)",
                    fontSize:"12px",
                    color:"rgba(255,255,255,0.85)",
                    wordBreak:"break-all"
                  }}>
                    {referralLink}
                  </div>
                ) : (
                  <div style={{ marginTop:"10px", fontSize:"12px", color:"var(--muted)" }}>
                    Если ссылка не показалась — задай переменную окружения <b>BOT_USERNAME</b> или проверь, что бот запущен (мы берём username через Telegram API).
                  </div>
                )}
                <Button
                  icon="📎"
                  label="Скопировать ссылку"
                  onClick={() => copyText(referralLink)}
                  disabled={!botUsername || !referralLink}
                />

                <Divider />

                <div style={{ fontSize:"14px", fontWeight:650 }}>💎 На что тратить баллы</div>
                <div style={{ marginTop:"8px", fontSize:"13px", color:"var(--muted)" }}>
                  • 🎁 Билет на розыгрыш — 500 баллов<br/>
                  • 🎡 Рулетка — 2000 баллов (лимит 1 раз в 3 часа)
                </div>

                <div style={{ marginTop:"12px", fontSize:"14px", fontWeight:650 }}>💎 Как получать баллы</div>
                <div style={{ marginTop:"8px", fontSize:"13px", color:"var(--muted)" }}>
                  • Ежедневный визит: +5 (раз в 24ч)<br/>
                  • 3 комментария в день: +10 (коммент от 25 символов)<br/>
                  • Просмотр постов из Mini App: +1 (до 15/день)<br/>
                  • Голосования: +3 (до 5/день)<br/>
                  • Challenge дня: +5 (1/день)<br/>
                  • Квест дня: +10 (1/день)<br/>
                  • Избранное: +2 (до 5/день)<br/>
                  • Оценки: +3 / +5 с текстом<br/>
                  • Реферал: +20 +10 за активность
                </div>

                <Divider />

                <div style={{ fontSize:"14px", fontWeight:650 }}>⚡ Активности</div>
                <div style={{ marginTop:"8px", display:"grid", gap:"8px" }}>
                  <Button icon="💬" label="Отметить комментарий" onClick={async () => {
                    const postId = parseInt(prompt("ID поста (message_id) из канала:", ""), 10);
                    if (!postId) return;
                    const txt = prompt("Текст комментария (минимум 25 символов):", "");
                    if (!txt) return;
                    await doActivity("/api/activity/comment", { post_id: postId, text: txt });
                  }} />
                  <Button icon="🗳" label="Голосование" onClick={async () => {
                    const pollId = prompt("ID/код голосования (любой идентификатор):", "");
                    if (!pollId) return;
                    await doActivity("/api/activity/vote", { poll_id: pollId });
                  }} />
                  <Button icon="🎯" label="Challenge дня" onClick={async () => {
                    const txt = prompt("Твой ответ / мини-мысль:", "");
                    if (!txt) return;
                    await doActivity("/api/activity/challenge", { text: txt });
                  }} />
                  <Button icon="🧾" label="Квест дня" onClick={async () => {
                    await doActivity("/api/activity/quest", {});
                  }} />
                  <Button icon="✨" label="Добавить бренд в избранное (+2)" onClick={async () => {
                    const brand = prompt("Тег бренда (например: #Dior):", "");
                    if (!brand) return;
                    await doActivity("/api/activity/favorite", { brand_tag: brand.trim() });
                  }} />
                  <Button icon="⭐" label="Оценить продукт" onClick={async () => {
                    const tag = prompt("Тег продукта (например: Dior Glow Maximizer Palette):", "");
                    if (!tag) return;
                    const stars = parseInt(prompt("Оценка 1-5:", "5"), 10);
                    if (!stars) return;
                    const txt = prompt("Комментарий (необязательно):", "");
                    await doActivity("/api/activity/rating", { product_tag: tag.trim(), stars, text: (txt || "").trim() });
                  }} />
                </div>

                <Divider />


                

                <div style={{ fontSize:"14px", fontWeight:650 }}>🎁 Розыгрыши</div>
                <div style={{ marginTop:"8px", fontSize:"13px", color:"var(--muted)" }}>
                  Билет = 500 баллов. Баллы списываются.
                </div>
                <div style={{ marginTop:"10px", fontSize:"13px", color:"var(--muted)" }}>
                  Твоих билетов: <b style={{ color:"rgba(255,255,255,0.92)" }}>{raffle?.ticket_count ?? 0}</b>
                </div>
                <Button
                  icon="🎟"
                  label="Купить билет (500)"
                  subtitle={busy ? "Подожди…" : ""}
                  onClick={buyTicket}
                  disabled={busy || (user.points || 0) < 500}
                />

                <Divider />

                <div style={{ fontSize:"14px", fontWeight:650 }}>🎡 Рулетка</div>
                <div style={{ marginTop:"8px", fontSize:"13px", color:"var(--muted)" }}>
                  1 спин = 2000 баллов. Лимит: 1 раз в 3 часа.
                </div>
                <Button
                  icon="🎡"
                  label="Крутить (2000)"
                  subtitle={busy ? "Подожди…" : ""}
                  onClick={spinRoulette}
                  disabled={busy || (user.points || 0) < 2000}
                />

                <PrizeTable />

                {msg && (
                  <div style={{
                    marginTop:"14px",
                    padding:"10px",
                    borderRadius:"14px",
                    border:"1px solid var(--stroke)",
                    background:"rgba(255,255,255,0.08)",
                    fontSize:"13px"
                  }}>{msg}</div>
                )}

                <Divider />

                <div style={{ fontSize:"14px", fontWeight:650 }}>🧾 История рулетки</div>
                {rouletteHistory.length === 0 ? (
                  <div style={{ marginTop:"8px", fontSize:"13px", color:"var(--muted)" }}>
                    Пока пусто.
                  </div>
                ) : (
                  <div style={{ marginTop:"10px", display:"grid", gap:"8px" }}>
                    {rouletteHistory.map((x) => (
                      <div key={x.id} style={{
                        padding:"10px",
                        borderRadius:"14px",
                        border:"1px solid var(--stroke)",
                        background:"rgba(255,255,255,0.08)"
                      }}>
                        <div style={{ fontSize:"12px", color:"var(--muted)" }}>{x.created_at}</div>
                        <div style={{ marginTop:"4px", fontSize:"14px", fontWeight:600 }}>{x.prize_label}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Sheet>

          <div style={{ marginTop:"20px", color:"var(--muted)", fontSize:"12px", textAlign:"center" }}>
            Открывается как Mini App внутри Telegram
          </div>
        </div>
      );
    };

    ReactDOM.render(<App />, document.getElementById("root"));
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



@app.post("/api/activity/view", response_model="ActivityResp")
async def api_activity_view(req: ActivityViewReq):
    day = utc_day_key()
    async with async_session_maker() as session:
        async with session.begin():
            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)
            views = _counter_get(stat.counters or {}, "views", 0)
            if views >= 15:
                user = await get_or_create_user(session, int(req.telegram_id))
                return ActivityResp(ok=False, message="Лимит просмотров на сегодня достигнут.", points=int(user.points or 0), awarded=0)

            ok, msg, awarded = await award_activity_points(
                session,
                int(req.telegram_id),
                "view",
                1,
                key=f"post:{int(req.post_id)}",
                day=day,
            )
            if ok and awarded > 0:
                c = dict(stat.counters or {})
                c["views"] = views + 1
                stat.counters = c
                stat.updated_at = datetime.utcnow()

            user = await get_or_create_user(session, int(req.telegram_id))
            return ActivityResp(ok=ok, message=msg, points=int(user.points or 0), awarded=awarded)


@app.post("/api/activity/vote", response_model="ActivityResp")
async def api_activity_vote(req: ActivityVoteReq):
    day = utc_day_key()
    async with async_session_maker() as session:
        async with session.begin():
            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)
            votes = _counter_get(stat.counters or {}, "votes", 0)
            if votes >= 5:
                user = await get_or_create_user(session, int(req.telegram_id))
                return ActivityResp(ok=False, message="Лимит голосований на сегодня достигнут.", points=int(user.points or 0), awarded=0)

            ok, msg, awarded = await award_activity_points(
                session,
                int(req.telegram_id),
                "vote",
                3,
                key=f"poll:{req.poll_id}",
                day=day,
            )
            if ok and awarded > 0:
                c = dict(stat.counters or {})
                c["votes"] = votes + 1
                stat.counters = c
                stat.updated_at = datetime.utcnow()

            user = await get_or_create_user(session, int(req.telegram_id))
            return ActivityResp(ok=ok, message=msg, points=int(user.points or 0), awarded=awarded)


@app.post("/api/activity/challenge", response_model="ActivityResp")
async def api_activity_challenge(req: ActivityChallengeReq):
    day = utc_day_key()
    async with async_session_maker() as session:
        async with session.begin():
            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)
            done = bool((stat.counters or {}).get("challenge_done", False))
            if done:
                user = await get_or_create_user(session, int(req.telegram_id))
                return ActivityResp(ok=False, message="Challenge дня уже выполнен.", points=int(user.points or 0), awarded=0)

            ok, msg, awarded = await award_activity_points(
                session,
                int(req.telegram_id),
                "challenge",
                5,
                key="daily",
                day=day,
            )
            if ok and awarded > 0:
                c = dict(stat.counters or {})
                c["challenge_done"] = True
                stat.counters = c
                stat.updated_at = datetime.utcnow()

            user = await get_or_create_user(session, int(req.telegram_id))
            return ActivityResp(ok=ok, message=msg, points=int(user.points or 0), awarded=awarded)


@app.post("/api/activity/quest", response_model="ActivityResp")
async def api_activity_quest(req: ActivityQuestReq):
    day = utc_day_key()
    async with async_session_maker() as session:
        async with session.begin():
            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)
            done = bool((stat.counters or {}).get("quest_done", False))
            if done:
                user = await get_or_create_user(session, int(req.telegram_id))
                return ActivityResp(ok=False, message="Квест дня уже выполнен.", points=int(user.points or 0), awarded=0)

            ok, msg, awarded = await award_activity_points(
                session,
                int(req.telegram_id),
                "quest",
                10,
                key="daily",
                day=day,
            )
            if ok and awarded > 0:
                c = dict(stat.counters or {})
                c["quest_done"] = True
                stat.counters = c
                stat.updated_at = datetime.utcnow()

            user = await get_or_create_user(session, int(req.telegram_id))
            return ActivityResp(ok=ok, message=msg, points=int(user.points or 0), awarded=awarded)


@app.post("/api/activity/favorite", response_model="ActivityResp")
async def api_activity_favorite(req: ActivityFavoriteReq):
    day = utc_day_key()
    brand = (req.brand_tag or "").strip()
    if not brand:
        raise HTTPException(status_code=400, detail="brand_tag required")

    async with async_session_maker() as session:
        async with session.begin():
            user = await get_or_create_user(session, int(req.telegram_id))
            favs = list(user.favorites or [])
            if brand in favs:
                return ActivityResp(ok=False, message="Этот бренд уже в избранном.", points=int(user.points or 0), awarded=0)

            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)
            cnt = _counter_get(stat.counters or {}, "favorites_added", 0)
            if cnt >= 5:
                return ActivityResp(ok=False, message="Лимит добавлений в избранное на сегодня достигнут.", points=int(user.points or 0), awarded=0)

            ok, msg, awarded = await award_activity_points(
                session,
                int(req.telegram_id),
                "favorite",
                2,
                key=f"brand:{brand}",
                day=day,
            )
            if ok and awarded > 0:
                favs.append(brand)
                user.favorites = favs
                c = dict(stat.counters or {})
                c["favorites_added"] = cnt + 1
                stat.counters = c
                stat.updated_at = datetime.utcnow()

            return ActivityResp(ok=ok, message=msg, points=int(user.points or 0), awarded=awarded)


@app.post("/api/activity/rating", response_model="ActivityResp")
async def api_activity_rating(req: ActivityRatingReq):
    day = utc_day_key()
    tag = (req.product_tag or "").strip()
    if not tag:
        raise HTTPException(status_code=400, detail="product_tag required")

    has_text = bool(req.text and req.text.strip())
    delta = 5 if has_text else 3

    async with async_session_maker() as session:
        async with session.begin():
            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)
            r_cnt = _counter_get(stat.counters or {}, "ratings", 0)
            rt_cnt = _counter_get(stat.counters or {}, "ratings_text", 0)

            if r_cnt >= 3:
                user = await get_or_create_user(session, int(req.telegram_id))
                return ActivityResp(ok=False, message="Лимит оценок на сегодня достигнут.", points=int(user.points or 0), awarded=0)
            if has_text and rt_cnt >= 2:
                user = await get_or_create_user(session, int(req.telegram_id))
                return ActivityResp(ok=False, message="Лимит оценок с текстом на сегодня достигнут.", points=int(user.points or 0), awarded=0)

            key = f"product:{tag}:stars:{int(req.stars)}"
            ok, msg, awarded = await award_activity_points(
                session,
                int(req.telegram_id),
                "rating_text" if has_text else "rating",
                delta,
                key=key,
                day=day,
            )
            if ok and awarded > 0:
                c = dict(stat.counters or {})
                c["ratings"] = r_cnt + 1
                if has_text:
                    c["ratings_text"] = rt_cnt + 1
                stat.counters = c
                stat.updated_at = datetime.utcnow()

            user = await get_or_create_user(session, int(req.telegram_id))
            return ActivityResp(ok=ok, message=msg, points=int(user.points or 0), awarded=awarded)


@app.post("/api/activity/comment", response_model="ActivityResp")
async def api_activity_comment(req: ActivityCommentReq):
    day = utc_day_key()
    text = (req.text or "").strip()
    if len(text) < 25:
        raise HTTPException(status_code=400, detail="comment text too short")

    # хеш для анти-дубликата
    h = secrets.token_hex(4)
    # deterministic-ish: take stable hash of text
    try:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    except Exception:
        pass

    async with async_session_maker() as session:
        async with session.begin():
            user = await get_or_create_user(session, int(req.telegram_id))
            stat = await get_or_create_daily_stat(session, int(req.telegram_id), day)

            # засчитываем комментарий (без баллов), но учитываем уникальность по посту+тексту
            key = f"{int(req.post_id)}:{h}"
            unique = await record_activity_event(session, int(req.telegram_id), day, "comment", key)
            if not unique:
                return ActivityResp(ok=False, message="Похоже на повтор — комментарий уже засчитан.", points=int(user.points or 0), awarded=0)

            c = dict(stat.counters or {})
            count = _counter_get(c, "comments_count", 0) + 1
            c["comments_count"] = count

            # бонус за 3 коммента/день (+10 один раз)
            bonus_paid = bool(c.get("comment_bonus_paid", False))
            awarded = 0
            msg = "Комментарий засчитан ✅"
            if (not bonus_paid) and count >= 3:
                ok2, msg2, awarded2 = await award_activity_points(
                    session,
                    int(req.telegram_id),
                    "comment_bonus",
                    10,
                    key="daily",
                    day=day,
                )
                if ok2:
                    c["comment_bonus_paid"] = True
                    awarded = awarded2
                    msg = "✅ +10 баллов — за 3 комментария сегодня."
                else:
                    msg = msg2

            stat.counters = c
            stat.updated_at = datetime.utcnow()

            # бонус за активного реферала (72 часа)
            try:
                inviter = int(user.referred_by) if user.referred_by else None
            except Exception:
                inviter = None
            if inviter and not bool(user.ref_active_bonus_paid):
                try:
                    joined = user.joined_at or datetime.utcnow()
                    if isinstance(joined, datetime):
                        if datetime.utcnow() - joined <= timedelta(hours=72):
                            await award_points_unlimited(
                                session,
                                inviter,
                                "referral_active_bonus",
                                10,
                                meta={"invited_telegram_id": int(req.telegram_id)},
                            )
                            user.ref_active_bonus_paid = True
                except Exception:
                    pass

            user = await get_or_create_user(session, int(req.telegram_id))
            return ActivityResp(ok=True, message=msg, points=int(user.points or 0), awarded=awarded)



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



# -----------------------------------------------------------------------------
# ACTIVITY (баллы за действия в Mini App)
# -----------------------------------------------------------------------------
class ActivityViewReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    post_id: int = Field(..., ge=1)  # message_id поста


class ActivityVoteReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    poll_id: str = Field(..., min_length=1, max_length=128)


class ActivityChallengeReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    text: str = Field(..., min_length=1, max_length=2000)


class ActivityQuestReq(BaseModel):
    telegram_id: int = Field(..., ge=1)


class ActivityFavoriteReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    brand_tag: str = Field(..., min_length=1, max_length=64)


class ActivityRatingReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    product_tag: str = Field(..., min_length=1, max_length=128)
    stars: int = Field(..., ge=1, le=5)
    text: Optional[str] = Field(default=None, max_length=2000)


class ActivityCommentReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    post_id: int = Field(..., ge=1)
    text: str = Field(..., min_length=25, max_length=4000)


class ActivityResp(BaseModel):
    ok: bool
    message: str
    points: int
    awarded: int = 0


def utc_day_key(dt: Optional[datetime] = None) -> str:
    d = dt or datetime.utcnow()
    return d.date().isoformat()


def _counter_get(counters: dict, k: str, default=0):
    try:
        return int(counters.get(k, default))
    except Exception:
        return default


async def get_or_create_daily_stat(session: AsyncSession, telegram_id: int, day: str) -> UserDailyStat:
    row = await session.scalar(
        select(UserDailyStat).where(UserDailyStat.telegram_id == telegram_id, UserDailyStat.day == day)
    )
    if row:
        return row
    row = UserDailyStat(telegram_id=telegram_id, day=day, earned_activity_points=0, counters={})
    session.add(row)
    await session.flush()
    return row


async def record_activity_event(session: AsyncSession, telegram_id: int, day: str, kind: str, key: str) -> bool:
    """True если событие уникальное (не было в этот day), иначе False.
    Делаем мягко через SELECT перед INSERT, чтобы не откатывать всю транзакцию.
    """
    existing = await session.scalar(
        select(ActivityEvent.id).where(
            ActivityEvent.telegram_id == telegram_id,
            ActivityEvent.day == day,
            ActivityEvent.kind == kind,
            ActivityEvent.key == key,
        )
    )
    if existing:
        return False
    session.add(ActivityEvent(telegram_id=telegram_id, day=day, kind=kind, key=key))
    await session.flush()
    return True


async def award_activity_points(
    session: AsyncSession,
    telegram_id: int,
    kind: str,
    delta: int,
    key: str,
    *,
    day: Optional[str] = None,
) -> tuple[bool, str, int]:
    """Начисление баллов за активности с дневным лимитом DAILY_ACTIVITY_CAP.
    Возвращает (ok, message, awarded_points).
    """
    day = day or utc_day_key()
    user = await get_or_create_user(session, telegram_id)
    stat = await get_or_create_daily_stat(session, telegram_id, day)

    # дневной лимит на активности
    remaining = max(0, int(DAILY_ACTIVITY_CAP) - int(stat.earned_activity_points or 0))
    if remaining <= 0:
        return False, "Дневной лимит баллов за активности достигнут.", 0

    award = min(int(delta), remaining)

    # дедупликация
    unique = await record_activity_event(session, telegram_id, day, kind, key)
    if not unique:
        return False, "Похоже на повтор — баллы не начислены.", 0

    # применяем
    user.points = int(user.points or 0) + award
    stat.earned_activity_points = int(stat.earned_activity_points or 0) + award
    stat.updated_at = datetime.utcnow()
    session.add(
        PointTransaction(
            telegram_id=telegram_id,
            type=f"activity_{kind}",
            delta=award,
            meta={"key": key, "day": day},
        )
    )
    await session.flush()
    return True, f"+{award} баллов начислено.", award


async def award_points_unlimited(
    session: AsyncSession,
    telegram_id: int,
    tx_type: str,
    delta: int,
    meta: Optional[dict] = None,
) -> None:
    user = await get_or_create_user(session, telegram_id)
    user.points = int(user.points or 0) + int(delta)
    session.add(PointTransaction(telegram_id=telegram_id, type=tx_type, delta=int(delta), meta=meta or {}))
    await session.flush()

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
    qty = int(req.qty)
    cost = RAFFLE_TICKET_COST * qty

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(
                    select(User).where(User.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            if (user.points or 0) < cost:
                raise HTTPException(status_code=400, detail=f"Недостаточно баллов. Нужно {cost}")

            raffle = (await session.execute(select(Raffle).where(Raffle.id == DEFAULT_RAFFLE_ID))).scalar_one()
            if not raffle.is_active:
                raise HTTPException(status_code=400, detail="Розыгрыш сейчас недоступен")

            user.points = (user.points or 0) - cost
            _recalc_tier(user)

            ticket_row = await get_ticket_row(session, tid, raffle.id)
            ticket_row.count = int(ticket_row.count or 0) + qty
            ticket_row.updated_at = datetime.utcnow()

            session.add(PointTransaction(telegram_id=tid, type="raffle_ticket", delta=-cost, meta={"qty": qty, "raffle_id": raffle.id}))

        await session.refresh(user)
        # refresh ticket
        async with session.begin():
            ticket_row2 = (
                await session.execute(
                    select(RaffleTicket).where(RaffleTicket.telegram_id == tid, RaffleTicket.raffle_id == DEFAULT_RAFFLE_ID)
                )
            ).scalar_one()

        return {"telegram_id": tid, "points": int(user.points or 0), "ticket_count": int(ticket_row2.count or 0)}


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
                hours_left = max(
                    0,
                    int(delta.total_seconds() // 3600) + (1 if (delta.total_seconds() % 3600) > 0 else 0),
                )
                raise HTTPException(status_code=400, detail=f"Рулетка доступна через ~{hours_left} ч")

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
                        status="awaiting_contact",
                    )
                )

        await session.refresh(user)

    if prize_type == "physical_dior_palette":
        uname = (user.username or "").strip()
        mention = f"@{uname}" if uname else "(без username)"
        await notify_admin(
            "💎 ТОП ПРИЗ: Dior палетка!\n"
            f"user: {mention} | {user.first_name or '-'}\n"
            f"telegram_id: {tid}\n"
            f"link: {tg_user_link(tid)}\n"
            f"claim: {claim_code}\n"
            f"roll: {roll}\n"
            "👉 Пользователю: отправьте /claim <код> и потом сообщение с контактами/адресом."
        )

        # Дублируем пользователю в ЛС, чтобы он 100% не потерял инструкцию
        await notify_user(
            tid,
            "💎 Вы выиграли: Dior палетка (ТОП приз)!\n\n"
            f"Чтобы забрать приз, отправьте в этот чат команду:\n/claim {claim_code}\n\n"
            "Затем одним сообщением пришлите удобный способ связи (Telegram/WhatsApp) и город/адрес доставки."
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
