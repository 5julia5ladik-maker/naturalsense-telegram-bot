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
ROULETTE_SPIN_COST = 3000
ROULETTE_LIMIT_WINDOW = timedelta(hours=24)
DEFAULT_RAFFLE_ID = 1

PrizeType = Literal["points", "raffle_ticket", "physical_dior_palette"]

# per 1_000_000
ROULETTE_DISTRIBUTION: list[dict[str, Any]] = [
    {"weight": 500_000, "type": "points", "value": 500, "label": "+500 баллов"},
    {"weight": 250_000, "type": "points", "value": 1000, "label": "+1000 баллов"},
    {"weight": 150_000, "type": "raffle_ticket", "value": 1, "label": "🎟 Билет на розыгрыш"},
    {"weight": 80_000, "type": "points", "value": 3000, "label": "+3000 баллов"},
    {"weight": 20_000, "type": "physical_dior_palette", "value": 1, "label": "💎 Dior палетка (ТОП приз)"},
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
    type = Column(String, nullable=False)
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

        # Postgres: int32 -> bigint
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

        session.add(PointTransaction(telegram_id=telegram_id, type="register", delta=REGISTER_BONUS_POINTS, meta={}))

        await session.commit()
        await session.refresh(user)
        return user, referral_paid


async def add_daily_bonus_and_update_streak(telegram_id: int) -> tuple[Optional[User], bool, int, int]:
    async with async_session_maker() as session:
        user: User | None = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalar_one_or_none()
        if not user:
            return None, False, 0, 0

        now = datetime.utcnow()
        last = user.last_daily_bonus_at

        # anti-farm: strictly 24h
        if last is not None and (now - last) < timedelta(days=1):
            delta = timedelta(days=1) - (now - last)
            hours_left = max(0, int(delta.total_seconds() // 3600) + (1 if (delta.total_seconds() % 3600) > 0 else 0))
            return user, False, hours_left, 0

        user.points = (user.points or 0) + DAILY_BONUS_POINTS

        # streak: 48h window
        if last is None:
            user.daily_streak = 1
        else:
            user.daily_streak = (user.daily_streak or 0) + 1 if (now - last) <= timedelta(days=2) else 1

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
    except Exception:
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
    return to_mark


async def sweeper_loop():
    while True:
        try:
            await sweep_deleted_posts(batch=80)
        except Exception as e:
            logger.error("Sweeper error: %s", e)
        await asyncio.sleep(300)


# -----------------------------------------------------------------------------
# TELEGRAM BOT
# -----------------------------------------------------------------------------
tg_app: Application | None = None
tg_task: asyncio.Task | None = None
sweeper_task: asyncio.Task | None = None


def is_admin(user_id: int) -> bool:
    return int(user_id) == int(ADMIN_CHAT_ID)


def get_main_keyboard():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("👤 Профиль"), KeyboardButton("ℹ️ Помощь")]],
        resize_keyboard=True,
    )


def build_start_inline_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("↩️ В канал", url=f"https://t.me/{CHANNEL_USERNAME}")]])


async def tg_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram handler error: %s", context.error)
    try:
        if ADMIN_CHAT_ID:
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"❌ Ошибка в боте:\n{repr(context.error)}")
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
        # ReplyKeyboard ставим отдельным сообщением и удаляем сразу (без "Меню")
        try:
            m = await context.bot.send_message(chat_id=update.effective_chat.id, text="\u200b", reply_markup=get_main_keyboard())
            await asyncio.sleep(0.2)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=m.message_id)
        except Exception:
            pass
        return

    user2, granted, hours_left, streak_bonus = await add_daily_bonus_and_update_streak(user.id)
    text_ = build_welcome_text(
        first_name=user.first_name,
        is_new=False,
        daily_granted=bool(granted),
        hours_left=int(hours_left),
        streak=(user2.daily_streak or 0) if user2 else 0,
        streak_bonus=int(streak_bonus),
        referral_paid=False,
    )
    await update.message.reply_text(text_, reply_markup=build_start_inline_kb())
    try:
        m = await context.bot.send_message(chat_id=update.effective_chat.id, text="\u200b", reply_markup=get_main_keyboard())
        await asyncio.sleep(0.2)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=m.message_id)
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


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await update.message.reply_text(build_help_text(), parse_mode="Markdown", reply_markup=get_main_keyboard())


async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    u = update.effective_user
    db_user = await get_user(u.id)
    if not db_user:
        await update.message.reply_text("Нажми /start для регистрации", reply_markup=get_main_keyboard())
        return

    tier_emoji = {"free": "🥉", "premium": "🥈", "vip": "🥇"}
    tier_name = {"free": "Bronze", "premium": "Silver", "vip": "Gold VIP"}

    streak = db_user.daily_streak or 0
    best = db_user.best_streak or 0
    refs = db_user.referral_count or 0

    text_ = f"""\
👤 **Твой профиль**

{tier_emoji.get(db_user.tier, "🥉")} Уровень: {tier_name.get(db_user.tier, "Bronze")}
💎 Баллы: **{db_user.points}**

🔥 Стрик: **{streak}** • Лучший: **{best}**
🎟 Приглашено: **{refs}**
"""
    await update.message.reply_text(text_, parse_mode="Markdown", reply_markup=get_main_keyboard())


async def on_text_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    txt = update.message.text.strip()
    if txt == "👤 Профиль":
        await cmd_profile(update, context)
    elif txt == "ℹ️ Помощь":
        await cmd_help(update, context)


# -----------------------------------------------------------------------------
# CHANNEL INDEXING
# -----------------------------------------------------------------------------
async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return
    text_ = msg.text or msg.caption or ""
    await upsert_post_from_channel(message_id=msg.message_id, date=msg.date, text_=text_)


async def on_edited_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.edited_channel_post
    if not msg:
        return
    text_ = msg.text or msg.caption or ""
    await upsert_post_from_channel(message_id=msg.message_id, date=msg.date, text_=text_)


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
    tg_app.add_handler(CommandHandler("profile", cmd_profile))

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

    if (tg) {
      tg.expand();
      tg.setHeaderColor("#0c0f14");
      tg.setBackgroundColor("#0c0f14");
    }

    const CHANNEL = "__CHANNEL__";
    const BOT_USERNAME = "__BOT_USERNAME__";

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
          background: "linear-gradient(180deg, rgba
