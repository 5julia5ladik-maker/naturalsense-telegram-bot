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
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
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



# -----------------------------------------------------------------------------
# MEDIA (thumbnails for Mini App)
# -----------------------------------------------------------------------------
MEDIA_CACHE_DIR = env_get("MEDIA_CACHE_DIR", "./media_cache")
os.makedirs(MEDIA_CACHE_DIR, exist_ok=True)

_file_path_cache: dict[str, str] = {}
_media_lock = asyncio.Lock()


async def _tg_get_file_path(file_id: str) -> str:
    """Telegram Bot API: getFile -> file_path"""
    if not BOT_TOKEN:
        raise HTTPException(status_code=500, detail="BOT_TOKEN is not set")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, params={"file_id": file_id})
        r.raise_for_status()
        data = r.json()
    if not data.get("ok") or not data.get("result") or not data["result"].get("file_path"):
        raise HTTPException(status_code=404, detail="Telegram file not found")
    return str(data["result"]["file_path"])


async def get_cached_media_file(file_id: str) -> str:
    """Downloads media file into MEDIA_CACHE_DIR once and returns local path."""
    file_id = (file_id or "").strip()
    if not file_id:
        raise HTTPException(status_code=404, detail="Empty file_id")

    async with _media_lock:
        # fast path
        cached = _file_path_cache.get(file_id)
        if cached and os.path.exists(cached):
            return cached

        # Determine local filename (safe)
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", file_id)[:120]
        local_path = os.path.join(MEDIA_CACHE_DIR, safe)

        if os.path.exists(local_path):
            _file_path_cache[file_id] = local_path
            return local_path

        file_path = await _tg_get_file_path(file_id)
        dl_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(dl_url)
            r.raise_for_status()
            content = r.content

        with open(local_path, "wb") as f:
            f.write(content)

        _file_path_cache[file_id] = local_path
        return local_path


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

        st = (claim.status or "").strip()
        if st == "submitted":
            await update.message.reply_text(
                "✅ Данные уже получены.\n\n"
                "Статус: ⏳ Ожидает подтверждения.\n"
                "Мы свяжемся с вами после проверки."
            )
            return

        if st == "closed":
            await update.message.reply_text("✅ Эта заявка уже закрыта.")
            return

        # помечаем как ожидание контакта и обновляем время
        claim.status = "awaiting_contact"
        claim.updated_at = datetime.utcnow()
        await session.commit()

    await update.message.reply_text(
        "🎁 Заявка на приз создана.\n\n"
        "Отправьте ОДНИМ сообщением:\n"
        "1) удобный способ связи (Telegram/WhatsApp)\n"
        "2) город и адрес доставки\n\n"
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

            await update.message.reply_text(
                "✅ Данные получены.\n\n"
                "Ваша заявка принята и ожидает подтверждения.\n"
                "Мы свяжемся с вами после проверки.\n\n"
                "Статус: ⏳ Ожидает подтверждения"
            )
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

    # Detect media (for Mini App previews)
    media_type = None
    media_file_id = None

    # Photos: take the largest size for best preview
    if getattr(msg, "photo", None):
        try:
            media_type = "photo"
            media_file_id = msg.photo[-1].file_id
        except Exception:
            media_type = None
            media_file_id = None
    elif getattr(msg, "video", None):
        try:
            media_type = "video"
            media_file_id = msg.video.file_id
        except Exception:
            pass
    elif getattr(msg, "document", None):
        # Some channels post images as documents; keep it for fallback preview attempts
        try:
            media_type = "document"
            media_file_id = msg.document.file_id
        except Exception:
            pass

    await upsert_post_from_channel(
        message_id=msg.message_id,
        date=msg.date,
        text_=text_,
        media_type=media_type,
        media_file_id=media_file_id,
    )


async def on_edited_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.edited_channel_post
    if not msg:
        return

    text_ = msg.text or msg.caption or ""

    # Detect media (for Mini App previews)
    media_type = None
    media_file_id = None

    if getattr(msg, "photo", None):
        try:
            media_type = "photo"
            media_file_id = msg.photo[-1].file_id
        except Exception:
            media_type = None
            media_file_id = None
    elif getattr(msg, "video", None):
        try:
            media_type = "video"
            media_file_id = msg.video.file_id
        except Exception:
            pass
    elif getattr(msg, "document", None):
        try:
            media_type = "document"
            media_file_id = msg.document.file_id
        except Exception:
            pass

    await upsert_post_from_channel(
        message_id=msg.message_id,
        date=msg.date,
        text_=text_,
        media_type=media_type,
        media_file_id=media_file_id,
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
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>NS · Natural Sense</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <script crossorigin src="https://cdn.jsdelivr.net/npm/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://cdn.jsdelivr.net/npm/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@babel/standalone/babel.min.js"></script>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    :root{
      --bg: #0c0f14;
      --card: rgba(255,255,255,0.08);
      --card2: rgba(255,255,255,0.06);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.60);
      --gold: rgba(230,193,128,0.90);
      --stroke: rgba(255,255,255,0.12);

      --sheetOverlay: rgba(12,15,20,0.55);
      --sheetCardBg: rgba(255,255,255,0.10);
      --glassStroke: rgba(255,255,255,0.18);
      --glassShadow: rgba(0,0,0,0.45);

      --r-lg: 22px;
      --r-md: 16px;
      --r-sm: 14px;
    }
    body{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background:
        radial-gradient(1200px 800px at 20% 10%, rgba(230,193,128,0.18), transparent 60%),
        radial-gradient(900px 600px at 80% 0%, rgba(255,255,255,0.06), transparent 55%),
        var(--bg);
      color: var(--text);
      overflow-x:hidden;
    }
    #root{ min-height:100vh; }
    a{ color: inherit; }

    .safePadBottom{ padding-bottom: 92px; } /* space for bottom nav */
    .container{ max-width: 560px; margin: 0 auto; padding: 16px 16px 24px; }
    .h1{ font-size: 18px; font-weight: 800; letter-spacing: 0.2px; }
    .sub{ margin-top: 6px; font-size: 13px; color: var(--muted); }
    .card{
      border: 1px solid var(--stroke);
      background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05));
      border-radius: var(--r-lg);
      padding: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      position: relative;
      overflow: hidden;
    }
    .card2{
      border: 1px solid var(--stroke);
      background: var(--card2);
      border-radius: var(--r-lg);
      padding: 12px;
    }
    .pill{
      display:inline-flex; align-items:center; gap:8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(230,193,128,0.25);
      background: rgba(230,193,128,0.10);
      font-size: 12px; font-weight: 700;
    }
    .row{ display:flex; justify-content:space-between; align-items:center; gap: 12px; }
    .btn{
      width:100%;
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.06);
      border-radius: 18px;
      padding: 14px;
      display:flex;
      justify-content:space-between;
      align-items:center;
      cursor:pointer;
      user-select:none;
    }
    .btn:active{ transform: translateY(1px); }
    .btnTitle{ font-size: 15px; font-weight: 750; }
    .btnSub{ margin-top: 4px; font-size: 12px; color: var(--muted); }
    .grid{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .tile{
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.06);
      border-radius: 18px;
      padding: 12px;
      cursor:pointer;
      user-select:none;
      min-height: 82px;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
    }
    .tileTitle{ font-size: 14px; font-weight: 800; }
    .tileSub{ font-size: 12px; color: var(--muted); margin-top: 6px; line-height: 1.25; }
    .hr{ height:1px; background: var(--stroke); margin: 14px 0; opacity: 0.8; }

    .hScroll{ display:flex; gap: 10px; overflow:auto; padding-bottom: 8px; -webkit-overflow-scrolling: touch; }
    .hScroll::-webkit-scrollbar{ display:none; }
    .thumbWrap{
  width: 100%;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.05);
  position: relative;
  aspect-ratio: 16 / 10;
  margin-bottom: 10px;
}
.thumbImg{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display:block;
  transform: scale(1.02);
  filter: saturate(1.05) contrast(1.02);
}
.thumbOverlay{
  position:absolute; inset:0;
  background: linear-gradient(180deg, rgba(0,0,0,0.00) 35%, rgba(0,0,0,0.72) 100%);
  pointer-events:none;
}
.thumbBadge{
  position:absolute;
  left:10px; bottom:10px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.20);
  background: rgba(18,22,30,0.55);
  backdrop-filter: blur(14px) saturate(160%);
  -webkit-backdrop-filter: blur(14px) saturate(160%);
  font-size: 12px;
  font-weight: 850;
  color: rgba(255,255,255,0.92);
  max-width: calc(100% - 20px);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.thumbFallback{
  width:100%;
  height:100%;
  display:flex;
  align-items:center;
  justify-content:center;
  color: rgba(255,255,255,0.70);
  font-weight: 900;
  letter-spacing: 0.8px;
  background:
    radial-gradient(420px 240px at 30% 10%, rgba(230,193,128,0.22), transparent 60%),
    radial-gradient(380px 220px at 80% 0%, rgba(255,255,255,0.12), transparent 55%),
    rgba(255,255,255,0.04);
}
.thumbNS{
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:6px;
  text-align:center;
  padding: 10px;
}
.thumbNS .mark{ font-size: 18px; }
.thumbNS .brand{ font-size: 12px; color: rgba(255,255,255,0.72); font-weight: 800; }

.miniCard{
      min-width: 220px;
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.06);
      border-radius: 18px;
      padding: 12px;
      cursor:pointer;
      user-select:none;
    }
    .miniMeta{ font-size: 12px; color: var(--muted); }
    .miniText{ margin-top: 8px; font-size: 14px; line-height: 1.3; }
    .postImg{ width:100%; height: 140px; object-fit: cover; border-radius: 16px; border: 1px solid var(--stroke); background: rgba(255,255,255,0.06); }
    .postImgSm{ width:100%; height: 110px; object-fit: cover; border-radius: 14px; border: 1px solid var(--stroke); background: rgba(255,255,255,0.06); }

    .chipRow{ margin-top: 10px; display:flex; gap: 6px; flex-wrap: wrap; }
    .chip{
      font-size: 12px;
      padding: 5px 8px;
      border-radius: 999px;
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.08);
    }

    .bottomNav{
      position: fixed;
      left: 0; right: 0; bottom: 0;
      padding: 10px 12px calc(10px + env(safe-area-inset-bottom));
      display:flex;
      justify-content:center;
      z-index: 9000;
      pointer-events: none;
    }
    .bottomNavInner{
      pointer-events: auto;
      width: min(560px, calc(100% - 24px));
      display:flex;
      gap: 10px;
      padding: 10px;
      border-radius: 22px;
      border: 1px solid var(--glassStroke);
      background: rgba(18,22,30,0.55);
      backdrop-filter: blur(22px) saturate(180%);
      -webkit-backdrop-filter: blur(22px) saturate(180%);
      box-shadow: 0 12px 40px var(--glassShadow);
    }
    .navItem{
      flex:1;
      border-radius: 16px;
      padding: 10px 8px;
      text-align:center;
      cursor:pointer;
      user-select:none;
      border: 1px solid transparent;
      background: rgba(255,255,255,0.05);
      display:flex;
      flex-direction:column;
      gap: 6px;
      align-items:center;
      justify-content:center;
    }
    .navItemActive{
      border: 1px solid rgba(230,193,128,0.35);
      background: rgba(230,193,128,0.12);
    }
    .navIcon{ font-size: 18px; line-height: 1; }
    .navLabel{ font-size: 11px; color: var(--muted); }
    .navItemActive .navLabel{ color: rgba(255,255,255,0.85); }

    .sheetOverlay{
      position: fixed; inset: 0;
      background: var(--sheetOverlay);
      backdrop-filter: blur(22px) saturate(180%);
      -webkit-backdrop-filter: blur(22px) saturate(180%);
      z-index: 9999;
      display:flex;
      justify-content:center;
      align-items:flex-end;
      padding: 10px;
    }
    .sheet{
      width: 100%;
      max-width: 560px;
      border-radius: 22px 22px 18px 18px;
      border: 1px solid var(--glassStroke);
      background: var(--sheetCardBg);
      backdrop-filter: blur(28px) saturate(180%);
      -webkit-backdrop-filter: blur(28px) saturate(180%);
      box-shadow: 0 12px 40px var(--glassShadow);
      padding: 14px 14px 10px;
      max-height: 84vh;
      overflow:auto;
    }
    .sheetHandle{
      width: 46px; height: 5px; border-radius: 999px;
      background: rgba(255,255,255,0.22);
      margin: 0 auto 10px;
    }
    .input{
      width: 100%;
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.06);
      border-radius: 16px;
      padding: 12px 12px;
      outline: none;
      color: var(--text);
      font-size: 14px;
    }
    .seg{
      display:flex; gap: 8px;
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.05);
      padding: 6px;
      border-radius: 18px;
    }
    .segBtn{
      flex:1;
      padding: 10px;
      border-radius: 14px;
      text-align:center;
      cursor:pointer;
      user-select:none;
      font-size: 13px;
      border: 1px solid transparent;
      color: var(--muted);
      background: transparent;
    }
    .segBtnActive{
      border: 1px solid rgba(230,193,128,0.35);
      background: rgba(230,193,128,0.12);
      color: rgba(255,255,255,0.9);
      font-weight: 750;
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useEffect, useMemo, useState } = React;
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
    const setVar = (k, v) => document.documentElement.style.setProperty(k, v);

    const applyTelegramTheme = () => {
      const scheme = tg?.colorScheme || "dark";
      const p = tg?.themeParams || {};
      const bg = p.bg_color || DEFAULT_BG;
      const text = p.text_color || (scheme === "dark" ? "rgba(255,255,255,0.92)" : "rgba(17,17,17,0.92)");
      const muted = p.hint_color || (scheme === "dark" ? "rgba(255,255,255,0.60)" : "rgba(0,0,0,0.55)");

      setVar("--bg", bg);
      setVar("--text", text);
      setVar("--muted", muted);
      setVar("--stroke", scheme === "dark" ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.10)");
      setVar("--card", scheme === "dark" ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.72)");
      setVar("--card2", scheme === "dark" ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.82)");

      setVar("--sheetOverlay", scheme === "dark" ? hexToRgba(bg, 0.55) : hexToRgba(bg, 0.45));
      setVar("--sheetCardBg", scheme === "dark" ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.86)");
      setVar("--glassStroke", scheme === "dark" ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.10)");
      setVar("--glassShadow", scheme === "dark" ? "rgba(0,0,0,0.45)" : "rgba(0,0,0,0.18)");

      if (tg) {
        tg.setHeaderColor(bg);
        tg.setBackgroundColor(bg);
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

    const haptic = (kind="light") => { try { tg?.HapticFeedback?.impactOccurred?.(kind); } catch(e){} };

    const Sheet = ({ open, onClose, children }) => {
      if (!open) return null;
      return (
        <div className="sheetOverlay" onClick={onClose}>
          <div className="sheet" onClick={(e) => e.stopPropagation()}>
            <div className="sheetHandle" />
            {children}
          </div>
        </div>
      );
    };

    const LockedClaimModal = ({ open, message, claimCode, onOk, onClaim }) => {
      if (!open) return null;
      return (
        <div
          onClick={(e) => { if (e.target === e.currentTarget) { /* no close */ } }}
          style={{
            position:"fixed", inset:0, background:"var(--sheetOverlay)",
            backdropFilter:"blur(22px) saturate(180%)",
            WebkitBackdropFilter:"blur(22px) saturate(180%)",
            zIndex:10000, display:"flex", justifyContent:"center", alignItems:"center", padding:"16px"
          }}
        >
          <div style={{
            width:"100%", maxWidth:"560px",
            borderRadius:"22px", border:"1px solid var(--glassStroke)",
            background:"var(--sheetCardBg)",
            backdropFilter:"blur(28px) saturate(180%)",
            WebkitBackdropFilter:"blur(28px) saturate(180%)",
            boxShadow:"0 12px 40px var(--glassShadow)",
            padding:"16px"
          }}>
            <div style={{ fontSize:"18px", fontWeight:850, marginBottom:"10px" }}>🎡 Рулетка</div>
            <div style={{ fontSize:"14px", lineHeight:"1.4", whiteSpace:"pre-line" }}>{message}</div>

            <div style={{ display:"flex", gap:"10px", marginTop:"16px" }}>
              <div onClick={onOk} className="btn" style={{ justifyContent:"center", fontWeight:850 }}>OK</div>
              <div
                onClick={onClaim}
                className="btn"
                style={{
                  justifyContent:"center",
                  fontWeight:900,
                  border:"1px solid rgba(230,193,128,0.35)",
                  background:"rgba(230,193,128,0.14)"
                }}
              >
                Получить приз
              </div>
            </div>
          </div>
        </div>
      );
    };

    const ConfirmClaimModal = ({ open, title, message, onCancel, onConfirm }) => {
      if (!open) return null;
      return (
        <div
          onClick={(e) => { /* no close */ }}
          style={{
            position:"fixed", inset:0, background:"var(--sheetOverlay)",
            backdropFilter:"blur(22px) saturate(180%)",
            WebkitBackdropFilter:"blur(22px) saturate(180%)",
            zIndex:10001, display:"flex", justifyContent:"center", alignItems:"center", padding:"16px"
          }}
        >
          <div style={{
            width:"100%", maxWidth:"560px",
            borderRadius:"22px", border:"1px solid var(--glassStroke)",
            background:"var(--sheetCardBg)",
            backdropFilter:"blur(28px) saturate(180%)",
            WebkitBackdropFilter:"blur(28px) saturate(180%)",
            boxShadow:"0 12px 40px var(--glassShadow)",
            padding:"16px"
          }}>
            <div style={{ fontSize:"18px", fontWeight:900, marginBottom:"10px" }}>{title || "Подтверждение"}</div>
            <div style={{ fontSize:"14px", lineHeight:"1.4", whiteSpace:"pre-line" }}>{message || ""}</div>
            <div style={{ display:"flex", gap:"10px", marginTop:"16px" }}>
              <div onClick={onCancel} className="btn" style={{ justifyContent:"center", fontWeight:900 }}>Отмена</div>
              <div
                onClick={onConfirm}
                className="btn"
                style={{
                  justifyContent:"center",
                  fontWeight:950,
                  border:"1px solid rgba(230,193,128,0.35)",
                  background:"rgba(230,193,128,0.14)"
                }}
              >
                Да, забрать
              </div>
            </div>
          </div>
        </div>
      );
    };

    const PostMiniCard = ({ post }) => {
  const [imgOk, setImgOk] = React.useState(true);
  const hasImg = !!post.media_url;
  const tagTitle = "#" + (post.tags?.[0] || "post");
  return (
    <div className="miniCard" onClick={() => { haptic(); openLink(post.url); }}>
      <div className="thumbWrap">
        {hasImg && imgOk ? (
          <img
            className="thumbImg"
            src={post.media_url}
            loading="lazy"
            onError={() => setImgOk(false)}
            alt={post.preview || tagTitle}
          />
        ) : (
          <div className="thumbFallback">
            <div className="thumbNS">
              <div className="mark">NS</div>
              <div className="brand">Natural Sense</div>
            </div>
          </div>
        )}
        <div className="thumbOverlay" />
        <div className="thumbBadge">{tagTitle}</div>
      </div>

      <div className="miniMeta">{tagTitle} • ID {post.message_id}</div>
      <div className="miniText">{post.preview || "Открыть пост →"}</div>
      <div className="chipRow">
        {(post.tags || []).slice(0,4).map(t => <div key={t} className="chip">#{t}</div>)}
      </div>
    </div>
  );
};

const PostFullCard = ({ post }) => {
  const [imgOk, setImgOk] = React.useState(true);
  const hasImg = !!post.media_url;
  const tagTitle = "#" + (post.tags?.[0] || "post");
  return (
    <div className="card2" style={{ cursor:"pointer" }} onClick={() => { haptic(); openLink(post.url); }}>
      <div className="thumbWrap" style={{ aspectRatio: "16 / 9" }}>
        {hasImg && imgOk ? (
          <img
            className="thumbImg"
            src={post.media_url}
            loading="lazy"
            onError={() => setImgOk(false)}
            alt={post.preview || tagTitle}
          />
        ) : (
          <div className="thumbFallback">
            <div className="thumbNS">
              <div className="mark">NS</div>
              <div className="brand">Natural Sense</div>
            </div>
          </div>
        )}
        <div className="thumbOverlay" />
        <div className="thumbBadge">{tagTitle}</div>
      </div>

      <div className="miniMeta">{tagTitle} • ID {post.message_id}</div>
      <div className="miniText">{post.preview || "Открыть пост →"}</div>
      <div className="chipRow">
        {(post.tags || []).slice(0,8).map(t => <div key={t} className="chip">#{t}</div>)}
      </div>
    </div>
  );
};

 = ({ post }) => (
      <div className="card2" style={{ cursor:"pointer" }} onClick={() => { haptic(); openLink(post.url); }}>
        <div className="miniMeta">{"#" + (post.tags?.[0] || "post")} • ID {post.message_id}</div>
        {post.media_url ? (
          <img className="postImg" src={post.media_url} alt="" loading="lazy" />
        ) : null}
        <div className="miniText">{post.preview || "Открыть пост →"}</div>
        <div className="chipRow">
          {(post.tags || []).slice(0,8).map(t => <div key={t} className="chip">#{t}</div>)}
        </div>
      </div>
    );

    const BottomNav = ({ tab, onTab }) => {
      const items = [
        { id:"journal", icon:"📰", label:"Journal" },
        { id:"discover", icon:"🧭", label:"Discover" },
        { id:"rewards", icon:"🎁", label:"Rewards" },
        { id:"profile", icon:"👤", label:"Profile" },
      ];
      return (
        <div className="bottomNav">
          <div className="bottomNavInner">
            {items.map(it => (
              <div
                key={it.id}
                className={"navItem " + (tab === it.id ? "navItemActive" : "")}
                onClick={() => { haptic(); onTab(it.id); }}
              >
                <div className="navIcon">{it.icon}</div>
                <div className="navLabel">{it.label}</div>
              </div>
            ))}
          </div>
        </div>
      );
    };

    const PrizeTable = () => (
      <div style={{ marginTop:"10px" }}>
        <div className="sub">Шансы рулетки (честно):</div>
        <div style={{ marginTop:"10px", display:"grid", gap:"8px" }}>
          {[
            ["50%", "+500"],
            ["35%", "+1000"],
            ["15%", "+1500"],
            ["10%", "+2000"],
            ["5%", "🎟 +1 билет"],
            ["3.5%", "+3000"],
            ["1.5%", "💎 главный приз"],
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
              <div style={{ fontWeight:700 }}>{t}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop:"10px", fontSize:"12px", color:"var(--muted)" }}>
          Лимит: 1 спин / 5с (тест)
        </div>
      </div>
    );

    const App = () => {
      const [tab, setTab] = useState("journal");

      // core data
      const [user, setUser] = useState(null);
      const [botUsername, setBotUsername] = useState(BOT_USERNAME || "");
      const tgUserId = tg?.initDataUnsafe?.user?.id;

      // overlays / modes
      const [postsSheet, setPostsSheet] = useState({ open:false, tag:null, title:"" });
      const [posts, setPosts] = useState([]);
      const [loadingPosts, setLoadingPosts] = useState(false);

      const [inventoryOpen, setInventoryOpen] = useState(false);
      const [inventory, setInventory] = useState(null);
      const [ticketQty, setTicketQty] = useState(1);
      const [invMsg, setInvMsg] = useState("");

      const [profileOpen, setProfileOpen] = useState(false);
      const [profileView, setProfileView] = useState("menu"); // menu|raffle|roulette|history

      const [raffle, setRaffle] = useState(null);
      const [rouletteHistory, setRouletteHistory] = useState([]);
      const [busy, setBusy] = useState(false);
      const [msg, setMsg] = useState("");

      const [claimModal, setClaimModal] = useState({ open:false, message:"", claim_code:"" });
      const [confirmClaim, setConfirmClaim] = useState({ open:false, claim_code:"", prize_label:"" });

      // Discover
      const [discoverMode, setDiscoverMode] = useState("brands"); // brands|categories
      const [q, setQ] = useState("");

      const refreshUser = () => {
        if (!tgUserId) return Promise.resolve();
        return fetch(`/api/user/${tgUserId}`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setUser(data))
          .catch(() => {});
      };

      const loadPosts = (tag) => {
        if (!tag) return;
        setLoadingPosts(true);
        fetch(`/api/posts?tag=${encodeURIComponent(tag)}`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setPosts(Array.isArray(data) ? data : []))
          .catch(() => setPosts([]))
          .finally(() => setLoadingPosts(false));
      };

      const openPosts = (tag, title) => {
        setPostsSheet({ open:true, tag, title: title || ("#" + tag) });
        loadPosts(tag);
      };

      const closePosts = () => {
        setPostsSheet({ open:false, tag:null, title:"" });
        setPosts([]);
        setLoadingPosts(false);
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

      const loadInventory = () => {
        if (!tgUserId) return Promise.resolve();
        return fetch(`/api/inventory?telegram_id=${encodeURIComponent(tgUserId)}`)
          .then(r => r.ok ? r.json() : Promise.reject())
          .then(data => setInventory(data))
          .catch(() => setInventory(null));
      };

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
          haptic("light");
          return;
        } catch (e) {
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
            if (ok) haptic("light");
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
          setRaffle((prev) => ({ ...(prev || {}), ticket_count: data.ticket_count }));
          await refreshUser();
          await loadRaffleStatus();
          haptic("light");
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

          try {
            if (data.claimable && data.claim_code) {
              const m = `Ваш приз: ${data.prize_label}

Чтобы забрать: откройте чат с ботом и отправьте
/claim ${data.claim_code}`;
              setClaimModal({ open:true, message:m, claim_code:data.claim_code });
            } else if (tg?.showPopup) {
              tg.showPopup({
                title: "🎡 Рулетка",
                message: `Ваш приз: ${data.prize_label}`,
                buttons: [{ type: "ok" }]
              });
            } else {
              alert(`Ваш приз: ${data.prize_label}`);
            }
          } catch (e) {}

          await refreshUser();
          await loadRaffleStatus();
          await loadRouletteHistory();
          haptic("light");
        } catch (e) {
          setMsg(`❌ ${e.message || "Ошибка"}`);
        } finally {
          setBusy(false);
        }
      };

      const incTicketQty = () => {
        const max = Math.max(1, Number(inventory?.ticket_count || 0));
        setTicketQty((x) => Math.min(max, x + 1));
      };
      const decTicketQty = () => setTicketQty((x) => Math.max(1, x - 1));
      const maxTicketQty = () => setTicketQty(Math.max(1, Number(inventory?.ticket_count || 0)));

      const convertTickets = async () => {
        if (!tgUserId) return;
        const have = Number(inventory?.ticket_count || 0);
        const qty = Math.max(1, Math.min(have, Number(ticketQty || 1)));
        if (!have) return;

        setBusy(true);
        setInvMsg("");
        try {
          const r = await fetch(`/api/inventory/convert_ticket`, {
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify({ telegram_id: tgUserId, qty })
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка");
          }
          const data = await r.json();
          setInvMsg(`✅ Обмен выполнен: +${data.added_points} баллов`);
          await refreshUser();
          await loadRaffleStatus();
          await loadInventory();
          haptic("light");
        } catch (e) {
          setInvMsg(`❌ ${e.message || "Ошибка"}`);
        } finally {
          setBusy(false);
        }
      };

      const convertPrize = async (claimCode) => {
        if (!tgUserId) return;
        const code = String(claimCode || "").trim();
        if (!code) return;

        setBusy(true);
        setInvMsg("");
        try {
          const r = await fetch(`/api/inventory/convert_prize`, {
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify({ telegram_id: tgUserId, claim_code: code })
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка");
          }
          const data = await r.json();
          setInvMsg(`✅ Приз превращён в бонусы: +${data.added_points} баллов`);
          await refreshUser();
          await loadInventory();
          haptic("light");
        } catch (e) {
          setInvMsg(`❌ ${e.message || "Ошибка"}`);
        } finally {
          setBusy(false);
        }
      };

      // bootstrap
      useEffect(() => {
        if (tgUserId) refreshUser();
      }, []);

      useEffect(() => {
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

      useEffect(() => {
        if (inventoryOpen) {
          setTicketQty(1);
          setInvMsg("");
          loadInventory();
        }
      }, [inventoryOpen]);

      // tab behavior
      useEffect(() => {
        if (tab === "profile") {
          setProfileOpen(true);
          setProfileView("menu");
          setTab("journal");
        }
      }, [tab]);

      // curated blocks for Journal
      const JOURNAL_BLOCKS = [
        { tag: "Новинка", title: "🆕 New arrivals" },
        { tag: "Люкс", title: "💎 Luxury picks" },
        { tag: "Тренд", title: "🔥 Trending" },
        { tag: "Оценка", title: "⭐ Personal review" },
        { tag: "Факты", title: "🧾 Facts" },
      ];

      const [blockPosts, setBlockPosts] = useState({}); // tag -> posts[]
      const loadJournalBlocks = () => {
        JOURNAL_BLOCKS.forEach(async (b) => {
          try {
            const r = await fetch(`/api/posts?tag=${encodeURIComponent(b.tag)}`);
            const data = r.ok ? await r.json() : [];
            setBlockPosts((prev) => ({ ...prev, [b.tag]: Array.isArray(data) ? data.slice(0, 8) : [] }));
          } catch(e) {
            setBlockPosts((prev) => ({ ...prev, [b.tag]: [] }));
          }
        });
      };

      useEffect(() => {
        loadJournalBlocks();
      }, []);

      // Discover datasets
      const BRANDS = [
        ["The Ordinary", "TheOrdinary", "Skincare essentials"],
        ["Dior", "Dior", "Couture beauty"],
        ["Chanel", "Chanel", "Iconic classics"],
        ["Kylie Cosmetics", "KylieCosmetics", "Pop-glam"],
        ["Gisou", "Gisou", "Honey haircare"],
        ["Rare Beauty", "RareBeauty", "Soft-focus makeup"],
        ["Yves Saint Laurent", "YSL", "Bold luxury"],
        ["Givenchy", "Givenchy", "Haute beauty"],
        ["Charlotte Tilbury", "CharlotteTilbury", "Red carpet glow"],
        ["NARS", "NARS", "Editorial makeup"],
        ["Sol de Janeiro", "SolDeJaneiro", "Body & scent"],
        ["Huda Beauty", "HudaBeauty", "Full glam"],
        ["Rhode", "Rhode", "Minimal skincare"],
        ["Tower 28 Beauty", "Tower28Beauty", "Sensitive skin"],
        ["Benefit Cosmetics", "BenefitCosmetics", "Brows & cheeks"],
        ["Estée Lauder", "EsteeLauder", "Skincare icons"],
        ["Sisley", "Sisley", "Ultra premium"],
        ["Kérastase", "Kerastase", "Salon haircare"],
        ["Armani Beauty", "ArmaniBeauty", "Soft luxury"],
        ["Hourglass", "Hourglass", "Ambient glow"],
        ["Shiseido", "Shiseido", "Japanese skincare"],
        ["Tom Ford Beauty", "TomFordBeauty", "Private blend vibe"],
        ["Tarte", "Tarte", "Everyday glam"],
        ["Sephora Collection", "SephoraCollection", "Smart basics"],
        ["Clinique", "Clinique", "Skin first"],
        ["Dolce & Gabbana", "DolceGabbana", "Italian glamour"],
        ["Kayali", "Kayali", "Fragrance focus"],
        ["Guerlain", "Guerlain", "Heritage luxury"],
        ["Fenty Beauty", "FentyBeauty", "Inclusive glam"],
        ["Too Faced", "TooFaced", "Playful makeup"],
        ["MAKE UP FOR EVER", "MakeUpForEver", "Pro artistry"],
        ["Erborian", "Erborian", "K-beauty meets EU"],
        ["Natasha Denona", "NatashaDenona", "Palette queen"],
        ["Lancôme", "Lancome", "French classics"],
        ["Kosas", "Kosas", "Clean makeup"],
        ["ONE/SIZE", "OneSize", "Stage-ready"],
        ["Laneige", "Laneige", "Hydration"],
        ["Makeup by Mario", "MakeupByMario", "Artist essentials"],
        ["Valentino Beauty", "ValentinoBeauty", "Couture color"],
        ["Drunk Elephant", "DrunkElephant", "Active skincare"],
        ["Olaplex", "Olaplex", "Bond repair"],
        ["Anastasia Beverly Hills", "AnastasiaBeverlyHills", "Brows & glam"],
        ["Amika", "Amika", "Hair styling"],
        ["BYOMA", "BYOMA", "Barrier care"],
        ["Glow Recipe", "GlowRecipe", "Fruity glow"],
        ["Milk Makeup", "MilkMakeup", "Cool minimal"],
        ["Summer Fridays", "SummerFridays", "Clean glow"],
        ["K18", "K18", "Repair tech"],
      ];

      const CATEGORIES = [
        ["Новинка", "Новинка", "New launches"],
        ["Люкс", "Люкс", "Luxury picks"],
        ["Тренд", "Тренд", "What’s trending"],
        ["История", "История", "Brand stories"],
        ["Оценка", "Оценка", "Personal reviews"],
        ["Факты", "Факты", "Short facts"],
        ["Состав", "Состав", "Ingredients / formulas"],
        ["Challenge", "Challenge", "Beauty challenges"],
        ["SephoraPromo", "SephoraPromo", "Sephora promos"],
      ];

      const PRODUCTS = [
        ["Праймер", "Праймер"], ["Тональная основа", "ТональнаяОснова"], ["Консилер", "Консилер"],
        ["Пудра", "Пудра"], ["Румяна", "Румяна"], ["Скульптор", "Скульптор"], ["Бронзер", "Бронзер"],
        ["Продукт для бровей", "ПродуктДляБровей"], ["Хайлайтер", "Хайлайтер"], ["Тушь", "Тушь"],
        ["Тени", "Тени"], ["Помада", "Помада"], ["Карандаш для губ", "КарандашДляГуб"], ["Палетка", "Палетка"], ["Фиксатор", "Фиксатор"],
      ];

      const filteredBrands = useMemo(() => {
        const s = q.trim().toLowerCase();
        if (!s) return BRANDS;
        return BRANDS.filter(([name, tag, sub]) =>
          name.toLowerCase().includes(s) || tag.toLowerCase().includes(s) || String(sub||"").toLowerCase().includes(s)
        );
      }, [q]);

      const filteredCats = useMemo(() => {
        const s = q.trim().toLowerCase();
        if (!s) return CATEGORIES;
        return CATEGORIES.filter(([name, tag, sub]) =>
          name.toLowerCase().includes(s) || tag.toLowerCase().includes(s) || String(sub||"").toLowerCase().includes(s)
        );
      }, [q]);

      const openInventory = () => { setInventoryOpen(true); setProfileOpen(false); };
      const closeInventory = () => { setInventoryOpen(false); setInvMsg(""); };

      const Journal = () => (
        <div>
          <div className="card" onClick={() => { if (user) { haptic(); setProfileOpen(true); setProfileView("menu"); } }}>
            <div style={{
              position:"absolute", inset:"-2px",
              background:"radial-gradient(600px 300px at 10% 0%, rgba(230,193,128,0.26), transparent 60%)",
              pointerEvents:"none"
            }} />
            <div style={{ position:"relative" }}>
              <div className="row">
                <div>
                  <div className="h1">NS · Natural Sense</div>
                  <div className="sub">Today’s Edit · luxury beauty magazine</div>
                </div>
                {user && <div className="pill">💎 {user.points} · {tierLabel(user.tier)}</div>}
              </div>
              <div style={{ marginTop:"12px", color:"var(--muted)", fontSize:"13px" }}>
                Curated picks, short facts, luxury reviews — inside Telegram.
              </div>

              <div style={{ marginTop:"12px", display:"grid", gap:"10px" }}>
                <div className="btn" onClick={(e) => { e.stopPropagation(); haptic(); openLink(`https://t.me/${CHANNEL}`); }}>
                  <div>
                    <div className="btnTitle">↩️ Open Channel</div>
                    <div className="btnSub">Return to Natural Sense feed</div>
                  </div>
                  <div style={{ opacity:0.85 }}>›</div>
                </div>

                <div className="grid">
                  <div className="tile" onClick={(e) => { e.stopPropagation(); haptic(); openPosts("Новинка","🆕 New arrivals"); }}>
                    <div className="tileTitle">🆕 New</div>
                    <div className="tileSub">Fresh launches & updates</div>
                  </div>
                  <div className="tile" onClick={(e) => { e.stopPropagation(); haptic(); openPosts("Люкс","💎 Luxury picks"); }}>
                    <div className="tileTitle">💎 Luxury</div>
                    <div className="tileSub">Short & premium</div>
                  </div>
                </div>

                <div className="grid">
                  <div className="tile" onClick={(e) => { e.stopPropagation(); haptic(); openPosts("Тренд","🔥 Trending"); }}>
                    <div className="tileTitle">🔥 Trending</div>
                    <div className="tileSub">What everyone wants</div>
                  </div>
                  <div className="tile" onClick={(e) => { e.stopPropagation(); haptic(); openInventory(); }}>
                    <div className="tileTitle">👜 Bag</div>
                    <div className="tileSub">Tickets & prizes</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {JOURNAL_BLOCKS.map((b) => (
            <div key={b.tag} style={{ marginTop:"14px" }}>
              <div className="row" style={{ alignItems:"baseline" }}>
                <div style={{ fontSize:"15px", fontWeight:850 }}>{b.title}</div>
                <div
                  style={{ fontSize:"12px", color:"var(--muted)", cursor:"pointer", userSelect:"none" }}
                  onClick={() => { haptic(); openPosts(b.tag, b.title); }}
                >
                  View all ›
                </div>
              </div>
              <div style={{ marginTop:"10px" }} className="hScroll">
                {(blockPosts[b.tag] || []).length === 0 ? (
                  <div className="miniCard" style={{ minWidth:"100%", cursor:"default" }}>
                    <div className="miniMeta">Пока пусто</div>
                    <div className="miniText" style={{ color:"var(--muted)" }}>Добавь посты с тегом #{b.tag} в канал.</div>
                  </div>
                ) : (
                  (blockPosts[b.tag] || []).map((p) => <PostMiniCard key={p.message_id} post={p} />)
                )}
              </div>
            </div>
          ))}
        </div>
      );

      const Discover = () => (
        <div className="card2">
          <div className="row">
            <div>
              <div className="h1">Discover</div>
              <div className="sub">Brands · Categories · Products</div>
            </div>
            <div className="pill" onClick={() => { haptic(); setInventoryOpen(true); }} style={{ cursor:"pointer" }}>
              👜 Bag
            </div>
          </div>

          <div style={{ marginTop:"12px" }}>
            <input
              className="input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search brands / tags…"
            />
          </div>

          <div style={{ marginTop:"12px" }} className="seg">
            <div className={"segBtn " + (discoverMode==="brands" ? "segBtnActive" : "")} onClick={() => { haptic(); setDiscoverMode("brands"); }}>
              Brands
            </div>
            <div className={"segBtn " + (discoverMode==="categories" ? "segBtnActive" : "")} onClick={() => { haptic(); setDiscoverMode("categories"); }}>
              Categories
            </div>
          </div>

          <div style={{ marginTop:"12px" }} className="grid">
            {(discoverMode === "brands" ? filteredBrands : filteredCats).map(([name, tag, sub]) => (
              <div key={tag} className="tile" onClick={() => { haptic(); openPosts(tag, name); }}>
                <div className="tileTitle">{name}</div>
                <div className="tileSub">{sub || ("#" + tag)}</div>
              </div>
            ))}
          </div>

          <div className="hr" />

          <div style={{ fontSize:"14px", fontWeight:850 }}>🧴 Product types</div>
          <div className="sub" style={{ marginTop:"6px" }}>Quick access</div>
          <div style={{ marginTop:"10px" }} className="grid">
            {PRODUCTS.map(([name, tag]) => (
              <div key={tag} className="tile" onClick={() => { haptic(); openPosts(tag, name); }}>
                <div className="tileTitle">{name}</div>
                <div className="tileSub">#{tag}</div>
              </div>
            ))}
          </div>
        </div>
      );

      const Rewards = () => (
        <div className="card2">
          <div className="row">
            <div>
              <div className="h1">Rewards</div>
              <div className="sub">Roulette · Raffle · Inventory</div>
            </div>
            {user && <div className="pill">💎 {user.points} pts</div>}
          </div>

          <div style={{ marginTop:"12px" }} className="grid">
            <div className="tile" onClick={() => { haptic(); setProfileOpen(true); setProfileView("roulette"); }}>
              <div className="tileTitle">🎡 Roulette</div>
              <div className="tileSub">Try your luck (2000)</div>
            </div>
            <div className="tile" onClick={() => { haptic(); setProfileOpen(true); setProfileView("raffle"); }}>
              <div className="tileTitle">🎁 Raffle</div>
              <div className="tileSub">Ticket (500)</div>
            </div>
            <div className="tile" onClick={() => { haptic(); setInventoryOpen(true); }}>
              <div className="tileTitle">👜 Bag</div>
              <div className="tileSub">Tickets & prizes</div>
            </div>
            <div className="tile" onClick={() => { haptic(); openPosts("Challenge","💎 Beauty Challenges"); }}>
              <div className="tileTitle">💎 Challenges</div>
              <div className="tileSub">Daily motivation</div>
            </div>
          </div>

          <div className="hr" />

          <div className="btn" onClick={() => { haptic(); openLink(`https://t.me/${CHANNEL}`); }}>
            <div>
              <div className="btnTitle">↩️ Open Channel</div>
              <div className="btnSub">Natural Sense feed</div>
            </div>
            <div style={{ opacity:0.85 }}>›</div>
          </div>
        </div>
      );

      const PostsSheetContent = () => (
        <div>
          <div className="row" style={{ alignItems:"baseline" }}>
            <div className="h1">{postsSheet.title || "Posts"}</div>
            <div style={{ fontSize:"13px", color:"var(--muted)", cursor:"pointer" }} onClick={() => { haptic(); closePosts(); }}>
              Close
            </div>
          </div>
          <div className="sub" style={{ marginTop:"6px" }}>
            Посты {postsSheet.tag ? ("#" + postsSheet.tag) : ""}
          </div>

          {loadingPosts && (
            <div className="sub" style={{ marginTop:"12px" }}>Загрузка…</div>
          )}

          {!loadingPosts && posts.length === 0 && (
            <div className="sub" style={{ marginTop:"12px" }}>Постов с этим тегом пока нет.</div>
          )}

          <div style={{ marginTop:"12px", display:"grid", gap:"10px" }}>
            {posts.map(p => <PostFullCard key={p.message_id} post={p} />)}
          </div>
        </div>
      );

      const InventorySheetContent = () => {
        const rate = Number(inventory?.ticket_convert_rate || 0) || 0;
        const diorValue = Number(inventory?.dior_convert_value || 0) || 0;
        const haveTickets = Number(inventory?.ticket_count || 0) || 0;
        const qty = Math.max(1, Math.min(haveTickets || 1, Number(ticketQty || 1)));
        const calc = rate ? (qty * rate) : 0;

        const statusLabel = (s) => {
          const v = String(s || "");
          if (v === "awaiting_contact") return "⏳ Доступен";
          if (v === "submitted") return "⏳ Ожидает подтверждения";
          if (v === "closed") return "✅ Закрыт";
          return v || "-";
        };

        return (
          <div>
            <div className="row" style={{ alignItems:"baseline" }}>
              <div className="h1">👜 My Bag</div>
              <div style={{ fontSize:"13px", color:"var(--muted)", cursor:"pointer" }} onClick={() => { haptic(); closeInventory(); }}>
                Close
              </div>
            </div>
            <div className="sub" style={{ marginTop:"6px" }}>Tickets & prizes</div>

            <div style={{ marginTop:"12px" }} className="card2">
              <div className="row">
                <div>
                  <div style={{ fontSize:"13px", color:"var(--muted)" }}>Balance</div>
                  <div style={{ marginTop:"6px", fontSize:"16px", fontWeight:900 }}>💎 {user?.points ?? 0} pts</div>
                </div>
                <div className="pill">{tierLabel(user?.tier)}</div>
              </div>
            </div>

            <div style={{ marginTop:"12px" }} className="card2">
              <div style={{ fontSize:"14px", fontWeight:900 }}>🎟 Tickets</div>
              <div className="sub" style={{ marginTop:"6px" }}>You have: <b style={{ color:"rgba(255,255,255,0.92)" }}>{haveTickets}</b></div>
              <div className="sub" style={{ marginTop:"6px" }}>Rate: 1 = {rate} pts</div>

              <div style={{ marginTop:"10px", display:"flex", gap:"8px", alignItems:"center" }}>
                <div className="tile" style={{ width:"62px", minHeight:"44px", alignItems:"center", justifyContent:"center" }}
                     onClick={() => { if (haveTickets) { haptic(); decTicketQty(); } }}>
                  <div style={{ fontWeight:950, fontSize:"18px" }}>–</div>
                </div>

                <div className="tile" style={{ flex:1, minHeight:"44px", alignItems:"center", justifyContent:"center" }}>
                  <div style={{ fontWeight:950, fontSize:"15px" }}>{haveTickets ? qty : 0}</div>
                </div>

                <div className="tile" style={{ width:"62px", minHeight:"44px", alignItems:"center", justifyContent:"center" }}
                     onClick={() => { if (haveTickets) { haptic(); incTicketQty(); } }}>
                  <div style={{ fontWeight:950, fontSize:"18px" }}>+</div>
                </div>

                <div className="tile" style={{ width:"76px", minHeight:"44px" }}
                     onClick={() => { if (haveTickets) { haptic(); maxTicketQty(); } }}>
                  <div style={{ fontWeight:950, fontSize:"13px" }}>MAX</div>
                  <div className="tileSub" style={{ marginTop:"4px" }}>{haveTickets || 0}</div>
                </div>
              </div>

              <div className="sub" style={{ marginTop:"10px" }}>
                You’ll get: <b style={{ color:"rgba(255,255,255,0.92)" }}>{calc}</b> pts
              </div>

              <div
                className="btn"
                onClick={() => { if (!busy && haveTickets) { haptic(); convertTickets(); } }}
                style={{
                  marginTop:"10px",
                  opacity: (busy || !haveTickets) ? 0.5 : 1,
                  cursor: (busy || !haveTickets) ? "not-allowed" : "pointer"
                }}
              >
                <div>
                  <div className="btnTitle">💎 Convert tickets</div>
                  <div className="btnSub">{busy ? "Подожди…" : `Convert (${haveTickets ? qty : 0})`}</div>
                </div>
                <div style={{ opacity:0.85 }}>›</div>
              </div>
            </div>

            <div style={{ marginTop:"12px" }} className="card2">
              <div style={{ fontSize:"14px", fontWeight:900 }}>🎁 Prizes</div>

              {(!inventory?.prizes || inventory.prizes.length === 0) ? (
                <div className="sub" style={{ marginTop:"10px" }}>Пока нет призов.</div>
              ) : (
                <div style={{ marginTop:"10px", display:"grid", gap:"10px" }}>
                  {inventory.prizes.map((p) => (
                    <div key={p.claim_code} className="card2" style={{
                      border:"1px solid rgba(230,193,128,0.22)",
                      background:"rgba(230,193,128,0.10)"
                    }}>
                      <div style={{ fontSize:"14px", fontWeight:950 }}>{p.prize_label || "💎 Главный приз"}</div>
                      <div className="sub" style={{ marginTop:"6px" }}>
                        Status: {statusLabel(p.status)} • Code: {p.claim_code}
                      </div>

                      {(String(p.status||"") === "submitted" || String(p.status||"") === "closed") ? null : (
                        <div style={{ display:"flex", gap:"10px", marginTop:"12px" }}>
                          <div
                            className="btn"
                            style={{ justifyContent:"center", fontWeight:900 }}
                            onClick={() => {
                              const st = String(p.status || "");
                              if (st === "submitted" || st === "closed") return;
                              setConfirmClaim({ open:true, claim_code: p.claim_code, prize_label: p.prize_label || "Приз" });
                              haptic();
                            }}
                          >
                            🎁 Claim
                          </div>
                          <div
                            className="btn"
                            style={{
                              justifyContent:"center",
                              fontWeight:950,
                              border:"1px solid rgba(230,193,128,0.35)",
                              background:"rgba(230,193,128,0.14)"
                            }}
                            onClick={() => { haptic(); convertPrize(p.claim_code); }}
                          >
                            💎 Convert (+{diorValue})
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {invMsg && (
              <div style={{ marginTop:"12px" }} className="card2">
                <div style={{ fontSize:"13px" }}>{invMsg}</div>
              </div>
            )}
          </div>
        );
      };

      const ProfileSheetContent = () => {
        if (!user) {
          return <div className="sub">Профиль недоступен.</div>;
        }

        const StatRow = ({ left, right }) => (
          <div className="row" style={{ marginTop:"10px", fontSize:"14px" }}>
            <div style={{ color:"var(--muted)" }}>{left}</div>
            <div style={{ fontWeight:800 }}>{right}</div>
          </div>
        );

        return (
          <div>
            <div className="row" style={{ alignItems:"baseline" }}>
              <div className="h1">👤 Profile</div>
              <div style={{ fontSize:"13px", color:"var(--muted)", cursor:"pointer" }} onClick={() => { haptic(); setProfileOpen(false); }}>
                Close
              </div>
            </div>
            <div className="sub" style={{ marginTop:"6px" }}>Members area</div>

            <div style={{ marginTop:"12px" }} className="card2">
              <div style={{ position:"relative" }}>
                <div style={{
                  position:"absolute", top:"0", right:"0",
                  padding:"6px 10px",
                  borderRadius:"999px",
                  border:"1px solid rgba(230,193,128,0.25)",
                  background:"rgba(230,193,128,0.10)",
                  fontSize:"13px",
                  fontWeight:850
                }}>
                  💎 {user.points}
                </div>

                <div style={{ fontSize:"13px", color:"var(--muted)" }}>Hello, {user.first_name}!</div>
                <div style={{ marginTop:"6px", fontSize:"13px", color:"var(--muted)" }}>{tierLabel(user.tier)}</div>

                <StatRow left="🔥 Streak" right={`${user.daily_streak || 0} (best ${user.best_streak || 0})`} />
                <StatRow left="🎟 Referrals" right={`${user.referral_count || 0}`} />
              </div>
            </div>

            <div className="hr" />

            <div style={{ fontSize:"14px", fontWeight:900 }}>🎟 Invite</div>
            <div className="sub" style={{ marginTop:"6px" }}>+20 points for each new user (once).</div>
            {botUsername ? (
              <div style={{ marginTop:"10px" }} className="card2">
                <div style={{ fontSize:"12px", color:"rgba(255,255,255,0.85)", wordBreak:"break-all" }}>{referralLink}</div>
              </div>
            ) : (
              <div className="sub" style={{ marginTop:"10px" }}>
                Если ссылка не показалась — задай переменную окружения <b>BOT_USERNAME</b>.
              </div>
            )}
            <div
              className="btn"
              style={{ marginTop:"10px", opacity: (!botUsername || !referralLink) ? 0.5 : 1, cursor: (!botUsername || !referralLink) ? "not-allowed" : "pointer" }}
              onClick={() => { if (botUsername && referralLink) { haptic(); copyText(referralLink); } }}
            >
              <div>
                <div className="btnTitle">📎 Copy link</div>
                <div className="btnSub">{msg || "Copy to clipboard"}</div>
              </div>
              <div style={{ opacity:0.85 }}>›</div>
            </div>

            <div className="hr" />

            {profileView === "menu" ? (
              <div style={{ display:"grid", gap:"10px" }}>
                <div className="btn" onClick={() => { haptic(); setInventoryOpen(true); setProfileOpen(false); }}>
                  <div>
                    <div className="btnTitle">👜 My Bag</div>
                    <div className="btnSub">Tickets & prizes</div>
                  </div>
                  <div style={{ opacity:0.85 }}>›</div>
                </div>
                <div className="btn" onClick={() => { haptic(); setProfileView("raffle"); }}>
                  <div>
                    <div className="btnTitle">🎁 Raffle</div>
                    <div className="btnSub">Buy tickets (500)</div>
                  </div>
                  <div style={{ opacity:0.85 }}>›</div>
                </div>
                <div className="btn" onClick={() => { haptic(); setProfileView("roulette"); }}>
                  <div>
                    <div className="btnTitle">🎡 Roulette</div>
                    <div className="btnSub">Spin (2000)</div>
                  </div>
                  <div style={{ opacity:0.85 }}>›</div>
                </div>
                <div className="btn" onClick={() => { haptic(); setProfileView("history"); }}>
                  <div>
                    <div className="btnTitle">🧾 Roulette history</div>
                    <div className="btnSub">Last spins</div>
                  </div>
                  <div style={{ opacity:0.85 }}>›</div>
                </div>
              </div>
            ) : (
              <div>
                <div
                  className="btn"
                  style={{ justifyContent:"center", fontWeight:900 }}
                  onClick={() => { haptic(); setProfileView("menu"); setMsg(""); }}
                >
                  ← Back
                </div>

                {profileView === "raffle" && (
                  <div style={{ marginTop:"12px" }}>
                    <div style={{ fontSize:"14px", fontWeight:900 }}>🎁 Raffle</div>
                    <div className="sub" style={{ marginTop:"6px" }}>Ticket = 500 points.</div>
                    <div className="sub" style={{ marginTop:"8px" }}>
                      Your tickets: <b style={{ color:"rgba(255,255,255,0.92)" }}>{raffle?.ticket_count ?? 0}</b>
                    </div>

                    <div
                      className="btn"
                      style={{
                        marginTop:"10px",
                        opacity: (busy || (user.points || 0) < 500) ? 0.5 : 1,
                        cursor: (busy || (user.points || 0) < 500) ? "not-allowed" : "pointer"
                      }}
                      onClick={() => { if (!busy && (user.points || 0) >= 500) { haptic(); buyTicket(); } }}
                    >
                      <div>
                        <div className="btnTitle">🎟 Buy ticket</div>
                        <div className="btnSub">{busy ? "Подожди…" : "Spend 500 points"}</div>
                      </div>
                      <div style={{ opacity:0.85 }}>›</div>
                    </div>

                    {msg && <div style={{ marginTop:"12px" }} className="card2">{msg}</div>}
                  </div>
                )}

                {profileView === "roulette" && (
                  <div style={{ marginTop:"12px" }}>
                    <div style={{ fontSize:"14px", fontWeight:900 }}>🎡 Roulette</div>
                    <div className="sub" style={{ marginTop:"6px" }}>Spin = 2000 points.</div>

                    <div
                      className="btn"
                      style={{
                        marginTop:"10px",
                        opacity: (busy || (user.points || 0) < 2000) ? 0.5 : 1,
                        cursor: (busy || (user.points || 0) < 2000) ? "not-allowed" : "pointer"
                      }}
                      onClick={() => { if (!busy && (user.points || 0) >= 2000) { haptic(); spinRoulette(); } }}
                    >
                      <div>
                        <div className="btnTitle">🎡 Spin</div>
                        <div className="btnSub">{busy ? "Подожди…" : "Try your luck"}</div>
                      </div>
                      <div style={{ opacity:0.85 }}>›</div>
                    </div>

                    <PrizeTable />
                    {msg && <div style={{ marginTop:"12px" }} className="card2">{msg}</div>}
                  </div>
                )}

                {profileView === "history" && (
                  <div style={{ marginTop:"12px" }}>
                    <div style={{ fontSize:"14px", fontWeight:900 }}>🧾 History</div>
                    {(rouletteHistory || []).length === 0 ? (
                      <div className="sub" style={{ marginTop:"10px" }}>Пока пусто.</div>
                    ) : (
                      <div style={{ marginTop:"10px", display:"grid", gap:"10px" }}>
                        {rouletteHistory.map((x) => (
                          <div key={x.id} className="card2">
                            <div className="sub">{x.created_at}</div>
                            <div style={{ marginTop:"6px", fontSize:"14px", fontWeight:850 }}>{x.prize_label}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      };

      const MainScreen = () => {
        if (tab === "journal") return <Journal />;
        if (tab === "discover") return <Discover />;
        if (tab === "rewards") return <Rewards />;
        return <Journal />;
      };

      return (
        <div className="safePadBottom">
          <div className="container">
            <MainScreen />
          </div>

          <BottomNav tab={tab} onTab={setTab} />

          <Sheet open={postsSheet.open} onClose={() => { haptic(); closePosts(); }}>
            <PostsSheetContent />
          </Sheet>

          <Sheet open={inventoryOpen} onClose={() => { haptic(); closeInventory(); }}>
            <InventorySheetContent />
          </Sheet>

          <Sheet open={profileOpen} onClose={() => { haptic(); setProfileOpen(false); }}>
            <ProfileSheetContent />
          </Sheet>

          <LockedClaimModal
            open={claimModal.open}
            message={claimModal.message}
            claimCode={claimModal.claim_code}
            onOk={() => setClaimModal({ open:false, message:"", claim_code:"" })}
            onClaim={() => {
              if (botUsername && tg?.openTelegramLink && claimModal.claim_code) {
                tg.openTelegramLink(`https://t.me/${botUsername}?start=claim_${claimModal.claim_code}`);
              }
              setClaimModal({ open:false, message:"", claim_code:"" });
            }}
          />

          <ConfirmClaimModal
            open={confirmClaim.open}
            title="🎁 Забрать приз?"
            message={`Вы уверены?

После подтверждения вы НЕ сможете:
• превратить приз в бонусы
• отменить действие

Приз: ${confirmClaim.prize_label}
Код: ${confirmClaim.claim_code}`}
            onCancel={() => setConfirmClaim({ open:false, claim_code:"", prize_label:"" })}
            onConfirm={() => {
              const code = String(confirmClaim.claim_code || "").trim();
              if (botUsername && tg?.openTelegramLink && code) {
                tg.openTelegramLink(`https://t.me/${botUsername}?start=claim_${code}`);
              } else if (code) {
                alert(`/claim ${code}`);
              }
              setConfirmClaim({ open:false, claim_code:"", prize_label:"" });
            }}
          />
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
    return HTMLResponse(
        get_webapp_html(),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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
        media_type = (p.media_type or "").strip().lower()
        media_url = f"/api/post_media/{int(p.message_id)}" if (media_type == "photo" and p.media_file_id) else None

        out.append({
            "message_id": int(p.message_id),
            "url": p.permalink or make_permalink(int(p.message_id)),
            "tags": p.tags or [],
            "preview": preview_text(p.text),
            "media_type": media_type or None,
            "media_url": media_url,
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
            if (claim.status or "") == "closed":
                raise HTTPException(status_code=400, detail="Этот приз уже закрыт")

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
                        status="awaiting_contact",
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
