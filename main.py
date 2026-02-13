"""Natural Sense Telegram Bot + Mini App (FastAPI + python-telegram-bot).

IMPORTANT:
- This file is intentionally kept as a single module for easy Railway deploy.
- Changes in this revision are **non-functional**: readability, structure, safety guards,
  and small refactors that do NOT alter UI/logic/percentages/flows.

Sections are separated with big headers (CONFIG / DB / TELEGRAM / WEBAPP / API).
"""

import os
import re
import html
import asyncio
import logging
import secrets
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal, Any

import httpx
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# HTTP CLIENT (shared) + retry/throttle for t.me embed fetches
# -----------------------------------------------------------------------------
_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_SEMAPHORE: asyncio.Semaphore | None = None

HTTP_MAX_CONCURRENCY = int(os.getenv("HTTP_MAX_CONCURRENCY", "4"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
HTTP_BACKOFF_BASE = float(os.getenv("HTTP_BACKOFF_BASE", "0.6"))

def _utcnow() -> datetime:
    """Return a *naive* UTC timestamp (kept naive for DB consistency in this project)."""
    return datetime.utcnow()

async def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT, _HTTP_SEMAPHORE
    if _HTTP_SEMAPHORE is None:
        _HTTP_SEMAPHORE = asyncio.Semaphore(max(1, HTTP_MAX_CONCURRENCY))

    if _HTTP_CLIENT is None:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            limits=limits,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    return _HTTP_CLIENT

async def _close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        try:
            await _HTTP_CLIENT.aclose()
        except Exception:
            pass
        _HTTP_CLIENT = None

async def _http_get_text(url: str) -> tuple[int, str]:
    """HTTP GET with concurrency limit + basic retries/backoff. Returns (status, text)."""
    client = await _get_http_client()
    sem = _HTTP_SEMAPHORE or asyncio.Semaphore(1)

    last_exc: Exception | None = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            async with sem:
                r = await client.get(url)
            return r.status_code, (r.text or "")
        except Exception as e:
            last_exc = e
            # backoff: 0.6, 1.2, 2.4 ...
            await asyncio.sleep(HTTP_BACKOFF_BASE * (2 ** attempt))
    raise last_exc or RuntimeError("HTTP failed")



from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response
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
# NOTE: We keep env parsing centralized and explicit to make later edits safer.
def env_get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None else default


BOT_TOKEN = env_get("BOT_TOKEN")
BOT_USERNAME = (env_get("BOT_USERNAME", "") or "").strip().lstrip("@")
COOP_BOT_USERNAME = (env_get("COOP_BOT_USERNAME", "") or env_get("ENV_BOT_USERNAME", "") or "").strip().lstrip("@")
PUBLIC_BASE_URL = (env_get("PUBLIC_BASE_URL", "") or "").rstrip("/")
CHANNEL_USERNAME = env_get("CHANNEL_USERNAME", "NaturalSense") or "NaturalSense"
CHANNEL_CHAT_ID = (env_get("CHANNEL_CHAT_ID", "") or "").strip()  # optional: -100... or @channelusername
CHANNEL_INVITE_URL = (env_get("CHANNEL_INVITE_URL", "") or "").strip()  # optional: https://t.me/+...
DATABASE_URL = env_get("DATABASE_URL", "sqlite+aiosqlite:///./ns.db") or "sqlite+aiosqlite:///./ns.db"
ADMIN_CHAT_ID = int(env_get("ADMIN_CHAT_ID", "5443870760") or "5443870760")

APP_VERSION = (env_get("APP_VERSION", "1.1") or "1.1").strip()
ASSET_VERSION = (env_get("ASSET_VERSION", APP_VERSION) or APP_VERSION).strip()

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
DAILY_BONUS_POINTS = 100
REGISTER_BONUS_POINTS = 100
# Рефералка платится только за активного (см. add_daily_bonus_and_update_streak)
REFERRAL_ACTIVE_BONUS_POINTS = 600
REFERRAL_WEEK_BONUS_POINTS = 300

# Referral revshare: inviter receives 10% of invitee BONUS roulette wins (only when invitee is ACTIVE)
REF_REVSHARE_PCT = 0.10
REF_REVSHARE_PER_REFERRED_DAY_CAP = 150   # max points/day per 1 referred user
REF_REVSHARE_PER_INVITER_DAY_CAP = 600    # max points/day total per inviter
REF_INACTIVE_AFTER_DAYS = 7               # if invitee doesn't show up for N days -> inactive (revshare paused)
REF_ACTIVE_MIN_LOGIN_DAYS = 3
REF_ACTIVE_MIN_SPINS = 1

STREAK_MILESTONES = {
    3: 100,
    7: 250,
    14: 600,
    30: 1500,
}

RAFFLE_TICKET_COST = 500
ROULETTE_SPIN_COST = 300
ROULETTE_LIMIT_WINDOW = timedelta(seconds=1)  # anti double-tap cooldown  # TEST: 5s cooldown
DEFAULT_RAFFLE_ID = 1

# -----------------------------------------------------------------------------
# INVENTORY / CONVERSION (FIXED RATES)
# -----------------------------------------------------------------------------
TICKET_CONVERT_RATE = 300          # 1 raffle ticket -> 300 points
DIOR_PALETTE_CONVERT_VALUE = 3000  # 1 Dior palette -> 3000 points (главный приз)

# -----------------------------------------------------------------------------
# DAILY TASKS CONFIG (max 400/day)
# -----------------------------------------------------------------------------
DAILY_MAX_POINTS_PER_DAY = 600

# Important: tasks are claimed manually ("Забрать"). Client only sends events; server validates and caps.
DAILY_TASKS: list[dict[str, Any]] = [
    {"key": "open_miniapp", "title": "Зайти в Mini App", "points": 20, "icon": "✨"},
    {"key": "open_channel", "title": "Перейти в канал", "points": 30, "icon": "↩️"},
    {"key": "use_search", "title": "Использовать поиск", "points": 30, "icon": "🔍"},
    {"key": "open_post", "title": "Открыть 3 поста", "points": 60, "icon": "📰", "need": 3},
    {"key": "open_inventory", "title": "Открыть Косметичку", "points": 20, "icon": "👜"},
    {"key": "open_profile", "title": "Открыть Профиль", "points": 20, "icon": "👤"},

    # Социальные (обязательные базовые daily)
    {"key": "comment_post", "title": "Написать комментарий", "points": 50, "icon": "💬"},

    # Игровые
    {"key": "spin_roulette", "title": "Крутить рулетку 1 раз", "points": 50, "icon": "🎡"},

    # Бонус дня (чтобы добить ровно до 400)
    {"key": "bonus_day", "title": "Собрать все задания дня", "points": 150, "icon": "🎁", "special": True},
]
# Total base points are capped by DAILY_MAX_POINTS_PER_DAY (600). Bonus_day = 150.


PrizeType = Literal["points", "raffle_ticket", "physical_dior_palette"]

# per 1_000_000 (проценты:
# +50=45%, +100=25%, +150=15%, +200=8%, +300=4%, билет=2%, Dior=1%)
ROULETTE_DISTRIBUTION: list[dict[str, Any]] = [
    {"weight": 450_000, "key": "points_500", "type": "points", "value": 50, "label": "+50 баллов"},
    {"weight": 250_000, "key": "points_1000", "type": "points", "value": 100, "label": "+100 баллов"},
    {"weight": 150_000, "key": "points_1500", "type": "points", "value": 150, "label": "+150 баллов"},
    {"weight": 80_000,  "key": "points_2000", "type": "points", "value": 200, "label": "+200 баллов"},
    {"weight": 40_000,  "key": "points_3000", "type": "points", "value": 300, "label": "+300 баллов"},
    {"weight": 20_000,  "key": "ticket_1",   "type": "raffle_ticket", "value": 1, "label": "🎟 +1 билет"},
    {"weight": 10_000,  "key": "dior_palette", "type": "physical_dior_palette", "value": 1, "label": "✨ Dior Palette"},
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
    ref_bonus_paid = Column(Boolean, default=False, nullable=False)  # реф-бонус за 3 дня активности invitee
    ref_week_bonus_paid = Column(Boolean, default=False, nullable=False)  # доп. реф-бонус за 7 дней активности invitee

    # activity + referrals analytics
    last_seen_at = Column(DateTime, nullable=True)  # naive UTC
    daily_login_total = Column(Integer, default=0)  # distinct daily bonus claims (1/day)
    roulette_spins_total = Column(Integer, default=0)  # total roulette spins
    ref_active_at = Column(DateTime, nullable=True)  # when user became "active referral" (for inviter revshare)


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

# -----------------------------------------------------------------------------
# DAILY TASKS (Mini App)
# -----------------------------------------------------------------------------
class DailyTaskLog(Base):
    __tablename__ = "daily_task_logs"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    day = Column(String, index=True, nullable=False)  # YYYY-MM-DD (UTC)
    task_key = Column(String, nullable=False)
    status = Column(String, nullable=False, default="done")  # done | claimed


    # Backward-compat: some DBs have NOT NULL column is_done (used by older schema)
    is_done = Column(Boolean, nullable=False, default=True)

    done_at = Column(DateTime, nullable=True)     # naive UTC
    claimed_at = Column(DateTime, nullable=True)  # naive UTC
    points = Column(Integer, nullable=False, default=0)
    meta = Column(JSON, default=dict)

    __table_args__ = (
        Index("ux_daily_task_logs_tid_day_key", "telegram_id", "day", "task_key", unique=True),
        Index("ix_daily_task_logs_tid_day", "telegram_id", "day"),
    )



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

    resolution = Column(String, default="pending", nullable=False)  # pending|converted|claim
    resolved_at = Column(DateTime, nullable=True)
    resolved_meta = Column(JSON, nullable=True)


Index("ix_roulette_spins_tid_created", RouletteSpin.telegram_id, RouletteSpin.created_at)


class RouletteClaim(Base):
    __tablename__ = "roulette_claims"

    id = Column(Integer, primary_key=True)
    claim_code = Column(String, unique=True, index=True, nullable=False)  # e.g. "NS-AB12CD34"
    telegram_id = Column(BigInteger, index=True, nullable=False)
    spin_id = Column(Integer, nullable=True)  # optional link to roulette_spins.id

    prize_type = Column(String, nullable=False)
    prize_label = Column(String, nullable=False)

    status = Column(String, default="draft", nullable=False)  # draft|submitted|approved|rejected|fulfilled
    contact_text = Column(String, nullable=True)

    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    country = Column(String, nullable=True)
    city = Column(String, nullable=True)
    address_line = Column(String, nullable=True)
    postal_code = Column(String, nullable=True)
    comment = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)

Index("ix_roulette_claims_tid_created", RouletteClaim.telegram_id, RouletteClaim.created_at)



class ReferralEarning(Base):
    __tablename__ = "ref_earnings"

    id = Column(Integer, primary_key=True)
    inviter_id = Column(BigInteger, index=True, nullable=False)   # telegram_id inviter
    referred_id = Column(BigInteger, index=True, nullable=False)  # telegram_id referred
    amount = Column(Integer, nullable=False)  # points credited to inviter
    created_at = Column(DateTime, default=lambda: datetime.utcnow(), nullable=False)  # naive UTC
    source = Column(String, nullable=False, default="roulette_win")  # roulette_win / other
    meta = Column(JSON, nullable=True)

Index("ix_ref_earnings_inviter_created", ReferralEarning.inviter_id, ReferralEarning.created_at)
Index("ix_ref_earnings_ref_created", ReferralEarning.referred_id, ReferralEarning.created_at)


# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
# Engine with keepalive (fix: "connection is closed" on asyncpg idle disconnects)
_engine_kwargs = {"echo": False, "pool_pre_ping": True}
# Recycle connections for Postgres/asyncpg to avoid stale closed sockets
if str(DATABASE_URL).startswith(("postgresql", "postgres")):
    _engine_kwargs.update({"pool_recycle": 1800, "pool_size": 5, "max_overflow": 10})
engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
# Backward-compatible alias (some handlers use async_session)
async_session = async_session_maker


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

        # daily_task_logs (for old DBs where create_all could be skipped/fail)
        await _safe_exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS daily_task_logs (
                id SERIAL PRIMARY KEY,
                telegram_id BIGINT NOT NULL,
                day VARCHAR NOT NULL,
                task_key VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'done',
        is_done BOOLEAN NOT NULL DEFAULT TRUE,
                is_done BOOLEAN NOT NULL DEFAULT TRUE,
                done_at TIMESTAMP NULL,
                claimed_at TIMESTAMP NULL,
                points INTEGER NOT NULL DEFAULT 0,
                meta JSON DEFAULT '{}'::json
            );
            """,
        )
        await _safe_exec(conn, "CREATE UNIQUE INDEX IF NOT EXISTS ux_daily_task_logs_tid_day_key ON daily_task_logs (telegram_id, day, task_key);")
        await _safe_exec(conn, "CREATE INDEX IF NOT EXISTS ix_daily_task_logs_tid_day ON daily_task_logs (telegram_id, day);")
        # daily_task_logs compatibility (older schemas expect is_done NOT NULL)
        await _safe_exec(conn, "ALTER TABLE daily_task_logs ADD COLUMN IF NOT EXISTS is_done BOOLEAN NOT NULL DEFAULT TRUE;")
        await _safe_exec(conn, "UPDATE daily_task_logs SET is_done = TRUE WHERE is_done IS NULL;")


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
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS ref_week_bonus_paid BOOLEAN NOT NULL DEFAULT FALSE;")

        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP NULL;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_login_total INTEGER NOT NULL DEFAULT 0;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS roulette_spins_total INTEGER NOT NULL DEFAULT 0;")
        await _safe_exec(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS ref_active_at TIMESTAMP NULL;")

        # referral earnings (revshare 10% from invitee roulette wins)
        await _safe_exec(
            conn,
            """
            CREATE TABLE IF NOT EXISTS ref_earnings (
                id SERIAL PRIMARY KEY,
                inviter_id BIGINT NOT NULL,
                referred_id BIGINT NOT NULL,
                amount INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT (NOW()),
                source TEXT NOT NULL DEFAULT 'roulette_win',
                meta JSONB NULL
            );
            """,
        )
        await _safe_exec(conn, "CREATE INDEX IF NOT EXISTS ix_ref_earnings_inviter_created ON ref_earnings (inviter_id, created_at);")
        await _safe_exec(conn, "CREATE INDEX IF NOT EXISTS ix_ref_earnings_ref_created ON ref_earnings (referred_id, created_at);")

        # daily tasks (Mini App)
        await _safe_exec(
            conn,
    """
    CREATE TABLE IF NOT EXISTS daily_task_logs (
        id SERIAL PRIMARY KEY,
        telegram_id BIGINT NOT NULL,
        day VARCHAR NOT NULL,
        task_key VARCHAR NOT NULL,
        status VARCHAR NOT NULL DEFAULT 'done',
        is_done BOOLEAN NOT NULL DEFAULT TRUE,
                is_done BOOLEAN NOT NULL DEFAULT TRUE,
        done_at TIMESTAMP NULL,
        claimed_at TIMESTAMP NULL,
        points INTEGER NOT NULL DEFAULT 0,
        meta JSON NULL
    );
    """,
)
        await _safe_exec(conn, "CREATE UNIQUE INDEX IF NOT EXISTS ux_daily_task_logs_tid_day_key ON daily_task_logs (telegram_id, day, task_key);")
        await _safe_exec(conn, "CREATE INDEX IF NOT EXISTS ix_daily_task_logs_tid_day ON daily_task_logs (telegram_id, day);")


        # ✅ Postgres: int32 -> bigint
        await _safe_exec(conn, "ALTER TABLE users ALTER COLUMN telegram_id TYPE BIGINT;")
        await _safe_exec(conn, "ALTER TABLE users ALTER COLUMN referred_by TYPE BIGINT;")
        await _safe_exec(conn, "ALTER TABLE posts ALTER COLUMN message_id TYPE BIGINT;")

        # roulette (LUX wheel + claims)
        await _safe_exec(conn, "ALTER TABLE roulette_spins ADD COLUMN IF NOT EXISTS resolution VARCHAR NOT NULL DEFAULT 'pending';")
        await _safe_exec(conn, "ALTER TABLE roulette_spins ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_spins ADD COLUMN IF NOT EXISTS resolved_meta JSON NULL;")

        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS full_name VARCHAR NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS phone VARCHAR NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS country VARCHAR NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS city VARCHAR NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS address_line VARCHAR NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS postal_code VARCHAR NULL;")
        await _safe_exec(conn, "ALTER TABLE roulette_claims ADD COLUMN IF NOT EXISTS comment VARCHAR NULL;")


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



def _utc_day_start(dt: datetime) -> datetime:
    """Naive UTC start of day for dt (dt must be naive UTC)."""
    return datetime(dt.year, dt.month, dt.day)

# -----------------------------------------------------------------------------
# DAILY TASKS HELPERS
# -----------------------------------------------------------------------------
def _today_key(now: datetime | None = None) -> str:
    n = now or datetime.utcnow()
    return f"{n.year:04d}-{n.month:02d}-{n.day:02d}"


def _daily_tasks_map() -> dict[str, dict[str, Any]]:
    return {t["key"]: t for t in DAILY_TASKS}


async def _daily_points_claimed(session: AsyncSession, telegram_id: int, day: str) -> int:
    try:
        q = select(func.coalesce(func.sum(DailyTaskLog.points), 0)).where(
            DailyTaskLog.telegram_id == telegram_id,
            DailyTaskLog.day == day,
            DailyTaskLog.status == "claimed",
        )
        return int((await session.execute(q)).scalar_one() or 0)
    except Exception as e:
        # If the table is not created yet (old DB) or any transient DB error happens,
        # do not break the Mini App — just return 0.
        logger.info("daily_points_claimed fallback: %s", e)
        return 0


async def _get_daily_logs(session: AsyncSession, telegram_id: int, day: str) -> dict[str, DailyTaskLog]:
    try:
        res = await session.execute(
            select(DailyTaskLog).where(DailyTaskLog.telegram_id == telegram_id, DailyTaskLog.day == day)
        )
        rows = res.scalars().all()
        return {r.task_key: r for r in rows}
    except Exception as e:
        # If the table is missing or DB temporarily unavailable — return empty logs.
        logger.info("daily_logs fallback: %s", e)
        return {}


async def _mark_daily_done(session: AsyncSession, telegram_id: int, day: str, task_key: str, meta: dict | None = None) -> None:
    existing = (await session.execute(
        select(DailyTaskLog).where(
            DailyTaskLog.telegram_id == telegram_id,
            DailyTaskLog.day == day,
            DailyTaskLog.task_key == task_key,
        )
    )).scalar_one_or_none()

    now = datetime.utcnow()
    if existing:
        # do not downgrade claimed
        if existing.status != "claimed":
            existing.status = "done"
            try:
                existing.is_done = True
            except Exception:
                pass
            existing.done_at = existing.done_at or now
            if meta:
                m = dict(existing.meta or {})
                m.update(meta)
                existing.meta = m
            session.add(existing)
        return

    session.add(
        DailyTaskLog(
            telegram_id=telegram_id,
            day=day,
            task_key=task_key,
            status="done",
            is_done=True,
            done_at=now,
            points=0,
            meta=meta or {},
        )
    )


async def _can_unlock_bonus_day(task_map: dict[str, dict[str, Any]], logs: dict[str, DailyTaskLog]) -> bool:
    # All non-special tasks must be claimed (or at least done?) -> to avoid abuse, require claimed.
    for k, t in task_map.items():
        if t.get("special"):
            continue
        lg = logs.get(k)
        if not lg or lg.status != "claimed":
            return False
    return True



def _days_ago(dt: Optional[datetime], now: datetime) -> Optional[int]:
    if not dt:
        return None
    return max(0, int((now - dt).total_seconds() // 86400))


def _referral_status(invitee: User, now: datetime) -> tuple[str, str, dict[str, int]]:
    """
    Returns: (status_code, status_label, progress)
    status_code: pending|active|inactive
    progress: {days_done, days_need, spins_done, spins_need}
    """
    days_done = int(invitee.daily_login_total or 0)
    spins_done = int(invitee.roulette_spins_total or 0)
    progress = {
        "days_done": min(days_done, REF_ACTIVE_MIN_LOGIN_DAYS),
        "days_need": REF_ACTIVE_MIN_LOGIN_DAYS,
        "spins_done": min(spins_done, REF_ACTIVE_MIN_SPINS),
        "spins_need": REF_ACTIVE_MIN_SPINS,
    }
    base_active = (days_done >= REF_ACTIVE_MIN_LOGIN_DAYS) and (spins_done >= REF_ACTIVE_MIN_SPINS)
    if not base_active:
        return "pending", "⏳ Не активирован", progress

    last_seen = invitee.last_seen_at or invitee.last_daily_bonus_at or invitee.joined_at
    inactive_days = _days_ago(last_seen, now) or 0
    if inactive_days >= REF_INACTIVE_AFTER_DAYS:
        return "inactive", "⚠️ Неактивный", progress

    return "active", "✅ Активный", progress


async def touch_user_seen(telegram_id: int, username: str | None = None, first_name: str | None = None) -> None:
    """Update last_seen_at and basic profile fields."""
    now = datetime.utcnow()
    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalar_one_or_none()
        if not user:
            return
        user.last_seen_at = now
        if username is not None:
            user.username = username.lower() if username else None
        if first_name is not None:
            user.first_name = first_name
        session.add(user)
        await session.commit()



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
            existing.username = (username.lower() if username else None)
            existing.first_name = first_name
            existing.last_seen_at = now
            session.add(existing)
            await session.commit()
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
            ref_week_bonus_paid=False,
            last_seen_at=now,
            daily_login_total=1,
            roulette_spins_total=0,
            ref_active_at=None,
        )
        _recalc_tier(user)
        session.add(user)
        await session.flush()

        if inviter:
            # Рефералка начисляется только за "активного" приглашённого (см. add_daily_bonus_and_update_streak).
            inviter.referral_count = (inviter.referral_count or 0) + 1
            _recalc_tier(inviter)

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
        user.last_seen_at = now
        user.daily_login_total = int(user.daily_login_total or 0) + 1

        # if invitee hits ACTIVE threshold for first time
        if user.referred_by and (user.ref_active_at is None):
            if int(user.daily_login_total or 0) >= REF_ACTIVE_MIN_LOGIN_DAYS and int(user.roulette_spins_total or 0) >= REF_ACTIVE_MIN_SPINS:
                user.ref_active_at = now

        streak_bonus = 0
        if user.daily_streak in STREAK_MILESTONES:
            streak_bonus = STREAK_MILESTONES[user.daily_streak]
            user.points = (user.points or 0) + streak_bonus


        # ------------------ REFERRAL: pay only for active invitee ------------------
        # 3 days streak => +REFERRAL_ACTIVE_BONUS_POINTS to inviter (once)
        # 7 days streak => +REFERRAL_WEEK_BONUS_POINTS to inviter (once)
        if user.referred_by:
            inviter = (await session.execute(select(User).where(User.telegram_id == int(user.referred_by)))).scalar_one_or_none()
            if inviter:
                # Active bonus (3 days)
                if (not user.ref_bonus_paid) and (user.daily_streak or 0) >= 3:
                    inviter.points = (inviter.points or 0) + int(REFERRAL_ACTIVE_BONUS_POINTS)
                    _recalc_tier(inviter)
                    user.ref_bonus_paid = True
                    session.add(PointTransaction(
                        telegram_id=int(inviter.telegram_id),
                        type="referral_active",
                        delta=int(REFERRAL_ACTIVE_BONUS_POINTS),
                        meta={"invited": int(user.telegram_id), "streak": int(user.daily_streak or 0)},
                    ))
                # Week bonus (7 days)
                if (not user.ref_week_bonus_paid) and (user.daily_streak or 0) >= 7:
                    inviter.points = (inviter.points or 0) + int(REFERRAL_WEEK_BONUS_POINTS)
                    _recalc_tier(inviter)
                    user.ref_week_bonus_paid = True
                    session.add(PointTransaction(
                        telegram_id=int(inviter.telegram_id),
                        type="referral_week",
                        delta=int(REFERRAL_WEEK_BONUS_POINTS),
                        meta={"invited": int(user.telegram_id), "streak": int(user.daily_streak or 0)},
                    ))
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

# Канонизация системных тегов: старые #люкс/#ЛЮКС будут считаться как #Люкс
CANON_TAGS = {
    "новинка": "Новинка",
    "люкс": "Люкс",
    "тренд": "Тренд",
    "история": "История",
    "оценка": "Оценка",
    "факты": "Факты",
    "состав": "Состав",
    "челенджи": "Челенджи",
    "челленджи": "Челленджи",
    "challenge": "Challenge",
    "sephorapromo": "SephoraPromo",
}

def _norm_tag(s: str) -> str:
    return (s or "").strip().casefold()

def extract_tags(text_: str | None) -> list[str]:
    if not text_:
        return []
    raw = [m.group(1) for m in TAG_RE.finditer(text_)]
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        key = _norm_tag(t)
        canon = CANON_TAGS.get(key, t.strip())
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out
def preview_text(text_: str | None, limit: int = 180) -> str:
    if not text_:
        return ""
    s = re.sub(r"\s+", " ", text_.strip())
    return (s[:limit] + "…") if len(s) > limit else s





def search_snippet(text_: str | None, q: str, radius: int = 80, hard_limit: int = 260) -> str:
    """Возвращает 'сниппет' вокруг первого совпадения (как поиск Windows)."""
    if not text_:
        return ""
    s = re.sub(r"\s+", " ", text_.strip())
    q = (q or "").strip()
    if not q:
        return preview_text(s, limit=min(hard_limit, 180))

    try:
        m = re.search(re.escape(q), s, flags=re.IGNORECASE)
    except re.error:
        m = None

    if not m:
        return preview_text(s, limit=min(hard_limit, 180))

    start = max(0, m.start() - radius)
    end = min(len(s), m.end() + radius)

    pre = "…" if start > 0 else ""
    post = "…" if end < len(s) else ""
    sn = pre + s[start:end] + post

    return (sn[:hard_limit] + "…") if len(sn) > hard_limit else sn
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



# -----------------------------------------------------------------------------
# HOT PATH CACHE (Mini App feed)
# -----------------------------------------------------------------------------
_LIST_POSTS_CACHE: dict[tuple[str | None, int], tuple[float, list[Any]]] = {}
_LIST_POSTS_CACHE_TTL = float(os.getenv("LIST_POSTS_CACHE_TTL", "20"))

async def list_posts(tag: str | None, limit: int = 50, offset: int = 0):
    if tag and tag in BLOCKED_TAGS:
        return []

    tag = (tag or "").strip() if tag else None
    want_norm: str | None = None
    if tag:
        # Канонизируем запрос тега (например, 'люкс' -> 'Люкс'), сравниваем без учета регистра
        tag_canon = CANON_TAGS.get(_norm_tag(tag), tag)
        want_norm = _norm_tag(tag_canon)

    # ВАЖНО: чтобы старые посты с тегом не пропадали, сначала берем расширенное окно,
    # затем фильтруем по тегу и только потом применяем offset/limit к отфильтрованному списку.
    window = max(1000, (offset + limit) * 40)
    window = min(window, 5000)

    cache_key = (want_norm, window)
    now_ts = asyncio.get_event_loop().time()
    cached = _LIST_POSTS_CACHE.get(cache_key)
    if cached and (now_ts - cached[0]) < _LIST_POSTS_CACHE_TTL:
        rows = cached[1]
        return rows[offset: offset + limit]

    async with async_session_maker() as session:
        q = (
            select(Post)
            .where(Post.is_deleted == False)  # noqa: E712
            .order_by(Post.message_id.desc())
            .limit(window)
        )
        rows = (await session.execute(q)).scalars().all()

    if want_norm:
        def _canon_norm(x: str) -> str:
            return _norm_tag(CANON_TAGS.get(_norm_tag(x), x))

        rows = [p for p in rows if any(_canon_norm(x) == want_norm for x in (p.tags or []))]

    _LIST_POSTS_CACHE[cache_key] = (now_ts, rows)
    return rows[offset: offset + limit]



# -----------------------------------------------------------------------------
# DELETE SWEEPER (AUTO CHECK)
# -----------------------------------------------------------------------------
async def message_exists_public(message_id: int) -> bool:
    url = f"https://t.me/{CHANNEL_USERNAME}/{message_id}?embed=1"
    try:
        status, body = await _http_get_text(url)
        if status == 404:
            return False
        if status != 200:
            # treat temporary issues as "exists" to avoid false deletions
            return True

        low = (body or "").lower()
        if "message not found" in low or "post not found" in low:
            return False
        if "join channel" in low or "this channel is private" in low or "private channel" in low:
            return True
        return True
    except Exception as e:
        logger.warning("Sweeper check failed for %s: %s", message_id, e)
        return True



async def sweep_deleted_posts(batch: int = 30):
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
    interval = int(os.getenv("SWEEPER_INTERVAL", "180"))
    while True:
        try:
            await sweep_deleted_posts(batch=30)
        except Exception as e:
            logger.error("Sweeper error: %s", e)
        await asyncio.sleep(max(30, interval))
# -----------------------------------------------------------------------------
# PUBLIC RECONCILE (AUTO DISCOVER NEW POSTS)
# -----------------------------------------------------------------------------
async def fetch_public_post_text(message_id: int) -> str | None:
    """Fetches public embed HTML and extracts visible text. Returns None if not accessible or not found."""
    url = f"https://t.me/{CHANNEL_USERNAME}/{message_id}?embed=1"
    try:
        status, html_text = await _http_get_text(url)
        if status == 404:
            return None
        if status != 200:
            return None
    except Exception:
        return None

    low = (html_text or "").lower()
    # Not found / private / need join
    if "message not found" in low or "post not found" in low:
        return None
    if "join channel" in low or "private channel" in low:
        return None

    # Very lightweight HTML-to-text extraction (no external deps)
    txt = html_text
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", txt)
    txt = re.sub(r"(?is)<br\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</p\s*>", "\n", txt)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t\r]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip() or None


# -----------------------------------------------------------------------------
# TELEGRAM BOT
# -----------------------------------------------------------------------------
tg_app: Application | None = None
tg_task: asyncio.Task | None = None
sweeper_task: asyncio.Task | None = None


def is_admin(user_id: int) -> bool:
    return int(user_id) == int(ADMIN_CHAT_ID)


def get_main_keyboard():
    # ✅ СНИЗУ: Профиль + Рефералы + Помощь
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("👤 Профиль"), KeyboardButton("👥 Рефералы")],
            [KeyboardButton("ℹ️ Помощь")],
        ],
        resize_keyboard=True,
    )


def build_start_inline_kb() -> InlineKeyboardMarkup:
    # ✅ “В канал” прикреплена к сообщению /start
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("↩️ В канал", url=f"https://t.me/{CHANNEL_USERNAME}")]]
    )

def get_channel_chat_id() -> str:
    """chat_id for Bot API (preferred: CHANNEL_CHAT_ID; fallback: @CHANNEL_USERNAME)."""
    if CHANNEL_CHAT_ID:
        return CHANNEL_CHAT_ID
    u = (CHANNEL_USERNAME or "").strip().lstrip("@")
    if not u:
        raise RuntimeError("CHANNEL_USERNAME is empty and CHANNEL_CHAT_ID not set")
    return f"@{u}"


def get_channel_url() -> str:
    """Public URL to the channel (preferred: invite link)."""
    if CHANNEL_INVITE_URL:
        return CHANNEL_INVITE_URL
    u = (CHANNEL_USERNAME or "").strip().lstrip("@")
    return f"https://t.me/{u}"


def get_bot_deeplink() -> Optional[str]:
    u = (BOT_USERNAME or "").strip().lstrip("@")
    if not u:
        return None
    return f"https://t.me/{u}?start=channel"

# Cache runtime bot username (fallback when BOT_USERNAME env is not set)
_BOT_USERNAME_RUNTIME: str | None = None


async def get_bot_deeplink_runtime(context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    """Return deeplink to bot. Uses BOT_USERNAME env if set, otherwise resolves via getMe."""
    global _BOT_USERNAME_RUNTIME
    u = (BOT_USERNAME or "").strip().lstrip("@")
    if u:
        return f"https://t.me/{u}?start=channel"
    if _BOT_USERNAME_RUNTIME:
        return f"https://t.me/{_BOT_USERNAME_RUNTIME}?start=channel"
    try:
        me = await context.bot.get_me()
        uname = (me.username or "").strip().lstrip("@")
        if uname:
            _BOT_USERNAME_RUNTIME = uname
            return f"https://t.me/{uname}?start=channel"
    except Exception:
        pass
    return None




_LAST_SILENT_KB: dict[int, float] = {}
async def set_keyboard_silent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Telegram показывает ReplyKeyboard только через сообщение.
    # Делаем невидимое сообщение и удаляем -> в чате ничего не видно, кнопки остаются.
    chat = update.effective_chat
    if not chat:
        return

    # Anti-flood: не шлём "тихое" сообщение слишком часто в один и тот же чат
    now_ts = asyncio.get_event_loop().time()
    last_ts = _LAST_SILENT_KB.get(chat.id, 0.0)
    if (now_ts - last_ts) < 8.0:
        return
    _LAST_SILENT_KB[chat.id] = now_ts

    try:
        m = await context.bot.send_message(chat_id=chat.id, text="\u200b", reply_markup=get_main_keyboard())
        # достаточно маленькой задержки, чтобы Telegram успел применить клавиатуру
        await asyncio.sleep(0.15)
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

💎 *Баллы*
• Первый /start: +{REGISTER_BONUS_POINTS} за регистрацию
• Ежедневный бонус: +{DAILY_BONUS_POINTS} (1 раз в 24 часа)

🔥 *Стрик (серия дней — доп. бонус к дневному)*
• 3 дня: +100
• 7 дней: +250
• 14 дней: +600
• 30 дней: +1500

🎟 *Рефералка*
/invite — твоя ссылка.
• На 3‑й день активности приглашённого: +{REFERRAL_ACTIVE_BONUS_POINTS}
• На 7‑й день активности приглашённого: +{REFERRAL_WEEK_BONUS_POINTS}
• Если приглашённый *АКТИВНЫЙ* — ты получаешь *10%* от его *бонусных выигрышей* в рулетке.
  (Если он не заходит 7 дней — статус становится неактивным, 10% замораживается.)
👥 Рефералы — отслеживание статуса и сколько ты получил.

🎰 *Рулетка*
Стоимость 1 спина: {ROULETTE_SPIN_COST} 💎
Крутить можно сколько угодно — пока хватает баллов.
""".format(
        REGISTER_BONUS_POINTS=REGISTER_BONUS_POINTS,
        DAILY_BONUS_POINTS=DAILY_BONUS_POINTS,
        REFERRAL_ACTIVE_BONUS_POINTS=REFERRAL_ACTIVE_BONUS_POINTS,
        REFERRAL_WEEK_BONUS_POINTS=REFERRAL_WEEK_BONUS_POINTS,
        ROULETTE_SPIN_COST=ROULETTE_SPIN_COST,
    )



async def _sum_ref_earnings(session: AsyncSession, inviter_id: int, referred_id: int | None = None, day_start: datetime | None = None) -> int:
    q = select(func.coalesce(func.sum(ReferralEarning.amount), 0)).where(ReferralEarning.inviter_id == int(inviter_id))
    if referred_id is not None:
        q = q.where(ReferralEarning.referred_id == int(referred_id))
    if day_start is not None:
        q = q.where(ReferralEarning.created_at >= day_start)
    return int((await session.execute(q)).scalar_one() or 0)


async def build_referrals_page(inviter_id: int, page: int = 0, page_size: int = 6) -> tuple[str, InlineKeyboardMarkup]:
    now = datetime.utcnow()
    day_start = _utc_day_start(now)
    page = max(0, int(page))

    async with async_session_maker() as session:
        # list of invitees
        total = int(
            (await session.execute(select(func.count()).select_from(User).where(User.referred_by == int(inviter_id)))).scalar_one()
            or 0
        )
        offset = page * page_size
        rows = (
            await session.execute(
                select(User)
                .where(User.referred_by == int(inviter_id))
                .order_by(User.joined_at.desc())
                .offset(offset)
                .limit(page_size)
            )
        ).scalars().all()

        active_cnt = 0
        inactive_cnt = 0
        pending_cnt = 0
        for u in (
            await session.execute(select(User).where(User.referred_by == int(inviter_id)))
        ).scalars().all():
            st, _, _ = _referral_status(u, now)
            if st == "active":
                active_cnt += 1
            elif st == "inactive":
                inactive_cnt += 1
            else:
                pending_cnt += 1

        earned_total = await _sum_ref_earnings(session, inviter_id)
        earned_today = await _sum_ref_earnings(session, inviter_id, day_start=day_start)

        lines: list[str] = []
        lines.append("👥 *Рефералы*")
        lines.append("")
        lines.append(f"Всего: *{total}* | ✅ активных: *{active_cnt}* | ⚠️ неактивных: *{inactive_cnt}* | ⏳ не актив.: *{pending_cnt}*")
        lines.append(f"Заработано всего: *+{earned_total}* | Сегодня: *+{earned_today}*")
        lines.append("")
        if not rows:
            lines.append("Пока нет приглашённых. Используй /invite чтобы получить ссылку.")
        else:
            for u in rows:
                uname = f"@{u.username}" if u.username else (u.first_name or "Пользователь")
                st_code, st_label, prog = _referral_status(u, now)
                last_seen = u.last_seen_at or u.last_daily_bonus_at or u.joined_at
                da = _days_ago(last_seen, now)
                last_seen_txt = f"{da} дн. назад" if da is not None else "—"
                utotal = await _sum_ref_earnings(session, inviter_id, referred_id=int(u.telegram_id))
                utoday = await _sum_ref_earnings(session, inviter_id, referred_id=int(u.telegram_id), day_start=day_start)

                lines.append(f"• {uname} — {st_label}")
                lines.append(f"  Последний визит: *{last_seen_txt}* | Получено: *+{utotal}* (сегодня *+{utoday}*)")

                if st_code == "pending":
                    days_left = max(0, REF_ACTIVE_MIN_LOGIN_DAYS - int(u.daily_login_total or 0))
                    spins_left = max(0, REF_ACTIVE_MIN_SPINS - int(u.roulette_spins_total or 0))
                    need_parts = []
                    if days_left > 0:
                        need_parts.append(f"{days_left} дн. входа")
                    if spins_left > 0:
                        need_parts.append(f"{spins_left} спин")
                    need = ", ".join(need_parts) if need_parts else "—"
                    lines.append(f"  До активации: *{need}*")
                elif st_code == "inactive":
                    lines.append("  10% заморожено: реферал не заходил 7 дней. Вернётся в актив при следующем входе.")

                lines.append("")

        # pagination kb
        btns = []
        nav = []
        if offset > 0:
            nav.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"ref_page:{page-1}"))
        if (offset + page_size) < total:
            nav.append(InlineKeyboardButton("Вперёд ➡️", callback_data=f"ref_page:{page+1}"))
        if nav:
            btns.append(nav)

        kb = InlineKeyboardMarkup(btns) if btns else InlineKeyboardMarkup([])
        return "\n".join(lines).strip(), kb


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
        ref_line = f"\n🎁 Тебя пригласили — твой друг получил +{REFERRAL_ACTIVE_BONUS_POINTS} баллов."

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

Реф-бонус начисляется только за активного приглашённого:
• на 3-й день: +{REFERRAL_ACTIVE_BONUS_POINTS}
• на 7-й день: +{REFERRAL_WEEK_BONUS_POINTS}
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

# -----------------------------------------------------------------------------
# MEDIA SYNC (admin forwards posts from channel to bot)
# -----------------------------------------------------------------------------
SYNC_KEY = "sync_media_until"


def _is_admin(update: Update) -> bool:
    try:
        return int(update.effective_user.id) == int(ADMIN_CHAT_ID)
    except Exception:
        return False


def _sync_enabled(context: ContextTypes.DEFAULT_TYPE) -> bool:
    until = context.application.bot_data.get(SYNC_KEY)
    if not until:
        return False
    try:
        return _utcnow() < until
    except Exception:
        return False


async def cmd_sync_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not _is_admin(update):
        await update.message.reply_text("⛔️ Только админ.")
        return
    context.application.bot_data[SYNC_KEY] = _utcnow() + timedelta(minutes=60)
    await update.message.reply_text(
        "✅ Режим синхронизации медиа ВКЛ на 60 минут.\n"
        "Теперь пересылай посты из канала обычным форвардом (со ссылкой на источник)."
    )


async def cmd_sync_media_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not _is_admin(update):
        await update.message.reply_text("⛔️ Только админ.")
        return
    context.application.bot_data.pop(SYNC_KEY, None)
    await update.message.reply_text("🛑 Режим синхронизации медиа ВЫКЛ.")


def _extract_forward_origin_message_id(msg) -> tuple[Optional[int], Optional[str]]:
    """Returns (origin_message_id, error_text)."""
    # New API: forward_origin
    try:
        fo = getattr(msg, "forward_origin", None)
        if fo and getattr(fo, "type", None) == "channel":
            # PTB: MessageOriginChannel has .message_id and .chat
            mid = getattr(fo, "message_id", None)
            if mid:
                return int(mid), None
    except Exception:
        pass

    # Legacy fields
    try:
        fmid = getattr(msg, "forward_from_message_id", None)
        fchat = getattr(msg, "forward_from_chat", None)
        if fmid and fchat:
            return int(fmid), None
    except Exception:
        pass

    return None, (
        "Перешли именно пост из канала обычным форвардом.\n"
        "Важно: при пересылке НЕ выбирай 'как копию' / 'скрыть источник'."
    )


def _extract_media(msg) -> tuple[Optional[str], Optional[str]]:
    """Returns (media_type, media_file_id)."""
    if getattr(msg, "photo", None):
        try:
            ph = msg.photo[-1]
            return "photo", ph.file_id
        except Exception:
            pass
    if getattr(msg, "video", None):
        try:
            return "video", msg.video.file_id
        except Exception:
            pass
    if getattr(msg, "document", None):
        try:
            return "document", msg.document.file_id
        except Exception:
            pass
    return None, None


async def on_sync_forward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Accept forwarded channel posts from admin and update media_file_id in DB."""
    msg = update.message
    if not msg:
        return
    if not _is_admin(update):
        return
    if not _sync_enabled(context):
        return

    origin_mid, err = _extract_forward_origin_message_id(msg)
    if not origin_mid:
        await msg.reply_text(err)
        return

    media_type, media_file_id = _extract_media(msg)
    if not media_file_id:
        await msg.reply_text("⚠️ В этом форварде нет фото/видео/файла. Нужен пост с медиа.")
        return

    async with async_session() as session:
        res = await session.execute(select(Post).where(Post.message_id == origin_mid))
        post = res.scalar_one_or_none()
        if not post:
            await msg.reply_text("⚠️ Этот пост ещё не найден в базе.\nСделай edit поста в канале и повтори.")
            return

        post.media_type = media_type
        post.media_file_id = media_file_id
        post.updated_at = _utcnow()
        await session.commit()

    await msg.reply_text(f"✅ Обновил пост {origin_mid}: {media_type}")


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
    marked = await sweep_deleted_posts(batch=30)
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



async def cmd_pin_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin-only: posts a pinned 'menu' message in the channel with buttons to bot+channel."""
    user = update.effective_user
    chat = update.effective_chat
    if not user or not is_admin(user.id):
        return
    if not chat or chat.type != "private":
        try:
            await update.message.reply_text("Напиши /pin_post мне в ЛС (private), чтобы я мог закрепить пост в канале.")
        except Exception:
            pass
        return

    bot_link = await get_bot_deeplink_runtime(context)
    if not bot_link:
        await update.message.reply_text("❌ Не могу определить username этого бота для кнопки 'Войти в Журнал'. Проверь что бот доступен (getMe) и что он не заблокирован.")
        return

    channel_chat_id = get_channel_chat_id()

    coop_username = (COOP_BOT_USERNAME or "").strip().lstrip("@")
    if not coop_username:
        await update.message.reply_text(
            "❌ Не задан ENV_BOT_USERNAME (или COOP_BOT_USERNAME). Добавь username бота сотрудничества (без @), чтобы создать кнопку \"🤝 Сотрудничество\"."
        )
        return
    coop_url = f"https://t.me/{coop_username}"

    # Текст поста можно передать после команды /pin_post (с переносами строк).
    # ВАЖНО: context.args "склеивает" текст, поэтому берём сырой текст сообщения и вырезаем команду.
    raw_msg_text = (update.message.text or "") if update.message else ""
    custom = ""
    if update.message and update.message.reply_to_message:
        # Если админ ответил на сообщение и вызвал /pin_post — берём текст/подпись из реплая.
        rt = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
        custom = rt
    if not custom:
        custom = re.sub(r"^/pin_post(@\w+)?\s*", "", raw_msg_text, flags=re.IGNORECASE | re.DOTALL)
    custom = custom.strip()
    post_text = custom or (
        "📰 Natural Sense — журнал косметических новинок\n\n"
        "• Смотри посты прямо в канале\n"
        "• Открывай Mini App, копи бонусы и участвуй в розыгрышах\n\n"
        "Жми кнопку ниже 👇"
    )

    kb = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📲 Войти в Журнал", url=bot_link)],
            [InlineKeyboardButton("🤝 Сотрудничество", url=coop_url)],
        ]
    )

    try:
        sent = await context.bot.send_message(
            chat_id=channel_chat_id,
            text=post_text,
            reply_markup=kb,
            disable_web_page_preview=True,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Не смог отправить пост в канал. Проверь что бот админ в канале и CHANNEL_CHAT_ID/CHANNEL_USERNAME. Ошибка: {e}")
        return

    try:
        await context.bot.pin_chat_message(
            chat_id=channel_chat_id,
            message_id=sent.message_id,
            disable_notification=True,
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Пост отправил, но закрепить не смог. Дай боту право 'Закреплять сообщения'. Ошибка: {e}")
        return

    await update.message.reply_text("✅ Готово: пост отправлен в канал и закреплён. Если нужно обновить — просто вызови /pin_post ещё раз.")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    await q.answer()

    uid = q.from_user.id

    data = q.data or ""

    # referrals pagination (available for everyone)
    if data.startswith("ref_page:"):
        try:
            page = int(data.split(":", 1)[1])
        except Exception:
            page = 0
        page_text, kb = await build_referrals_page(q.from_user.id, page=page)
        try:
            await q.edit_message_text(page_text, parse_mode="Markdown", reply_markup=kb)
        except Exception:
            # if editing fails, send new message
            await q.message.reply_text(page_text, parse_mode="Markdown", reply_markup=kb)
        return

    if not is_admin(uid):
        await q.edit_message_text("⛔️ Нет доступа.")
        return

    if data == "admin_stats":
        await q.edit_message_text((await admin_stats_text()), parse_mode="Markdown")
        return

    if data == "admin_sweep":
        marked = await sweep_deleted_posts(batch=30)
        if not marked:
            await q.edit_message_text("🧹 Sweep: ничего не найдено.")
        else:
            await q.edit_message_text(f"🧹 Sweep: помечены удалёнными: {marked}")
        return


async def on_text_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    txt = update.message.text.strip()

    # update last seen
    try:
        await touch_user_seen(update.effective_user.id, update.effective_user.username, update.effective_user.first_name)
    except Exception:
        pass

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

    if txt == "👥 Рефералы":
        page_text, kb = await build_referrals_page(update.effective_user.id, page=0)
        await update.message.reply_text(page_text, parse_mode="Markdown", reply_markup=kb)
        return

    if txt == "ℹ️ Помощь":
        await cmd_help(update, context)
        return


async def on_discussion_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Tracks comments in the linked discussion group to unlock Daily tasks:
    - comment_post: reply to the forwarded channel post (sender_chat == channel)
    NOTE: Telegram does not provide a perfect "comment vs reply" signal in all cases,
    but this logic is reliable for linked discussions.
    """
    msg = update.message
    if not msg or not msg.text:
        return
    if not update.effective_user:
        return

    uid = int(update.effective_user.id)
    text_ = (msg.text or "").strip()

    # basic anti-spam for daily tasks: ignore very short messages
    if len(text_) < 10:
        return

    # update last seen (best-effort)
    try:
        await touch_user_seen(uid, update.effective_user.username, update.effective_user.first_name)
    except Exception:
        pass

    # We only care about replies (comments)
    if not msg.reply_to_message:
        return

    rt = msg.reply_to_message

    # Determine if it's a comment to the channel post (reply to forwarded channel message)
    is_reply_to_channel_post = False
    try:
        if getattr(rt, "sender_chat", None) and getattr(rt.sender_chat, "type", None) == "channel":
            is_reply_to_channel_post = True
    except Exception:
        is_reply_to_channel_post = False

    # Determine if it's a reply to another user's comment
    is_reply_to_user_comment = False
    try:
        if (not is_reply_to_channel_post) and getattr(rt, "from_user", None) and int(rt.from_user.id) != uid:
            is_reply_to_user_comment = True
    except Exception:
        is_reply_to_user_comment = False

    if not (is_reply_to_channel_post or is_reply_to_user_comment):
        return

    day = _today_key()

    async with async_session_maker() as session:
        # ensure user exists (Mini App может быть открыт до /start)
        user = (await session.execute(select(User).where(User.telegram_id == uid))).scalar_one_or_none()
        if not user:
            user = User(telegram_id=uid, points=10)
            session.add(user)
            await session.commit()

        await _mark_daily_done(
            session,
            uid,
            day,
            task_key,
            meta={
                "chat_id": int(msg.chat_id),
                "message_id": int(msg.message_id),
            },
        )
        await session.commit()


# -----------------------------------------------------------------------------
# CHANNEL INDEXING
# -----------------------------------------------------------------------------
async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return

    text_ = (msg.caption if msg.caption is not None else msg.text) or ""

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

    text_ = (msg.caption if msg.caption is not None else msg.text) or ""

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
        # Ensure webhook is disabled when using polling
        try:
            await tg_app.bot.delete_webhook(drop_pending_updates=False)
        except Exception:
            pass
        await tg_app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)
        logger.info("✅ Telegram bot started (polling)")
        while True:
            # Keep runner alive; channel_post handlers index posts in real-time
            await asyncio.sleep(int(os.getenv('BOT_IDLE_TICK', '30')))
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
    tg_app.add_handler(CommandHandler("sync_media", cmd_sync_media))
    tg_app.add_handler(CommandHandler("sync_media_off", cmd_sync_media_off))

    tg_app.add_handler(CommandHandler("admin", cmd_admin))
    tg_app.add_handler(CommandHandler("admin_stats", cmd_admin_stats))
    tg_app.add_handler(CommandHandler("admin_sweep", cmd_admin_sweep))
    tg_app.add_handler(CommandHandler("admin_user", cmd_admin_user))
    tg_app.add_handler(CommandHandler("admin_add", cmd_admin_add))
    tg_app.add_handler(CommandHandler("find", cmd_admin_find))
    tg_app.add_handler(CommandHandler("pin_post", cmd_pin_post))

    tg_app.add_handler(CallbackQueryHandler(on_callback))
    tg_app.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND, on_text_button))
    # Media sync: accept forwarded posts with media from admin (private chat)
    tg_app.add_handler(MessageHandler(filters.ChatType.PRIVATE & (filters.PHOTO | filters.VIDEO | filters.Document.ALL), on_sync_forward))

    # Discussion comments (linked group) -> unlock Daily tasks (comment/reply)
    tg_app.add_handler(MessageHandler((filters.ChatType.GROUP | filters.ChatType.SUPERGROUP) & filters.TEXT & ~filters.COMMAND, on_discussion_message))

    tg_app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST, on_channel_post))
    tg_app.add_handler(MessageHandler(filters.UpdateType.EDITED_CHANNEL_POST, on_edited_channel_post))

    try:
        me = await tg_app.bot.get_me()
        logger.info('🤖 Bot identity: @%s (id=%s)', me.username, me.id)
    except Exception as e:
        logger.warning('Bot getMe failed (token/webhook issue?): %s', e)

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
    <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0" />
    <meta http-equiv="Pragma" content="no-cache" />
    <meta http-equiv="Expires" content="0" />
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <title>NS · Natural Sense</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    :root{
      --bg:#0c0f14;
      --card:rgba(255,255,255,0.08);
      --card2:rgba(255,255,255,0.06);
      --text:rgba(255,255,255,0.92);
      --muted:rgba(255,255,255,0.60);
      --gold:rgba(230,193,128,0.90);
      --stroke:rgba(255,255,255,0.12);
      --sheetOverlay:rgba(12,15,20,0.55);
      --sheetCardBg:rgba(255,255,255,0.10);
      --glassStroke:rgba(255,255,255,0.18);
      --glassShadow:rgba(0,0,0,0.45);
      --r-lg:22px;
      --r-md:16px;
      --r-sm:14px;
    }
    body{
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      background:
        radial-gradient(1200px 800px at 20% 10%, rgba(230,193,128,0.18), transparent 60%),
        radial-gradient(900px 600px at 80% 0%, rgba(255,255,255,0.06), transparent 55%),
        var(--bg);
      color:var(--text);
      overflow-x:hidden;
    }
    #root{min-height:100vh}
    .safePadBottom{padding-bottom:92px}
    .container{max-width:560px;margin:0 auto;padding:16px 16px 24px}

    .h1{font-size:18px;font-weight:800;letter-spacing:.2px}
    .sub{margin-top:6px;font-size:13px;color:var(--muted)}
    .row{display:flex;justify-content:space-between;align-items:center;gap:12px}
    .hr{height:1px;background:var(--stroke);margin:14px 0;opacity:.8}

    .card{
      border:1px solid var(--stroke);
      background:linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05));
      border-radius:var(--r-lg);
      padding:14px;
      box-shadow:0 10px 30px rgba(0,0,0,0.35);
      position:relative;
      overflow:hidden;
    }
    .card2{
      border:1px solid var(--stroke);
      background:var(--card2);
      border-radius:var(--r-lg);
      padding:12px;
    }
    .pill{
      display:inline-flex;align-items:center;gap:8px;
      padding:7px 10px;border-radius:999px;
      border:1px solid rgba(230,193,128,0.25);
      background:rgba(230,193,128,0.10);
      font-size:12px;font-weight:700;
      user-select:none;
    }
    .btn{
      width:100%;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      border-radius:18px;
      padding:14px;
      display:flex;
      justify-content:space-between;
      align-items:center;
      cursor:pointer;
      user-select:none;
    }
    .btn:active{transform:translateY(1px)}
    .btnTitle{font-size:15px;font-weight:750}
    .btnSub{margin-top:4px;font-size:12px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .tile{
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      border-radius:18px;
      padding:12px;
      cursor:pointer;
      user-select:none;
      min-height:82px;
      display:flex;
      flex-direction:column;
      justify-content:space-between;
    }
    .tileTitle{font-size:14px;font-weight:800}
    .tileSub{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.25}
    .hScroll{display:flex;gap:10px;overflow:auto;padding-bottom:8px;-webkit-overflow-scrolling:touch}
    .hScroll::-webkit-scrollbar{display:none}

    .miniCard{
      min-width:220px;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      border-radius:18px;
      padding:12px;
      cursor:pointer;
      user-select:none;
    }
    .miniMeta{font-size:12px;color:var(--muted)}
    .miniText{margin-top:8px;font-size:14px;line-height:1.3}
    .chipRow{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}
    .chip{font-size:12px;padding:5px 8px;border-radius:999px;border:1px solid var(--stroke);background:rgba(255,255,255,0.08)}

    /* Image preview */
    .thumbWrap{
      width:100%;
      border-radius:16px;
      overflow:hidden;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.05);
      position:relative;
      aspect-ratio:16/10;
      margin-bottom:10px;
    }
    .thumbImg{
      width:100%;height:100%;
      object-fit:cover;display:block;
      transform:scale(1.02);
      filter:saturate(1.05) contrast(1.02);
    }
    .thumbOverlay{position:absolute;inset:0;background:linear-gradient(180deg, rgba(0,0,0,0.00) 35%, rgba(0,0,0,0.72) 100%);pointer-events:none}
    .thumbBadge{
      position:absolute;left:10px;bottom:10px;
      padding:6px 10px;border-radius:999px;
      border:1px solid rgba(255,255,255,0.20);
      background:rgba(18,22,30,0.55);
      backdrop-filter:blur(14px) saturate(160%);
      -webkit-backdrop-filter:blur(14px) saturate(160%);
      font-size:12px;font-weight:850;color:rgba(255,255,255,0.92);
      max-width:calc(100% - 20px);
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    }
    .thumbFallback{
      width:100%;height:100%;
      display:flex;align-items:center;justify-content:center;
      font-weight:900;letter-spacing:.8px;
      color:rgba(18,22,28,0.72);
      background:
        radial-gradient(520px 300px at 18% 0%, rgba(255,255,255,0.95), rgba(255,255,255,0) 62%),
        radial-gradient(420px 260px at 82% 6%, rgba(255,224,190,0.88), rgba(255,255,255,0) 58%),
        linear-gradient(135deg,
          rgba(252,248,242,0.96) 0%,
          rgba(244,236,226,0.94) 42%,
          rgba(236,224,212,0.94) 100%
        );
      position:relative;
      overflow:hidden;
    }
    .thumbFallback::before{
      content:"";
      position:absolute;inset:0;
      background:
        repeating-linear-gradient(0deg, rgba(0,0,0,0.035), rgba(0,0,0,0.035) 1px, rgba(255,255,255,0) 1px, rgba(255,255,255,0) 3px),
        radial-gradient(120px 120px at 40% 30%, rgba(255,255,255,0.55), rgba(255,255,255,0) 70%),
        radial-gradient(140px 140px at 70% 70%, rgba(255,255,255,0.35), rgba(255,255,255,0) 72%);
      opacity:0.12;
      mix-blend-mode:multiply;
      pointer-events:none;
    }
    .thumbFallback > *{position:relative;z-index:1}
    .thumbFallback .thumbNS .brand{color:rgba(18,22,28,0.56)}
    .thumbNS{display:flex;flex-direction:column;align-items:center;gap:6px;text-align:center;padding:10px}
    .thumbNS .mark{font-size:18px}
    .thumbNS .brand{font-size:12px;color:rgba(255,255,255,0.72);font-weight:800}

    /* Bottom nav */
    .bottomNav{
      position:fixed;left:0;right:0;bottom:0;
      padding:10px 12px calc(10px + env(safe-area-inset-bottom));
      display:flex;justify-content:center;
      z-index:9000;pointer-events:none;
    }
    .bottomNavInner{
      pointer-events:auto;
      width:min(560px, calc(100% - 24px));
      display:flex;gap:10px;padding:10px;
      border-radius:22px;border:1px solid var(--glassStroke);
      background:rgba(18,22,30,0.55);
      backdrop-filter:blur(22px) saturate(180%);
      -webkit-backdrop-filter:blur(22px) saturate(180%);
      box-shadow:0 12px 40px var(--glassShadow);
    }
    .navItem{
      flex:1;border-radius:16px;padding:10px 8px;text-align:center;
      cursor:pointer;user-select:none;border:1px solid transparent;
      background:rgba(255,255,255,0.05);
      display:flex;flex-direction:column;gap:6px;align-items:center;justify-content:center;
    }
    .navItemActive{border:1px solid rgba(230,193,128,0.35);background:rgba(230,193,128,0.12)}
    .navIcon{font-size:18px;line-height:1}
    .navLabel{font-size:11px;color:var(--muted)}
    .navItemActive .navLabel{color:rgba(255,255,255,0.85)}

    /* Sheets */
    .sheetOverlay{
      position:fixed;inset:0;
      background:var(--sheetOverlay);
      backdrop-filter:blur(22px) saturate(180%);
      -webkit-backdrop-filter:blur(22px) saturate(180%);
      z-index:9999;display:none;
      justify-content:center;align-items:flex-end;padding:10px;
    }
    .sheetOverlay.open{display:flex}
    .sheet{
      width:100%;max-width:560px;
      border-radius:22px 22px 18px 18px;
      border:1px solid var(--glassStroke);
      background:var(--sheetCardBg);
      backdrop-filter:blur(28px) saturate(180%);
      -webkit-backdrop-filter:blur(28px) saturate(180%);
      box-shadow:0 12px 40px var(--glassShadow);
      padding:14px 14px 10px;
      max-height:84vh;overflow:auto;
    }
    .sheetHandle{width:46px;height:5px;border-radius:999px;background:rgba(255,255,255,0.22);margin:0 auto 10px}
    .input{
      width:100%;
      border:1px solid var(--stroke);
      background:rgba(255,255,255,0.06);
      border-radius:16px;
      padding:12px 12px;
      outline:none;color:var(--text);font-size:14px;
    }
    .seg{display:flex;gap:8px;border:1px solid var(--stroke);background:rgba(255,255,255,0.05);padding:6px;border-radius:18px}
    .segBtn{
      flex:1;padding:10px;border-radius:14px;text-align:center;
      cursor:pointer;user-select:none;font-size:13px;border:1px solid transparent;
      color:var(--muted);background:transparent;
    }
    .segBtnActive{border:1px solid rgba(230,193,128,0.35);background:rgba(230,193,128,0.12);color:rgba(255,255,255,0.9);font-weight:750}
    .hidden{display:none!important}

    /* Splash loader */
    .nsSplash{
      position:fixed; inset:0; z-index:100000;
      display:flex; align-items:center; justify-content:center;
      background:inherit; /* uses body background */
      transition:opacity .35s ease, visibility .35s ease;
    }
    .nsSplash.hide{opacity:0; visibility:hidden; pointer-events:none}
    .nsSplashInner{display:flex; flex-direction:column; align-items:center; gap:10px; text-align:center; padding:24px}
    .nsMarkWrap{position:relative; width:88px; height:88px; display:flex; align-items:center; justify-content:center}
    .nsMark{
      width:74px; height:74px; border-radius:24px;
      border:1px solid rgba(255,255,255,0.14);
      background:linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
      box-shadow:0 18px 60px rgba(0,0,0,0.55);
      display:flex; align-items:center; justify-content:center;
      font-weight:950; letter-spacing:1px; font-size:22px;
      position:relative; overflow:hidden;
      color:rgba(255,255,255,0.92);
    }
    .nsMark:before{
      content:"";
      position:absolute; inset:-40%;
      background:conic-gradient(from 180deg, transparent, rgba(230,193,128,0.35), transparent 60%);
      animation:nsSpin 1.6s linear infinite;
      opacity:.9;
      z-index:0;
    }
    .nsMark:after{
      content:"";
      position:absolute; inset:1px; border-radius:23px;
      background:rgba(12,15,20,0.88);
      backdrop-filter:blur(16px) saturate(180%);
      -webkit-backdrop-filter:blur(16px) saturate(180%);
      z-index:1;
    }
    .nsMarkTxt{position:relative; z-index:2}
    .nsRing{
      position:absolute; inset:0;
      border-radius:999px;
      border:1px solid rgba(230,193,128,0.30);
      filter:drop-shadow(0 10px 40px rgba(230,193,128,0.12));
      animation:nsPulse 1.4s ease-in-out infinite;
      pointer-events:none;
    }
    .nsTitle{font-size:16px;font-weight:900; letter-spacing:.2px}
    .nsSub{font-size:12px;color:var(--muted)}
    .nsDots{display:flex; gap:6px; margin-top:2px}
    .nsDots span{
      width:6px; height:6px; border-radius:999px;
      background:rgba(255,255,255,0.45);
      animation:nsDot 1.1s ease-in-out infinite;
    }
    .nsDots span:nth-child(2){animation-delay:.15s}
    .nsDots span:nth-child(3){animation-delay:.30s}
    @keyframes nsSpin{to{transform:rotate(360deg)}}
    @keyframes nsPulse{
      0%,100%{transform:scale(1); opacity:.55}
      50%{transform:scale(1.08); opacity:.95}
    }
    @keyframes nsDot{
      0%,100%{transform:translateY(0); opacity:.45}
      50%{transform:translateY(-4px); opacity:.95}
    }

  
    /* Search highlight */
    mark.hl{
      padding:0 3px;
      border-radius:6px;
      background:rgba(230,193,128,0.22);
      color:rgba(255,255,255,0.95);
      border:1px solid rgba(230,193,128,0.25);
    }

    /* --- Premium "Личный кабинет" card (Quiet luxury, no gold) --- */
.cabinetFrame{
  position:relative;
  border-radius:24px;
  padding:14px;
  background:
    radial-gradient(900px 340px at 50% -18%, rgba(140,170,255,0.14), transparent 62%),
    radial-gradient(520px 260px at 12% 18%, rgba(255,255,255,0.06), transparent 60%),
    radial-gradient(520px 260px at 92% 22%, rgba(170,120,255,0.08), transparent 62%),
    linear-gradient(180deg, rgba(28,34,48,0.90), rgba(12,15,22,0.92));
  border:1px solid rgba(255,255,255,0.10);
  box-shadow:
    0 14px 34px rgba(0,0,0,0.42),
    inset 0 1px 0 rgba(255,255,255,0.08),
    inset 0 0 0 1px rgba(0,0,0,0.30);
  overflow:hidden;
}
/* subtle premium frame (no gold) */
.cabinetFrame:before{
  content:"";
  position:absolute;
  inset:0;
  padding:1px;
  border-radius:24px;
  background:linear-gradient(135deg,
    rgba(255,255,255,0.18),
    rgba(140,170,255,0.10),
    rgba(255,255,255,0.08)
  );
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity:0.55;
  pointer-events:none;
}
/* soft gloss + micro texture (quiet) */
.cabinetFrame:after{
  content:"";
  position:absolute;
  inset:-60px;
  background:
    radial-gradient(320px 240px at 18% 22%, rgba(255,255,255,0.06), transparent 62%),
    radial-gradient(320px 240px at 84% 26%, rgba(140,170,255,0.07), transparent 64%),
    linear-gradient(135deg, rgba(255,255,255,0.06), transparent 40%, rgba(0,0,0,0.26)),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 8px);
  opacity:0.70;
  pointer-events:none;
}

.cabinetHeader{
  position:relative;
  display:flex;
  align-items:center;
  justify-content:center;
  margin-bottom:8px;
  color:rgba(255,255,255,0.72);
  font-weight:900;
  letter-spacing:0.35px;
  font-size:13px;
  text-shadow:0 1px 0 rgba(0,0,0,0.55);
}
.cabinetMain{
  position:relative;
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:10px;
}
.cabinetGreet{
  font-size:18px;
  font-weight:950;
  color:rgba(255,255,255,0.96);
  text-shadow:0 1px 0 rgba(0,0,0,0.65);
}
.cabinetTier{
  margin-top:6px;
  font-size:13px;
  color:rgba(255,255,255,0.70);
  display:flex;
  align-items:center;
  gap:6px;
  font-weight:850;
}
.cabinetBalanceRow{
  position:relative;
  margin-top:12px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
}
.cabinetBalanceLabel{
  font-size:12px;
  color:rgba(255,255,255,0.58);
  font-weight:850;
  letter-spacing:0.2px;
}
.cabinetBalancePill{
  display:flex;
  align-items:center;
  gap:8px;
  padding:7px 10px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.14);
  background:
    radial-gradient(220px 120px at 30% 0%, rgba(140,170,255,0.16), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 18px rgba(0,0,0,0.36);
  font-size:13px;
  font-weight:950;
  color:rgba(255,255,255,0.94);
  white-space:nowrap;
}
.cabinetBalanceGem{
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.45));
}

.cabinetStats{
  position:relative;
  margin-top:12px;
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap:10px;
}
.cabinetStat{
  padding:9px 10px 9px;
  border-radius:16px;
  border:1px solid rgba(255,255,255,0.12);
  background:
    radial-gradient(260px 140px at 50% 0%, rgba(255,255,255,0.06), transparent 62%),
    rgba(255,255,255,0.035);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
  min-height:52px;
}
.cabinetStatLabel{
  font-size:11px;
  color:rgba(255,255,255,0.58);
  display:flex;
  align-items:center;
  gap:6px;
  letter-spacing:0.2px;
  font-weight:850;
}
.cabinetStatVal{
  margin-top:4px;
  font-size:14px;
  font-weight:950;
  color:rgba(255,255,255,0.95);
}



    /* ------------------ ROULETTE LUX (Obsidian Glass) ------------------ */
    .rouletteWrap{margin-top:12px}
    .wheelStage{display:flex;flex-direction:column;align-items:center;gap:10px}
    .wheelBox{position:relative; width:min(78vw, 360px); aspect-ratio: 1 / 1; border-radius:999px;
      background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.10), rgba(255,255,255,0.02) 42%, rgba(0,0,0,0.0) 70%),
                  rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.10);
      box-shadow: 0 18px 40px rgba(0,0,0,0.55);
      overflow:hidden;
    }
    .wheelCanvas{width:100%; height:100%; display:block}
    .wheelCenter{position:absolute; inset:28%; border-radius:999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 10px 22px rgba(0,0,0,0.35);
      display:flex; align-items:center; justify-content:center;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      color: rgba(255,255,255,0.92);
      font-weight: 900;
      letter-spacing: 0.6px;
    }
    
    .wheelPointer{position:absolute; top:-22px; left:50%; transform:translateX(-50%);
      width:0; height:0; z-index:3;
      border-left:10px solid transparent;
      border-right:10px solid transparent;
      border-bottom:28px solid rgba(235,245,255,0.86);
      filter: drop-shadow(0 12px 20px rgba(0,0,0,0.62));
    }
    /* wheelPointerDot removed: pointer should be arrow only */
    .microHud{margin-top:10px; font-size:12px; color: rgba(255,255,255,0.62); text-align:center}
    .ticker{margin-top:10px; height:32px; width:100%;
      border-radius:14px; padding:0 12px;
      display:flex; align-items:center;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.10);
      overflow:hidden;
    }
    .tickerText{white-space:nowrap; will-change:transform; color: rgba(255,255,255,0.70); font-size:12px}
    .chipsRow{margin-top:10px; display:flex; gap:8px; overflow:auto; padding-bottom:2px}
    .chip{flex:0 0 auto; padding:9px 12px; border-radius:16px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.10);
      color: rgba(255,255,255,0.86);
      font-size:12px;
    }
    .resultSheetOverlay{position:fixed; inset:0; background: rgba(0,0,0,0.0); pointer-events:none; transition:background 180ms ease; z-index:10050}
    .resultSheetOverlay.on{background: rgba(0,0,0,0.22); pointer-events:auto}
    /* Dior: subtle cold sparkle (quiet luxury) */
    .resultSheetOverlay.on.dior::before{
      content:"";
      position:absolute;
      inset:0;
      pointer-events:none;
      background-image:
        radial-gradient(circle at 20% 30%, rgba(235,245,255,0.22) 0 2px, transparent 3px),
        radial-gradient(circle at 72% 22%, rgba(235,245,255,0.18) 0 2px, transparent 3px),
        radial-gradient(circle at 58% 62%, rgba(235,245,255,0.16) 0 2px, transparent 3px),
        radial-gradient(circle at 35% 70%, rgba(235,245,255,0.14) 0 2px, transparent 3px),
        radial-gradient(circle at 82% 74%, rgba(235,245,255,0.14) 0 2px, transparent 3px);
      opacity:0;
      animation: sparklePop 900ms ease-out 1;
    }
    @keyframes sparklePop{
      0%{opacity:0; transform:translateY(8px) scale(0.98)}
      20%{opacity:1;}
      100%{opacity:0; transform:translateY(-10px) scale(1.02)}
    }
    .resultSheet{position:fixed; left:0; right:0; bottom:-420px; padding:16px; transition:bottom 260ms cubic-bezier(.2,.9,.2,1);
      z-index: 10051;
    }
    .resultSheet.on{bottom:0}
    .resultCard{max-width:520px; margin:0 auto; border-radius:22px; padding:14px 14px 12px 14px;
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.12);
      box-shadow: 0 18px 50px rgba(0,0,0,0.6);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }
    .resultTitle{font-weight:900; font-size:13px; color: rgba(255,255,255,0.92)}
    .resultValue{margin-top:8px; font-weight:1000; font-size:22px; letter-spacing:0.4px; color: rgba(255,255,255,0.96)}
    .resultSub{margin-top:6px; font-size:12px; color: rgba(255,255,255,0.62)}
    .resultBtns{display:flex; gap:10px; margin-top:12px}
    .btnGhost{flex:1; padding:12px 14px; border-radius:18px; border:1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.92); font-weight:900; text-align:center;
      cursor:pointer;
    }
    .btnGhost[disabled]{opacity:0.55; cursor:not-allowed}
    .btnPrimary{flex:1; padding:12px 14px; border-radius:18px; border:1px solid rgba(235,245,255,0.22);
      background: rgba(235,245,255,0.10); color: rgba(255,255,255,0.96); font-weight:900; text-align:center;
      cursor:pointer;
    }
    .sparkle{position:absolute; inset:0; pointer-events:none}

</style>
</head>
<body>
  <div id="nsSplash" class="nsSplash">
    <div class="nsSplashInner">
      <div class="nsMarkWrap">
        <div class="nsMark"><span class="nsMarkTxt">NS</span></div>
        <div class="nsRing"></div>
      </div>
      <div class="nsTitle">Natural Sense</div>
      <div class="nsSub">Загрузка…</div>
      <div class="nsDots"><span></span><span></span><span></span></div>
    </div>
  </div>
  <div id="root"></div>

  <script>
  (function(){
    const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;

    const CHANNEL = "__CHANNEL__";
    const BOT_USERNAME = "__BOT_USERNAME__"; // может быть пустым

    const DEFAULT_BG = "#0c0f14";

    function showSplash(){
      const s = document.getElementById("nsSplash");
      if(s) s.classList.remove("hide");
    }
    function hideSplash(){
      const s = document.getElementById("nsSplash");
      if(!s) return;
      s.classList.add("hide");
      setTimeout(()=>{ try{ s.parentNode && s.parentNode.removeChild(s); }catch(e){} }, 450);
    }

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
      setVar("--card2", scheme==="dark" ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.82)");

      setVar("--sheetOverlay", scheme==="dark" ? hexToRgba(bg,0.55) : hexToRgba(bg,0.45));
      setVar("--sheetCardBg", scheme==="dark" ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.86)");
      setVar("--glassStroke", scheme==="dark" ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.10)");
      setVar("--glassShadow", scheme==="dark" ? "rgba(0,0,0,0.45)" : "rgba(0,0,0,0.18)");

      try{
        if(tg){
          tg.setHeaderColor(bg);
          tg.setBackgroundColor(bg);
        }
      }catch(e){}
    }

    function haptic(kind){
      try{ tg && tg.HapticFeedback && tg.HapticFeedback.impactOccurred && tg.HapticFeedback.impactOccurred(kind||"light"); }catch(e){}
    }
    function openLink(url){
      if(!url) return;
      if(tg && tg.openTelegramLink) tg.openTelegramLink(url);
      else window.open(url,"_blank");
    }

    async function askConfirm(title, message, okText){
      okText = okText || "Да";
      // Prefer native Telegram popup if available
      if(tg && tg.showPopup){
        return await new Promise((resolve)=>{
          try{
            tg.showPopup({
              title: title || "Подтверждение",
              message: message || "",
              buttons: [
                {id:"ok", type:"destructive", text: okText},
                {id:"cancel", type:"cancel", text:"Отмена"}
              ]
            }, (btnId)=>{
              resolve(btnId==="ok");
            });
          }catch(e){
            resolve(window.confirm((title?title+"\\n\\n":"") + (message||"")));
          }
        });
      }
      return window.confirm((title?title+"\\n\\n":"") + (message||""));
    }
    
    function fmtNum(n){
      try{
        const x = Number(n||0);
        return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
      }catch(e){
        return String(n||0);
      }
    }
function esc(s){
      return String(s||"").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
    }
    function el(tag, cls, html){
      const n = document.createElement(tag);
      if(cls) n.className = cls;
      if(html!==undefined) n.innerHTML = html;
      return n;
    }
    function tierLabel(t){
      // Статусы по просьбе: оставляем на английском
      return ( {free:"🥉 Bronze", premium:"🥈 Silver", vip:"🥇 Gold VIP"}[t] ) || "🥉 Bronze";
    }

    let postsRefreshTimer = null;
        const DIOR_CONVERT_VALUE = 3000;
    const state = {
      tab: "journal",
      user: null,
      botUsername: (BOT_USERNAME||"").replace(/^@/,""),
      postsSheet: {open:false, tag:null, title:""},
      posts: [],
      loadingPosts: false,
      profileOpen:false,
      profileView:"menu",
      raffle:null,
      rouletteHistory:[],

      rouletteOddsOpen:false,

      rouletteRecent: [],
      rouletteWheel: {angle:0, spinning:false, mode:"idle", lastTick:-1, startedAt:0, targetKey:null, spinId:null, prize:null, overlay:false},
      rouletteStatus: {can_spin:true, seconds_left:0, enough_points:true, points:0, spin_cost:300},
      rouletteCooldownTick: 0,
      claim: {open:false, claim_id:null, claim_code:null, status:null, prize_label:null, data:null, step:1, form:{full_name:"", phone:"", country:"", city:"", address_line:"", postal_code:"", comment:""}},
      inventoryOpen:false,
      inventory:null,
      invMsg:"",
      q:"",
      searchResults: [],
      searchLoading: false,
      searchLastQ: "",
      // discoverJump: куда прокрутить на экране "Поиск" (brands/categories/products)
      discoverJump:null,
      // Daily tasks (isolated; must never break UI)
      dailyOpen:false,
      daily:null,
      dailyMsg:"",
      dailyBusy:false,
      msg:"",
      busy:false
    };
    // -------------------------------------------------------------------------
    // SEARCH UI (fix input bug: do not recreate input on each keypress)
    // -------------------------------------------------------------------------
    let searchInputEl = null;
    let searchResultsBoxEl = null;
    let searchAbortController = null;

    function setInputValuePreserveCaret(inp, v){
      try{
        if(document.activeElement === inp){
          const s = inp.selectionStart, e = inp.selectionEnd;
          inp.value = v;
          if(typeof s === "number" && typeof e === "number"){
            inp.setSelectionRange(s, e);
          }
        }else{
          inp.value = v;
        }
      }catch(_){
        inp.value = v;
      }
    }

    function escRegExp(s){
      return (s||"").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    function makeSnippet(text, q, radius=70){
      const t = (text||"").toString();
      const query = (q||"").trim();
      if(!t) return "";
      if(!query) return t;
      try{
        const reQ = new RegExp(escRegExp(query), "i");
        const m = reQ.exec(t);
        if(!m) return t.length > (radius*2+30) ? (t.slice(0, radius*2) + "…") : t;
        const idx = m.index;
        const start = Math.max(0, idx - radius);
        const end = Math.min(t.length, idx + m[0].length + radius);
        const pre = start>0 ? "…" : "";
        const post = end<t.length ? "…" : "";
        return pre + t.slice(start, end) + post;
      }catch(e){
        return t.length > (radius*2+30) ? (t.slice(0, radius*2) + "…") : t;
      }
    }

    function highlightHTML(text, q){
      const t = (text||"").toString();
      const query = (q||"").trim();
      if(!query) return esc(t);
      const safe = esc(t);
      try{
        const reQ = new RegExp(escRegExp(query), "ig");
        return safe.replace(reQ, (m)=>'<mark class="hl">'+m+'</mark>');
      }catch(e){
        return safe;
      }
    }

    function updateSearchBox(){
      if(state.tab !== "discover") return;
      if(!searchResultsBoxEl) return;

      searchResultsBoxEl.innerHTML = "";
      const q = (state.q||"").trim();

      if(!q){
        const hint = el("div","sub","Например: &laquo;консилер&raquo;, &laquo;SPF&raquo;, &laquo;Drunk Elephant&raquo; …");
        hint.style.marginTop="12px";
        searchResultsBoxEl.appendChild(hint);
        return;
      }

      if(state.searchLoading){
        const l = el("div","sub","Ищу посты…");
        l.style.marginTop="12px";
        searchResultsBoxEl.appendChild(l);
        return;
      }

      if(!state.searchResults || state.searchResults.length===0){
        const empty = el("div","sub","Ничего не найдено.");
        empty.style.marginTop="12px";
        searchResultsBoxEl.appendChild(empty);
        return;
      }

      const list = el("div");
      list.style.marginTop="12px";
      for(const p of state.searchResults){
        list.appendChild(postCard(p, true));
      }
      searchResultsBoxEl.appendChild(list);
    }


    const tgUserId = tg && tg.initDataUnsafe && tg.initDataUnsafe.user ? tg.initDataUnsafe.user.id : null;

    // Data sets
    const JOURNAL_BLOCKS = [
      { tag: "Новинка", title: "🆕 Новинки" },
      { tag: "Люкс", title: "💎 Люкс" },
      { tag: "Тренд", title: "🔥 Тренд" },
      { tag: "Оценка", title: "⭐ Оценка" },
      { tag: "Факты", title: "🧾 Факты" }
    ];
    const BRANDS = [
      ["The Ordinary","TheOrdinary","Skincare essentials"],
      ["Dior","Dior","Couture beauty"],
      ["Chanel","Chanel","Iconic classics"],
      ["Kylie Cosmetics","KylieCosmetics","Pop-glam"],
      ["Gisou","Gisou","Honey haircare"],
      ["Rare Beauty","RareBeauty","Soft-focus makeup"],
      ["Yves Saint Laurent","YSL","Bold luxury"],
      ["Givenchy","Givenchy","Haute beauty"],
      ["Charlotte Tilbury","CharlotteTilbury","Red carpet glow"],
      ["NARS","NARS","Editorial makeup"],
      ["Sol de Janeiro","SolDeJaneiro","Body & scent"],
      ["Huda Beauty","HudaBeauty","Full glam"],
      ["Rhode","Rhode","Minimal skincare"],
      ["Tower 28 Beauty","Tower28Beauty","Sensitive skin"],
      ["Benefit Cosmetics","BenefitCosmetics","Brows & cheeks"],
      ["Estée Lauder","EsteeLauder","Skincare icons"],
      ["Sisley","Sisley","Ultra premium"],
      ["Kérastase","Kerastase","Salon haircare"],
      ["Armani Beauty","ArmaniBeauty","Soft luxury"],
      ["Hourglass","Hourglass","Ambient glow"],
      ["Shiseido","Shiseido","Japanese skincare"],
      ["Tom Ford Beauty","TomFordBeauty","Private blend vibe"],
      ["Tarte","Tarte","Everyday glam"],
      ["Sephora Collection","SephoraCollection","Smart basics"],
      ["Clinique","Clinique","Skin first"],
      ["Dolce & Gabbana","DolceGabbana","Italian glamour"],
      ["Kayali","Kayali","Fragrance focus"],
      ["Guerlain","Guerlain","Heritage luxury"],
      ["Fenty Beauty","FentyBeauty","Inclusive glam"],
      ["Too Faced","TooFaced","Playful makeup"],
      ["MAKE UP FOR EVER","MakeUpForEver","Pro artistry"],
      ["Erborian","Erborian","K-beauty meets EU"],
      ["Natasha Denona","NatashaDenona","Palette queen"],
      ["Lancôme","Lancome","French classics"],
      ["Kosas","Kosas","Clean makeup"],
      ["ONE/SIZE","OneSize","Stage-ready"],
      ["Laneige","Laneige","Hydration"],
      ["Makeup by Mario","MakeupByMario","Artist essentials"],
      ["Valentino Beauty","ValentinoBeauty","Couture color"],
      ["Drunk Elephant","DrunkElephant","Active skincare"],
      ["Olaplex","Olaplex","Bond repair"],
      ["Anastasia Beverly Hills","AnastasiaBeverlyHills","Brows & glam"],
      ["Amika","Amika","Hair styling"],
      ["BYOMA","BYOMA","Barrier care"],
      ["Glow Recipe","GlowRecipe","Fruity glow"],
      ["Milk Makeup","MilkMakeup","Cool minimal"],
      ["Summer Fridays","SummerFridays","Clean glow"],
      ["K18","K18","Repair tech"]
    ];
    const CATEGORIES = [
      ["Новинка","Новинка","Новые релизы"],
      ["Люкс","Люкс","Люкс подборка"],
      ["Тренд","Тренд","Что в тренде"],
      ["История","История","Истории брендов"],
      ["Оценка","Оценка","Личные обзоры"],
      ["Факты","Факты","Короткие факты"],
      ["Состав","Состав","Состав / формулы"],
      ["Challenge","Челленджи","Бьюти челленджи"],
      ["SephoraPromo","SephoraPromo","Промо Sephora"]
    ];
    const PRODUCTS = [
      ["Праймер","Праймер"],["Тональная основа","ТональнаяОснова"],["Консилер","Консилер"],
      ["Пудра","Пудра"],["Румяна","Румяна"],["Скульптор","Скульптор"],["Бронзер","Бронзер"],
      ["Продукт для бровей","ПродуктДляБровей"],["Хайлайтер","Хайлайтер"],["Тушь","Тушь"],
      ["Тени","Тени"],["Помада","Помада"],["Карандаш для губ","КарандашДляГуб"],["Палетка","Палетка"],["Фиксатор","Фиксатор"]
    ];

    const journalCache = {}; // tag->posts preview list

    async function fetchJson(url, opts){
      const controller = new AbortController();
      const timeoutMs = 12000;
      const t = setTimeout(()=>controller.abort(), timeoutMs);
      try{
        const r = await fetch(url, Object.assign({}, opts||{}, {signal: controller.signal, cache:"no-store"}));
        const data = await r.json().catch(()=> ({}));
        if(!r.ok) throw new Error(data.detail || ("HTTP "+r.status));
        return data;
      }catch(e){
        if(String(e && e.name) === "AbortError") throw new Error("Таймаут сети");
        throw e;
      }finally{
        clearTimeout(t);
      }
    }

    async function apiGet(url, init){
      const sep = url.includes("?") ? "&" : "?";
      const bust = "t="+Date.now();
      return fetchJson(url + sep + bust, Object.assign({method:"GET"}, (init||{})));
    }
    async function apiPost(url, body){
      const sep = url.includes("?") ? "&" : "?";
      const bust = "t="+Date.now();
      return fetchJson(url + sep + bust, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})});
    }

    async function refreshUser(){
      if(!tgUserId) return;
      try{
        state.user = await apiGet("/api/user/"+encodeURIComponent(tgUserId));
      }catch(e){}
    }
    async function loadBotUsername(){
      try{
        const d = await apiGet("/api/bot/username");
        const u = String(d.bot_username||"").trim().replace(/^@/,"");
        if(u) state.botUsername = u;
      }catch(e){}
    }

    async function loadЖурналBlocks(){
      for(const b of JOURNAL_BLOCKS){
        try{
          const arr = await apiGet("/api/posts?tag="+encodeURIComponent(b.tag));
          journalCache[b.tag] = Array.isArray(arr) ? arr.slice(0,8) : [];
        }catch(e){
          journalCache[b.tag] = [];
        }
      }
    }

    async function openPosts(tag, title){
      state.postsSheet.open = true;
      state.postsSheet.tag = tag;
      state.postsSheet.title = title || ("#"+tag);
      state.posts = [];
      state.loadingPosts = true;

      // автообновление каждые 30 секунд (редактирование/удаление)
      if(postsRefreshTimer) { try{ clearInterval(postsRefreshTimer); }catch(e){} postsRefreshTimer=null; }

      render();

      async function loadOnce(){
        try{
          const arr = await apiGet("/api/posts?tag="+encodeURIComponent(tag));
          state.posts = Array.isArray(arr) ? arr : [];
        }catch(e){
          state.posts = [];
        }finally{
          state.loadingPosts = false;
          render();
        }
      }

      await loadOnce();
      postsRefreshTimer = setInterval(()=>{ if(state.postsSheet.open && state.postsSheet.tag===tag){ loadOnce(); } }, 30000);
    }

    function closePosts(){
      state.postsSheet = {open:false, tag:null, title:""};
      state.posts = [];
      state.loadingPosts = false;
      if(postsRefreshTimer) { try{ clearInterval(postsRefreshTimer); }catch(e){} postsRefreshTimer=null; }
      render();
    }

    async function openInventory(){
      state.inventoryOpen = true;
      try{ dailyEvent('open_inventory'); }catch(e){}
      state.invMsg = "";
      render();
      if(!tgUserId) return;
      try{
        state.inventory = await apiGet("/api/inventory?telegram_id="+encodeURIComponent(tgUserId));
      }catch(e){
        state.inventory = null;
      }
      render();
    }
    function closeInventory(){
      state.inventoryOpen = false;
      state.invMsg = "";
      render();
    }

    // Roulette odds sheet
    function openRouletteOdds(){
      state.rouletteOddsOpen = true;
      render();
    }
    function closeRouletteOdds(){
      state.rouletteOddsOpen = false;
      render();
    }

    async function loadRaffleStatus(){
      if(!tgUserId) return;
      try{
        state.raffle = await apiGet("/api/raffle/status?telegram_id="+encodeURIComponent(tgUserId));
      }catch(e){
        state.raffle = null;
      }
    }
    async function loadRouletteHistory(){
      if(!tgUserId) return;
      try{
        const arr = await apiGet("/api/roulette/history?telegram_id="+encodeURIComponent(tgUserId)+"&limit=5");
        state.rouletteHistory = Array.isArray(arr) ? arr : [];
      }catch(e){
        state.rouletteHistory = [];
      }
    }

    async function openПрофиль(view){
      state.profileOpen = true;
      try{ dailyEvent('open_profile'); }catch(e){}
      state.profileView = view || "menu";
      render();
      await Promise.all([loadRaffleStatus(), loadRouletteHistory(), loadRouletteRecent()]);
      render();
    }
    function closeПрофиль(){
      state.profileOpen = false;
      state.msg = "";
      render();
    }

    async function buyTicket(){
      if(!tgUserId || state.busy) return;
      state.busy = true; state.msg = ""; render();
      try{
        const d = await apiPost("/api/raffle/buy_ticket", {telegram_id: tgUserId, qty: 1});
        state.msg = "✅ Билет куплен. Твоих билетов: "+d.ticket_count;
        try{ dailyEvent('spin_roulette'); }catch(e){}
        await refreshUser();
        await loadRaffleStatus();
        haptic("light");
      }catch(e){
        state.msg = "❌ "+(e.message||"Ошибка");
      }finally{
        state.busy = false;
        render();
      }
    }

    async function spinRoulette(){
      if(!tgUserId || state.busy) return;
      state.busy = true; state.msg = ""; render();
      try{
        const d = await apiPost("/api/roulette/spin", {telegram_id: tgUserId});
        state.msg = "🎡 Выпало: "+d.prize_label;
        try{
          if(tg && tg.showPopup){
            tg.showPopup({title:"🎡 Рулетка", message:"Ваш приз: "+d.prize_label, buttons:[{type:"ok"}]});
          }
        }catch(e){}
        await refreshUser();
        await loadRaffleStatus();
        await loadRouletteHistory();
        haptic("light");
      }catch(e){
        state.msg = "❌ "+(e.message||"Ошибка");
      }finally{
        state.busy = false;
        render();
      }
    }


    // ------------------ ROULETTE LUX (Obsidian Glass) ------------------
    // NOTE: UI/logic must stay stable. Percentages are controlled by `chance` values below.
    // Chances (percent): +50=45, +100=25, +150=15, +200=8, +300=4, ticket=2, Dior=1
    const ROULETTE_SEGMENTS = [
      {key:"points_500",  icon:"💎", text:"+50",  chance:45},
      {key:"points_1000", icon:"💎", text:"+100", chance:25},
      {key:"points_1500", icon:"💎", text:"+150", chance:15},
      {key:"points_2000", icon:"💎", text:"+200", chance:8},
      {key:"ticket_1",    icon:"🎟", text:"+1",   chance:2},
      {key:"points_3000", icon:"💎", text:"+300", chance:4},
      {key:"dior_palette",icon:"✨", text:"Dior", chance:1},
    ];
    const SEG_N = ROULETTE_SEGMENTS.length;
    const SEG_ANGLE = (Math.PI*2)/SEG_N;

    let _wheelRaf = null;
    let _tickerRaf = null;

    function wheelIndexAtPointer(angle){
      // angle: rotation applied to wheel (radians). Segments start at -PI/2.
      // pointer is fixed at -PI/2. We want segment whose center aligns with pointer.
      let a = (-(angle) % (Math.PI*2) + (Math.PI*2)) % (Math.PI*2);
      // shift by half segment so boundaries map correctly
      let idx = Math.floor((a + SEG_ANGLE/2) / SEG_ANGLE) % SEG_N;
      return idx;
    }

    function keyToIndex(key){
      const i = ROULETTE_SEGMENTS.findIndex(s=>s.key===key);
      return i>=0?i:0;
    }

    
function drawWheel(canvas, angle){
  if(!canvas) return;
  const dpr = window.devicePixelRatio || 1;

  // --- Stable canvas sizing (prevents jitter / drifting text on some devices) ---
  // We resize only when the *CSS size* changed noticeably, not on every frame.
  const rect = canvas.getBoundingClientRect();
  const cssW = rect.width;
  const cssH = rect.height;
  canvas._cssW = canvas._cssW || 0;
  canvas._cssH = canvas._cssH || 0;
  if(Math.abs(cssW - canvas._cssW) > 0.5 || Math.abs(cssH - canvas._cssH) > 0.5){
    canvas._cssW = cssW;
    canvas._cssH = cssH;
    const w = Math.max(10, Math.round(cssW * dpr));
    const h = Math.max(10, Math.round(cssH * dpr));
    canvas.width = w;
    canvas.height = h;
  }

  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext("2d");
  if(!ctx) return;

  ctx.clearRect(0,0,w,h);
  const cx = w/2, cy = h/2;
  const r = Math.min(w,h)*0.48;
  const innerR = r*0.46;                // hub radius matches .wheelCenter inset
  const labelR = (innerR + r) / 2;      // EXACT center of each sector ring

  // ---------------- Segments (wheel rotates) ----------------
  ctx.save();
  ctx.translate(cx,cy);
  ctx.rotate(angle);

  for(let i=0;i<SEG_N;i++){
    const start = -Math.PI/2 + i*SEG_ANGLE;
    const end = start + SEG_ANGLE;

    ctx.beginPath();
    ctx.moveTo(0,0);
    ctx.arc(0,0,r,start,end,false);
    ctx.closePath();

    const isAlt = (i%2===0);
    ctx.fillStyle = isAlt ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.085)";
    ctx.fill();

    // divider
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = Math.max(1, Math.floor(1*dpr));
    ctx.stroke();
  }

  // rim highlight (rotates with wheel)
  ctx.beginPath();
  ctx.arc(0,0,r,0,Math.PI*2);
  ctx.strokeStyle = "rgba(235,245,255,0.16)";
  ctx.lineWidth = Math.max(1, Math.floor(2*dpr));
  ctx.stroke();

  ctx.restore();

  // ---------------- Labels (locked to sector centers, NO drift) ----------------
  // We draw labels in screen space using the CURRENT wheel angle to compute their positions.
  // This makes icons/text "nailed" to each segment center and prevents any wobble.
  ctx.save();
  ctx.translate(cx,cy);

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for(let i=0;i<SEG_N;i++){
    const seg = ROULETTE_SEGMENTS[i];
    const mid = -Math.PI/2 + (i+0.5)*SEG_ANGLE;
    const a = angle + mid;
    const x = Math.cos(a) * labelR;
    const y = Math.sin(a) * labelR;

    // icon
    ctx.font = `900 ${Math.floor(16*dpr)}px system-ui, -apple-system, Segoe UI, Roboto, Arial`;
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillText(seg.icon, x, y - Math.floor(10*dpr));

    // label
    ctx.font = `900 ${Math.floor(12*dpr)}px system-ui, -apple-system, Segoe UI, Roboto, Arial`;
    ctx.fillStyle = "rgba(255,255,255,0.82)";
    ctx.fillText(seg.text, x, y + Math.floor(10*dpr));

    if(seg.key==="dior_palette"){
      ctx.fillStyle = "rgba(220,235,255,0.20)";
      ctx.fillRect(
        x - Math.floor(18*dpr),
        y + Math.floor(22*dpr),
        Math.floor(36*dpr),
        Math.floor(2*dpr)
      );
    }
  }

  ctx.restore();
}


function easeOutCubic(t){ return 1 - Math.pow(1-t,3); }
    function easeInOut(t){ return t<0.5 ? 2*t*t : 1 - Math.pow(-2*t+2,2)/2; }

    function stopWheelRaf(){
      if(_wheelRaf){ cancelAnimationFrame(_wheelRaf); _wheelRaf=null; }
    }

    function stopTickerRaf(){
      if(_tickerRaf){ cancelAnimationFrame(_tickerRaf); _tickerRaf=null; }
      const elTxt = document.getElementById("rouletteTickerText");
      if(elTxt){ try{ elTxt.style.transform = "translateX(0px)"; }catch(e){} }
    }

    function startFreeSpin(){
      const w = state.rouletteWheel;
      w.spinning = true;
      w.mode = "free";
      w.startedAt = Date.now();
      w.lastTick = -1;
      let lastTs = performance.now();
      let vel = 0; // rad/sec
      const accelDur = 700; // ms
      const targetVel = 14.5; // rad/sec (fast)
      function frame(ts){
        const dt = Math.max(0, (ts-lastTs)/1000);
        lastTs = ts;
        const elapsed = Date.now() - w.startedAt;
        if(elapsed < accelDur){
          const p = elapsed/accelDur;
          vel = targetVel * easeOutCubic(p);
        }else{
          vel = targetVel;
        }
        w.angle += vel*dt;

        // tick
        const idx = wheelIndexAtPointer(w.angle);
        if(idx !== w.lastTick){
          w.lastTick = idx;
          haptic("light");
        }

        const canvas = document.getElementById("wheelCanvas");
        drawWheel(canvas, w.angle);
        _wheelRaf = requestAnimationFrame(frame);
      }
      stopWheelRaf();
      _wheelRaf = requestAnimationFrame(frame);
    }

    function animateToAngle(finalAngle, durationMs, onDone){
      const w = state.rouletteWheel;
      const startAngle = w.angle;
      const delta = finalAngle - startAngle;
      const t0 = performance.now();
      function frame(ts){
        const p = Math.min(1, (ts-t0)/durationMs);
        const e = (p<0.5) ? easeInOut(p) : easeOutCubic(p);
        w.angle = startAngle + delta*e;

        const idx = wheelIndexAtPointer(w.angle);
        if(idx !== w.lastTick){
          w.lastTick = idx;
          haptic("light");
        }

        const canvas = document.getElementById("wheelCanvas");
        drawWheel(canvas, w.angle);
        if(p<1){
          _wheelRaf = requestAnimationFrame(frame);
        }else{
          w.spinning = false;
          w.mode = "idle";
          stopWheelRaf();
          if(onDone) onDone();
        }
      }
      stopWheelRaf();
      _wheelRaf = requestAnimationFrame(frame);
    }

    async function loadRouletteRecent(){
      try{
        const arr = await apiGet("/api/roulette/recent_wins?limit=12");
        state.rouletteRecent = Array.isArray(arr) ? arr : [];
      }catch(e){
        state.rouletteRecent = [];
      }
    }

    function startTicker(){
      // simple ticker transform loop
      stopTickerRaf();
      const elTxt = document.getElementById("rouletteTickerText");
      if(!elTxt) return;
      let x = 0;
      let last = performance.now();
      function frame(ts){
        const dt = (ts-last)/1000; last=ts;
        const speed = 18; // px/sec
        x -= speed*dt;
        const w = elTxt.scrollWidth;
        if(w>0 && Math.abs(x) > w) x = elTxt.parentElement ? elTxt.parentElement.clientWidth : 0;
        elTxt.style.transform = "translateX("+x+"px)";
        _tickerRaf = requestAnimationFrame(frame);
      }
      _tickerRaf = requestAnimationFrame(frame);
    }

    function openClaimForm(claim_id, claim_code, prize_label){
      // Open profile sheet and switch to claim flow (works from Cosmetics/Inventory too)
      state.profileOpen = true;
      state.inventoryOpen = false;
      state.invMsg = "";
      state.profileView = "claim";
      state.claim.open = true;
      state.claim.claim_id = claim_id;
      state.claim.claim_code = claim_code;
      state.claim.prize_label = prize_label;
      state.claim.status = "draft";
            state.claim.step = 1;
      state.claim.form = {full_name:"", phone:"", country:"", city:"", address_line:"", postal_code:"", comment:""};
      render();
      renderПрофильSheet();
      renderDailySheet();
      // load claim data
      (async ()=>{
        try{
          const d = await apiGet("/api/roulette/claim/"+encodeURIComponent(claim_id)+"?telegram_id="+encodeURIComponent(tgUserId));
          state.claim.data = d;
          state.claim.status = d.status || "draft";
          // hydrate form from server draft
          state.claim.form = {
            full_name: d.full_name || "",
            phone: d.phone || "",
            country: d.country || "",
            city: d.city || "",
            address_line: d.address_line || "",
            postal_code: d.postal_code || "",
            comment: d.comment || ""
          };
          state.claim.step = 1;
        }catch(e){}
        renderПрофильSheet();
      })();
    }

    
    async function updateRouletteStatus(){
      if(!tgUserId) return;
      try{
        const st = await apiGet("/api/roulette/status?telegram_id="+encodeURIComponent(tgUserId));
        state.rouletteStatus = st;
      }catch(e){
        // ignore UI status errors
      }
    }

    function startRouletteCooldownTicker(){
      // One ticker for countdown UI (prevents multiple intervals)
      if(state._rouletteCooldownTimer) return;
      state._rouletteCooldownTimer = setInterval(()=>{
        try{
          if(!state.rouletteStatus) return;
          if(state.rouletteStatus.seconds_left && state.rouletteStatus.seconds_left > 0){
            state.rouletteStatus.seconds_left = Math.max(0, (state.rouletteStatus.seconds_left|0) - 1);
            state.rouletteStatus.can_spin_time = state.rouletteStatus.seconds_left===0;
            state.rouletteStatus.can_spin = state.rouletteStatus.can_spin_time && !!state.rouletteStatus.enough_points;
            state.rouletteCooldownTick = (state.rouletteCooldownTick||0)+1;
            // rerender only if roulette screen visible
            if(state.profileView==="roulette"){ renderПрофильSheet(); }
          }
        }catch(e){}
      }, 1000);
    }

async function spinRouletteLux(){
      if(!tgUserId || state.busy) return;
      state.busy = true; state.msg=""; state.rouletteWheel.overlay=false;
      renderПрофильSheet();

      // IMPORTANT: no "fake spins"
      // We do NOT start wheel animation until the server confirms the spin.
      const started = Date.now();
      let resp = null, err = null;
      try{
        resp = await apiPost("/api/roulette/spin", {telegram_id: tgUserId});
      }catch(e){ err = e; }

      if(err || !resp){
        stopWheelRaf();
        stopTickerRaf();
        state.busy=false;
        state.msg = "❌ "+(err && err.message ? err.message : "Ошибка");
        await updateRouletteStatus();
        renderПрофильSheet();
        return;
      }

      // Start wheel only after successful server response
      startFreeSpin();

      // ensure minimum spin time (so it feels like a real spin)
      const minTime = 900;
      const elapsed = Date.now()-started;
      if(elapsed < minTime){
        await new Promise(r=>setTimeout(r, minTime-elapsed));
      }

      // compute final angle for target
      const key = resp.prize_key || "";
      const idx = keyToIndex(key);
      const fullTurns = 3 + Math.floor(Math.random()*3); // 3..5
      // Stop strictly at the CENTER of the winning segment.
      // Derivation: pointer is at -PI/2, segment i center is at (-PI/2 + i*SEG_ANGLE + SEG_ANGLE/2) + angle.
      // So we need angle = -i*SEG_ANGLE - SEG_ANGLE/2 (mod 2PI).
      let base = state.rouletteWheel.angle;
      let desired = (-idx*SEG_ANGLE) - (SEG_ANGLE/2);
      // normalize desired near base
      const twoPi = Math.PI*2;
      while(desired < base) desired += twoPi;
      // add turns
      let finalAngle = desired + twoPi*fullTurns;

      // IMPORTANT: do not offset finalAngle (must snap to exact center).

      state.rouletteWheel.spinId = resp.spin_id;
      state.rouletteWheel.targetKey = key;
      state.rouletteWheel.prize = resp;

      // decel animation
      animateToAngle(finalAngle, 1850, async ()=>{
        // show result sheet
        state.busy = false;
        state.rouletteWheel.overlay = true; // show result sheet for ANY prize
        // refresh user / history / recent
        await Promise.all([refreshUser(), loadRaffleStatus(), loadRouletteHistory(), loadRouletteRecent(), updateRouletteStatus()]);
        renderПрофильSheet();
        // popup light
        try{
          if(tg && tg.HapticFeedback && key==="dior_palette"){
            tg.HapticFeedback.notificationOccurred("success");
          }
        }catch(e){}
      });
    }


    async function claimFromResult(){
      const resp = state.rouletteWheel.prize;
      if(!resp || !resp.spin_id) return;
      // IMPORTANT: This is the "Забрать" flow.
      // No conversion confirmation here — conversion has its own button and confirmation.
      state.busy = true; renderПрофильSheet();
      try{
        const d = await apiPost("/api/roulette/claim/create", {telegram_id: tgUserId, spin_id: resp.spin_id});
        haptic("medium");
        state.busy = false;
        openClaimForm(d.claim_id, d.claim_code, resp.prize_label);
      }catch(e){
        state.busy = false;
        state.msg = "❌ "+(e.message||"Ошибка");
        renderПрофильSheet();
      }
    }

    async function convertFromResult(){
      const resp = state.rouletteWheel.prize;
      if(!resp || !resp.spin_id) return;

      const ok = await askConfirm(
        "Конвертировать приз?",
        "Вы получите +"+fmtNum(DIOR_CONVERT_VALUE)+" баллов. После конвертации забрать приз будет нельзя.",
        "Да, конвертировать"
      );
      if(!ok) return;

      state.busy = true; renderПрофильSheet();
      try{
        const d = await apiPost("/api/roulette/convert", {telegram_id: tgUserId, spin_id: resp.spin_id});
        state.msg = "✅ Конвертировано: +"+d.converted_value;
        haptic("light");
        await Promise.all([refreshUser(), loadRaffleStatus(), loadRouletteHistory()]);
      }catch(e){
        state.msg = "❌ "+(e.message||"Ошибка");
      }finally{
        state.busy = false;
        // close result
        state.rouletteWheel.prize = null;
        state.rouletteWheel.overlay = false;
        renderПрофильSheet();
      }
    }

    function closeResultSheet(){
      state.rouletteWheel.prize = null;
      state.rouletteWheel.overlay = false;
      renderПрофильSheet();
    }

    // Rendering helpers
    function postCard(post, full){
      const tagTitle = "#"+((post.tags && post.tags[0]) ? post.tags[0] : "post");
      const wrap = el("div", full ? "card2" : "miniCard");
      wrap.style.cursor = "pointer";
      wrap.addEventListener("click", ()=>{ haptic(); try{ dailyEvent('open_post', {message_id: post.message_id}); }catch(e){} openLink(post.url); });

      const tw = el("div","thumbWrap");
      if(full) tw.style.aspectRatio = "16 / 9";

      const badge = el("div","thumbBadge", esc(tagTitle));
      const overlay = el("div","thumbOverlay");

      if(post.media_url){
        const img = document.createElement("img");
        img.className = "thumbImg";
        img.loading = "lazy";
        img.src = post.media_url;
        img.alt = post.preview || tagTitle;
        img.onerror = ()=>{ tw.innerHTML=""; tw.appendChild(fallbackThumb()); tw.appendChild(overlay); tw.appendChild(badge); };
        tw.appendChild(img);
      }else{
        tw.appendChild(fallbackThumb());
      }

      tw.appendChild(overlay);
      tw.appendChild(badge);

      wrap.appendChild(tw);
      wrap.appendChild(el("div","miniMeta", esc(tagTitle)+" • ID "+esc(post.message_id)));
            // В поиске подсвечиваем совпадение (как Windows)
      if(full && state.tab==="discover" && (state.q||"").trim()){
        const sn = makeSnippet(post.preview || "", state.q, 80);
        const t = el("div","miniText");
        t.innerHTML = highlightHTML(sn || (post.preview||"Открыть пост →"), state.q);
        wrap.appendChild(t);
      }else{
        wrap.appendChild(el("div","miniText", esc(post.preview || "Открыть пост →")));
      }

      const chips = el("div","chipRow");
      const tags = Array.isArray(post.tags) ? post.tags.slice(0, full?8:4) : [];
      for(const t of tags){
        chips.appendChild(el("div","chip", "#"+esc(t)));
      }
      wrap.appendChild(chips);

      return wrap;
    }
    function fallbackThumb(){
      const f = el("div","thumbFallback");
      const ns = el("div","thumbNS");
      ns.appendChild(el("div","mark","NS"));
      ns.appendChild(el("div","brand","Natural Sense"));
      f.appendChild(ns);
      return f;
    }

    function renderЖурнал(main){
      const hero = el("div","card");
      hero.addEventListener("click", ()=>{ if(state.user){ haptic(); openПрофиль("menu"); } });

      const glow = el("div");
      glow.style.position="absolute";
      glow.style.inset="-2px";
      glow.style.background="radial-gradient(600px 300px at 10% 0%, rgba(230,193,128,0.26), transparent 60%)";
      glow.style.pointerEvents="none";
      hero.appendChild(glow);

      const inner = el("div");
      inner.style.position="relative";

      const topRow = el("div","row");
      const left = el("div");
      left.appendChild(el("div","h1","NS · Natural Sense"));
      left.appendChild(el("div","sub","Выпуск дня · люкс-журнал"));
      topRow.appendChild(left);

      if(state.user){
        topRow.appendChild(el("div","pill","💎 "+esc(state.user.points)+" · "+esc(tierLabel(state.user.tier))));
      }
      inner.appendChild(topRow);

      inner.appendChild(el("div",null,'<div style="margin-top:12px;color:var(--muted);font-size:13px">Подборки, факты и люкс-обзоры — прямо в Telegram.</div>'));

      const actions = el("div");
      actions.style.marginTop="12px";
      actions.style.display="grid";
      actions.style.gap="10px";

      const openCh = el("div","btn");
      openCh.addEventListener("click",(e)=>{ e.stopPropagation(); haptic(); dailyEvent('open_channel'); openLink("https://t.me/"+CHANNEL); });
      openCh.appendChild(el("div",null,'<div class="btnTitle">↩️ В канал</div><div class="btnSub">Вернуться в ленту Natural Sense</div>'));
      openCh.appendChild(el("div",null,'<div style="opacity:0.85">›</div>'));
      actions.appendChild(openCh);

      const grid1 = el("div","grid");
      const tNew = el("div","tile");
      // Главный экран: Новинки -> Категории
      tNew.addEventListener("click",(e)=>{
        e.stopPropagation();
        haptic();
        state.tab = "categories";
        render();
      });
      tNew.appendChild(el("div","tileTitle","📚 Категории"));
      tNew.appendChild(el("div","tileSub","Темы и разделы журнала"));
      const tLux = el("div","tile");
      // Главный экран: Люкс -> Бренды
      tLux.addEventListener("click",(e)=>{
        e.stopPropagation();
        haptic();
        state.tab = "brands";
        render();
      });
      tLux.appendChild(el("div","tileTitle","🏷️ Бренды"));
      tLux.appendChild(el("div","tileSub","Все бренды и теги"));
      grid1.appendChild(tNew); grid1.appendChild(tLux);

      const grid2 = el("div","grid");
      const tTrend = el("div","tile");
      // Главный экран: Тренд -> Типы продуктов
      tTrend.addEventListener("click",(e)=>{
        e.stopPropagation();
        haptic();
        state.tab = "products";
        render();
      });
      tTrend.appendChild(el("div","tileTitle","🧴 Продукты"));
      tTrend.appendChild(el("div","tileSub","Типы продуктов"));
      const tBag = el("div","tile");
      tBag.addEventListener("click",(e)=>{ e.stopPropagation(); haptic(); openInventory(); });
      tBag.appendChild(el("div","tileTitle","👜 Косметичка"));
      tBag.appendChild(el("div","tileSub","Призы и билеты"));
      grid2.appendChild(tTrend); grid2.appendChild(tBag);

      actions.appendChild(grid1);
      actions.appendChild(grid2);
      inner.appendChild(actions);

      hero.appendChild(inner);
      main.appendChild(hero);

      for(const b of JOURNAL_BLOCKS){
        const block = el("div");
        block.style.marginTop="14px";

        const hdr = el("div","row");
        hdr.style.alignItems="baseline";
        hdr.appendChild(el("div",null,'<div style="font-size:15px;font-weight:850">'+esc(b.title)+'</div>'));

        const viewAll = el("div",null,'<div style="font-size:12px;color:var(--muted);cursor:pointer;user-select:none">Смотреть всё ›</div>');
        viewAll.addEventListener("click", ()=>{ haptic(); openPosts(b.tag, b.title); });
        hdr.appendChild(viewAll);
        block.appendChild(hdr);

        const sc = el("div","hScroll");
        sc.style.marginTop="10px";
        const arr = journalCache[b.tag] || [];
        if(arr.length===0){
          const empty = el("div","miniCard");
          empty.style.minWidth="100%";
          empty.style.cursor="default";
          empty.appendChild(el("div","miniMeta","Пока пусто"));
          empty.appendChild(el("div","miniText",'<span style="color:var(--muted)">Добавь посты с тегом #'+esc(b.tag)+' в канал.</span>'));
          sc.appendChild(empty);
        }else{
          for(const p of arr){
            sc.appendChild(postCard(p,false));
          }
        }
        block.appendChild(sc);
        main.appendChild(block);
      }
    }

    
function renderПоиск(main){
      const wrap = el("div","card2");

      const top = el("div","row");
      const tl = el("div");
      tl.appendChild(el("div","h1","Поиск"));
      tl.appendChild(el("div","sub","Поиск по тексту постов в канале"));
      top.appendChild(tl);

      const bag = el("div","pill","👜 Косметичка");
      bag.style.cursor="pointer";
      bag.addEventListener("click", ()=>{ haptic(); openInventory(); });
      top.appendChild(bag);

      wrap.appendChild(top);

      if(!searchInputEl){
        searchInputEl = document.createElement("input");
        searchInputEl.className="input";
        searchInputEl.placeholder="Введите слово или фразу…";
        searchInputEl.addEventListener("input", (e)=>{
          state.q = e.target.value;
          scheduleSearch();
          updateSearchBox(); // без полного render() — иначе ломается ввод
        });
        searchInputEl.addEventListener("keydown", (e)=>{
          // Esc очищает
          if(e.key === "Escape"){
            e.preventDefault();
            state.q = "";
            setInputValuePreserveCaret(searchInputEl, "");
            scheduleSearch();
            updateSearchBox();
          }
          // Enter — мгновенный поиск (без debounce)
          if(e.key === "Enter"){
            const q = (state.q||"").trim();
            if(q){
              try{ if(searchDebounce){ clearTimeout(searchDebounce); } }catch(_){}
              runSearch(q, true);
            }
          }
        });
      }

      // синхронизируем значение, не сбивая каретку
      setInputValuePreserveCaret(searchInputEl, state.q || "");

      const inpWrap = el("div");
      inpWrap.style.marginTop="12px";
      inpWrap.appendChild(searchInputEl);
      wrap.appendChild(inpWrap);

      // контейнер результатов (перерисовываем только его)
      searchResultsBoxEl = el("div");
      wrap.appendChild(searchResultsBoxEl);

      updateSearchBox();
      main.appendChild(wrap);
    }

    let searchDebounce = null;
    function scheduleSearch(){
      const q = (state.q||"").trim();
      if(searchDebounce){ try{ clearTimeout(searchDebounce);}catch(e){} searchDebounce=null; }

      if(!q){
        // отменяем текущий запрос
        if(searchAbortController){ try{ searchAbortController.abort(); }catch(_){}
          searchAbortController = null;
        }
        state.searchResults = [];
        state.searchLoading = false;
        state.searchLastQ = "";
        updateSearchBox();
        return;
      }

      searchDebounce = setTimeout(()=>{ runSearch(q); }, 250);
    }

    async function runSearch(q, force){
      q = (q||"").trim();
      if(!q) return;

      // не делаем повторно тот же запрос
      if(!force && state.searchLastQ === q && Array.isArray(state.searchResults)) return;

      // отменяем предыдущий запрос
      if(searchAbortController){ try{ searchAbortController.abort(); }catch(_){}
        searchAbortController = null;
      }
      searchAbortController = new AbortController();

      state.searchLoading = true;
      state.searchLastQ = q;
      updateSearchBox();

      try{
        const arr = await apiGet("/api/search?q="+encodeURIComponent(q), { signal: searchAbortController.signal });
        state.searchResults = Array.isArray(arr) ? arr : [];
        try{ dailyEvent('use_search', {q:q}); }catch(e){}
      }catch(e){
        // если отменили — тихо выходим
        if(e && (e.name === "AbortError" || (""+e).includes("AbortError"))){
          return;
        }
        state.searchResults = [];
      }finally{
        state.searchLoading = false;
        updateSearchBox();
      }
    }


    
    function renderКатегории(main){
      const wrap = el("div","card2");

      const top = el("div","row");
      const tl = el("div");
      tl.appendChild(el("div","h1","Категории"));
      tl.appendChild(el("div","sub","Разделы журнала"));
      top.appendChild(tl);

      const back = el("div","pill","← Назад");
      back.style.cursor="pointer";
      back.addEventListener("click", ()=>{ haptic(); state.tab="journal"; render(); });
      top.appendChild(back);

      wrap.appendChild(top);

      const grid = el("div","grid");
      grid.style.marginTop="12px";

      const data = CATEGORIES;
      for(const item of data){
        const obj = ((x)=>({name:x[0], tag:x[1], sub:x[2]}))(item);
        const t = el("div","tile");
        t.addEventListener("click", ()=>{ haptic(); openPosts(obj.tag, obj.name); });
        t.appendChild(el("div","tileTitle", esc(obj.name)));
        t.appendChild(el("div","tileSub", esc(obj.sub || ("#"+obj.tag))));
        grid.appendChild(t);
      }

      wrap.appendChild(grid);
      main.appendChild(wrap);
    }

    function renderБренды(main){
      const wrap = el("div","card2");

      const top = el("div","row");
      const tl = el("div");
      tl.appendChild(el("div","h1","Бренды"));
      tl.appendChild(el("div","sub","Все бренды и теги"));
      top.appendChild(tl);

      const back = el("div","pill","← Назад");
      back.style.cursor="pointer";
      back.addEventListener("click", ()=>{ haptic(); state.tab="journal"; render(); });
      top.appendChild(back);

      wrap.appendChild(top);

      const grid = el("div","grid");
      grid.style.marginTop="12px";

      const data = BRANDS;
      for(const item of data){
        const obj = ((x)=>({name:x[0], tag:x[1], sub:x[2]}))(item);
        const t = el("div","tile");
        t.addEventListener("click", ()=>{ haptic(); openPosts(obj.tag, obj.name); });
        t.appendChild(el("div","tileTitle", esc(obj.name)));
        t.appendChild(el("div","tileSub", esc(obj.sub || ("#"+obj.tag))));
        grid.appendChild(t);
      }

      wrap.appendChild(grid);
      main.appendChild(wrap);
    }

    function renderПродукты(main){
      const wrap = el("div","card2");

      const top = el("div","row");
      const tl = el("div");
      tl.appendChild(el("div","h1","Продукты"));
      tl.appendChild(el("div","sub","Типы продуктов"));
      top.appendChild(tl);

      const back = el("div","pill","← Назад");
      back.style.cursor="pointer";
      back.addEventListener("click", ()=>{ haptic(); state.tab="journal"; render(); });
      top.appendChild(back);

      wrap.appendChild(top);

      const grid = el("div","grid");
      grid.style.marginTop="12px";

      const data = PRODUCTS.map(p=>[p[0], p[1], "#"+p[1]]);
      for(const item of data){
        const obj = ((x)=>({name:x[0], tag:x[1], sub:x[2]}))(item);
        const t = el("div","tile");
        t.addEventListener("click", ()=>{ haptic(); openPosts(obj.tag, obj.name); });
        t.appendChild(el("div","tileTitle", esc(obj.name)));
        t.appendChild(el("div","tileSub", esc(obj.sub || ("#"+obj.tag))));
        grid.appendChild(t);
      }

      wrap.appendChild(grid);
      main.appendChild(wrap);
    }

function renderБонусы(main){
      const wrap = el("div","card2");
      const top = el("div","row");
      const tl = el("div");
      tl.appendChild(el("div","h1","Бонусы"));
      tl.appendChild(el("div","sub","Рулетка · Билеты · Косметичка"));
      top.appendChild(tl);
      if(state.user) top.appendChild(el("div","pill","💎 "+esc(state.user.points)+" баллов"));
      wrap.appendChild(top);

      const grid = el("div","grid");
      grid.style.marginTop="12px";

      const t1 = el("div","tile");
      t1.addEventListener("click", ()=>{ haptic(); openПрофиль("roulette"); });
      t1.appendChild(el("div","tileTitle","🎡 Рулетка"));
      t1.appendChild(el("div","tileSub",`Испытать удачу (${state.rouletteStatus.spin_cost||0})`));
      const t2 = el("div","tile");
      t2.addEventListener("click", ()=>{ haptic(); openПрофиль("raffle"); });
      t2.appendChild(el("div","tileTitle","🎁 Розыгрыши"));
      t2.appendChild(el("div","tileSub","Билет (500)"));
      const t3 = el("div","tile");
      t3.addEventListener("click", ()=>{ haptic(); openInventory(); });
      t3.appendChild(el("div","tileTitle","👜 Косметичка"));
      t3.appendChild(el("div","tileSub","Призы и билеты"));
      const t4 = el("div","tile");
      t4.addEventListener("click", ()=>{ haptic(); openPosts("Challenge","💎 Челленджи"); });
      t4.appendChild(el("div","tileTitle","💎 Челленджи"));
      t4.appendChild(el("div","tileSub","Ежедневная мотивация"));



      const t5 = el("div","tile");
      t5.addEventListener("click", ()=>{ haptic(); openDaily(); });
      t5.appendChild(el("div","tileTitle","🎯 Daily бонусы"));
      t5.appendChild(el("div","tileSub","Задания на +400/день"));

            grid.appendChild(t1);grid.appendChild(t2);grid.appendChild(t3);grid.appendChild(t4);grid.appendChild(t5);
      wrap.appendChild(grid);

      wrap.appendChild(el("div","hr"));
      const openCh = el("div","btn");
      openCh.addEventListener("click", ()=>{ haptic(); dailyEvent('open_channel'); openLink("https://t.me/"+CHANNEL); });
      openCh.appendChild(el("div",null,'<div class="btnTitle">↩️ В канал</div><div class="btnSub">Natural Sense feed</div>'));
      openCh.appendChild(el("div",null,'<div style="opacity:0.85">›</div>'));
      wrap.appendChild(openCh);

      main.appendChild(wrap);
    }


// -----------------------------------------------------------------------
// DAILY TASKS (isolated module: must not break main UI)
// -----------------------------------------------------------------------
async function dailyEvent(event, data){
  try{
    if(!tgUserId) return;
    await apiPost("/api/daily/event", {telegram_id: tgUserId, event: event, data: (data||{})});

    // If Daily sheet is open — refresh tasks so the user sees completion instantly.
    if(state && state.dailyOpen){
      clearTimeout(state.__dailyRefreshT);
      state.__dailyRefreshT = setTimeout(async ()=>{
        try{
          state.daily = await apiGet("/api/daily/tasks?telegram_id="+encodeURIComponent(tgUserId));
          render();
        }catch(e){}
      }, 450);
    }
  }catch(e){
    // silent (must never break main UI)
  }
}

async function openDaily(){
  state.dailyOpen = true;
  state.dailyMsg = "";
  render();
  try{
    if(!tgUserId) return;
    // Ensure "Зайти в Mini App" is always recorded before loading the list.
    try{ await dailyEvent('open_miniapp'); }catch(e){}
    state.daily = await apiGet("/api/daily/tasks?telegram_id="+encodeURIComponent(tgUserId));
  }catch(e){
    state.daily = null;
    state.dailyMsg = "❌ Не удалось загрузить задания";
  }
  render();
}
function closeDaily(){
  state.dailyOpen = false;
  state.dailyMsg = "";
  render();
}

async function claimDaily(taskKey){
  if(!tgUserId || state.dailyBusy) return;
  state.dailyBusy = true;
  state.dailyMsg = "";
  render();
  try{
    const resp = await apiPost("/api/daily/claim", {telegram_id: tgUserId, task_key: taskKey});
    await refreshUser();
    state.daily = await apiGet("/api/daily/tasks?telegram_id="+encodeURIComponent(tgUserId));
    state.dailyMsg = resp.awarded ? ("✅ +" + resp.awarded + " бонусов") : "✅ Уже получено";
    haptic("light");
  }catch(e){
    state.dailyMsg = "❌ "+(e.message||"Ошибка");
  }finally{
    state.dailyBusy = false;
    render();
  }
}

function renderDailySheet(){
  const overlay = document.getElementById("dailyOverlay");
  if(!overlay) return;
  overlay.classList.toggle("open", !!state.dailyOpen);
  const content = document.getElementById("dailyContent");
  if(!content) return;
  content.innerHTML = "";
  if(!state.dailyOpen) return;

  const hdr = el("div","row");
  hdr.style.alignItems="baseline";
  hdr.appendChild(el("div","h1","🎯 Daily бонусы"));
  const close = el("div",null,'<div style="font-size:13px;color:var(--muted);cursor:pointer">Закрыть</div>');
  close.addEventListener("click", ()=>{ haptic(); closeDaily(); });
  hdr.appendChild(close);
  content.appendChild(hdr);

  if(state.daily && typeof state.daily.claimed_points==="number"){
    content.appendChild(el("div","sub","Сегодня: "+esc(state.daily.claimed_points)+" / "+esc(state.daily.max_points)+" · осталось "+esc(state.daily.remaining_points)));
  }else{
    content.appendChild(el("div","sub","Ежедневные задания, чтобы набрать до 400 бонусов."));
  }

  if(state.dailyMsg){
    const m = el("div","sub", esc(state.dailyMsg));
    m.style.marginTop="10px";
    content.appendChild(m);
  }

  if(!state.daily){
    const b = el("div","sub","Загрузка…");
    b.style.marginTop="12px";
    content.appendChild(b);
    return;
  }

  const list = el("div");
  list.style.marginTop="12px";
  list.style.display="grid";
  list.style.gap="10px";

  const tasks = Array.isArray(state.daily.tasks) ? state.daily.tasks : [];
  for(const t of tasks){
    const card = el("div","card2");
    const row = el("div","row");
    const left = el("div");
    left.appendChild(el("div",null,'<div style="font-size:14px;font-weight:900">'+esc(t.icon||"🎯")+' '+esc(t.title)+'</div>'));
    let sub = "";
    if((t.need||1) > 1){
      sub = (t.progress||0) + " / " + t.need;
    }else{
      sub = t.claimed ? "Получено" : (t.done ? "Выполнено" : "Не выполнено");
    }
    left.appendChild(el("div","sub", esc(sub)));
    row.appendChild(left);

    const pill = el("div","pill","💎 +"+esc(t.points));
    row.appendChild(pill);

    card.appendChild(row);

    const btnRow = el("div");
    btnRow.style.marginTop="10px";
    btnRow.style.display="grid";
    btnRow.style.gridTemplateColumns="1fr";
    btnRow.style.gap="8px";

    const btn = el("div","btn");
    const canClaim = !!t.done && !t.claimed;
    btn.style.opacity = canClaim ? "1" : "0.55";
    btn.style.pointerEvents = canClaim ? "auto" : "none";
    btn.addEventListener("click", ()=>{ haptic(); claimDaily(t.key); });
    btn.appendChild(el("div",null,'<div class="btnTitle">'+(t.claimed ? "✅ Получено" : (t.done ? "🎁 Забрать" : "🔒 Выполни чтобы забрать"))+'</div><div class="btnSub">'+(t.claimed ? "Награда уже начислена" : (t.done ? "Нажми, чтобы получить бонусы" : "Сначала выполни задание"))+'</div>'));
    btn.appendChild(el("div",null,'<div style="opacity:0.85">›</div>'));
    btnRow.appendChild(btn);

    card.appendChild(btnRow);

    list.appendChild(card);
  }

  content.appendChild(list);
}

    function renderPostsSheet(){
      const overlay = document.getElementById("postsOverlay");
      overlay.classList.toggle("open", !!state.postsSheet.open);
      const content = document.getElementById("postsContent");
      content.innerHTML = "";

      if(!state.postsSheet.open) return;

      const hdr = el("div","row");
      hdr.style.alignItems="baseline";
      hdr.appendChild(el("div","h1", esc(state.postsSheet.title || "Посты")));
      const close = el("div",null,'<div style="font-size:13px;color:var(--muted);cursor:pointer">Закрыть</div>');
      close.addEventListener("click", ()=>{ haptic(); closePosts(); });
      hdr.appendChild(close);
      content.appendChild(hdr);

      content.appendChild(el("div","sub","Посты "+(state.postsSheet.tag ? "#"+esc(state.postsSheet.tag) : "")));

      if(state.loadingPosts){
        content.appendChild(el("div","sub",'Загрузка…'));
        return;
      }
      if(!state.loadingPosts && state.posts.length===0){
        content.appendChild(el("div","sub",'Постов с этим тегом пока нет.'));
        return;
      }
      const list = el("div");
      list.style.marginTop="12px";
      list.style.display="grid";
      list.style.gap="10px";
      for(const p of state.posts){
        list.appendChild(postCard(p,true));
      }
      content.appendChild(list);
    }

    function renderInventorySheet(){
      const overlay = document.getElementById("invOverlay");
      overlay.classList.toggle("open", !!state.inventoryOpen);
      const content = document.getElementById("invContent");
      content.innerHTML = "";
      if(!state.inventoryOpen) return;

      const hdr = el("div","row"); hdr.style.alignItems="baseline";
      hdr.appendChild(el("div","h1","👜 Моя косметичка"));
      const close = el("div",null,'<div style="font-size:13px;color:var(--muted);cursor:pointer">Закрыть</div>');
      close.addEventListener("click", ()=>{ haptic(); closeInventory(); });
      hdr.appendChild(close);
      content.appendChild(hdr);
      content.appendChild(el("div","sub","Призы и билеты"));

      const bal = el("div","card2");
      const r1 = el("div","row");
      const left = el("div");
      left.appendChild(el("div",null,'<div style="font-size:13px;color:var(--muted)">Баланс</div>'));
      left.appendChild(el("div",null,'<div style="margin-top:6px;font-size:16px;font-weight:900">💎 '+esc(state.user ? state.user.points : 0)+' баллов</div>'));
      r1.appendChild(left);
      r1.appendChild(el("div","pill", esc(tierLabel(state.user ? state.user.tier : "free"))));
      bal.appendChild(r1);
      bal.style.marginTop="12px";
      content.appendChild(bal);

      const inv = state.inventory || {};
      const haveTickets = Number(inv.ticket_count || 0) || 0;
      const rate = Number(inv.ticket_convert_rate || 0) || 0;
      const diorValue = Number(inv.dior_convert_value || 0) || 0;

      const tCard = el("div","card2");
      tCard.style.marginTop="12px";
      tCard.appendChild(el("div",null,'<div style="font-size:14px;font-weight:900">🎟 Билеты</div>'));
      tCard.appendChild(el("div","sub",'У вас: <b style="color:rgba(255,255,255,0.92)">'+haveTickets+'</b>'));
      tCard.appendChild(el("div","sub",'Курс: 1 = '+rate+' баллов'));

      // Simple convert all button (no qty selector in vanilla to keep it stable)
      const convBtn = el("div","btn");
      convBtn.style.marginTop="10px";
      convBtn.style.opacity = (!haveTickets || state.busy) ? 0.5 : 1;
      convBtn.style.cursor = (!haveTickets || state.busy) ? "not-allowed" : "pointer";
      convBtn.appendChild(el("div",null,'<div class="btnTitle">💎 Конвертировать все билеты</div><div class="btnSub">'+(haveTickets ? ('Будет ~'+(haveTickets*rate)+' баллов') : 'Нет билетов')+'</div>'));
      convBtn.appendChild(el("div",null,'<div style="opacity:0.85">›</div>'));
      convBtn.addEventListener("click", async ()=>{
        if(!haveTickets || state.busy) return;
        state.busy=true; state.invMsg=""; renderInventorySheet();
        try{
          const d = await apiPost("/api/inventory/convert_ticket", {telegram_id: tgUserId, qty: haveTickets});
          state.invMsg = "✅ Обмен выполнен: +"+d.added_points+" баллов";
          await refreshUser();
          state.inventory = await apiGet("/api/inventory?telegram_id="+encodeURIComponent(tgUserId));
          haptic("light");
        }catch(e){
          state.invMsg = "❌ "+(e.message||"Ошибка");
        }finally{
          state.busy=false;
          renderInventorySheet();
        }
      });
      tCard.appendChild(convBtn);
      content.appendChild(tCard);

      // Prizes list (convert only in vanilla, claim stays in bot)
      const pCard = el("div","card2");
      pCard.style.marginTop="12px";
      pCard.appendChild(el("div",null,'<div style="font-size:14px;font-weight:900">🎁 Призы</div>'));

      const prizes = Array.isArray(inv.prizes) ? inv.prizes : [];
      if(prizes.length===0){
        pCard.appendChild(el("div","sub","Пока нет призов."));
      }else{
        const list = el("div");
        list.style.marginTop="10px";
        list.style.display="grid";
        list.style.gap="10px";
        for(const p of prizes){
          const pc = el("div","card2");
          pc.style.border="1px solid rgba(230,193,128,0.22)";
          pc.style.background="rgba(230,193,128,0.08)";
          pc.style.padding="14px";

          const title = el("div",null,'<div style="font-size:14px;font-weight:950">'+esc(p.prize_label||"✨ Dior Palette")+'</div>');
          pc.appendChild(title);

          // meta + status
          const st = String(p.status||"draft").trim() || "draft";
          const stMap = {
            "draft":"Доступно",
            "awaiting_contact":"Нужны данные",
            "submitted":"Заявка отправлена",
            "approved":"Подтверждено",
            "fulfilled":"Отправлено",
            "rejected":"Отклонено",
            "closed":"Закрыто"
          };
          const stLabel = stMap[st] || st;

          const meta = el("div","sub", 'Код: '+esc(p.claim_code||"-")+' • '+esc(stLabel));
          meta.style.marginTop="6px";
          pc.appendChild(meta);

          const canAct = (st==="draft" || st==="awaiting_contact");

          if(!canAct){
            const statusPill = el("div",null,'<div style="margin-top:12px;display:inline-flex;gap:8px;align-items:center;padding:10px 12px;border-radius:999px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.10);color:rgba(255,255,255,0.88);font-weight:900;font-size:12px">🟡 '+esc(stLabel)+'</div>');
            pc.appendChild(statusPill);
          }else{
            // Action zone (Variant A): primary + secondary
            const actions = el("div");
            actions.style.marginTop="12px";
            actions.style.display="grid";
            actions.style.gap="10px";

            const claimBtn = el("div","btn");
            claimBtn.style.justifyContent="center";
            claimBtn.style.fontWeight="950";
            claimBtn.style.border="1px solid rgba(235,245,255,0.22)";
            claimBtn.style.background="rgba(235,245,255,0.10)";
            claimBtn.innerHTML = (st==='awaiting_contact' ? '✍️ Продолжить оформление' : '🎁 Забрать приз');
            claimBtn.addEventListener("click", ()=>{
              haptic();
              const cid = p.claim_id;
              const code = String(p.claim_code||"").trim();
              const label = String(p.prize_label||"Приз");
              if(cid){
                openClaimForm(cid, code, label);
              }else if(state.botUsername && tg && tg.openTelegramLink && code){
                tg.openTelegramLink("https://t.me/"+state.botUsername+"?start=claim_"+encodeURIComponent(code));
              }else{
                alert("/claim "+code);
              }
            });

            const convLink = el("div",null,'<div style="text-align:center;font-weight:900;font-size:12.5px;color:rgba(255,255,255,0.78);padding:10px 12px;border-radius:16px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);cursor:pointer">💎 Конвертировать • +'+esc(diorValue)+'</div>');
            convLink.style.opacity = state.busy ? 0.6 : 1;
            convLink.style.cursor = state.busy ? "not-allowed" : "pointer";
            convLink.addEventListener("click", async ()=>{
              const code = String(p.claim_code||"").trim();
              if(!code || state.busy) return;

              const ok = await askConfirm(
                "Конвертировать приз?",
                "Вы получите +"+diorValue+" баллов. После конвертации забрать приз будет нельзя.",
                "Да, конвертировать"
              );
              if(!ok) return;

              state.busy=true; state.invMsg=""; renderInventorySheet();
              try{
                state.invMsg = "✅ Приз превращён в бонусы: +"+d.added_points+" баллов";
                await refreshUser();
                state.inventory = await apiGet("/api/inventory?telegram_id="+encodeURIComponent(tgUserId));
                haptic("light");
              }catch(e){
                state.invMsg = "❌ "+(e.message||"Ошибка");
              }finally{
                state.busy=false;
                renderInventorySheet();
              }
            });

            actions.appendChild(claimBtn);
            actions.appendChild(convLink);
            pc.appendChild(actions);
          }

          list.appendChild(pc);
        }
        pCard.appendChild(list);
      }
      content.appendChild(pCard);

      if(state.invMsg){
        const m = el("div","card2", esc(state.invMsg));
        m.style.marginTop="12px";
        content.appendChild(m);
      }
    }

    function renderRouletteOddsSheet(){
      const overlay = document.getElementById("oddsOverlay");
      if(!overlay) return;
      overlay.classList.toggle("open", !!state.rouletteOddsOpen);
      const content = document.getElementById("oddsContent");
      if(!content) return;
      content.innerHTML = "";
      if(!state.rouletteOddsOpen) return;

      const hdr = el("div","row");
      hdr.style.alignItems="baseline";
      hdr.appendChild(el("div","h1","📊 Шансы рулетки"));
      const close = el("div",null,'<div style="font-size:13px;color:var(--muted);cursor:pointer">Закрыть</div>');
      close.addEventListener("click", ()=>{ haptic(); closeRouletteOdds(); });
      hdr.appendChild(close);
      content.appendChild(hdr);

      // list
      const list = el("div");
      list.style.marginTop = "12px";
      list.style.display = "grid";
      list.style.gap = "10px";

      // We show exactly the configured chances on the wheel (sorted: biggest → smallest).
      const sortedSegs = [...ROULETTE_SEGMENTS].sort((a,b)=>(Number(b.chance)||0)-(Number(a.chance)||0));
      for(const s of sortedSegs){
        const row = el("div","card2");
        row.style.display = "flex";
        row.style.justifyContent = "space-between";
        row.style.alignItems = "center";
        row.style.gap = "10px";

        const left = el("div");
        left.style.display = "flex";
        left.style.alignItems = "center";
        left.style.gap = "10px";

        const ic = el("div",null,'<div style="font-size:18px;line-height:1">'+esc(String(s.icon||""))+'</div>');
        const title = el("div");
        const sub = (s.key||"").startsWith("points_") ? "бонусы" : ((s.key||"")==="ticket_1" ? "билет" : "приз");
        title.innerHTML = '<div style="font-weight:900">'+esc(String(s.text||""))+'</div><div class="sub" style="margin-top:4px">'+esc(sub)+'</div>';
        left.appendChild(ic);
        left.appendChild(title);

        const pct = el("div",null,'<div class="pill" style="border-color:rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);font-weight:950">'+esc(String(s.chance))+'%</div>');

        row.appendChild(left);
        row.appendChild(pct);
        list.appendChild(row);
      }
      content.appendChild(list);

      const pctLine = sortedSegs.map(s=>{
        const label = (s.key||"")==="ticket_1" ? "билет" : ((s.key||"")==="dior_palette" ? "Dior" : String(s.text||""));
        return label+"="+String(s.chance)+"%";
      }).join(", ");
      content.appendChild(el("div","sub", "Проценты: "+pctLine));
    }

    function renderПрофильSheet(){
      const overlay = document.getElementById("profileOverlay");
      overlay.classList.toggle("open", !!state.profileOpen);
      const content = document.getElementById("profileContent");
      content.innerHTML = "";
      // cleanup floating overlays from previous render
      if(state._cleanup && Array.isArray(state._cleanup)){
        try{ state._cleanup.forEach(fn=>{ try{ fn(); }catch(e){} }); }catch(e){}
      }
      state._cleanup = [];

      // Stop roulette animations when profile is closed or when we are not on the roulette view
      if(!state.profileOpen){
        try{ stopWheelRaf(); }catch(e){}
        try{ stopTickerRaf(); }catch(e){}
        return;
      }
      if(state.profileView !== "roulette"){
        try{ stopWheelRaf(); }catch(e){}
        try{ stopTickerRaf(); }catch(e){}
      }

      if(!state.user){
        content.appendChild(el("div","sub","Профиль недоступен."));
        return;
      }

const hdr = el("div","row"); hdr.style.alignItems="baseline";
hdr.appendChild(el("div","h1","👤 Профиль"));
const close = el("div",null,'<div style="font-size:13px;color:var(--muted);cursor:pointer">Закрыть</div>');
close.addEventListener("click", ()=>{ haptic(); closeПрофиль(); });
hdr.appendChild(close);
content.appendChild(hdr);

const info = el("div","card2");
info.style.marginTop="12px";
info.innerHTML =
  '<div class="cabinetFrame">'+
    '<div class="cabinetHeader">Личный кабинет</div>'+
    '<div class="cabinetMain">'+
      '<div>'+
        '<div class="cabinetGreet">Привет, '+esc(state.user.first_name)+'!</div>'+
        '<div class="cabinetTier">👑 '+esc(tierLabel(state.user.tier))+'</div>'+
      '</div>'+
    '</div>'+
    '<div class="cabinetBalanceRow">'+
      '<div class="cabinetBalanceLabel">Balance</div>'+
      '<div class="cabinetBalancePill"><span class="cabinetBalanceGem">💎</span>'+esc(state.user.points)+'</div>'+
    '</div>'+
    '<div class="cabinetStats">'+
      '<div class="cabinetStat">'+
        '<div class="cabinetStatLabel">🔥 Streak</div>'+
        '<div class="cabinetStatVal">'+esc(state.user.daily_streak||0)+'</div>'+
      '</div>'+
      '<div class="cabinetStat">'+
        '<div class="cabinetStatLabel">🏁 Best</div>'+
        '<div class="cabinetStatVal">'+esc(state.user.best_streak||0)+'</div>'+
      '</div>'+
      '<div class="cabinetStat">'+
        '<div class="cabinetStatLabel">👥 Реф</div>'+
        '<div class="cabinetStatVal">'+esc(state.user.referral_count||0)+'</div>'+
      '</div>'+
    '</div>'+
  '</div>';
content.appendChild(info);

      content.appendChild(el("div","hr"));

      // Menu / views


      // Menu / views
      if(state.profileView==="menu"){
        const list = el("div");
        list.style.display="grid";
        list.style.gap="10px";

        function menuBtn(title, sub, onClick){
          const b = el("div","btn");
          b.innerHTML = '<div><div class="btnTitle">'+title+'</div><div class="btnSub">'+sub+'</div></div><div style="opacity:0.85">›</div>';
          b.addEventListener("click", ()=>{ haptic(); onClick(); });
          return b;
        }
                list.appendChild(menuBtn("👥 Рефералы","Ссылка и бонус +20", ()=>{ state.profileView="referrals"; state.msg=""; renderПрофильSheet(); }));
list.appendChild(menuBtn("👜 Моя косметичка","Призы и билеты", ()=>{ state.profileOpen=false; render(); openInventory(); }));
        list.appendChild(menuBtn("🎁 Розыгрыши","Купить билеты (500)", ()=>{ state.profileView="raffle"; renderПрофильSheet(); }));
        list.appendChild(menuBtn("🎡 Рулетка","Крутить (300)", ()=>{ state.profileView="roulette"; renderПрофильSheet(); }));
        list.appendChild(menuBtn("🧾 История рулетки","Последние спины", ()=>{ state.profileView="history"; renderПрофильSheet(); }));
        content.appendChild(list);
      }else{
        const back = el("div","btn");
        back.style.justifyContent="center";
        back.style.fontWeight="900";
        back.textContent = "← Назад";
        back.addEventListener("click", ()=>{ haptic(); state.profileView="menu"; state.msg=""; renderПрофильSheet(); });
        content.appendChild(back);

                if(state.profileView==="referrals"){
          const box = el("div");
          box.style.marginTop="12px";
          box.innerHTML =
            '<div style="display:flex;align-items:center;justify-content:space-between;gap:10px">'+
              '<div style="font-size:14px;font-weight:950">👥 Рефералы</div>'+
              '<div class="pill" id="refReload" style="cursor:pointer">↻</div>'+
            '</div>'+
            '<div class="sub" style="margin-top:6px">Твой доход: <b>10%</b> от бонусных выигрышей активных друзей. Если друг не заходит 7 дней — статус замораживается.</div>';
          content.appendChild(box);

          // Invite link
          const ref = (tgUserId && state.botUsername) ? ("https://t.me/"+state.botUsername+"?start="+tgUserId) : "";
          if(ref){
            const linkBox = el("div","card2");
            linkBox.style.marginTop="10px";
            linkBox.innerHTML = '<div style="font-size:12px;color:rgba(255,255,255,0.85);word-break:break-all">'+esc(ref)+'</div>';
            content.appendChild(linkBox);
          }else{
            content.appendChild(el("div","sub",'Если ссылка не показалась — задай переменную окружения <b>BOT_USERNAME</b>.'));
          }

          const copy = el("div","btn");
          copy.style.marginTop="10px";
          copy.style.opacity = ref ? 1 : 0.5;
          copy.style.cursor = ref ? "pointer" : "not-allowed";
          copy.innerHTML = '<div><div class="btnTitle">📎 Скопировать ссылку</div><div class="btnSub">'+esc(state.msg || "Скопировать в буфер")+'</div></div><div style="opacity:0.85">›</div>';
          copy.addEventListener("click", async ()=>{
            if(!ref) return;
            try{
              await navigator.clipboard.writeText(ref);
              state.msg = "✅ Скопировано";
              haptic("light");
            }catch(e){
              state.msg = "ℹ️ Не удалось скопировать";
            }
            renderПрофильSheet();
          });
          content.appendChild(copy);

          const statsWrap = el("div");
          statsWrap.style.marginTop="12px";
          content.appendChild(statsWrap);

          const listWrap = el("div");
          listWrap.style.marginTop="10px";
          content.appendChild(listWrap);

          const renderStatusPill = (st)=>{
            if(st==="active") return '<span class="pill" style="background:rgba(76,175,80,0.22);border-color:rgba(76,175,80,0.35)">✅ Активный</span>';
            if(st==="inactive") return '<span class="pill" style="background:rgba(255,152,0,0.18);border-color:rgba(255,152,0,0.30)">⚠️ Неактивный</span>';
            return '<span class="pill" style="background:rgba(255,255,255,0.08);border-color:rgba(255,255,255,0.16)">⏳ До активации</span>';
          };

          const timeAgo = (iso)=>{
            if(!iso) return "—";
            const t = new Date(iso).getTime();
            if(!t) return "—";
            const diff = Math.max(0, Date.now()-t);
            const d = Math.floor(diff/86400000);
            const h = Math.floor((diff%86400000)/3600000);
            if(d>0) return d+"д назад";
            if(h>0) return h+"ч назад";
            return "только что";
          };

          async function loadReferrals(){
            listWrap.innerHTML = '<div class="sub">Загружаю…</div>';
            statsWrap.innerHTML = "";
            try{
              const r = await fetch("/api/referrals?telegram_id="+encodeURIComponent(tgUserId));
              const data = await r.json();

              // Summary
              const s = el("div","card2");
              s.innerHTML =
                '<div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">'+
                  '<div>'+
                    '<div style="font-weight:900">Итог</div>'+
                    '<div class="sub" style="margin-top:4px">Всего: <b>'+data.total_referrals+'</b> · ✅ '+data.active+' · ⏳ '+data.pending+' · ⚠️ '+data.inactive+'</div>'+
                  '</div>'+
                  '<div style="text-align:right">'+
                    '<div style="font-weight:950">+'+data.earned_total+' 💎</div>'+
                    '<div class="sub">сегодня: +'+data.earned_today+'</div>'+
                  '</div>'+
                '</div>';
              statsWrap.appendChild(s);

              // List
              listWrap.innerHTML = "";
              if(!data.items || data.items.length===0){
                listWrap.innerHTML = '<div class="sub">Пока нет рефералов. Поделись ссылкой выше.</div>';
                return;
              }

              data.items.forEach((it)=>{
                const card = el("div","card2");
                const name = (it.username ? "@"+it.username : (it.name || ("ID "+it.telegram_id)));
                const last = timeAgo(it.last_seen_at);

                let progress = "";
                if(it.status==="pending"){
                  const lleft = it.progress?.login_left ?? 0;
                  const needSpin = it.progress?.need_spin;
                  const parts = [];
                  if(lleft>0) parts.push("ещё "+lleft+" дн. входа");
                  if(needSpin) parts.push("нужен 1 спин");
                  progress = parts.length ? ("<div class='sub' style='margin-top:4px'>До активации: <b>"+parts.join(" · ")+"</b></div>") : "";
                }else if(it.status==="inactive"){
                  progress = "<div class='sub' style='margin-top:4px'>Не заходил 7 дней — доход заморожен</div>";
                }else{
                  progress = "<div class='sub' style='margin-top:4px'>10% с бонусных выигрышей активен</div>";
                }

                card.innerHTML =
                  '<div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">'+
                    '<div style="min-width:0">'+
                      '<div style="font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+esc(name)+'</div>'+
                      '<div class="sub">Последний визит: '+esc(last)+'</div>'+
                    '</div>'+
                    '<div style="text-align:right;white-space:nowrap">'+
                      renderStatusPill(it.status)+
                    '</div>'+
                  '</div>'+
                  progress+
                  '<div style="display:flex;justify-content:space-between;gap:10px;margin-top:10px">'+
                    '<div class="pill">сегодня: <b>+'+it.earned_today+'</b></div>'+
                    '<div class="pill">всего: <b>+'+it.earned_total+'</b></div>'+
                  '</div>';
                listWrap.appendChild(card);
              });

            }catch(e){
              listWrap.innerHTML = '<div class="sub">Ошибка загрузки. Попробуй позже.</div>';
            }
          }

          // hook reload
          setTimeout(()=>{
            const rr = document.getElementById("refReload");
            if(rr) rr.onclick = ()=>loadReferrals();
          },0);

          loadReferrals();

          if(state.msg && state.msg.startsWith("ℹ️")){
            const m = el("div","card2", esc(state.msg));
            m.style.marginTop="12px";
            content.appendChild(m);
          }
        }

if(state.profileView==="raffle"){
          const box = el("div");
          box.style.marginTop="12px";
          box.innerHTML =
            '<div style="font-size:14px;font-weight:900">🎁 Розыгрыши</div>'+
            '<div class="sub" style="margin-top:6px">Билет = 500 баллов.</div>'+
            '<div class="sub" style="margin-top:8px">Ваши билеты: <b style="color:rgba(255,255,255,0.92)">'+esc((state.raffle && state.raffle.ticket_count) ? state.raffle.ticket_count : 0)+'</b></div>';
          content.appendChild(box);

          const can = (state.user.points||0) >= 500 && !state.busy;
          const b = el("div","btn");
          b.style.marginTop="10px";
          b.style.opacity = can ? 1 : 0.5;
          b.style.cursor = can ? "pointer" : "not-allowed";
          b.innerHTML = '<div><div class="btnTitle">🎟 Buy ticket</div><div class="btnSub">'+(state.busy?"Подожди…":"Потратить 500 баллов")+'</div></div><div style="opacity:0.85">›</div>';
          b.addEventListener("click", ()=>{ if(can){ buyTicket(); } });
          content.appendChild(b);

          if(state.msg){
            const m = el("div","card2", esc(state.msg));
            m.style.marginTop="12px";
            content.appendChild(m);
          }
        }

if(state.profileView==="roulette"){
          const wrap = el("div","rouletteWrap");

          const title = el("div");
          title.style.marginTop="12px";
          const titleRow = el("div","row");
          titleRow.style.alignItems = "flex-start";
          const titleLeft = el("div");
          titleLeft.innerHTML =
            '<div style="font-size:14px;font-weight:900">Рулетка</div>'+
            '<div class="sub" style="margin-top:6px">Крутить = '+String((state.rouletteStatus?.spin_cost)||0)+' баллов.</div>';
          const oddsBtn = el("div","pill","📊 Шансы");
          oddsBtn.style.cursor = "pointer";
          oddsBtn.style.borderColor = "rgba(255,255,255,0.14)";
          oddsBtn.style.background = "rgba(255,255,255,0.06)";
          oddsBtn.addEventListener("click", ()=>{ haptic(); openRouletteOdds(); });
          titleRow.appendChild(titleLeft);
          titleRow.appendChild(oddsBtn);
          title.appendChild(titleRow);
          wrap.appendChild(title);

          const stage = el("div","wheelStage");

          const wheelBox = el("div","wheelBox");
          const canvas = document.createElement("canvas");
          canvas.id = "wheelCanvas";
          canvas.className = "wheelCanvas";
          wheelBox.appendChild(canvas);

          const pointer = el("div","wheelPointer");
          wheelBox.appendChild(pointer);


          const center = el("div","wheelCenter","NS");
          wheelBox.appendChild(center);

          stage.appendChild(wheelBox);

          const micro = el("div","microHud",
            "Баланс: "+esc(String(state.user?.points||0))+" 💎   •   Стоимость: "+esc(String((state.rouletteStatus?.spin_cost)||0))+" 💎"
          );
          stage.appendChild(micro);

          // ticker
          const ticker = el("div","ticker");
          ticker.style.cursor = "pointer";
          ticker.addEventListener("click", ()=>{ haptic(); state.profileView="history"; renderПрофильSheet(); });
          const recent = (state.rouletteRecent||[]).map(x=>x.prize_label).filter(Boolean);
          const tickerText = "Последние выигрыши: "+(recent.length?recent.join(" • "):"—");
          const tt = el("div","tickerText", esc(tickerText));
          tt.id = "rouletteTickerText";
          ticker.appendChild(tt);
          stage.appendChild(ticker);

          // chips
          const chips = el("div","chipsRow");
          (state.rouletteHistory||[]).slice(0,10).forEach(it=>{
            const c = el("div","chip", esc((it.prize_label||"").replace("баллов","").trim() || "приз"));
            c.style.cursor="pointer";
            c.addEventListener("click", ()=>{ haptic(); state.profileView="history"; renderПрофильSheet(); });
            chips.appendChild(c);
          });
          stage.appendChild(chips);

          wrap.appendChild(stage);

          // CTA
          const st = state.rouletteStatus || {can_spin:true, seconds_left:0, enough_points:true};
          const can = !!st.can_spin && !state.busy;
          const b = el("div","btn");
          b.style.marginTop="14px";
          b.style.opacity = (can || state.busy) ? 1 : 0.55;
          b.style.cursor = (can && !state.busy) ? "pointer" : "not-allowed";
          b.innerHTML =
            '<div><div class="btnTitle">'+(state.busy?"Крутим…":"Крутить")+
            '</div><div class="btnSub">'+(state.busy?"":`−${state.rouletteStatus.spin_cost||0} 💎`)+'</div></div><div style="opacity:0.85">›</div>';
          b.addEventListener("click", ()=>{ if(can && !state.busy){ spinRouletteLux(); } });
          wrap.appendChild(b);

                    // status + cooldown UI
          updateRouletteStatus();
          startRouletteCooldownTicker();

content.appendChild(wrap);

          // draw wheel now
          setTimeout(()=>{
            drawWheel(document.getElementById("wheelCanvas"), state.rouletteWheel.angle||0);
            startTicker();
          }, 0);

          // Result overlay + sheet
          // Remove previous result overlays (avoid stacking / z-index bugs)
          try{
            const oldO = document.getElementById("nsResultOverlay"); if(oldO) oldO.remove();
            const oldS = document.getElementById("nsResultSheet"); if(oldS) oldS.remove();
          }catch(e){}

          const prize = state.rouletteWheel.prize;
          const isDiorPrize = !!(prize && prize.prize_key==="dior_palette");
          // Show result ONLY when the spin animation is fully finished.
          const showResult = !!(state.rouletteWheel.overlay && prize);
          // Overlay should appear for ANY prize (popup), Dior gets extra sparkle class.
          const overlay = el("div","resultSheetOverlay"+(showResult?" on":"")+(isDiorPrize?" dior":""));
          overlay.id="nsResultOverlay";
          overlay.style.zIndex = "200000";
          overlay.addEventListener("click", ()=>{ if(!state.busy){ closeResultSheet(); } });
          document.body.appendChild(overlay);

          const sheet = el("div","resultSheet"+(showResult?" on":""));
          sheet.id="nsResultSheet";
          sheet.style.zIndex = "200001";
          const card = el("div","resultCard");
          if(showResult){
            const isDior = isDiorPrize;
            card.appendChild(el("div","resultTitle", isDior ? "Главный приз" : "Выпало:"));
            card.appendChild(el("div","resultValue", esc(prize.prize_label)));
            card.appendChild(el("div","resultSub", isDior ? "Оформи получение или конвертируй в баллы." : "Готово."));
            const btns = el("div","resultBtns");

            if(isDior){
              const b1 = document.createElement("button");
              b1.className="btnPrimary";
              b1.textContent = state.busy ? "Подожди…" : "Забрать";
              b1.disabled = !!state.busy;
              b1.addEventListener("click", ()=>{ if(!state.busy){ claimFromResult(); } });

              const b2 = document.createElement("button");
              b2.className="btnGhost";
              b2.textContent = state.busy ? "Подожди…" : "Конвертировать";
              b2.disabled = !!state.busy;
              b2.addEventListener("click", ()=>{ if(!state.busy){ convertFromResult(); } });

              btns.appendChild(b1);
              btns.appendChild(b2);
            }else{
              const ok = document.createElement("button");
              ok.className="btnPrimary";
              ok.textContent = "Принять";
              ok.addEventListener("click", ()=>{ closeResultSheet(); });
              btns.appendChild(ok);
            }

            card.appendChild(btns);
          }
          sheet.appendChild(card);
          document.body.appendChild(sheet);

          // cleanup on next render
          state._cleanup = state._cleanup || [];
          state._cleanup.push(()=>{ try{ overlay.remove(); sheet.remove(); }catch(e){} });
        }

        if(state.profileView==="claim"){
          const box = el("div");
          box.style.marginTop="12px";
          box.innerHTML =
            '<div style="font-size:14px;font-weight:900">Получение приза</div>'+
            '<div class="sub" style="margin-top:6px">Заполни анкету — мы подтвердим заявку.</div>';
          content.appendChild(box);

          const card = el("div","card2");
          card.style.marginTop="12px";

          const prize = state.claim.prize_label || "Приз";
          const code = state.claim.claim_code ? ("Код: "+state.claim.claim_code) : "";
          card.appendChild(el("div","miniMeta", esc(prize)));
          if(code) card.appendChild(el("div","sub", esc(code)));

          const st = (state.claim.status||"draft");
          if(st && st !== "draft" && st !== "awaiting_contact"){
            const s = el("div","sub", "Статус: "+esc(st));
            s.style.marginTop="10px";
            card.appendChild(s);

            const back = el("div","btn");
            back.style.marginTop="12px";
            back.innerHTML = '<div><div class="btnTitle">Назад</div><div class="btnSub">Вернуться</div></div><div style="opacity:0.85">›</div>';
            back.addEventListener("click", ()=>{ state.profileView="roulette"; renderПрофильSheet(); });
            card.appendChild(back);
            content.appendChild(card);
          }else{
            const form = el("div");
            form.style.marginTop="10px";

            // Stepper (quiet-lux)
            const step = Math.max(1, Math.min(3, Number(state.claim.step||1)||1));
            const stepHdr = el("div",null,
              '<div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px">'+
                '<div style="font-size:12px;color:rgba(255,255,255,0.68)">Шаг '+step+' из 3</div>'+
                '<div style="display:flex;gap:6px">'+
                  '<span style="width:8px;height:8px;border-radius:99px;background:'+(step>=1?'rgba(235,245,255,0.86)':'rgba(255,255,255,0.20)')+'"></span>'+
                  '<span style="width:8px;height:8px;border-radius:99px;background:'+(step>=2?'rgba(235,245,255,0.86)':'rgba(255,255,255,0.20)')+'"></span>'+
                  '<span style="width:8px;height:8px;border-radius:99px;background:'+(step>=3?'rgba(235,245,255,0.86)':'rgba(255,255,255,0.20)')+'"></span>'+
                '</div>'+
              '</div>'
            );
            form.appendChild(stepHdr);

            function inputRow(ph, id, val, onChange){
              const inp = document.createElement("input");
              inp.id=id; inp.placeholder=ph;
              inp.value = val || "";
              inp.style.width="100%";
              inp.style.padding="12px 12px";
              inp.style.marginTop="10px";
              inp.style.borderRadius="14px";
              inp.style.border="1px solid rgba(255,255,255,0.12)";
              inp.style.background="rgba(255,255,255,0.04)";
              inp.style.color="rgba(255,255,255,0.92)";
              inp.style.outline="none";
              inp.autocomplete = "off";
              inp.addEventListener("input", ()=>{ try{ onChange && onChange(inp.value); }catch(e){} });
              return inp;
            }

            const f = state.claim.form || (state.claim.form = {full_name:"", phone:"", country:"", city:"", address_line:"", postal_code:"", comment:""});

            if(step===1){
              form.appendChild(inputRow("Имя и фамилия", "c_full_name", f.full_name, (v)=>{ f.full_name=v; }));
              form.appendChild(inputRow("Телефон", "c_phone", f.phone, (v)=>{ f.phone=v; }));
            }else if(step===2){
              form.appendChild(inputRow("Страна", "c_country", f.country, (v)=>{ f.country=v; }));
              form.appendChild(inputRow("Город", "c_city", f.city, (v)=>{ f.city=v; }));
            }else{
              form.appendChild(inputRow("Адрес", "c_address", f.address_line, (v)=>{ f.address_line=v; }));
              form.appendChild(inputRow("Индекс", "c_postal", f.postal_code, (v)=>{ f.postal_code=v; }));
              form.appendChild(inputRow("Комментарий (необязательно)", "c_comment", f.comment, (v)=>{ f.comment=v; }));
              const agree = document.createElement("label");
              agree.style.display="flex";
              agree.style.gap="10px";
              agree.style.alignItems="flex-start";
              agree.style.marginTop="12px";
              agree.style.padding="12px 12px";
              agree.style.borderRadius="14px";
              agree.style.border="1px solid rgba(255,255,255,0.10)";
              agree.style.background="rgba(255,255,255,0.03)";
              agree.style.cursor="pointer";
              agree.innerHTML = '<input id="c_agree" type="checkbox" style="margin-top:2px"> <div style="font-size:12px;color:rgba(255,255,255,0.78)"><b style="color:rgba(255,255,255,0.90)">Подтверждаю</b>, что данные указаны верно.</div>';
              form.appendChild(agree);
            }

            // Nav buttons
            const nav = el("div","row");
            nav.style.marginTop="12px";
            nav.style.gap="10px";

            const backStep = el("div","btn");
            backStep.style.flex="1";
            backStep.style.justifyContent="center";
            backStep.style.fontWeight="900";
            backStep.style.opacity = (step===1 || state.busy) ? 0.6 : 1;
            backStep.style.cursor = (step===1 || state.busy) ? "not-allowed" : "pointer";
            backStep.innerHTML = "Назад";
            backStep.addEventListener("click", ()=>{
              if(state.busy || step===1) return;
              haptic();
              state.claim.step = step-1;
              renderПрофильSheet();
            });

            const nextStep = el("div","btn");
            nextStep.style.flex="1";
            nextStep.style.justifyContent="center";
            nextStep.style.fontWeight="950";
            nextStep.style.border="1px solid rgba(235,245,255,0.22)";
            nextStep.style.background="rgba(235,245,255,0.10)";
            nextStep.style.opacity = state.busy ? 0.6 : 1;
            nextStep.style.cursor = state.busy ? "not-allowed" : "pointer";
            nextStep.innerHTML = (step===3 ? "Отправить заявку" : "Продолжить");

            nextStep.addEventListener("click", async ()=>{
              if(state.busy) return;

              // simple validation per step
              if(step===1){
                if((f.full_name||"").trim().length < 2){ state.msg="❌ Укажи имя и фамилию."; renderПрофильSheet(); return; }
                if((f.phone||"").trim().length < 3){ state.msg="❌ Укажи телефон."; renderПрофильSheet(); return; }
                state.claim.step = 2; haptic("light"); renderПрофильSheet(); return;
              }
              if(step===2){
                if((f.country||"").trim().length < 2){ state.msg="❌ Укажи страну."; renderПрофильSheet(); return; }
                if((f.city||"").trim().length < 1){ state.msg="❌ Укажи город."; renderПрофильSheet(); return; }
                state.claim.step = 3; haptic("light"); renderПрофильSheet(); return;
              }

              // step 3 submit
              const agreeEl = document.getElementById("c_agree");
              if(!agreeEl || !agreeEl.checked){
                state.msg="❌ Подтверди корректность данных.";
                renderПрофильSheet();
                return;
              }
              if((f.address_line||"").trim().length < 5){ state.msg="❌ Укажи адрес."; renderПрофильSheet(); return; }
              if((f.postal_code||"").trim().length < 2){ state.msg="❌ Укажи индекс."; renderПрофильSheet(); return; }

              state.busy=true; state.msg=""; renderПрофильSheet();
              try{
                await apiPost("/api/roulette/claim/submit", {
                  telegram_id: tgUserId,
                  claim_id: state.claim.claim_id,
                  full_name: (f.full_name||"").trim(),
                  phone: (f.phone||"").trim(),
                  country: (f.country||"").trim(),
                  city: (f.city||"").trim(),
                  address_line: (f.address_line||"").trim(),
                  postal_code: (f.postal_code||"").trim(),
                  comment: (f.comment||"").trim()
                });
                state.claim.status = "submitted";
                state.msg = "✅ Заявка отправлена.";
                haptic("light");
                // refresh inventory so buttons -> status
                try{ state.inventory = await apiGet("/api/inventory?telegram_id="+encodeURIComponent(tgUserId)); }catch(e){}
              }catch(e){
                state.msg = "❌ "+(e.message||"Ошибка");
              }finally{
                state.busy=false;
                renderПрофильSheet();
              }
            });

            nav.appendChild(backStep);
            nav.appendChild(nextStep);

            form.appendChild(nav);
            card.appendChild(form);
            card.appendChild(back);
            content.appendChild(card);
          }

          if(state.msg){
            const m = el("div","card2", esc(state.msg));
            m.style.marginTop="12px";
            content.appendChild(m);
          }
        }

if(state.profileView==="history"){
          const box = el("div");
          box.style.marginTop="12px";
          box.innerHTML = '<div style="font-size:14px;font-weight:900">🧾 History</div>';
          content.appendChild(box);

          const arr = Array.isArray(state.rouletteHistory) ? state.rouletteHistory : [];
          if(arr.length===0){
            content.appendChild(el("div","sub","Пока пусто."));
          }else{
            const list = el("div");
            list.style.marginTop="10px";
            list.style.display="grid";
            list.style.gap="10px";
            for(const x of arr){
              const it = el("div","card2");
              it.innerHTML = '<div class="sub">'+esc(x.created_at)+'</div><div style="margin-top:6px;font-size:14px;font-weight:850">'+esc(x.prize_label)+'</div>';
              list.appendChild(it);
            }
            content.appendChild(list);
          }
        }
      }
    }

    function renderBottomNav(root){
      const nav = el("div","bottomNav");
      const inner = el("div","bottomNavInner");
      const items = [
        {id:"journal", icon:"📰", label:"Журнал"},
        {id:"discover", icon:"🧭", label:"Поиск"},
        {id:"rewards", icon:"🎁", label:"Бонусы"},
        {id:"profile", icon:"👤", label:"Профиль"}
      ];
      for(const it of items){
        const n = el("div","navItem"+(state.tab===it.id?" navItemActive":""));
        n.addEventListener("click", ()=>{
          haptic();
          if(it.id==="profile") openПрофиль("menu");
          else { state.tab = it.id; render(); }
        });
        n.appendChild(el("div","navIcon", it.icon));
        n.appendChild(el("div","navLabel", it.label));
        inner.appendChild(n);
      }
      nav.appendChild(inner);
      root.appendChild(nav);
    }

    function ensureSheets(root){
      // Posts
      const pO = el("div","sheetOverlay"); pO.id="postsOverlay";
      pO.addEventListener("click", (e)=>{ if(e.target===pO){ haptic(); closePosts(); }});
      const pS = el("div","sheet");
      pS.addEventListener("click",(e)=>e.stopPropagation());
      pS.appendChild(el("div","sheetHandle"));
      const pC = el("div"); pC.id="postsContent";
      pS.appendChild(pC); pO.appendChild(pS); root.appendChild(pO);

      // Inventory
      const iO = el("div","sheetOverlay"); iO.id="invOverlay";
      iO.addEventListener("click", (e)=>{ if(e.target===iO){ haptic(); closeInventory(); }});
      const iS = el("div","sheet");
      iS.addEventListener("click",(e)=>e.stopPropagation());
      iS.appendChild(el("div","sheetHandle"));
      const iC = el("div"); iC.id="invContent";
      iS.appendChild(iC); iO.appendChild(iS); root.appendChild(iO);

      // Профиль
      const prO = el("div","sheetOverlay"); prO.id="profileOverlay";
      prO.addEventListener("click", (e)=>{ if(e.target===prO){ haptic(); closeПрофиль(); }});
      const prS = el("div","sheet");
      prS.addEventListener("click",(e)=>e.stopPropagation());
      prS.appendChild(el("div","sheetHandle"));
      const prC = el("div"); prC.id="profileContent";
      prS.appendChild(prC); prO.appendChild(prS); root.appendChild(prO);
// Daily
const dO = el("div","sheetOverlay"); dO.id="dailyOverlay";
dO.addEventListener("click", (e)=>{ if(e.target===dO){ haptic(); closeDaily(); }});
const dS = el("div","sheet");
dS.addEventListener("click",(e)=>e.stopPropagation());
dS.appendChild(el("div","sheetHandle"));
const dC = el("div"); dC.id="dailyContent";
dS.appendChild(dC); dO.appendChild(dS); root.appendChild(dO);

      // Roulette odds
      const oO = el("div","sheetOverlay"); oO.id="oddsOverlay";
      oO.addEventListener("click", (e)=>{ if(e.target===oO){ haptic(); closeRouletteOdds(); }});
      const oS = el("div","sheet");
      oS.addEventListener("click",(e)=>e.stopPropagation());
      oS.appendChild(el("div","sheetHandle"));
      const oC = el("div"); oC.id="oddsContent";
      oS.appendChild(oC); oO.appendChild(oS); root.appendChild(oO);

    }

    function render(){
      const root = document.getElementById("root");
      root.innerHTML = "";

      const app = el("div","safePadBottom");
      const cont = el("div","container");
      if(state.tab==="journal") renderЖурнал(cont);
      else if(state.tab==="discover") renderПоиск(cont);
      else if(state.tab==="categories") renderКатегории(cont);
      else if(state.tab==="brands") renderБренды(cont);
      else if(state.tab==="products") renderПродукты(cont);
      else if(state.tab==="rewards") renderБонусы(cont);
      else renderЖурнал(cont);

      app.appendChild(cont);

      ensureSheets(app);
      renderBottomNav(app);

      root.appendChild(app);

      renderPostsSheet();
      renderInventorySheet();
      renderRouletteOddsSheet();
      renderПрофильSheet();
      renderDailySheet();
    }

    showSplash();
    setTimeout(()=>{ hideSplash(); }, 6000);

    async function boot(){
      if(tg){
        try{ tg.expand(); }catch(e){}
        applyTelegramTheme();
        try{ tg.onEvent && tg.onEvent("themeChanged", applyTelegramTheme); }catch(e){}
      }
      await Promise.all([refreshUser(), loadBotUsername()]);
      try{ await dailyEvent('open_miniapp'); }catch(e){}
      await loadЖурналBlocks();
      render();
      hideSplash();
    }

    document.addEventListener("DOMContentLoaded", boot);
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
        await _close_http_client()

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
    # mark activity (mini app open / profile refresh)
    try:
        await touch_user_seen(int(telegram_id))
    except Exception:
        pass
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
        media_url = f"/api/post_media/{int(p.message_id)}?fid={p.media_file_id}&v={ASSET_VERSION}" if (media_type == "photo" and p.media_file_id) else None
        if not media_url:
            main_tag = (p.tags or [None])[0] or "Пост"
            media_url = f"/api/tag_card/{main_tag}?v={ASSET_VERSION}"

        out.append({
            "message_id": int(p.message_id),
            "url": p.permalink or make_permalink(int(p.message_id)),
            "tags": p.tags or [],
            "preview": preview_text(p.text),
            "media_type": media_type or None,
            "media_url": media_url,
        })
    return out

@app.get("/api/search")
async def api_search(q: str, limit: int = 50, offset: int = 0):
    """
    Глобальный поиск по тексту постов (по всему каналу).
    Возвращает те же поля, что и /api/posts.
    """
    q = (q or "").strip()
    if not q:
        return []

    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    # Поиск по тексту. SQLite: LIKE, Postgres: ILIKE (через func.lower для универсальности).
    q_like = f"%{q.lower()}%"

    async with async_session() as session:
        stmt = (
            select(Post)
            .where(Post.is_deleted.is_(False))
            .where(func.lower(func.coalesce(Post.text, "")).like(q_like))
            .order_by(Post.date.desc())
            .limit(limit)
            .offset(offset)
        )
        res = await session.execute(stmt)
        rows = res.scalars().all()

    out = []
    for p in rows:
        media_type = (p.media_type or "").strip().lower()
        media_url = f"/api/post_media/{int(p.message_id)}?v={ASSET_VERSION}" if (media_type == "photo" and p.media_file_id) else None
        if not media_url:
            main_tag = (p.tags or [None])[0] or "Пост"
            media_url = f"/api/tag_card/{main_tag}?v={ASSET_VERSION}"

        out.append({
            "message_id": int(p.message_id),
            "url": p.permalink or make_permalink(int(p.message_id)),
            "tags": p.tags or [],
            "preview": search_snippet(p.text, q),
            "media_type": media_type or None,
            "media_url": media_url,
        })
    return out




@app.get("/api/post_media/{message_id}")
async def api_post_media(message_id: int, fid: str | None = None, v: str | None = None):
    """
    Возвращает картинку поста (если сохранён media_file_id).

    Важно для Telegram WebView:
    - Если отдавать no-store/no-cache, то при возврате назад картинки грузятся заново.
    - Поэтому для реальных фото включаем нормальный публичный кэш + versioning по fid.
    """
    message_id = int(message_id)

    async with async_session() as session:
        res = await session.execute(select(Post).where(Post.message_id == message_id))
        post = res.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    media_type = (post.media_type or "").strip().lower()
    if media_type != "photo" or not post.media_file_id:
        raise HTTPException(status_code=404, detail="No photo for this post")

    # Если клиент пришёл со старым fid (картинка у поста сменилась),
    # перекидываем на актуальный URL, чтобы кэш был консистентным.
    if fid and fid != post.media_file_id:
        return RedirectResponse(
            url=f"/api/post_media/{message_id}?fid={post.media_file_id}&v={ASSET_VERSION}",
            status_code=302,
        )

    local_path = await get_cached_media_file(post.media_file_id)

    etag = hashlib.sha1(post.media_file_id.encode("utf-8")).hexdigest()
    headers = {
        # Делаем WebView счастливым: не перекачивает при back/forward.
        # Обновление контролируем через fid в URL.
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": f'W/"{etag}"',
    }

    # Telegram фото может быть jpeg/webp; большинство клиентов понимают image/jpeg.
    return FileResponse(local_path, media_type="image/jpeg", headers=headers)

@app.get("/api/tag_card/{tag}")
async def api_tag_card(tag: str, v: str | None = None):
    """
    PNG-заглушка для постов без картинки: светлая premium-карта под NS.

    Почему так:
    - Telegram WebView агрессивно кэширует картинки.
    - Поэтому URL содержит ?v=... (ASSET_VERSION), а ответ отдаём с no-store.
    - Внутри сервера держим небольшой in-memory cache по (tag+version),
      чтобы не рендерить PNG на каждый запрос.
    """
    t = (tag or "").strip() or "Пост"
    key = _norm_tag(t)

    subtitle_map = {
        "люкс": "Luxury Selection",
        "новинка": "New Drop",
        "тренд": "Trending Now",
        "факты": "Beauty Facts",
        "состав": "Ingredients",
        "история": "Brand Story",
        "оценка": "Review",
        "челленджи": "Challenge",
        "challenge": "Challenge",
        "sephorapromo": "Sephora Promo",
    }
    subtitle = subtitle_map.get(key, "Natural Sense")

    # Accent (soft champagne / pearl)
    accent_map = {
        "люкс": "#9A7A3A",
        "новинка": "#8C6A4F",
        "тренд": "#4E7A74",
        "факты": "#6E5A8A",
        "состав": "#7B6A48",
        "история": "#52667A",
        "оценка": "#7A4E5B",
        "челленджи": "#4F7A5B",
        "challenge": "#4F7A5B",
        "sephorapromo": "#7A4E6A",
    }
    accent_hex = accent_map.get(key, "#8C7A5A")

    # Version for cache-busting
    version = (v or ASSET_VERSION or "v").strip()

    # --- server in-memory cache (version-aware) ---
    cache_key = f"{version}|{t}|{subtitle}|{accent_hex}"
    if not hasattr(api_tag_card, "_cache"):
        api_tag_card._cache = {}
    cache: dict[str, bytes] = api_tag_card._cache  # type: ignore[attr-defined]
    if cache_key in cache:
        return Response(
            content=cache[cache_key],
            media_type="image/png",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        # Fallback: SVG (на случай если Pillow отсутствует)
        svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#F8F4EE"/>
      <stop offset="55%" stop-color="#EFE6DA"/>
      <stop offset="100%" stop-color="#E5D2BE"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="1280" height="720" fill="url(#g)"/>
  <rect x="72" y="72" width="1136" height="576" rx="44" fill="rgba(255,255,255,0.80)" stroke="rgba(0,0,0,0.10)" stroke-width="2"/>
  <text x="120" y="240" fill="{accent_hex}" font-family="Arial" font-size="36" font-weight="700">#{html.escape(t)}</text>
  <text x="120" y="340" fill="#101418" font-family="Arial" font-size="64" font-weight="800">NS•Natural Sense</text>
  <text x="120" y="420" fill="rgba(16,20,24,0.72)" font-family="Arial" font-size="30" font-weight="600">{html.escape(subtitle)}</text>
</svg>"""
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    W, H = 1280, 720

    def hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def lerp(a: int, b: int, t_: float) -> int:
        return int(a + (b - a) * t_)

    # Premium light background gradient (ivory -> pearl -> champagne)
    c1 = hex_to_rgb("#F8F4EE")  # ivory
    c2 = hex_to_rgb("#EFE6DA")  # pearl
    c3 = hex_to_rgb("#E5D2BE")  # champagne nude

    img = Image.new("RGB", (W, H), c1)
    d = ImageDraw.Draw(img)

    # vertical blend c1 -> c2
    for y in range(H):
        t_ = y / (H - 1)
        col = (lerp(c1[0], c2[0], t_), lerp(c1[1], c2[1], t_), lerp(c1[2], c2[2], t_))
        d.line([(0, y), (W, y)], fill=col)

    # soft diagonal overlay c2 -> c3 (subtle)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(H):
        t_ = y / (H - 1)
        # more champagne towards bottom-right
        alpha = int(70 * t_)  # 0..70
        col = (*c3, alpha)
        od.line([(0, y), (W, y)], fill=col)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGBA")

    # very light noise to avoid "flat" look
    try:
        import random
        px = img.load()
        for _ in range(9000):  # sparse
            x = random.randrange(0, W)
            y = random.randrange(0, H)
            r, g, b, a = px[x, y]
            n = random.randint(-8, 8)
            px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)), a)
    except Exception:
        pass

    # Card area (glass / premium)
    pad = 72
    card = [pad, pad, W - pad, H - pad]
    # soft shadow
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [card[0] + 8, card[1] + 12, card[2] + 8, card[3] + 12],
        radius=44,
        fill=(0, 0, 0, 38),
    )
    img = Image.alpha_composite(img, shadow)

    d = ImageDraw.Draw(img)
    d.rounded_rectangle(card, radius=44, fill=(255, 255, 255, 210), outline=(0, 0, 0, 28), width=2)

    # Accent micro-dots
    acc = hex_to_rgb(accent_hex)
    d.ellipse((120, 120, 140, 140), fill=(*acc, 255))
    d.ellipse((156, 126, 168, 138), fill=(*acc, 255))

    # Fonts (best-effort)
    def load_font(size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in candidates:
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    f_tag = load_font(34, bold=True)
    f_title = load_font(66, bold=True)
    f_sub = load_font(32, bold=False)

    # Text (dark premium)
    d.text((120, 200), f"#{t}", font=f_tag, fill=(*acc, 255))
    d.text((120, 300), "NS•Natural Sense", font=f_title, fill=(16, 20, 24, 255))
    d.text((120, 390), subtitle, font=f_sub, fill=(16, 20, 24, 185))

    # Export
    import io as _io
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    png = buf.getvalue()
    cache[cache_key] = png

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )







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



# -----------------------------------------------------------------------------
# REFERRALS API (Mini App)
# -----------------------------------------------------------------------------
def _naive_utc_now() -> datetime:
    return datetime.utcnow()

def _naive_utc_midnight(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day)

@app.get("/api/referrals")
async def api_referrals(telegram_id: int):
    """
    Returns inviter referrals list with clear status/progress and earnings.
    Used by Mini App profile -> referrals view.
    """
    try:
        async with SessionLocal() as session:
            inviter = await get_or_create_user(session, telegram_id=telegram_id)
            await session.commit()

            # Referred users
            q = (
                select(User)
                .where(User.referred_by == inviter.telegram_id)
                .order_by(User.joined_at.desc())
            )
            referred_users = (await session.execute(q)).scalars().all()

            now = _naive_utc_now()
            inactive_cutoff = now - timedelta(days=7)
            today_mid = _naive_utc_midnight(now)

            # Earnings totals
            total_sum = await session.execute(
                select(func.coalesce(func.sum(ReferralEarning.amount), 0)).where(
                    ReferralEarning.inviter_id == inviter.telegram_id
                )
            )
            total_earned = int(total_sum.scalar() or 0)

            today_sum = await session.execute(
                select(func.coalesce(func.sum(ReferralEarning.amount), 0)).where(
                    ReferralEarning.inviter_id == inviter.telegram_id,
                    ReferralEarning.created_at >= today_mid,
                )
            )
            earned_today = int(today_sum.scalar() or 0)

            # Per referred aggregates (total + today)
            per_total_rows = await session.execute(
                select(ReferralEarning.referred_id, func.sum(ReferralEarning.amount))
                .where(ReferralEarning.inviter_id == inviter.telegram_id)
                .group_by(ReferralEarning.referred_id)
            )
            per_total = {int(rid): int(amount or 0) for rid, amount in per_total_rows.all()}

            per_today_rows = await session.execute(
                select(ReferralEarning.referred_id, func.sum(ReferralEarning.amount))
                .where(
                    ReferralEarning.inviter_id == inviter.telegram_id,
                    ReferralEarning.created_at >= today_mid,
                )
                .group_by(ReferralEarning.referred_id)
            )
            per_today = {int(rid): int(amount or 0) for rid, amount in per_today_rows.all()}

            items = []
            active_count = 0
            pending_count = 0
            inactive_count = 0

            for u in referred_users:
                # Determine status
                if u.ref_active_at is not None:
                    if u.last_seen_at is not None and u.last_seen_at < inactive_cutoff:
                        status = "inactive"
                        inactive_count += 1
                    else:
                        status = "active"
                        active_count += 1
                else:
                    status = "pending"
                    pending_count += 1

                # Progress to active
                login_days = int(u.daily_login_total or 0)
                login_done = min(login_days, 3)
                login_left = max(0, 3 - login_done)
                spin_done = int(u.roulette_spins_total or 0) >= 1
                need_spin = not spin_done

                last_seen_iso = u.last_seen_at.isoformat() if u.last_seen_at else None

                name = (u.first_name or (u.username or "") or f"ID {u.telegram_id}")
                username = (u.username or "").lstrip("@")

                items.append(
                    {
                        "telegram_id": int(u.telegram_id),
                        "name": name,
                        "username": username,
                        "status": status,  # pending / active / inactive
                        "last_seen_at": last_seen_iso,
                        "progress": {
                            "login_done": login_done,
                            "login_left": login_left,
                            "need_spin": need_spin,
                        },
                        "earned_total": int(per_total.get(int(u.telegram_id), 0)),
                        "earned_today": int(per_today.get(int(u.telegram_id), 0)),
                    }
                )

            return {
                "inviter_id": int(inviter.telegram_id),
                "total_referrals": len(items),
                "active": active_count,
                "pending": pending_count,
                "inactive": inactive_count,
                "earned_total": total_earned,
                "earned_today": earned_today,
                "items": items,
                "revshare_percent": 10,
                "inactive_after_days": 7,
                "activation_rule": {"login_days_required": 3, "spins_required": 1},
            }

    except Exception as e:
        logger.exception("/api/referrals failed: %s", e)
        raise HTTPException(status_code=500, detail="referrals_error")


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




# -----------------------------------------------------------------------------
# DAILY TASKS API (Mini App)
# -----------------------------------------------------------------------------
class DailyTaskItem(BaseModel):
    key: str
    title: str
    points: int
    icon: str = "🎯"
    need: int = 1
    progress: int = 0
    done: bool = False
    claimed: bool = False


class DailyTasksResp(BaseModel):
    day: str
    max_points: int = DAILY_MAX_POINTS_PER_DAY
    claimed_points: int = 0
    remaining_points: int = 0
    tasks: list[DailyTaskItem]


class DailyEventReq(BaseModel):
    telegram_id: int
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class DailyClaimReq(BaseModel):
    telegram_id: int
    task_key: str


class DailyClaimResp(BaseModel):
    ok: bool = True
    task_key: str
    awarded: int = 0
    claimed_points: int = 0
    remaining_points: int = 0
    user_points: int = 0
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
    spin_id: int
    points: int
    prize_key: str
    prize_type: str
    prize_value: int
    prize_label: str
    roll: int
    claimable: bool = False
    claim_id: Optional[int] = None
    claim_code: Optional[str] = None



class ClaimCreateReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    spin_id: int = Field(..., ge=1)

class ClaimCreateResp(BaseModel):
    telegram_id: int
    spin_id: int
    claim_id: int
    claim_code: str
    status: str

class ClaimSubmitReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    claim_id: int = Field(..., ge=1)
    full_name: str = Field(..., min_length=2, max_length=120)
    phone: str = Field(..., min_length=3, max_length=60)
    country: str = Field(..., min_length=2, max_length=80)
    city: str = Field(..., min_length=1, max_length=80)
    address_line: str = Field(..., min_length=5, max_length=240)
    postal_code: str = Field(..., min_length=2, max_length=20)
    comment: Optional[str] = Field(None, max_length=240)

class ConvertReq(BaseModel):
    telegram_id: int = Field(..., ge=1)
    spin_id: int = Field(..., ge=1)


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
                "claim_id": int(c.id),
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

# -----------------------------------------------------------------------------
# DAILY TASKS API
# -----------------------------------------------------------------------------
@app.get("/api/daily/tasks", response_model=DailyTasksResp)
async def daily_tasks_api(telegram_id: int):
    tid = int(telegram_id)
    day = _today_key()
    task_map = _daily_tasks_map()

    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            # Mini App can be opened before /start in bot -> create minimal user
            user = User(telegram_id=tid, points=10)
            session.add(user)
            await session.commit()

        logs = await _get_daily_logs(session, tid, day)

        # progress counters (stored in meta)
        post_opened = int((logs.get("open_post").meta or {}).get("count", 0) if logs.get("open_post") else 0)

        items: list[DailyTaskItem] = []
        for cfg in DAILY_TASKS:
            key = cfg["key"]
            need = int(cfg.get("need", 1))
            lg = logs.get(key)
            done = bool(lg) and (lg.status in ("done", "claimed"))
            claimed = bool(lg) and (lg.status == "claimed")
            progress = 0
            if key == "open_post":
                progress = min(post_opened, need)
                done = progress >= need and (lg is not None)
            if cfg.get("special"):
                # bonus_day becomes done only if all base tasks claimed
                done = await _can_unlock_bonus_day(task_map, logs)
                claimed = bool(lg) and (lg.status == "claimed")
                progress = 1 if done else 0
                need = 1

            items.append(
                DailyTaskItem(
                    key=key,
                    title=str(cfg["title"]),
                    points=int(cfg["points"]),
                    icon=str(cfg.get("icon") or "🎯"),
                    need=need,
                    progress=progress,
                    done=done,
                    claimed=claimed,
                )
            )

        claimed_points = await _daily_points_claimed(session, tid, day)
        remaining = max(0, DAILY_MAX_POINTS_PER_DAY - claimed_points)
        return DailyTasksResp(
            day=day,
            claimed_points=claimed_points,
            remaining_points=remaining,
            tasks=items,
        )


@app.post("/api/daily/event")
async def daily_event_api(req: DailyEventReq):
    tid = int(req.telegram_id)
    day = _today_key()
    ev = (req.event or "").strip().lower()

    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            # Mini App can be opened before /start in bot -> create minimal user
            user = User(telegram_id=tid, points=10)
            session.add(user)
            await session.commit()

        # NOTE: client events are best-effort; we still cap rewards on claim.
        if ev == "open_miniapp":
            await _mark_daily_done(session, tid, day, "open_miniapp")
        elif ev == "open_channel":
            await _mark_daily_done(session, tid, day, "open_channel")
        elif ev == "use_search":
            await _mark_daily_done(session, tid, day, "use_search", {"q": (req.data or {}).get("q", "")[:64]})
        elif ev == "open_inventory":
            await _mark_daily_done(session, tid, day, "open_inventory")
        elif ev == "open_profile":
            await _mark_daily_done(session, tid, day, "open_profile")
        elif ev == "comment_post":
            await _mark_daily_done(session, tid, day, "comment_post")
        elif ev == "spin_roulette":
            await _mark_daily_done(session, tid, day, "spin_roulette")
        elif ev == "open_post":
            # count up to need (3)
            logs = await _get_daily_logs(session, tid, day)
            lg = logs.get("open_post")
            cnt = int((lg.meta or {}).get("count", 0) if lg else 0)
            cnt = min(3, cnt + 1)
            await _mark_daily_done(session, tid, day, "open_post", {"count": cnt})
        else:
            # ignore unknown events
            return {"ok": True, "ignored": True}

        await session.commit()
        return {"ok": True}


@app.post("/api/daily/claim", response_model=DailyClaimResp)
async def daily_claim_api(req: DailyClaimReq):
    tid = int(req.telegram_id)
    key = (req.task_key or "").strip()
    day = _today_key()
    task_map = _daily_tasks_map()
    cfg = task_map.get(key)
    if not cfg:
        raise HTTPException(status_code=400, detail="unknown_task")

    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            # Mini App can be opened before /start in bot -> create minimal user
            user = User(telegram_id=tid, points=10)
            session.add(user)
            await session.commit()

        logs = await _get_daily_logs(session, tid, day)
        lg = logs.get(key)

        # determine if task is done server-side
        done = False
        if cfg.get("special"):
            done = await _can_unlock_bonus_day(task_map, logs)
        elif key == "open_post":
            cnt = int((lg.meta or {}).get("count", 0) if lg else 0)
            done = cnt >= int(cfg.get("need", 1))
        else:
            done = lg is not None

        if not done:
            raise HTTPException(status_code=400, detail="task_not_done")
        if lg and lg.status == "claimed":
            claimed_points = await _daily_points_claimed(session, tid, day)
            remaining = max(0, DAILY_MAX_POINTS_PER_DAY - claimed_points)
            return DailyClaimResp(
                ok=True,
                task_key=key,
                awarded=0,
                claimed_points=claimed_points,
                remaining_points=remaining,
                user_points=int(user.points or 0),
            )

        claimed_points = await _daily_points_claimed(session, tid, day)
        remaining = max(0, DAILY_MAX_POINTS_PER_DAY - claimed_points)
        award = int(cfg["points"])
        if award > remaining:
            award = remaining
        if award <= 0:
            raise HTTPException(status_code=400, detail="daily_cap_reached")

        now = datetime.utcnow()
        if not lg:
            lg = DailyTaskLog(
                telegram_id=tid,
                day=day,
                task_key=key,
                status="claimed",
                is_done=True,
                done_at=now,
                claimed_at=now,
                points=award,
                meta={},
            )
        else:
            lg.status = "claimed"
            try:
                lg.is_done = True
            except Exception:
                pass
            lg.claimed_at = now
            lg.points = award
        session.add(lg)

        user.points = int(user.points or 0) + award
        _recalc_tier(user)
        session.add(user)

        session.add(
            PointTransaction(
                telegram_id=tid,
                type=f"daily_task:{key}",
                delta=award,
                meta={"day": day, "task_key": key},
            )
        )

        await session.commit()

        new_claimed = claimed_points + award
        new_remaining = max(0, DAILY_MAX_POINTS_PER_DAY - new_claimed)
        return DailyClaimResp(
            ok=True,
            task_key=key,
            awarded=award,
            claimed_points=new_claimed,
            remaining_points=new_remaining,
            user_points=int(user.points or 0),
        )



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

            # если заявка уже отправлена/в обработке — конвертация недоступна
            if str(claim.status) not in {"draft", "awaiting_contact"}:
                raise HTTPException(status_code=400, detail="Заявка уже в обработке — конвертация недоступна")

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
                "resolution": getattr(r, "resolution", "pending"),
            }
        )
    return out


@app.get("/api/roulette/recent_wins")
async def roulette_recent_wins(limit: int = 12):
    limit = max(3, min(int(limit), 30))
    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(RouletteSpin.prize_label, RouletteSpin.created_at)
                .order_by(RouletteSpin.created_at.desc())
                .limit(limit)
            )
        ).all()
    out = []
    for (label, dt) in rows:
        out.append({"prize_label": str(label), "created_at": dt.isoformat() if dt else None})
    return out



@app.get("/api/roulette/status")
async def roulette_status(telegram_id: int):
    """
    UI helper: tells if user can spin now and how many seconds left until next spin.
    Also returns current points and spin cost.
    """
    tid = int(telegram_id)
    now = datetime.utcnow()
    async with async_session_maker() as session:
        user = (
            await session.execute(select(User).where(User.telegram_id == tid))
        ).scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        last_spin = (
            await session.execute(
                select(RouletteSpin.created_at)
                .where(RouletteSpin.telegram_id == tid)
                .order_by(RouletteSpin.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    secs_left = 0
    can_spin_time = True
    if last_spin and (now - last_spin) < ROULETTE_LIMIT_WINDOW:
        delta = ROULETTE_LIMIT_WINDOW - (now - last_spin)
        secs_left = max(1, int(delta.total_seconds()) + (1 if (delta.total_seconds() % 1) > 0 else 0))
        can_spin_time = False

    points = int(user.points or 0)
    enough_points = points >= int(ROULETTE_SPIN_COST)
    can_spin = can_spin_time and enough_points

    return {
        "telegram_id": tid,
        "points": points,
        "spin_cost": int(ROULETTE_SPIN_COST),
        "cooldown_seconds": int(ROULETTE_LIMIT_WINDOW.total_seconds()),
        "seconds_left": int(secs_left),
        "can_spin": bool(can_spin),
        "can_spin_time": bool(can_spin_time),
        "enough_points": bool(enough_points),
    }



@app.post("/api/roulette/spin", response_model=SpinResp)
async def roulette_spin(req: SpinReq):
    tid = int(req.telegram_id)
    now = datetime.utcnow()

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

            # activity tracking
            user.last_seen_at = now
            user.roulette_spins_total = int(user.roulette_spins_total or 0) + 1

            # if this user is invitee and reaches ACTIVE threshold for the first time -> mark time
            if user.referred_by and (user.ref_active_at is None):
                if int(user.daily_login_total or 0) >= REF_ACTIVE_MIN_LOGIN_DAYS and int(user.roulette_spins_total or 0) >= REF_ACTIVE_MIN_SPINS:
                    user.ref_active_at = now

            roll = secrets.randbelow(1_000_000)
            prize = pick_roulette_prize(roll)
            prize_key = str(prize.get("key") or "")
            prize_type: PrizeType = prize["type"]
            prize_value = int(prize["value"])
            prize_label = str(prize["label"])

            # выдача приза
            # ВАЖНО: для Dior (физический приз) создаём запись в "косметичке" сразу (claim со статусом awaiting_contact),
            # чтобы приз появился в инвентаре сразу после выигрыша. Конвертация всё равно доступна до отправки анкеты.
            claim_id_for_resp = None
            claim_code_for_resp = None

            if prize_type == "points":
                user.points = (user.points or 0) + prize_value
                session.add(
                    PointTransaction(
                        telegram_id=tid,
                        type="roulette_prize",
                        delta=prize_value,
                        meta={"roll": roll, "prize": prize_label, "key": prize_key},
                    )
                )
            elif prize_type == "raffle_ticket":
                ticket_row = await get_ticket_row(session, tid, DEFAULT_RAFFLE_ID)
                ticket_row.count = int(ticket_row.count or 0) + prize_value
                ticket_row.updated_at = now
                session.add(
                    PointTransaction(
                        telegram_id=tid,
                        type="roulette_prize",
                        delta=0,
                        meta={"roll": roll, "prize": "raffle_ticket", "qty": prize_value, "key": prize_key},
                    )
                )
            else:
                session.add(
                    PointTransaction(
                        telegram_id=tid,
                        type="roulette_prize",
                        delta=0,
                        meta={"roll": roll, "prize": "physical_dior_palette", "key": prize_key},
                    )
                )

            _recalc_tier(user)

            spin_row = RouletteSpin(
                telegram_id=tid,
                created_at=now,
                cost_points=ROULETTE_SPIN_COST,
                roll=roll,
                prize_type=prize_type,
                prize_value=prize_value,
                prize_label=prize_label,
                resolution="pending",
                resolved_at=None,
                resolved_meta=None,
            )
            session.add(spin_row)
            await session.flush()

            spin_id = int(spin_row.id)


            # ------------------ REF REVSHARE: 10% from invitee bonus wins ------------------
            # Pay to inviter ONLY if invitee is ACTIVE (3 days logins + >=1 spin) and not inactive (7 days)
            if prize_type == "points" and prize_value > 0 and user.referred_by:
                st_code, _, _ = _referral_status(user, now)
                if st_code == "active":
                    inviter_id = int(user.referred_by)
                    raw_share = int(prize_value * REF_REVSHARE_PCT)
                    if raw_share > 0:
                        # caps
                        day_start = _utc_day_start(now)
                        per_user_today = int(
                            (await session.execute(
                                select(func.coalesce(func.sum(ReferralEarning.amount), 0))
                                .where(ReferralEarning.inviter_id == inviter_id)
                                .where(ReferralEarning.referred_id == tid)
                                .where(ReferralEarning.created_at >= day_start)
                            )).scalar_one() or 0
                        )
                        inviter_today = int(
                            (await session.execute(
                                select(func.coalesce(func.sum(ReferralEarning.amount), 0))
                                .where(ReferralEarning.inviter_id == inviter_id)
                                .where(ReferralEarning.created_at >= day_start)
                            )).scalar_one() or 0
                        )
                        remaining = min(
                            max(0, REF_REVSHARE_PER_REFERRED_DAY_CAP - per_user_today),
                            max(0, REF_REVSHARE_PER_INVITER_DAY_CAP - inviter_today),
                        )
                        pay = min(raw_share, remaining)
                        if pay > 0:
                            inviter = (
                                await session.execute(
                                    select(User).where(User.telegram_id == inviter_id).with_for_update()
                                )
                            ).scalar_one_or_none()
                            if inviter:
                                inviter.points = (inviter.points or 0) + int(pay)
                                _recalc_tier(inviter)
                                session.add(
                                    ReferralEarning(
                                        inviter_id=inviter_id,
                                        referred_id=tid,
                                        amount=int(pay),
                                        created_at=now,
                                        source="roulette_win",
                                        meta={
                                            "spin_id": spin_id,
                                            "invitee_win": int(prize_value),
                                            "pct": int(REF_REVSHARE_PCT * 100),
                                            "prize": prize_label,
                                        },
                                    )
                                )
                                session.add(
                                    PointTransaction(
                                        telegram_id=inviter_id,
                                        type="ref_revshare",
                                        delta=int(pay),
                                        meta={
                                            "from": tid,
                                            "spin_id": spin_id,
                                            "invitee_win": int(prize_value),
                                            "pct": int(REF_REVSHARE_PCT * 100),
                                        },
                                    )
                                )

            # -----------------------------------------------------------------------------

            # Auto-create inventory item for Dior (physical prize) so it appears in Cosmetics Bag immediately.
            if prize_type == "physical_dior_palette":
                existing_claim = (
                    await session.execute(
                        select(RouletteClaim).where(RouletteClaim.spin_id == spin_id)
                    )
                ).scalar_one_or_none()
                if existing_claim:
                    claim_id_for_resp = int(existing_claim.id)
                    claim_code_for_resp = str(existing_claim.claim_code)
                else:
                    claim_code_for_resp = generate_claim_code()
                    claim = RouletteClaim(
                        claim_code=claim_code_for_resp,
                        telegram_id=tid,
                        spin_id=spin_id,
                        prize_type="physical_dior_palette",
                        prize_label=prize_label,
                        status="awaiting_contact",
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(claim)
                    await session.flush()
                    claim_id_for_resp = int(claim.id)

        await session.refresh(user)

    return {
        "telegram_id": tid,
        "spin_id": spin_id,
        "points": int(user.points or 0),
        "prize_key": prize_key,
        "prize_type": prize_type,
        "prize_value": prize_value,
        "prize_label": prize_label,
        "roll": int(roll),
        "claimable": bool(prize_type == "physical_dior_palette"),
        "claim_id": claim_id_for_resp,
        "claim_code": claim_code_for_resp,
    }


@app.post("/api/roulette/convert")
async def roulette_convert(req: ConvertReq):
    tid = int(req.telegram_id)
    spin_id = int(req.spin_id)
    now = datetime.utcnow()

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(
                    select(User).where(User.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            spin = (
                await session.execute(
                    select(RouletteSpin)
                    .where(RouletteSpin.id == spin_id, RouletteSpin.telegram_id == tid)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if not spin:
                raise HTTPException(status_code=404, detail="Spin not found")

            if spin.prize_type != "physical_dior_palette":
                raise HTTPException(status_code=400, detail="Этот приз нельзя конвертировать")

            if getattr(spin, "resolution", "pending") != "pending":
                raise HTTPException(status_code=400, detail="Этот спин уже обработан")

            existing_claim = (
                await session.execute(
                    select(RouletteClaim).where(RouletteClaim.spin_id == spin_id)
                )
            ).scalar_one_or_none()
            if existing_claim is not None:
                # Разрешаем конвертацию, если заявка ещё не отправлена (draft/awaiting_contact).
                if str(existing_claim.status) not in {"draft", "awaiting_contact"}:
                    raise HTTPException(status_code=400, detail="Заявка уже в обработке — конвертация недоступна")
                # удаляем "резерв" из косметички
                await session.delete(existing_claim)

            # конвертация Dior -> points (фиксированная стоимость)
            user.points = int(user.points or 0) + int(DIOR_PALETTE_CONVERT_VALUE)
            session.add(
                PointTransaction(
                    telegram_id=tid,
                    type="roulette_convert",
                    delta=int(DIOR_PALETTE_CONVERT_VALUE),
                    meta={"spin_id": spin_id, "convert": "dior_palette"},
                )
            )

            spin.resolution = "converted"
            spin.resolved_at = now
            spin.resolved_meta = {"converted_value": int(DIOR_PALETTE_CONVERT_VALUE)}

            _recalc_tier(user)

        await session.refresh(user)

    return {"ok": True, "balance_after": int(user.points or 0), "converted_value": int(DIOR_PALETTE_CONVERT_VALUE)}


@app.post("/api/roulette/claim/create", response_model=ClaimCreateResp)
async def roulette_claim_create(req: ClaimCreateReq):
    tid = int(req.telegram_id)
    spin_id = int(req.spin_id)
    now = datetime.utcnow()

    async with async_session_maker() as session:
        async with session.begin():
            user = (
                await session.execute(
                    select(User).where(User.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            spin = (
                await session.execute(
                    select(RouletteSpin)
                    .where(RouletteSpin.id == spin_id, RouletteSpin.telegram_id == tid)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if not spin:
                raise HTTPException(status_code=404, detail="Spin not found")

            if spin.prize_type != "physical_dior_palette":
                raise HTTPException(status_code=400, detail="Этот приз нельзя забрать как физический")

            # если уже есть claim — вернуть его
            existing = (
                await session.execute(
                    select(RouletteClaim).where(RouletteClaim.spin_id == spin_id)
                )
            ).scalar_one_or_none()
            if existing:
                # If claim was auto-created on spin (inventory), make sure the spin is marked as "claim"
                # so history/inventory stay consistent.
                if getattr(spin, "resolution", "pending") == "pending":
                    spin.resolution = "claim"
                    spin.resolved_at = now
                    spin.resolved_meta = {"claim_code": str(existing.claim_code)}
                return {
                    "telegram_id": tid,
                    "spin_id": spin_id,
                    "claim_id": int(existing.id),
                    "claim_code": str(existing.claim_code),
                    "status": str(existing.status),
                }

            if getattr(spin, "resolution", "pending") != "pending":
                raise HTTPException(status_code=400, detail="Этот спин уже обработан")

            claim_code = generate_claim_code()
            claim = RouletteClaim(
                claim_code=claim_code,
                telegram_id=tid,
                spin_id=spin_id,
                prize_type=spin.prize_type,
                prize_label=spin.prize_label,
                status="draft",
                created_at=now,
                updated_at=now,
            )
            session.add(claim)

            spin.resolution = "claim"
            spin.resolved_at = now
            spin.resolved_meta = {"claim_code": claim_code}

            await session.flush()

            claim_id = int(claim.id)

    return {
        "telegram_id": tid,
        "spin_id": spin_id,
        "claim_id": claim_id,
        "claim_code": claim_code,
        "status": "draft",
    }


@app.get("/api/roulette/claim/{claim_id}")
async def roulette_claim_get(claim_id: int, telegram_id: int):
    tid = int(telegram_id)
    cid = int(claim_id)
    async with async_session_maker() as session:
        claim = (
            await session.execute(
                select(RouletteClaim).where(RouletteClaim.id == cid, RouletteClaim.telegram_id == tid)
            )
        ).scalar_one_or_none()
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")

    return {
        "claim_id": int(claim.id),
        "claim_code": str(claim.claim_code),
        "spin_id": int(claim.spin_id) if claim.spin_id is not None else None,
        "status": str(claim.status),
        "prize_label": str(claim.prize_label),
        "full_name": claim.full_name,
        "phone": claim.phone,
        "country": claim.country,
        "city": claim.city,
        "address_line": claim.address_line,
        "postal_code": claim.postal_code,
        "comment": claim.comment,
        "created_at": claim.created_at.isoformat() if claim.created_at else None,
        "updated_at": claim.updated_at.isoformat() if claim.updated_at else None,
    }


@app.post("/api/roulette/claim/submit")
async def roulette_claim_submit(req: ClaimSubmitReq):
    tid = int(req.telegram_id)
    now = datetime.utcnow()

    async with async_session_maker() as session:
        async with session.begin():
            claim = (
                await session.execute(
                    select(RouletteClaim).where(RouletteClaim.id == int(req.claim_id), RouletteClaim.telegram_id == tid).with_for_update()
                )
            ).scalar_one_or_none()
            if not claim:
                raise HTTPException(status_code=404, detail="Claim not found")

            if str(claim.status) not in {"draft", "awaiting_contact"}:
                # уже отправлено/закрыто
                return {"ok": True, "status": str(claim.status)}

            # сохраняем контактные данные
            claim.full_name = req.full_name.strip()
            claim.phone = req.phone.strip()
            claim.country = req.country.strip()
            claim.city = req.city.strip()
            claim.address_line = req.address_line.strip()
            claim.postal_code = req.postal_code.strip()
            claim.comment = (req.comment or "").strip() or None

            claim.contact_text = (
                f"Имя: {claim.full_name}\n"
                f"Телефон: {claim.phone}\n"
                f"Страна: {claim.country}\n"
                f"Город: {claim.city}\n"
                f"Адрес: {claim.address_line}\n"
                f"Индекс: {claim.postal_code}\n"
                + (f"Комментарий: {claim.comment}\n" if claim.comment else "")
            )

            claim.status = "submitted"
            claim.updated_at = now

        # уведомление админу — уже после commit
        uname = ""
        async with async_session_maker() as s2:
            u = (await s2.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
            uname = (u.username or "").strip() if u else ""
            fname = (u.first_name or "-") if u else "-"

        mention = f"@{uname}" if uname else "(без username)"
        await notify_admin(
            "📦 ЗАЯВКА НА ПРИЗ (Рулетка)\n"
            f"prize: {claim.prize_label}\n"
            f"user: {mention} | {fname}\n"
            f"telegram_id: {tid}\n"
            f"link: {tg_user_link(tid)}\n"
            f"claim: {claim.claim_code}\n\n"
            f"{claim.contact_text}"
        )

    return {"ok": True, "status": "submitted"}
