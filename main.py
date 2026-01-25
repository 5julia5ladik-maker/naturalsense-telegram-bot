# main.py
import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    WebAppInfo,
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
PUBLIC_BASE_URL = (env_get("PUBLIC_BASE_URL", "") or "").rstrip("/")
CHANNEL_USERNAME = env_get("CHANNEL_USERNAME", "NaturalSense") or "NaturalSense"
DATABASE_URL = env_get("DATABASE_URL", "sqlite+aiosqlite:///./ns.db") or "sqlite+aiosqlite:///./ns.db"
ADMIN_CHAT_ID = int(env_get("ADMIN_CHAT_ID", "5443870760") or "5443870760")

# ✅ Mini App: НЕ ТРОГАЕМ. Просто отдаём готовую сборку из папки.
# Положи туда build (index.html + assets).
MINI_APP_DIR = env_get("MINI_APP_DIR", "./webapp_build") or "./webapp_build"
MINI_APP_INDEX = env_get("MINI_APP_INDEX", "index.html") or "index.html"

# Fix Railway postgres schemes for async SQLAlchemy
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

tok = BOT_TOKEN or ""
logger.info(
    "ENV CHECK: BOT_TOKEN_present=%s BOT_TOKEN_len=%s PUBLIC_BASE_URL_present=%s DATABASE_URL_present=%s CHANNEL=%s ADMIN=%s MINI_APP_DIR=%s",
    bool(BOT_TOKEN),
    len(tok),
    bool(PUBLIC_BASE_URL),
    bool(DATABASE_URL),
    CHANNEL_USERNAME,
    ADMIN_CHAT_ID,
    MINI_APP_DIR,
)

# -----------------------------------------------------------------------------
# BLOCKED TAGS (не отдаём эти теги наружу)
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

# -----------------------------------------------------------------------------
# BRAND TAG MAP (справка по тегам)
# -----------------------------------------------------------------------------
BRAND_TAGS: dict[str, str] = {
    "The Ordinary": "TheOrdinary",
    "Dior": "Dior",
    "Chanel": "Chanel",
    "Kylie Cosmetics": "KylieCosmetics",
    "Gisou": "Gisou",
    "Rare Beauty": "RareBeauty",
    "Yves Saint Laurent": "YSL",
    "Givenchy": "Givenchy",
    "Charlotte Tilbury": "CharlotteTilbury",
    "NARS": "NARS",
    "Sol de Janeiro": "SolDeJaneiro",
    "Huda Beauty": "HudaBeauty",
    "Rhode": "Rhode",
    "Tower 28 Beauty": "Tower28Beauty",
    "Benefit Cosmetics": "BenefitCosmetics",
    "Estée Lauder": "EsteeLauder",
    "Sisley": "Sisley",
    "Kérastase": "Kerastase",
    "Armani Beauty": "ArmaniBeauty",
    "Hourglass": "Hourglass",
    "Shiseido": "Shiseido",
    "Tom Ford Beauty": "TomFordBeauty",
    "Tarte": "Tarte",
    "Sephora Collection": "SephoraCollection",
    "Clinique": "Clinique",
    "Dolce & Gabbana": "DolceGabbana",
    "Kayali": "Kayali",
    "Guerlain": "Guerlain",
    "Fenty Beauty": "FentyBeauty",
    "Too Faced": "TooFaced",
    "MAKE UP FOR EVER": "MakeUpForEver",
    "Erborian": "Erborian",
    "Natasha Denona": "NatashaDenona",
    "Lancôme": "Lancome",
    "Kosas": "Kosas",
    "ONE/SIZE": "OneSize",
    "Laneige": "Laneige",
    "Makeup by Mario": "MakeupByMario",
    "Valentino Beauty": "ValentinoBeauty",
    "Drunk Elephant": "DrunkElephant",
    "Olaplex": "Olaplex",
    "Anastasia Beverly Hills": "AnastasiaBeverlyHills",
    "Amika": "Amika",
    "BYOMA": "BYOMA",
    "Glow Recipe": "GlowRecipe",
    "Milk Makeup": "MilkMakeup",
    "Summer Fridays": "SummerFridays",
    "K18": "K18",
}

# -----------------------------------------------------------------------------
# DATABASE MODELS
# -----------------------------------------------------------------------------
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, index=True, nullable=False)  # ✅ BIGINT

    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)

    tier = Column(String, default="free")
    points = Column(Integer, default=10)
    favorites = Column(JSON, default=list)
    joined_at = Column(DateTime, default=lambda: datetime.utcnow())  # naive UTC

    # антифарм + стрик
    last_daily_bonus_at = Column(DateTime, nullable=True)  # naive UTC
    daily_streak = Column(Integer, default=0)
    best_streak = Column(Integer, default=0)

    # рефералка
    referred_by = Column(BigInteger, nullable=True)
    referral_count = Column(Integer, default=0)
    ref_bonus_paid = Column(Boolean, default=False, nullable=False)  # чтобы не платить повторно

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, unique=True, index=True, nullable=False)  # ✅ BIGINT

    date = Column(DateTime, nullable=True)  # naive UTC
    text = Column(String, nullable=True)
    media_type = Column(String, nullable=True)
    media_file_id = Column(String, nullable=True)
    permalink = Column(String, nullable=True)

    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.utcnow())  # naive UTC

    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

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

    logger.info("✅ Database initialized")

# -----------------------------------------------------------------------------
# USER / POINTS / STREAK / REFERRAL
# -----------------------------------------------------------------------------
def _recalc_tier(user: User):
    # Bronze / Silver / Gold VIP
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
    """
    Новый пользователь:
    - получает +10
    - стрик = 1
    - daily бонус считается выданным сейчас (чтобы антифарм работал)
    - если есть валидный inviter и не self-ref: inviter +20 и referral_count +1
      бонус платится 1 раз за каждого (ref_bonus_paid у нового юзера)
    """
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

        # платим реф. бонус пригласившему (1 раз)
        if inviter and not user.ref_bonus_paid:
            inviter.points = (inviter.points or 0) + REFERRAL_BONUS_POINTS
            inviter.referral_count = (inviter.referral_count or 0) + 1
            _recalc_tier(inviter)
            user.ref_bonus_paid = True
            referral_paid = True

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
        await session.commit()
        await session.refresh(user)
        return user

async def add_daily_bonus_and_update_streak(telegram_id: int) -> tuple[Optional[User], bool, int, int]:
    """
    Антифарм: строго 1 раз в 24 часа.
    Стрик: если визит <= 48 часов от последнего бонуса — продолжаем, иначе сброс.
    """
    async with async_session_maker() as session:
        user: User | None = (await session.execute(select(User).where(User.telegram_id == telegram_id))).scalar_one_or_none()
        if not user:
            return None, False, 0, 0

        now = datetime.utcnow()
        last = user.last_daily_bonus_at

        # антифарм
        if last is not None and (now - last) < timedelta(days=1):
            delta = timedelta(days=1) - (now - last)
            hours_left = max(
                0,
                int(delta.total_seconds() // 3600) + (1 if (delta.total_seconds() % 3600) > 0 else 0),
            )
            return user, False, hours_left, 0

        # выдаём ежедневный бонус
        user.points = (user.points or 0) + DAILY_BONUS_POINTS

        # стрик
        if last is None:
            user.daily_streak = 1
        else:
            if (now - last) <= timedelta(days=2):  # 48ч окно
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
    # публичная проверка по embed
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
            # если канал приватный или требует join — считаем, что ок (не можем проверить)
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

# ✅ "В канал" без спама: редактируем одно и то же сообщение
_last_channel_msg_id: dict[int, int] = {}

def is_admin(user_id: int) -> bool:
    return int(user_id) == int(ADMIN_CHAT_ID)

def get_main_keyboard():
    webapp_url = f"{PUBLIC_BASE_URL}/webapp" if PUBLIC_BASE_URL else "/webapp"
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📲 Открыть журнал", web_app=WebAppInfo(url=webapp_url))],
            [KeyboardButton("👤 Профиль"), KeyboardButton("ℹ️ Помощь")],
            [KeyboardButton("↩️ В канал")],
        ],
        resize_keyboard=True,
    )

def build_help_text() -> str:
    return """\
ℹ️ *Помощь / Как пользоваться*

1) Нажми *📲 Открыть журнал* — откроется Mini App внутри Telegram.
2) Внутри Mini App выбирай категории/бренды и открывай посты.
3) *👤 Профиль* — твои баллы, уровень, стрик.
4) *↩️ В канал* — кнопка для быстрого перехода в канал.

💎 *Баллы и антифарм*
• Первый /start: +10 за регистрацию
• Далее: +5 за визит, строго 1 раз в 24 часа

🔥 *Стрик (серия дней)*
За ежедневный вход растёт стрик. Бонусы:
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

async def open_channel_clean(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    ✅ "Чистая" кнопка: не плодим сообщения.
    Редактируем один и тот же message (последний, где уже показывали кнопку).
    """
    url = f"https://t.me/{CHANNEL_USERNAME}"
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Открыть канал ↗️", url=url)]])

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    # если пользователь нажал кнопку с клавиатуры — это сообщение пользователя.
    # мы просто дадим/обновим одно "служебное" сообщение бота с кнопкой.
    prev_id = _last_channel_msg_id.get(user_id)
    if prev_id:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=prev_id,
                text="↩️ В канал:",
                reply_markup=kb,
            )
            return
        except Exception:
            _last_channel_msg_id.pop(user_id, None)

    # отправляем один раз и запоминаем
    if update.message:
        msg = await update.message.reply_text("↩️ В канал:", reply_markup=kb)
        _last_channel_msg_id[user_id] = msg.message_id

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(build_help_text(), parse_mode="Markdown", reply_markup=get_main_keyboard())

async def cmd_invite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    me = await context.bot.get_me()
    bot_username = me.username or ""
    if not bot_username:
        await update.message.reply_text("Не удалось получить username бота. Проверь настройки.", reply_markup=get_main_keyboard())
        return

    link = f"https://t.me/{bot_username}?start={user.id}"
    text_ = f"""\
🎟 Твоя реферальная ссылка:

{link}

За каждого нового пользователя по этой ссылке: +{REFERRAL_BONUS_POINTS} баллов (1 раз за каждого).
"""
    await update.message.reply_text(text_, reply_markup=get_main_keyboard())

async def cmd_brandtags(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["🏷 Теги брендов (пиши в постах так: #TAG):\n"]
    for name, tag in BRAND_TAGS.items():
        lines.append(f"• {name} — #{tag}")
    await update.message.reply_text("\n".join(lines), reply_markup=get_main_keyboard())

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

Я — Natural Sense Assistant.
• открываю мини-журнал внутри Telegram
• показываю профиль и баллы
• даю бонусы за ежедневные визиты и стрик
• веду в канал одним нажатием

Как пользоваться:
1) Нажми «📲 Открыть журнал»
2) Выбирай категории/бренды и открывай посты
3) «👤 Профиль» — баллы, уровень, стрик
4) «ℹ️ Помощь» — правила и фишки

{bonus_line}
{streak_line}{ref_line}
"""

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user

    referred_by: int | None = None
    if context.args:
        arg0 = (context.args[0] or "").strip()
        if arg0.isdigit():
            referred_by = int(arg0)

    db_user = await get_user(user.id)

    # новый пользователь
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
            daily_granted=True,  # регистрация уже даёт стартовый “визит”
            hours_left=0,
            streak=created_user.daily_streak or 1,
            streak_bonus=0,
            referral_paid=referral_paid,
        )
        await update.message.reply_text(text_, reply_markup=get_main_keyboard())
        return

    # существующий — выдаём ежедневный бонус/стрик по антифарму
    user2, granted, hours_left, streak_bonus = await add_daily_bonus_and_update_streak(user.id)
    if not user2:
        await update.message.reply_text("Ошибка пользователя. Нажми /start ещё раз.", reply_markup=get_main_keyboard())
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
    await update.message.reply_text(text_, reply_markup=get_main_keyboard())

async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("⛔️ Нет доступа.", reply_markup=get_main_keyboard())
        return
    await update.message.reply_text(await admin_stats_text(), parse_mode="Markdown", reply_markup=get_main_keyboard())

async def cmd_admin_sweep(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    text_ = f"""\
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
    await update.message.reply_text(text_, reply_markup=get_main_keyboard())

async def cmd_admin_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    if txt == "👤 Профиль":
        await cmd_profile(update, context)
        return

    if txt == "ℹ️ Помощь":
        await cmd_help(update, context)
        return

    if txt == "↩️ В канал":
        await open_channel_clean(update, context)
        return

# -----------------------------------------------------------------------------
# CHANNEL INDEXING (авто)
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
        await tg_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
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

    # errors
    tg_app.add_error_handler(tg_error_handler)

    # user commands
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("help", cmd_help))
    tg_app.add_handler(CommandHandler("invite", cmd_invite))
    tg_app.add_handler(CommandHandler("brandtags", cmd_brandtags))
    tg_app.add_handler(CommandHandler("profile", cmd_profile))
    tg_app.add_handler(CommandHandler("myid", cmd_myid))
    tg_app.add_handler(CommandHandler("id", cmd_id))

    # admin commands (дополнение, админ остаётся юзером)
    tg_app.add_handler(CommandHandler("admin", cmd_admin))
    tg_app.add_handler(CommandHandler("admin_stats", cmd_admin_stats))
    tg_app.add_handler(CommandHandler("admin_sweep", cmd_admin_sweep))
    tg_app.add_handler(CommandHandler("admin_user", cmd_admin_user))
    tg_app.add_handler(CommandHandler("admin_add", cmd_admin_add))
    tg_app.add_handler(CommandHandler("find", cmd_admin_find))

    # callbacks
    tg_app.add_handler(CallbackQueryHandler(on_callback))

    # text buttons
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text_button))

    # channel posts indexing
    tg_app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST, on_channel_post))
    tg_app.add_handler(MessageHandler(filters.UpdateType.EDITED_CHANNEL_POST, on_edited_channel_post))

    tg_task = asyncio.create_task(_telegram_runner())

# -----------------------------------------------------------------------------
# FASTAPI (MINI APP) — НЕ МЕНЯЕМ, ОТДАЁМ ГОТОВУЮ СБОРКУ
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _mini_app_exists() -> bool:
    index_path = os.path.join(MINI_APP_DIR, MINI_APP_INDEX)
    return os.path.isdir(MINI_APP_DIR) and os.path.isfile(index_path)

# отдаём статику мини-аппа как есть
if os.path.isdir(MINI_APP_DIR):
    app.mount("/webapp", StaticFiles(directory=MINI_APP_DIR, html=True), name="webapp_static")
else:
    logger.warning("MINI_APP_DIR not found: %s (Mini App will return fallback message)", MINI_APP_DIR)

@app.get("/", response_class=HTMLResponse)
async def root():
    # удобный редирект: если мини-апп есть — отдаём его index
    if _mini_app_exists():
        return FileResponse(os.path.join(MINI_APP_DIR, MINI_APP_INDEX))
    return HTMLResponse(
        "<h3>Mini App build not found</h3>"
        "<p>Put your React/HTML build into <b>webapp_build/</b> or set MINI_APP_DIR.</p>"
    )

@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# -----------------------------------------------------------------------------
# API для Mini App (НЕ меняем фронт, но бэкенд-ручки должны быть)
# -----------------------------------------------------------------------------
@app.get("/api/posts")
async def api_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    tag = (tag or "").strip() or None

    rows = await list_posts(tag=tag, limit=limit, offset=offset)
    items = []
    for p in rows:
        items.append(
            {
                "message_id": int(p.message_id),
                "date": (p.date.strftime("%d.%m.%Y %H:%M") if p.date else None),
                "text": p.text or "",
                "preview": preview_text(p.text, 220),
                "permalink": p.permalink,
                "tags": p.tags or [],
            }
        )
    return {"ok": True, "items": items}

@app.get("/api/profile")
async def api_profile(telegram_id: int):
    u = await get_user(int(telegram_id))
    if not u:
        return {"ok": False, "error": "not_registered"}
    return {
        "ok": True,
        "telegram_id": int(u.telegram_id),
        "username": u.username,
        "first_name": u.first_name,
        "tier": u.tier,
        "points": u.points,
        "daily_streak": u.daily_streak,
        "best_streak": u.best_streak,
        "referral_count": u.referral_count,
        "joined_at": u.joined_at.isoformat() if u.joined_at else None,
        "last_daily_bonus_at": u.last_daily_bonus_at.isoformat() if u.last_daily_bonus_at else None,
    }

@app.get("/api/brands")
async def api_brands():
    items = []
    for name, tag in BRAND_TAGS.items():
        if tag in BLOCKED_TAGS:
            continue
        items.append({"name": name, "tag": tag})
    items.sort(key=lambda x: x["name"].lower())
    return {"ok": True, "items": items}

# -----------------------------------------------------------------------------
# APP LIFECYCLE (init db + start bot + start sweeper)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app_: FastAPI):
    global sweeper_task
    await init_db()

    # sweeper auto loop
    sweeper_task = asyncio.create_task(sweeper_loop())

    # telegram bot
    await start_telegram_bot()

    try:
        yield
    finally:
        if sweeper_task:
            sweeper_task.cancel()
            try:
                await sweeper_task
            except Exception:
                pass

        if tg_task:
            tg_task.cancel()
            try:
                await tg_task
            except Exception:
                pass

        try:
            await engine.dispose()
        except Exception:
            pass

app.router.lifespan_context = lifespan

# -----------------------------------------------------------------------------
# Local run (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
