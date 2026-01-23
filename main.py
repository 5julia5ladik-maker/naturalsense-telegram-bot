import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, select, text as sql_text, update
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
    if v is not None:
        return v
    return default

BOT_TOKEN = env_get("BOT_TOKEN")
PUBLIC_BASE_URL = (env_get("PUBLIC_BASE_URL", "") or "").rstrip("/")
CHANNEL_USERNAME = env_get("CHANNEL_USERNAME", "NaturalSense") or "NaturalSense"
DATABASE_URL = env_get("DATABASE_URL", "sqlite+aiosqlite:///./ns.db") or "sqlite+aiosqlite:///./ns.db"

# Fix Railway postgres schemes for async SQLAlchemy
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

tok = BOT_TOKEN or ""
logger.info(
    "ENV CHECK: BOT_TOKEN_present=%s BOT_TOKEN_len=%s PUBLIC_BASE_URL_present=%s DATABASE_URL_present=%s CHANNEL=%s",
    bool(BOT_TOKEN), len(tok), bool(PUBLIC_BASE_URL), bool(DATABASE_URL), CHANNEL_USERNAME
)

# -----------------------------------------------------------------------------
# TAGS BLOCKLIST (убираем "Актуальные цены (TR)" и "Гайды/как покупать" + их теги)
# -----------------------------------------------------------------------------
BLOCKED_TAGS = {"SephoraTR", "SephoraGuide"}

# -----------------------------------------------------------------------------
# DATABASE MODELS
# -----------------------------------------------------------------------------
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    tier = Column(String, default="free")
    points = Column(Integer, default=10)
    favorites = Column(JSON, default=list)
    joined_at = Column(DateTime, default=lambda: datetime.utcnow())  # naive UTC

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)

    # используем message_id
    message_id = Column(Integer, unique=True, index=True, nullable=False)

    # naive UTC
    date = Column(DateTime, nullable=True)

    text = Column(String, nullable=True)
    media_type = Column(String, nullable=True)
    media_file_id = Column(String, nullable=True)
    permalink = Column(String, nullable=True)

    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.utcnow())  # naive UTC

    # УДАЛЕНИЕ
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # добавляем колонки, если их нет
        try:
            await conn.execute(sql_text("ALTER TABLE posts ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;"))
        except Exception:
            pass
        try:
            await conn.execute(sql_text("ALTER TABLE posts ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP NULL;"))
        except Exception:
            pass

    logger.info("✅ Database initialized")

# -----------------------------------------------------------------------------
# USER QUERIES
# -----------------------------------------------------------------------------
async def get_user(telegram_id: int):
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        return result.scalar_one_or_none()

async def create_user(telegram_id: int, username: str | None = None, first_name: str | None = None):
    async with async_session_maker() as session:
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            points=10
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        logger.info("✅ New user created: %s", telegram_id)
        return user

async def add_points(telegram_id: int, points: int):
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if not user:
            return None

        user.points += points

        if user.points >= 500:
            user.tier = "vip"
        elif user.points >= 100:
            user.tier = "premium"

        await session.commit()
        await session.refresh(user)
        return user

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
    permalink = make_permalink(message_id)
    date_naive = to_naive_utc(date)

    async with async_session_maker() as session:
        res = await session.execute(select(Post).where(Post.message_id == message_id))
        p = res.scalar_one_or_none()

        if p:
            p.date = date_naive
            p.text = text_
            p.media_type = media_type
            p.media_file_id = media_file_id
            p.permalink = permalink
            p.tags = tags

            # если пост снова виден — считаем не удалён
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
    # полностью блокируем эти 2 тега
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
        q = (
            select(Post)
            .where(Post.is_deleted == False)  # noqa: E712
            .order_by(Post.message_id.desc())
            .limit(batch)
        )
        posts = (await session.execute(q)).scalars().all()

    if not posts:
        return

    to_mark: list[int] = []
    for p in posts:
        ok = await message_exists_public(p.message_id)
        if not ok:
            to_mark.append(p.message_id)

    if not to_mark:
        return

    async with async_session_maker() as session:
        now = datetime.utcnow()
        await session.execute(
            update(Post)
            .where(Post.message_id.in_(to_mark))
            .values(is_deleted=True, deleted_at=now)
        )
        await session.commit()

    logger.info("🧹 Marked deleted posts: %s", to_mark)

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

def get_main_keyboard():
    webapp_url = f"{PUBLIC_BASE_URL}/webapp" if PUBLIC_BASE_URL else "/webapp"
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📲 Открыть журнал", web_app=WebAppInfo(url=webapp_url))],
            [KeyboardButton("👤 Профиль"), KeyboardButton("🎁 Челленджи")],
            [KeyboardButton("↩️ В канал")],
        ],
        resize_keyboard=True
    )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    db_user = await get_user(user.id)

    if not db_user:
        await create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name
        )
        text_ = f"Добро пожаловать, {user.first_name}! 🖤\n\n+10 баллов за регистрацию ✨"
    else:
        await add_points(user.id, 5)
        text_ = f"С возвращением, {user.first_name}!\n+5 баллов за визит ✨"

    await update.message.reply_text(text_, reply_markup=get_main_keyboard())

async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    db_user = await get_user(user.id)

    if not db_user:
        await update.message.reply_text("Нажми /start для регистрации")
        return

    tier_emoji = {"free": "🥉", "premium": "🥈", "vip": "🥇"}
    tier_name = {"free": "Bronze", "premium": "Silver", "vip": "Gold VIP"}

    next_tier_points = {
        "free": (100, "Silver"),
        "premium": (500, "Gold VIP"),
        "vip": (1000, "Platinum"),
    }

    next_points, next_name = next_tier_points.get(db_user.tier, (0, "Max"))
    remaining = max(0, next_points - db_user.points)

    text_ = f"""\
👤 **Твой профиль**

{tier_emoji.get(db_user.tier, "🥉")} Уровень: {tier_name.get(db_user.tier, "Bronze")}
💎 Баллы: **{db_user.points}**

📊 До {next_name}: {remaining} баллов
📅 С нами: {db_user.joined_at.strftime("%d.%m.%Y")}

Продолжай активничать! 🚀
"""
    await update.message.reply_text(text_, parse_mode="Markdown")

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

async def start_telegram_bot():
    global tg_app, tg_task

    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN not set; starting API WITHOUT Telegram bot")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("profile", cmd_profile))

    tg_app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST, on_channel_post))
    tg_app.add_handler(MessageHandler(filters.UpdateType.EDITED_CHANNEL_POST, on_edited_channel_post))

    async def run():
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Telegram bot started (polling)")
        while True:
            await asyncio.sleep(3600)

    tg_task = asyncio.create_task(run())

async def stop_telegram_bot():
    global tg_app, tg_task
    if tg_task:
        tg_task.cancel()
        tg_task = None
    if tg_app:
        try:
            await tg_app.updater.stop()
            await tg_app.stop()
            await tg_app.shutdown()
            logger.info("✅ Telegram bot stopped")
        except Exception as e:
            logger.error("Error stopping bot: %s", e)
        finally:
            tg_app = None

# -----------------------------------------------------------------------------
# WEBAPP HTML (ДИЗАЙН/КНОПКИ НЕ ТРОГАЕМ — только:
# 1) убрали SephoraTR и SephoraGuide
# 2) добавили бренды
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
    const { useState, useEffect } = React;
    const tg = window.Telegram?.WebApp;

    if (tg) {
      tg.expand();
      tg.setHeaderColor("#0c0f14");
      tg.setBackgroundColor("#0c0f14");
    }

    const CHANNEL = "__CHANNEL__";

    const openLink = (url) => {
      if (tg?.openTelegramLink) tg.openTelegramLink(url);
      else window.open(url, "_blank");
    };

    const Hero = ({ user }) => (
      <div style={{
        border: "1px solid var(--stroke)",
        background: "linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05))",
        borderRadius: "22px",
        padding: "16px 14px",
        boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
        position: "relative",
        overflow: "hidden"
      }}>
        <div style={{
          position: "absolute", inset: "-2px",
          background: "radial-gradient(600px 300px at 10% 0%, rgba(230,193,128,0.26), transparent 60%)",
          pointerEvents: "none"
        }} />
        <div style={{ position: "relative" }}>
          <div style={{ fontSize: "20px", fontWeight: 650, letterSpacing: "0.2px" }}>NS · Natural Sense</div>
          <div style={{ marginTop: "6px", fontSize: "13px", color: "var(--muted)" }}>luxury beauty magazine</div>

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
                💎 {user.points} баллов • {(
                  { free: "🥉 Bronze", premium: "🥈 Silver", vip: "🥇 Gold VIP" }[user.tier]
                ) || "🥉 Bronze"}
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
        { id: "sephora", label: "Sephora" }
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

    const Button = ({ icon, label, onClick, subtitle }) => (
      <div
        onClick={onClick}
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
          cursor: "pointer"
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
        onClick={() => openLink(post.url)}
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
              background:"rgba(255,255,255,0.05)"
            }}>#{t}</div>
          ))}
        </div>
      </div>
    );

    const App = () => {
      const [activeTab, setActiveTab] = useState("home");
      const [user, setUser] = useState(null);

      // ✅ отдельный режим экрана "ПОСТЫ"
      const [postsMode, setPostsMode] = useState(false);
      const [selectedTag, setSelectedTag] = useState(null);
      const [posts, setPosts] = useState([]);
      const [loading, setLoading] = useState(false);

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

      useEffect(() => {
        if (tg?.initDataUnsafe?.user) {
          const tgUser = tg.initDataUnsafe.user;
          fetch(`/api/user/${tgUser.id}`)
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => setUser(data))
            .catch(() => setUser({
              telegram_id: tgUser.id,
              first_name: tgUser.first_name,
              points: 10,
              tier: "free"
            }));
        }
      }, []);

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
                <Button icon="🧴" label="Тип продукта / факты" onClick={() => openPosts("Факты")} />
                <Button icon="🧪" label="Составы продуктов" onClick={() => openPosts("Состав")} />
              </Panel>
            );

          case "brand":
            return (
              <Panel>
                <Button icon="✨" label="The Ordinary" onClick={() => openPosts("TheOrdinary")} />
                <Button icon="✨" label="Dior" onClick={() => openPosts("Dior")} />
                <Button icon="✨" label="Chanel" onClick={() => openPosts("Chanel")} />
                <Button icon="✨" label="Kylie Cosmetics" onClick={() => openPosts("KylieCosmetics")} />
                <Button icon="✨" label="Gisou" onClick={() => openPosts("Gisou")} />
                <Button icon="✨" label="Rare Beauty" onClick={() => openPosts("RareBeauty")} />
                <Button icon="✨" label="Yves Saint Laurent" onClick={() => openPosts("YSL")} />
                <Button icon="✨" label="Givenchy" onClick={() => openPosts("Givenchy")} />
                <Button icon="✨" label="Charlotte Tilbury" onClick={() => openPosts("CharlotteTilbury")} />
                <Button icon="✨" label="NARS" onClick={() => openPosts("NARS")} />
                <Button icon="✨" label="Sol de Janeiro" onClick={() => openPosts("SolDeJaneiro")} />
                <Button icon="✨" label="Huda Beauty" onClick={() => openPosts("HudaBeauty")} />
                <Button icon="✨" label="Rhode" onClick={() => openPosts("Rhode")} />
                <Button icon="✨" label="Tower 28 Beauty" onClick={() => openPosts("Tower28Beauty")} />
                <Button icon="✨" label="Benefit Cosmetics" onClick={() => openPosts("BenefitCosmetics")} />
                <Button icon="✨" label="Estée Lauder" onClick={() => openPosts("EsteeLauder")} />
                <Button icon="✨" label="Sisley" onClick={() => openPosts("Sisley")} />
                <Button icon="✨" label="Kérastase" onClick={() => openPosts("Kerastase")} />
                <Button icon="✨" label="Armani Beauty" onClick={() => openPosts("ArmaniBeauty")} />
                <Button icon="✨" label="Hourglass" onClick={() => openPosts("Hourglass")} />
                <Button icon="✨" label="Shiseido" onClick={() => openPosts("Shiseido")} />
                <Button icon="✨" label="Tom Ford Beauty" onClick={() => openPosts("TomFordBeauty")} />
                <Button icon="✨" label="Tarte" onClick={() => openPosts("Tarte")} />
                <Button icon="✨" label="Sephora Collection" onClick={() => openPosts("SephoraCollection")} />
                <Button icon="✨" label="Clinique" onClick={() => openPosts("Clinique")} />
                <Button icon="✨" label="Dolce & Gabbana" onClick={() => openPosts("DolceGabbana")} />
                <Button icon="✨" label="Kayali" onClick={() => openPosts("Kayali")} />
                <Button icon="✨" label="Guerlain" onClick={() => openPosts("Guerlain")} />
                <Button icon="✨" label="Fenty Beauty" onClick={() => openPosts("FentyBeauty")} />
                <Button icon="✨" label="Too Faced" onClick={() => openPosts("TooFaced")} />
                <Button icon="✨" label="MAKE UP FOR EVER" onClick={() => openPosts("MakeUpForEver")} />
                <Button icon="✨" label="Erborian" onClick={() => openPosts("Erborian")} />
                <Button icon="✨" label="Natasha Denona" onClick={() => openPosts("NatashaDenona")} />
                <Button icon="✨" label="Lancôme" onClick={() => openPosts("Lancome")} />
                <Button icon="✨" label="Kosas" onClick={() => openPosts("Kosas")} />
                <Button icon="✨" label="ONE/SIZE" onClick={() => openPosts("OneSize")} />
                <Button icon="✨" label="Laneige" onClick={() => openPosts("Laneige")} />
                <Button icon="✨" label="Makeup by Mario" onClick={() => openPosts("MakeupByMario")} />
                <Button icon="✨" label="Valentino Beauty" onClick={() => openPosts("ValentinoBeauty")} />
                <Button icon="✨" label="Drunk Elephant" onClick={() => openPosts("DrunkElephant")} />
                <Button icon="✨" label="Olaplex" onClick={() => openPosts("Olaplex")} />
                <Button icon="✨" label="Anastasia Beverly Hills" onClick={() => openPosts("AnastasiaBeverlyHills")} />
                <Button icon="✨" label="Amika" onClick={() => openPosts("Amika")} />
                <Button icon="✨" label="BYOMA" onClick={() => openPosts("BYOMA")} />
                <Button icon="✨" label="Glow Recipe" onClick={() => openPosts("GlowRecipe")} />
                <Button icon="✨" label="Milk Makeup" onClick={() => openPosts("MilkMakeup")} />
                <Button icon="✨" label="Summer Fridays" onClick={() => openPosts("SummerFridays")} />
                <Button icon="✨" label="K18" onClick={() => openPosts("K18")} />
              </Panel>
            );

          case "sephora":
            return (
              <Panel>
                <Button icon="🎁" label="Подарки / акции" onClick={() => openPosts("SephoraPromo")} />
              </Panel>
            );

          default:
            return null;
        }
      };

      return (
        <div style={{ padding:"18px 16px 26px", maxWidth:"520px", margin:"0 auto" }}>
          <Hero user={user} />
          <Tabs active={activeTab} onChange={changeTab} />
          {renderContent()}
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
    return html.replace("__CHANNEL__", CHANNEL_USERNAME)

# -----------------------------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sweeper_task
    await init_db()
    await start_telegram_bot()

    sweeper_task = asyncio.create_task(sweeper_loop())

    logger.info("✅ NS · Natural Sense started")
    yield

    if sweeper_task:
        sweeper_task.cancel()
        sweeper_task = None

    await stop_telegram_bot()
    logger.info("✅ NS · Natural Sense stopped")

app = FastAPI(title="NS · Natural Sense API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"app": "NS · Natural Sense", "status": "running", "version": "2.0.0"}

@app.get("/webapp", response_class=HTMLResponse)
async def webapp():
    return HTMLResponse(get_webapp_html())

@app.get("/api/user/{telegram_id}")
async def get_user_api(telegram_id: int):
    user = await get_user(telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "telegram_id": user.telegram_id,
        "username": user.username,
        "first_name": user.first_name,
        "tier": user.tier,
        "points": user.points,
        "favorites": user.favorites,
        "joined_at": user.joined_at.isoformat(),
    }

@app.post("/api/user/{telegram_id}/points")
async def add_points_api(telegram_id: int, points: int):
    user = await add_points(telegram_id, points)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "new_total": user.points, "tier": user.tier}

@app.get("/api/posts")
async def api_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    if not tag:
        # без тега не отдаём ничего — чтобы случайно нигде не показывались
        return []

    # полностью запрещаем эти два тега
    if tag in BLOCKED_TAGS:
        return []

    rows = await list_posts(tag=tag, limit=limit, offset=offset)
    out = []
    for p in rows:
        out.append({
            "message_id": p.message_id,
            "url": p.permalink or make_permalink(p.message_id),
            "tags": p.tags or [],
            "preview": preview_text(p.text),
        })
    return out

@app.get("/health")
async def health():
    return {"status": "healthy"}
