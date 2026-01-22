import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
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

from sqlalchemy import Column, Integer, String, DateTime, JSON, select
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

# Fix Railway postgres schemes for async SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

tok = BOT_TOKEN or ""
logger.info(
    "ENV CHECK: BOT_TOKEN_present=%s BOT_TOKEN_len=%s PUBLIC_BASE_URL_present=%s DATABASE_URL_present=%s",
    bool(BOT_TOKEN), len(tok), bool(PUBLIC_BASE_URL), bool(DATABASE_URL)
)

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
    joined_at = Column(DateTime, default=lambda: datetime.utcnow())

class Post(Base):
    """
    Под твою реальную таблицу Railway:
    id, message_id, date, text, media_type, media_file_id, permalink, tags, created_at
    """
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, unique=True, index=True, nullable=False)

    # TIMESTAMP WITHOUT TIME ZONE -> кладём naive UTC
    date = Column(DateTime, nullable=True)

    text = Column(String, nullable=True)
    media_type = Column(String, nullable=True)
    media_file_id = Column(String, nullable=True)
    permalink = Column(String, nullable=True)

    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.utcnow())

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
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
# POSTS / TAGS
# -----------------------------------------------------------------------------
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")

def extract_tags(text: str | None) -> list[str]:
    if not text:
        return []
    tags = [m.group(1) for m in TAG_RE.finditer(text)]
    out, seen = [], set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def build_permalink(message_id: int) -> str:
    return f"https://t.me/{CHANNEL_USERNAME}/{message_id}"

def preview_text(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.strip())
    return (s[:limit] + "…") if len(s) > limit else s

def to_naive_utc(dt: datetime | None) -> datetime | None:
    """
    Telegram msg.date -> timezone-aware UTC.
    В Postgres TIMESTAMP WITHOUT TIME ZONE -> кладём naive UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

async def upsert_post_from_channel(
    message_id: int,
    date_dt: datetime | None,
    text: str | None,
    media_type: str | None,
    media_file_id: str | None,
):
    tags = extract_tags(text)
    permalink = build_permalink(message_id)
    now = datetime.utcnow()  # naive UTC
    date_naive = to_naive_utc(date_dt)

    async with async_session_maker() as session:
        res = await session.execute(select(Post).where(Post.message_id == message_id))
        p = res.scalar_one_or_none()

        if p:
            p.date = date_naive
            p.text = text
            p.media_type = media_type
            p.media_file_id = media_file_id
            p.permalink = permalink
            p.tags = tags
            p.created_at = now
            await session.commit()
            return p

        p = Post(
            message_id=message_id,
            date=date_naive,
            text=text,
            media_type=media_type,
            media_file_id=media_file_id,
            permalink=permalink,
            tags=tags,
            created_at=now,
        )
        session.add(p)
        await session.commit()
        await session.refresh(p)
        logger.info("✅ Indexed post %s tags=%s", message_id, tags)
        return p

async def list_posts(tag: str, limit: int = 100, offset: int = 0):
    """
    Берём последние по message_id desc, фильтруем по tag в Python.
    """
    async with async_session_maker() as session:
        q = select(Post).order_by(Post.message_id.desc()).limit(limit).offset(offset)
        rows = (await session.execute(q)).scalars().all()
        rows = [p for p in rows if tag in (p.tags or [])]
        return rows

# -----------------------------------------------------------------------------
# TELEGRAM BOT
# -----------------------------------------------------------------------------
tg_app: Application | None = None
tg_task: asyncio.Task | None = None

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
        text = f"Добро пожаловать, {user.first_name}! 🖤\n\n+10 баллов за регистрацию ✨"
    else:
        await add_points(user.id, 5)
        text = f"С возвращением, {user.first_name}!\n+5 баллов за визит ✨"

    await update.message.reply_text(text, reply_markup=get_main_keyboard())

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

    text = f"""\
👤 **Твой профиль**

{tier_emoji.get(db_user.tier, "🥉")} Уровень: {tier_name.get(db_user.tier, "Bronze")}
💎 Баллы: **{db_user.points}**

📊 До {next_name}: {remaining} баллов
📅 С нами: {db_user.joined_at.strftime("%d.%m.%Y")}

Продолжай активничать! 🚀
"""
    await update.message.reply_text(text, parse_mode="Markdown")

async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return

    text = msg.text or msg.caption or None
    media_type = None
    media_file_id = None

    if msg.photo:
        media_type = "photo"
        media_file_id = msg.photo[-1].file_id
    elif msg.video:
        media_type = "video"
        media_file_id = msg.video.file_id
    elif msg.document:
        media_type = "document"
        media_file_id = msg.document.file_id

    await upsert_post_from_channel(
        message_id=msg.message_id,
        date_dt=msg.date,
        text=text,
        media_type=media_type,
        media_file_id=media_file_id,
    )

async def on_edited_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.edited_channel_post
    if not msg:
        return

    text = msg.text or msg.caption or None
    media_type = None
    media_file_id = None

    if msg.photo:
        media_type = "photo"
        media_file_id = msg.photo[-1].file_id
    elif msg.video:
        media_type = "video"
        media_file_id = msg.video.file_id
    elif msg.document:
        media_type = "document"
        media_file_id = msg.document.file_id

    await upsert_post_from_channel(
        message_id=msg.message_id,
        date_dt=msg.date,
        text=text,
        media_type=media_type,
        media_file_id=media_file_id,
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
# WEBAPP HTML (ДИЗАЙН СОХРАНЁН, ПОСТЫ ТОЛЬКО ПО НАЖАТИЮ)
# -----------------------------------------------------------------------------
def get_webapp_html() -> str:
    html = """<!DOCTYPE html>
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
                💎 {user.points} баллов • {
                  ({
                    free: "🥉 Bronze",
                    premium: "🥈 Silver",
                    vip: "🥇 Gold VIP"
                  }[user.tier]) || "🥉 Bronze"
                }
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
        onClick={() => openLink(post.permalink || post.url)}
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
          {(post.tags && post.tags.length) ? ("#" + post.tags[0]) : "Пост"} • ID {post.message_id}
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

    const PostsScreen = ({ title, posts, loading }) => (
      <Panel>
        <div style={{
          padding: "6px 4px 10px",
          borderBottom: "1px solid var(--stroke)",
          marginBottom: "10px"
        }}>
          <div style={{ fontSize: "16px", fontWeight: 650 }}>{title}</div>
        </div>

        {loading && (
          <div style={{ color: "var(--muted)", fontSize: "13px", padding: "6px 4px" }}>
            Загрузка…
          </div>
        )}

        {!loading && posts.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: "13px", padding: "6px 4px" }}>
            Пока нет постов по этому тегу.
          </div>
        )}

        {!loading && posts.map(p => <PostCard key={p.id || p.message_id} post={p} />)}
      </Panel>
    );

    const App = () => {
      const [activeTab, setActiveTab] = useState("home");
      const [user, setUser] = useState(null);

      // экран: "menu" или "posts"
      const [screen, setScreen] = useState("menu");

      const [selectedTag, setSelectedTag] = useState(null);
      const [selectedTitle, setSelectedTitle] = useState("");
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

      const openPosts = (tag, title) => {
        setSelectedTag(tag);
        setSelectedTitle(title);
        setPosts([]);
        setScreen("posts");
        loadPosts(tag);
      };

      // при смене вкладки — возвращаемся в меню вкладки (без постов)
      const changeTab = (tabId) => {
        setActiveTab(tabId);
        setScreen("menu");
        setSelectedTag(null);
        setSelectedTitle("");
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

      const renderMenu = () => {
        switch (activeTab) {
          case "home":
            return (
              <Panel>
                <Button icon="📂" label="Категории" onClick={() => changeTab("cat")} />
                <Button icon="🏷" label="Бренды" onClick={() => changeTab("brand")} />
                <Button icon="💸" label="Sephora" onClick={() => changeTab("sephora")} />
                <Button icon="💎" label="Beauty Challenges" onClick={() => openPosts("Challenge", "Посты Challenge")} />
                <Button icon="↩️" label="В канал" onClick={() => openLink(`https://t.me/${CHANNEL}`)} />
              </Panel>
            );

          case "cat":
            return (
              <Panel>
                <Button icon="🆕" label="Новинка" onClick={() => openPosts("Новинка", "Посты Новинка")} />
                <Button icon="💎" label="Кратко о люкс продукте" onClick={() => openPosts("Люкс", "Посты Люкс")} />
                <Button icon="🔥" label="Тренд" onClick={() => openPosts("Тренд", "Посты Тренд")} />
                <Button icon="🏛" label="История бренда" onClick={() => openPosts("История", "Посты История")} />
                <Button icon="⭐" label="Личная оценка" onClick={() => openPosts("Оценка", "Посты Оценка")} />
                <Button icon="🧴" label="Тип продукта / факты" onClick={() => openPosts("Факты", "Посты Факты")} />
                <Button icon="🧪" label="Составы продуктов" onClick={() => openPosts("Состав", "Посты Состав")} />
              </Panel>
            );

          case "brand":
            return (
              <Panel>
                <Button icon="✨" label="Dior" onClick={() => openPosts("Dior", "Посты Dior")} />
                <Button icon="✨" label="Chanel" onClick={() => openPosts("Chanel", "Посты Chanel")} />
                <Button icon="✨" label="Charlotte Tilbury" onClick={() => openPosts("CharlotteTilbury", "Посты Charlotte Tilbury")} />
              </Panel>
            );

          case "sephora":
            return (
              <Panel>
                <Button icon="🇹🇷" label="Актуальные цены (TR)" subtitle="Ежедневное обновление"
                  onClick={() => openPosts("SephoraTR", "Посты Sephora TR")} />
                <Button icon="🎁" label="Подарки / акции" onClick={() => openPosts("SephoraPromo", "Посты Sephora Promo")} />
                <Button icon="🧾" label="Гайды / как покупать" onClick={() => openPosts("SephoraGuide", "Посты Sephora Guide")} />
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

          {screen === "menu" && renderMenu()}

          {screen === "posts" && (
            <PostsScreen
              title={selectedTitle || (selectedTag ? ("Посты " + selectedTag) : "Посты")}
              posts={posts}
              loading={loading}
            />
          )}

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
    await init_db()
    await start_telegram_bot()
    logger.info("✅ NS · Natural Sense started")
    yield
    await stop_telegram_bot()
    logger.info("✅ NS · Natural Sense stopped")

app = FastAPI(title="NS · Natural Sense API", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"app": "NS · Natural Sense", "status": "running", "version": "3.1.0"}

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

@app.get("/api/posts")
async def api_posts(
    tag: str = Query(..., description="Tag without #, example: Dior"),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    rows = await list_posts(tag=tag, limit=limit, offset=offset)
    return [
        {
            "id": p.id,
            "message_id": p.message_id,
            "date": p.date.isoformat() if p.date else None,
            "text": p.text,
            "preview": preview_text(p.text),
            "media_type": p.media_type,
            "media_file_id": p.media_file_id,
            "permalink": p.permalink or build_permalink(p.message_id),
            "url": p.permalink or build_permalink(p.message_id),
            "tags": p.tags or [],
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in rows
    ]

@app.get("/health")
async def health():
    return {"status": "healthy"}
