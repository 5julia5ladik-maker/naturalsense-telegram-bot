import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
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
    joined_at = Column(DateTime, default=datetime.utcnow)

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    channel_message_id = Column(Integer, unique=True, index=True, nullable=False)
    text = Column(String, nullable=True)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

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
# POSTS (TAGS) INDEX
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

def post_url(message_id: int) -> str:
    return f"https://t.me/{CHANNEL_USERNAME}/{message_id}"

def preview_text(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.strip())
    return (s[:limit] + "…") if len(s) > limit else s

async def save_channel_post(message_id: int, text: str | None):
    async with async_session_maker() as session:
        res = await session.execute(select(Post).where(Post.channel_message_id == message_id))
        p = res.scalar_one_or_none()

        if p:
            p.text = text
            p.tags = extract_tags(text)
            await session.commit()
            return p

        p = Post(
            channel_message_id=message_id,
            text=text,
            tags=extract_tags(text),
        )
        session.add(p)
        await session.commit()
        await session.refresh(p)
        logger.info("✅ Indexed channel post id=%s tags=%s", message_id, p.tags)
        return p

async def list_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    async with async_session_maker() as session:
        q = select(Post).order_by(Post.channel_message_id.desc()).limit(limit).offset(offset)
        rows = (await session.execute(q)).scalars().all()
        if tag:
            rows = [p for p in rows if tag in (p.tags or [])]
        return rows

# -----------------------------------------------------------------------------
# TELEGRAM BOT (polling) + channel posts handler
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
        db_user = await create_user(
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
    """
    Bot (as admin in channel) receives channel posts here.
    Index by message_id and tags from text/caption.
    """
    msg = update.channel_post
    if not msg:
        return

    text = msg.text or msg.caption or None
    await save_channel_post(msg.message_id, text)

async def start_telegram_bot():
    global tg_app, tg_task

    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN not set; starting API WITHOUT Telegram bot")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("profile", cmd_profile))

    # IMPORTANT: index channel posts (bot must be admin in channel)
    tg_app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST, on_channel_post))

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
# WEBAPP HTML (NO f-string; safe placeholders)
# -----------------------------------------------------------------------------
def get_webapp_html():
    html = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <title>NS · Natural Sense</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    :root {
      --bg:#0c0f14;
      --text:rgba(255,255,255,0.92);
      --muted:rgba(255,255,255,0.60);
      --stroke:rgba(255,255,255,0.10);
      --card:rgba(255,255,255,0.06);
      --gold:rgba(230,193,128,0.9);
    }
    body {
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
      background: radial-gradient(1200px 800px at 20% 10%, rgba(230,193,128,0.18), transparent 60%), var(--bg);
      color:var(--text);
      overflow-x:hidden;
    }
    #root { min-height:100vh; }
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

    const CHANNEL = "__CHANNEL_USERNAME__";

    const openLink = (url) => {
      if (tg?.openTelegramLink) tg.openTelegramLink(url);
      else window.open(url, "_blank");
    };

    const Chip = ({ text, onClick, active }) => (
      <div onClick={onClick} style={{
        padding:"8px 10px",
        borderRadius:"999px",
        fontSize:"12px",
        cursor:"pointer",
        userSelect:"none",
        border: active ? "1px solid rgba(230,193,128,0.45)" : "1px solid var(--stroke)",
        background: active ? "rgba(230,193,128,0.12)" : "rgba(255,255,255,0.06)",
        color: active ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.85)"
      }}>
        {text}
      </div>
    );

    const Card = ({ post }) => (
      <div onClick={() => openLink(post.url)} style={{
        border:"1px solid var(--stroke)",
        background:"var(--card)",
        borderRadius:"18px",
        padding:"14px",
        marginTop:"10px",
        cursor:"pointer"
      }}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <div style={{fontSize:"13px",color:"var(--muted)"}}>📝 Пост</div>
          <div style={{fontSize:"12px",color:"var(--muted)"}}>#{post.channel_message_id}</div>
        </div>

        <div style={{marginTop:"10px",fontSize:"14px",lineHeight:1.35,whiteSpace:"pre-wrap"}}>
          {post.preview || "Открыть пост →"}
        </div>

        <div style={{marginTop:"10px",display:"flex",gap:"8px",flexWrap:"wrap"}}>
          {(post.tags || []).slice(0,6).map(t => (
            <div key={t} style={{
              fontSize:"12px",
              padding:"6px 8px",
              borderRadius:"999px",
              border:"1px solid var(--stroke)",
              background:"rgba(255,255,255,0.05)",
              color:"rgba(255,255,255,0.85)"
            }}>#{t}</div>
          ))}
        </div>

        <div style={{marginTop:"12px",fontSize:"13px",color:"rgba(230,193,128,0.9)"}}>
          Открыть в канале →
        </div>
      </div>
    );

    const App = () => {
      const [activeTab, setActiveTab] = useState("home");
      const [tag, setTag] = useState(null);
      const [posts, setPosts] = useState([]);
      const [loading, setLoading] = useState(false);

      const tabs = useMemo(() => ([
        { id: "home", label: "Главное" },
        { id: "cat", label: "Категории" },
        { id: "brand", label: "Бренды" },
        { id: "sephora", label: "Sephora" },
      ]), []);

      const loadPosts = (t) => {
        setLoading(true);
        const url = t ? `/api/posts?tag=${encodeURIComponent(t)}` : `/api/posts`;
        fetch(url)
          .then(r => r.json())
          .then(data => setPosts(Array.isArray(data) ? data : []))
          .catch(() => setPosts([]))
          .finally(() => setLoading(false));
      };

      useEffect(() => {
        // load latest on start
        loadPosts(null);
      }, []);

      const openTag = (t) => {
        setTag(t);
        loadPosts(t);
      };

      const renderContent = () => {
        switch (activeTab) {
          case "home":
            return (
              <div style={{
                marginTop:"14px",
                border:"1px solid var(--stroke)",
                background:"rgba(255,255,255,0.05)",
                borderRadius:"22px",
                padding:"12px"
              }}>
                <div style={{fontSize:"13px", color:"var(--muted)"}}>Последние посты</div>
                {loading && <div style={{marginTop:"10px", color:"var(--muted)"}}>Загрузка…</div>}
                {!loading && posts.map(p => <Card key={p.id} post={p} />)}
              </div>
            );

          case "cat":
            return (
              <div style={{
                marginTop:"14px",
                border:"1px solid var(--stroke)",
                background:"rgba(255,255,255,0.05)",
                borderRadius:"22px",
                padding:"12px"
              }}>
                <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
                  <Chip text="🆕 Новинка" active={tag==="Новинка"} onClick={() => openTag("Новинка")} />
                  <Chip text="💎 Люкс" active={tag==="Люкс"} onClick={() => openTag("Люкс")} />
                  <Chip text="🔥 Тренд" active={tag==="Тренд"} onClick={() => openTag("Тренд")} />
                  <Chip text="🏛 История" active={tag==="История"} onClick={() => openTag("История")} />
                  <Chip text="⭐ Оценка" active={tag==="Оценка"} onClick={() => openTag("Оценка")} />
                  <Chip text="🧴 Факты" active={tag==="Факты"} onClick={() => openTag("Факты")} />
                  <Chip text="🧪 Состав" active={tag==="Состав"} onClick={() => openTag("Состав")} />
                </div>

                {loading && <div style={{marginTop:"10px", color:"var(--muted)"}}>Загрузка…</div>}
                {!loading && posts.length === 0 && (
                  <div style={{marginTop:"10px", color:"var(--muted)"}}>
                    Нет постов по тегу #{tag}. Бот индексирует только новые посты после запуска.
                  </div>
                )}
                {!loading && posts.map(p => <Card key={p.id} post={p} />)}
              </div>
            );

          case "brand":
            return (
              <div style={{
                marginTop:"14px",
                border:"1px solid var(--stroke)",
                background:"rgba(255,255,255,0.05)",
                borderRadius:"22px",
                padding:"12px"
              }}>
                <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
                  <Chip text="✨ Dior" active={tag==="Dior"} onClick={() => openTag("Dior")} />
                  <Chip text="✨ Chanel" active={tag==="Chanel"} onClick={() => openTag("Chanel")} />
                  <Chip text="✨ CharlotteTilbury" active={tag==="CharlotteTilbury"} onClick={() => openTag("CharlotteTilbury")} />
                </div>

                {loading && <div style={{marginTop:"10px", color:"var(--muted)"}}>Загрузка…</div>}
                {!loading && posts.length === 0 && (
                  <div style={{marginTop:"10px", color:"var(--muted)"}}>
                    Нет постов по тегу #{tag}.
                  </div>
                )}
                {!loading && posts.map(p => <Card key={p.id} post={p} />)}
              </div>
            );

          case "sephora":
            return (
              <div style={{
                marginTop:"14px",
                border:"1px solid var(--stroke)",
                background:"rgba(255,255,255,0.05)",
                borderRadius:"22px",
                padding:"12px"
              }}>
                <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
                  <Chip text="🇹🇷 SephoraTR" active={tag==="SephoraTR"} onClick={() => openTag("SephoraTR")} />
                  <Chip text="🎁 SephoraPromo" active={tag==="SephoraPromo"} onClick={() => openTag("SephoraPromo")} />
                  <Chip text="🧾 SephoraGuide" active={tag==="SephoraGuide"} onClick={() => openTag("SephoraGuide")} />
                </div>

                {loading && <div style={{marginTop:"10px", color:"var(--muted)"}}>Загрузка…</div>}
                {!loading && posts.length === 0 && (
                  <div style={{marginTop:"10px", color:"var(--muted)"}}>
                    Нет постов по тегу #{tag}.
                  </div>
                )}
                {!loading && posts.map(p => <Card key={p.id} post={p} />)}
              </div>
            );

          default:
            return null;
        }
      };

      return (
        <div style={{ padding:"18px 16px 26px", maxWidth:"520px", margin:"0 auto" }}>
          <div style={{
            border:"1px solid var(--stroke)",
            background:"linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05))",
            borderRadius:"22px",
            padding:"16px 14px",
            boxShadow:"0 10px 30px rgba(0,0,0,0.35)"
          }}>
            <div style={{ fontSize:"20px", fontWeight:650 }}>NS · Natural Sense</div>
            <div style={{ marginTop:"6px", fontSize:"13px", color:"var(--muted)" }}>
              Мини-журнал: кликай категории/бренды → смотри посты с #тегом
            </div>
          </div>

          <div style={{ display:"flex", gap:"8px", marginTop:"14px" }}>
            {tabs.map(t => (
              <div
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                style={{
                  flex: 1,
                  border: activeTab === t.id ? "1px solid rgba(230,193,128,0.40)" : "1px solid var(--stroke)",
                  background: activeTab === t.id ? "rgba(230,193,128,0.12)" : "rgba(255,255,255,0.06)",
                  padding: "10px",
                  borderRadius: "14px",
                  fontSize: "13px",
                  textAlign: "center",
                  cursor: "pointer",
                  userSelect: "none",
                }}
              >
                {t.label}
              </div>
            ))}
          </div>

          {renderContent()}

          <div style={{ marginTop:"20px", color:"var(--muted)", fontSize:"12px", textAlign:"center" }}>
            Канал: @{CHANNEL}
          </div>
        </div>
      );
    };

    ReactDOM.render(<App />, document.getElementById("root"));
  </script>
</body>
</html>
"""
    return html.replace("__CHANNEL_USERNAME__", CHANNEL_USERNAME)

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

app = FastAPI(title="NS · Natural Sense API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMIDDLEWARE,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"app": "NS · Natural Sense", "status": "running", "version": "2.1.0"}

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
    rows = await list_posts(tag=tag, limit=limit, offset=offset)
    return [
        {
            "id": p.id,
            "channel_message_id": p.channel_message_id,
            "url": post_url(p.channel_message_id),
            "tags": p.tags or [],
            "preview": preview_text(p.text, 160),
            "created_at": p.created_at.isoformat(),
        }
        for p in rows
    ]

@app.get("/health")
async def health():
    return {"status": "healthy"}
