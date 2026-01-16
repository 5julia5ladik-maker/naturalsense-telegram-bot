import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from sqlalchemy import Column, Integer, String, DateTime, JSON, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL", "") or "").rstrip("/")
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "NaturalSense") or "NaturalSense"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ns.db") or "sqlite+aiosqlite:///./ns.db"

# Fix Railway postgres scheme for async SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

logger.info(
    "ENV CHECK: BOT_TOKEN_present=%s PUBLIC_BASE_URL=%s CHANNEL=%s DB=%s",
    bool(BOT_TOKEN),
    "yes" if PUBLIC_BASE_URL else "no",
    CHANNEL_USERNAME,
    "yes" if DATABASE_URL else "no",
)

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
Base = declarative_base()

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    channel_message_id = Column(Integer, index=True, unique=True, nullable=False)
    text = Column(String, nullable=True)
    media_type = Column(String, nullable=True)  # text/photo/video/document
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionMaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database initialized")

# -----------------------------------------------------------------------------
# TAGS + HELPERS
# -----------------------------------------------------------------------------
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")

def extract_tags(text: str | None) -> list[str]:
    if not text:
        return []
    raw = [m.group(1) for m in TAG_RE.finditer(text)]
    out, seen = [], set()
    for t in raw:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def make_post_url(message_id: int) -> str:
    return f"https://t.me/{CHANNEL_USERNAME}/{message_id}"

def make_preview(text: str | None, limit: int = 140) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.strip())
    return (s[:limit] + "…") if len(s) > limit else s

async def save_channel_post(message_id: int, text: str | None, media_type: str):
    async with SessionMaker() as session:
        res = await session.execute(select(Post).where(Post.channel_message_id == message_id))
        post = res.scalar_one_or_none()

        if post:
            post.text = text
            post.media_type = media_type
            post.tags = extract_tags(text)
            await session.commit()
            return post

        post = Post(
            channel_message_id=message_id,
            text=text,
            media_type=media_type,
            tags=extract_tags(text),
        )
        session.add(post)
        await session.commit()
        await session.refresh(post)
        logger.info("✅ Indexed post %s tags=%s", message_id, post.tags)
        return post

async def list_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    async with SessionMaker() as session:
        q = select(Post).order_by(Post.channel_message_id.desc()).limit(limit).offset(offset)
        rows = (await session.execute(q)).scalars().all()
        if tag:
            rows = [p for p in rows if tag in (p.tags or [])]
        return rows

# -----------------------------------------------------------------------------
# TELEGRAM BOT (WEBHOOK)
# -----------------------------------------------------------------------------
tg_app: Application | None = None

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("NS · Natural Sense 🖤\nОткрой Mini App через Menu Button.")

async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Профиль позже. Сейчас журнал постов по тегам 🖤")

async def start_bot_webhook():
    """Init bot and set webhook. Never crash the API if token missing."""
    global tg_app
    if not BOT_TOKEN or not PUBLIC_BASE_URL:
        logger.error("❌ BOT_TOKEN or PUBLIC_BASE_URL missing -> bot webhook not started")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("profile", cmd_profile))

    await tg_app.initialize()
    await tg_app.start()

    webhook_url = f"{PUBLIC_BASE_URL}/telegram/webhook"
    await tg_app.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
    logger.info("✅ Telegram webhook set: %s", webhook_url)

async def stop_bot():
    global tg_app
    if not tg_app:
        return
    try:
        await tg_app.bot.delete_webhook(drop_pending_updates=False)
        await tg_app.stop()
        await tg_app.shutdown()
        logger.info("✅ Telegram bot stopped")
    except Exception as e:
        logger.error("Error stopping bot: %s", e)
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
    const { useEffect, useMemo, useState } = React;
    const tg = window.Telegram?.WebApp;

    if (tg) {
      tg.expand();
      tg.setHeaderColor("#0c0f14");
      tg.setBackgroundColor("#0c0f14");
    }

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
          <div style={{fontSize:"13px",color:"var(--muted)"}}>
            {post.media_type === "video" ? "🎬 Видео" : post.media_type === "photo" ? "🖼 Фото" : "📝 Пост"}
          </div>
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
      const [tag, setTag] = useState(null);
      const [posts, setPosts] = useState([]);
      const [loading, setLoading] = useState(false);

      const chips = useMemo(() => ([
        { id: null, label: "Главное" },
        { id: "Новинка", label: "🆕 Новинка" },
        { id: "Тренд", label: "🔥 Тренд" },
        { id: "Оценка", label: "⭐ Оценка" },
        { id: "Dior", label: "✨ Dior" },
        { id: "Chanel", label: "✨ Chanel" },
        { id: "SephoraTR", label: "🇹🇷 Sephora TR" },
        { id: "SephoraPromo", label: "🎁 Sephora Promo" },
      ]), []);

      const load = (t) => {
        setLoading(true);
        const url = t ? `/api/posts?tag=${encodeURIComponent(t)}` : `/api/posts`;
        fetch(url)
          .then(r => r.json())
          .then(data => setPosts(Array.isArray(data) ? data : []))
          .catch(() => setPosts([]))
          .finally(() => setLoading(false));
      };

      useEffect(() => { load(tag); }, [tag]);

      return (
        <div style={{padding:"18px 16px 26px",maxWidth:"520px",margin:"0 auto"}}>
          <div style={{
            border:"1px solid var(--stroke)",
            background:"linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05))",
            borderRadius:"22px",
            padding:"16px 14px",
            boxShadow:"0 10px 30px rgba(0,0,0,0.35)"
          }}>
            <div style={{fontSize:"20px",fontWeight:650}}>NS · Natural Sense</div>
            <div style={{marginTop:"6px",fontSize:"13px",color:"var(--muted)"}}>
              Лента постов по тегам • нажми тег и смотри
            </div>
          </div>

          <div style={{marginTop:"14px",display:"flex",gap:"8px",flexWrap:"wrap"}}>
            {chips.map(c => (
              <Chip
                key={String(c.id)}
                text={c.label}
                active={c.id === tag}
                onClick={() => setTag(c.id)}
              />
            ))}
          </div>

          <div style={{marginTop:"14px"}}>
            {loading && <div style={{color:"var(--muted)",fontSize:"13px"}}>Загрузка…</div>}
            {!loading && posts.length === 0 && (
              <div style={{color:"var(--muted)",fontSize:"13px"}}>
                Пока нет постов по этому тегу. Бот индексирует только новые посты после запуска.
              </div>
            )}
            {posts.map(p => <Card key={p.id} post={p} />)}
          </div>

          <div style={{marginTop:"18px",color:"var(--muted)",fontSize:"12px",textAlign:"center"}}>
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
    return html

# -----------------------------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await start_bot_webhook()
    logger.info("✅ NS started")
    yield
    await stop_bot()
    logger.info("✅ NS stopped")

app = FastAPI(title="NS · Natural Sense", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/webapp", response_class=HTMLResponse)
async def webapp():
    return HTMLResponse(get_webapp_html())

# Telegram webhook endpoint
@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    if not tg_app:
        return {"ok": False, "error": "bot not initialized"}

    data = await req.json()
    update = Update.de_json(data, tg_app.bot)

    # process commands etc
    await tg_app.process_update(update)

    # index channel posts
    msg = None
    if update.channel_post:
        msg = update.channel_post
    elif update.edited_channel_post:
        msg = update.edited_channel_post

    if msg:
        text = msg.text or msg.caption or None
        if msg.video:
            media_type = "video"
        elif msg.photo:
            media_type = "photo"
        elif msg.document:
            media_type = "document"
        else:
            media_type = "text"

        await save_channel_post(msg.message_id, text, media_type)

    return {"ok": True}

# API for Mini App
@app.get("/api/posts")
async def api_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    rows = await list_posts(tag=tag, limit=limit, offset=offset)
    return [
        {
            "id": p.id,
            "channel_message_id": p.channel_message_id,
            "url": make_post_url(p.channel_message_id),
            "tags": p.tags or [],
            "media_type": p.media_type or "text",
            "preview": make_preview(p.text),
            "created_at": p.created_at.isoformat(),
        }
        for p in rows
    ]
