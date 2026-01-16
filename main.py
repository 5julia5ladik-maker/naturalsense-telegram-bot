import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
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
# CONFIG (ENV)
# -----------------------------------------------------------------------------
def env_get(name: str, default: str | None = None) -> str | None:
    """Read env var; also tries trimmed key as safety."""
    v = os.getenv(name)
    if v is not None:
        return v
    # safety: sometimes people accidentally create keys with spaces in name in UI.
    # This won't fix that, but we keep it simple and explicit.
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

# ENV diagnostics (safe)
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

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database initialized")

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

        # Auto tier upgrade
        if user.points >= 500:
            user.tier = "vip"
        elif user.points >= 100:
            user.tier = "premium"

        await session.commit()
        await session.refresh(user)
        return user

# -----------------------------------------------------------------------------
# TELEGRAM BOT
# -----------------------------------------------------------------------------
tg_app: Application | None = None
tg_task: asyncio.Task | None = None

def get_main_keyboard():
    # If PUBLIC_BASE_URL not set, webapp button still shown but opens relative path (can be useless)
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

async def cmd_profile(update: Update, context: ContextTypes.DEFAULTi_TYPE if False else ContextTypes.DEFAULT_TYPE):
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

async def start_telegram_bot():
    """Start bot polling only if BOT_TOKEN is present. Never crash the API."""
    global tg_app, tg_task

    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN not set; starting API WITHOUT Telegram bot")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("profile", cmd_profile))

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
# WEBAPP HTML (React via CDN)
# -----------------------------------------------------------------------------
def get_webapp_html():
    return f"""<!DOCTYPE html>
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
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    :root {{
      --bg: #0c0f14;
      --card: rgba(255,255,255,0.08);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.60);
      --gold: rgba(230, 193, 128, 0.9);
      --stroke: rgba(255,255,255,0.10);
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: radial-gradient(1200px 800px at 20% 10%, rgba(230,193,128,0.18), transparent 60%),
                  var(--bg);
      color: var(--text);
      overflow-x: hidden;
    }}
    #root {{ min-height: 100vh; }}
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const {{ useState, useEffect }} = React;
    const tg = window.Telegram?.WebApp;

    if (tg) {{
      tg.expand();
      tg.setHeaderColor("#0c0f14");
      tg.setBackgroundColor("#0c0f14");
    }}

    const CHANNEL = "{CHANNEL_USERNAME}";

    const openLink = (url) => {{
      if (tg?.openTelegramLink) tg.openTelegramLink(url);
      else window.open(url, "_blank");
    }};

    const searchLink = (tag) => {{
      const clean = tag.startsWith("#") ? tag.slice(1) : tag;
      return `https://t.me/${{CHANNEL}}?q=%23${{clean}}`;
    }};

    const Hero = ({{ user }}) => (
      <div style={{{{
        border: "1px solid var(--stroke)",
        background: "linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.05))",
        borderRadius: "22px",
        padding: "16px 14px",
        boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
        position: "relative",
        overflow: "hidden"
      }}}}>
        <div style={{{{
          position: "absolute", inset: "-2px",
          background: "radial-gradient(600px 300px at 10% 0%, rgba(230,193,128,0.26), transparent 60%)",
          pointerEvents: "none"
        }}}} />
        <div style={{{{ position: "relative" }}}}>
          <div style={{{{ fontSize: "20px", fontWeight: 650, letterSpacing: "0.2px" }}}}>NS · Natural Sense</div>
          <div style={{{{ marginTop: "6px", fontSize: "13px", color: "var(--muted)" }}}}>luxury beauty magazine</div>

          {{user && (
            <div style={{{{
              marginTop: "14px",
              padding: "12px",
              background: "rgba(230, 193, 128, 0.1)",
              borderRadius: "14px",
              border: "1px solid rgba(230, 193, 128, 0.2)"
            }}}}>
              <div style={{{{ fontSize: "13px", color: "var(--muted)" }}}}>Привет, {{user.first_name}}!</div>
              <div style={{{{ fontSize: "16px", fontWeight: 600, marginTop: "4px" }}}}>
                💎 {{user.points}} баллов • {{
                  ({{
                    free: "🥉 Bronze",
                    premium: "🥈 Silver",
                    vip: "🥇 Gold VIP"
                  }}[user.tier]) || "🥉 Bronze"
                }}
              </div>
            </div>
          )}}
        </div>
      </div>
    );

    const Tabs = ({{ active, onChange }}) => {{
      const tabs = [
        {{ id: "home", label: "Главное" }},
        {{ id: "cat", label: "Категории" }},
        {{ id: "brand", label: "Бренды" }},
        {{ id: "sephora", label: "Sephora" }}
      ];
      return (
        <div style={{{{ display: "flex", gap: "8px", marginTop: "14px" }}}}>
          {{tabs.map(tab => (
            <div
              key={{tab.id}}
              onClick={{() => onChange(tab.id)}}
              style={{{{
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
              }}}}
            >
              {{tab.label}}
            </div>
          ))}}
        </div>
      );
    }};

    const Button = ({{ icon, label, onClick, subtitle }}) => (
      <div
        onClick={{onClick}}
        style={{{{
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
        }}}}
      >
        <div>
          <div>{{icon}} {{label}}</div>
          {{subtitle && <div style={{{{ fontSize:"12px", color:"var(--muted)", marginTop:"4px" }}}}>{{subtitle}}</div>}}
        </div>
        <span style={{{{ opacity: 0.8 }}}}>›</span>
      </div>
    );

    const Panel = ({{ children }}) => (
      <div style={{{{
        marginTop: "14px",
        border: "1px solid var(--stroke)",
        background: "rgba(255,255,255,0.05)",
        borderRadius: "22px",
        padding: "12px"
      }}}}>
        {{children}}
      </div>
    );

    const App = () => {{
      const [activeTab, setActiveTab] = useState("home");
      const [user, setUser] = useState(null);

      useEffect(() => {{
        if (tg?.initDataUnsafe?.user) {{
          const tgUser = tg.initDataUnsafe.user;
          fetch(`/api/user/${{tgUser.id}}`)
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => setUser(data))
            .catch(() => setUser({{
              telegram_id: tgUser.id,
              first_name: tgUser.first_name,
              points: 10,
              tier: "free"
            }}));
        }}
      }}, []);

      const renderContent = () => {{
        switch (activeTab) {{
          case "home":
            return (
              <Panel>
                <Button icon="📂" label="Категории" onClick={{() => setActiveTab("cat")}} />
                <Button icon="🏷" label="Бренды" onClick={{() => setActiveTab("brand")}} />
                <Button icon="💸" label="Sephora" onClick={{() => setActiveTab("sephora")}} />
                <Button icon="💎" label="Beauty Challenges" onClick={{() => openLink(`https://t.me/${{CHANNEL}}?q=%23Challenge`)}} />
                <Button icon="↩️" label="В канал" onClick={{() => openLink(`https://t.me/${{CHANNEL}}`)}} />
              </Panel>
            );
          case "cat":
            return (
              <Panel>
                <Button icon="🆕" label="Новинка" onClick={{() => openLink(searchLink("Новинка"))}} />
                <Button icon="💎" label="Кратко о люкс продукте" onClick={{() => openLink(searchLink("Люкс"))}} />
                <Button icon="🔥" label="Тренд" onClick={{() => openLink(searchLink("Тренд"))}} />
                <Button icon="🏛" label="История бренда" onClick={{() => openLink(searchLink("История"))}} />
                <Button icon="⭐" label="Личная оценка" onClick={{() => openLink(searchLink("Оценка"))}} />
                <Button icon="🧴" label="Тип продукта / факты" onClick={{() => openLink(searchLink("Факты"))}} />
                <Button icon="🧪" label="Составы продуктов" onClick={{() => openLink(searchLink("Состав"))}} />
              </Panel>
            );
          case "brand":
            return (
              <Panel>
                <Button icon="✨" label="Dior" onClick={{() => openLink(searchLink("Dior"))}} />
                <Button icon="✨" label="Chanel" onClick={{() => openLink(searchLink("Chanel"))}} />
                <Button icon="✨" label="Charlotte Tilbury" onClick={{() => openLink(searchLink("CharlotteTilbury"))}} />
              </Panel>
            );
          case "sephora":
            return (
              <Panel>
                <Button icon="🇹🇷" label="Актуальные цены (TR)" subtitle="Ежедневное обновление" onClick={{() => openLink(searchLink("SephoraTR"))}} />
                <Button icon="🎁" label="Подарки / акции" onClick={{() => openLink(searchLink("SephoraPromo"))}} />
                <Button icon="🧾" label="Гайды / как покупать" onClick={{() => openLink(searchLink("SephoraGuide"))}} />
              </Panel>
            );
          default:
            return null;
        }}
      }};

      return (
        <div style={{{{ padding:"18px 16px 26px", maxWidth:"520px", margin:"0 auto" }}}}>
          <Hero user={{user}} />
          <Tabs active={{activeTab}} onChange={{setActiveTab}} />
          {{renderContent()}}
          <div style={{{{ marginTop:"20px", color:"var(--muted)", fontSize:"12px", textAlign:"center" }}}}>
            Открывается как Mini App внутри Telegram
          </div>
        </div>
      );
    }};

    ReactDOM.render(<App />, document.getElementById("root"));
  </script>
</body>
</html>
"""

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

@app.get("/health")
async def health():
    return {"status": "healthy"}
