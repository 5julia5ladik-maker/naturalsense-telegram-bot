# app/main.py - Точка входа
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.bot.telegram_app import start_telegram_bot, stop_telegram_bot
from app.api import content, users, analytics
from app.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await start_telegram_bot()
    yield
    # Shutdown
    await stop_telegram_bot()

app = FastAPI(
    title="NS · Natural Sense API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API роуты
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])

# WebApp endpoint
from app.bot.webapp import get_webapp_html

@app.get("/")
async def root():
    return {"status": "ok", "app": "NS · Natural Sense"}

@app.get("/webapp")
async def webapp():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(get_webapp_html())


# ─────────────────────────────────────────────────────
# app/bot/telegram_app.py - Telegram Bot
# ─────────────────────────────────────────────────────
import os
import asyncio
from telegram.ext import Application
from app.bot.handlers import register_handlers

tg_app = None
tg_task = None

async def start_telegram_bot():
    global tg_app, tg_task
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    
    tg_app = Application.builder().token(BOT_TOKEN).build()
    register_handlers(tg_app)
    
    async def run():
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.updater.start_polling(drop_pending_updates=True)
        while True:
            await asyncio.sleep(3600)
    
    tg_task = asyncio.create_task(run())

async def stop_telegram_bot():
    global tg_app, tg_task
    if tg_task:
        tg_task.cancel()
    if tg_app:
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()


# ─────────────────────────────────────────────────────
# app/bot/handlers.py - Bot команды
# ─────────────────────────────────────────────────────
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes
from app.bot.keyboards import get_main_keyboard
from app.database.queries import create_user, get_user

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    # Сохраняем юзера в БД
    db_user = await get_user(user.id)
    if not db_user:
        db_user = await create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name
        )
        welcome_text = f"Добро пожаловать, {user.first_name}! 🖤\n\n+10 баллов за регистрацию"
    else:
        welcome_text = f"С возвращением, {user.first_name}! ✨"
    
    kb = get_main_keyboard()
    await update.message.reply_text(welcome_text, reply_markup=kb)

async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    db_user = await get_user(user.id)
    
    tier_emoji = {"free": "🥉", "premium": "🥈", "vip": "🥇"}
    
    text = f"""
👤 Твой профиль

Уровень: {tier_emoji.get(db_user.tier, "🥉")} {db_user.tier.upper()}
Баллы: {db_user.points}
Дата регистрации: {db_user.joined_at.strftime("%d.%m.%Y")}

Твои достижения скоро появятся! 💎
    """
    await update.message.reply_text(text)

def register_handlers(app):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("profile", cmd_profile))


# ─────────────────────────────────────────────────────
# app/bot/keyboards.py - Клавиатуры
# ─────────────────────────────────────────────────────
import os
from telegram import ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

def get_main_keyboard():
    webapp_url = f"{os.getenv('PUBLIC_BASE_URL')}/webapp"
    
    return ReplyKeyboardMarkup([
        [KeyboardButton("📲 Открыть журнал", web_app=WebAppInfo(url=webapp_url))],
        [KeyboardButton("👤 Профиль"), KeyboardButton("🎁 Челленджи")],
        [KeyboardButton("↩️ В канал")]
    ], resize_keyboard=True)


# ─────────────────────────────────────────────────────
# app/database/models.py - SQLAlchemy модели
# ─────────────────────────────────────────────────────
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, index=True)
    username = Column(String, nullable=True)
    first_name = Column(String)
    tier = Column(String, default="free")  # free, premium, vip
    points = Column(Integer, default=10)
    favorites = Column(JSON, default=list)
    joined_at = Column(DateTime, default=datetime.utcnow)

class Challenge(Base):
    __tablename__ = "challenges"
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(String)
    reward_points = Column(Integer)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    active = Column(Integer, default=1)


# ─────────────────────────────────────────────────────
# app/database/__init__.py
# ─────────────────────────────────────────────────────
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ns.db")
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ─────────────────────────────────────────────────────
# app/database/queries.py - Database операции
# ─────────────────────────────────────────────────────
from sqlalchemy import select
from app.database import async_session
from app.database.models import User

async def get_user(telegram_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.telegram_id == telegram_id)
        )
        return result.scalar_one_or_none()

async def create_user(telegram_id: int, username: str, first_name: str):
    async with async_session() as session:
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            points=10  # стартовые баллы
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

async def add_points(telegram_id: int, points: int):
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.telegram_id == telegram_id)
        )
        user = result.scalar_one()
        user.points += points
        
        # Автоапгрейд тира
        if user.points >= 500:
            user.tier = "vip"
        elif user.points >= 100:
            user.tier = "premium"
        
        await session.commit()
        return user


# ─────────────────────────────────────────────────────
# app/api/users.py - API для фронта
# ─────────────────────────────────────────────────────
from fastapi import APIRouter
from app.database.queries import get_user, add_points

router = APIRouter()

@router.get("/{telegram_id}")
async def get_user_profile(telegram_id: int):
    user = await get_user(telegram_id)
    if not user:
        return {"error": "User not found"}
    
    return {
        "id": user.id,
        "telegram_id": user.telegram_id,
        "username": user.username,
        "first_name": user.first_name,
        "tier": user.tier,
        "points": user.points,
        "favorites": user.favorites,
        "joined_at": user.joined_at.isoformat()
    }

@router.post("/{telegram_id}/points")
async def award_points(telegram_id: int, points: int):
    user = await add_points(telegram_id, points)
    return {"success": True, "new_total": user.points}


# ─────────────────────────────────────────────────────
# app/services/gamification.py - Геймификация
# ─────────────────────────────────────────────────────
from datetime import datetime
from app.database.queries import add_points

POINT_REWARDS = {
    "daily_visit": 5,
    "read_post": 2,
    "join_challenge": 20,
    "purchase": 50,
    "referral": 30,
}

async def reward_user(telegram_id: int, action: str):
    points = POINT_REWARDS.get(action, 0)
    if points > 0:
        await add_points(telegram_id, points)
        return points
    return 0
