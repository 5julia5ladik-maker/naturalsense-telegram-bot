# Main.ру 1.0.10.0 BASE 1.0.8.5 + DAILY AUTO TRACK FIX.py
import os
import re
import json
import math
import time
import hmac
import base64
import hashlib
import random
import asyncio
import logging
from typing import Any, Optional, Literal
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
    JSON as SAJSON,
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
# ENV
# -----------------------------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "").strip().lstrip("@")
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "5443870760"))  # from memory

if not BOT_TOKEN:
    logger.warning("BOT_TOKEN is not set")
if not DATABASE_URL:
    logger.warning("DATABASE_URL is not set")
if not CHANNEL_USERNAME:
    logger.warning("CHANNEL_USERNAME is not set")
if not PUBLIC_BASE_URL:
    logger.warning("PUBLIC_BASE_URL is not set (WebApp url may be relative)")

logger.info(
    "ENV CHECK: BOT_TOKEN_present=%s BOT_TOKEN_len=%s PUBLIC_BASE_URL_present=%s DATABASE_URL_present=%s CHANNEL_USERNAME=%s ADMIN_CHAT_ID=%s",
    bool(BOT_TOKEN),
    len(BOT_TOKEN),
    bool(PUBLIC_BASE_URL),
    bool(DATABASE_URL),
    CHANNEL_USERNAME,
    ADMIN_CHAT_ID,
)

# -----------------------------------------------------------------------------
# DAILY TASKS CONFIG (max 400/day)
# -----------------------------------------------------------------------------
DAILY_MAX_POINTS_PER_DAY = 400

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
    {"key": "reply_comment", "title": "Ответить на комментарий", "points": 50, "icon": "↩️💬"},

    # Игровые
    {"key": "spin_roulette", "title": "Крутить рулетку 1 раз", "points": 50, "icon": "🎡"},
    {"key": "convert_prize", "title": "Конвертировать приз/билет", "points": 40, "icon": "🔁"},

    # Бонус дня (чтобы добить ровно до 400)
    {"key": "bonus_day", "title": "Собрать все задания дня", "points": 30, "icon": "🎁", "special": True},
]
# Total base (excluding bonus_day) = 370; with bonus_day = 400


PrizeType = Literal["points", "raffle_ticket", "physical_dior_palette"]

# per 1_000_000
ROULETTE_DISTRIBUTION: list[dict[str, Any]] = [
    {"weight": 416_667, "key": "points_500", "type": "points", "value": 50, "label": "+50 баллов"},
    {"weight": 291_667, "key": "points_1000", "type": "points", "value": 100, "label": "+100 баллов"},
    {"weight": 125_000, "key": "points_1500", "type": "points", "value": 150, "label": "+150 баллов"},
    {"weight": 83_333,  "key": "points_2000", "type": "points", "value": 200, "label": "+200 баллов"},
    {"weight": 41_667,  "key": "ticket_1",   "type": "raffle_ticket", "value": 1, "label": "🎟 +1 билет"},
    {"weight": 29_166,  "key": "points_3000", "type": "points", "value": 300, "label": "+300 баллов"},
    {"weight": 12_500,  "key": "dior_palette", "type": "physical_dior_palette", "value": 1, "label": "✨ Dior Palette"},
]
ROULETTE_TOTAL = sum(x["weight"] for x in ROULETTE_DISTRIBUTION)
if ROULETTE_TOTAL != 1_000_000:
    raise RuntimeError("ROULETTE_DISTRIBUTION must sum to 1_000_000")

Base = declarative_base()

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
def _normalize_db_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url

DATABASE_URL_ASYNC = _normalize_db_url(DATABASE_URL)
engine = create_async_engine(DATABASE_URL_ASYNC, echo=False, pool_pre_ping=True, pool_recycle=1800)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    points = Column(Integer, default=10, nullable=False)
    tier = Column(String, default="Base", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, unique=True, index=True, nullable=False)
    url = Column(String, nullable=False)
    tags = Column(SAJSON, default=list, nullable=False)
    title = Column(String, nullable=True)
    text = Column(String, nullable=True)
    has_photo = Column(Boolean, default=False, nullable=False)
    photo_url = Column(String, nullable=True)
    deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

Index("ix_posts_deleted_message", Post.deleted, Post.message_id)

class Raffle(Base):
    __tablename__ = "raffles"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class RaffleTicket(Base):
    __tablename__ = "raffle_tickets"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    raffle_id = Column(Integer, index=True, nullable=False)
    count = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

Index("ix_tickets_user_raffle", RaffleTicket.telegram_id, RaffleTicket.raffle_id, unique=True)

class RouletteSpin(Base):
    __tablename__ = "roulette_spins"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    prize_key = Column(String, nullable=False)
    prize_type = Column(String, nullable=False)
    prize_value = Column(Integer, nullable=False)
    prize_label = Column(String, nullable=False)
    roll = Column(Integer, nullable=False)
    resolution = Column(String, default="pending", nullable=False)  # pending/converted/claim
    resolved_at = Column(DateTime, nullable=True)
    resolved_meta = Column(SAJSON, default=dict, nullable=False)

class RouletteClaim(Base):
    __tablename__ = "roulette_claims"
    id = Column(Integer, primary_key=True)
    claim_code = Column(String, unique=True, index=True, nullable=False)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    spin_id = Column(Integer, index=True, nullable=True)
    prize_type = Column(String, nullable=False)
    prize_label = Column(String, nullable=False)
    status = Column(String, default="draft", nullable=False)  # draft/awaiting_contact/submitted/closed
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    country = Column(String, nullable=True)
    city = Column(String, nullable=True)
    address_line = Column(String, nullable=True)
    postal_code = Column(String, nullable=True)
    comment = Column(String, nullable=True)
    contact_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class PointTransaction(Base):
    __tablename__ = "point_transactions"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    type = Column(String, nullable=False)
    delta = Column(Integer, nullable=False)
    meta = Column(SAJSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class DailyTaskLog(Base):
    __tablename__ = "daily_task_logs"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, index=True, nullable=False)
    day = Column(String, index=True, nullable=False)  # YYYY-MM-DD (UTC)
    task_key = Column(String, index=True, nullable=False)
    status = Column(String, default="done", nullable=False)  # done/claimed
    done_at = Column(DateTime, nullable=True)
    claimed_at = Column(DateTime, nullable=True)
    points = Column(Integer, default=0, nullable=False)
    meta = Column(SAJSON, default=dict, nullable=False)

Index("ix_daily_unique", DailyTaskLog.telegram_id, DailyTaskLog.day, DailyTaskLog.task_key, unique=True)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
DEFAULT_RAFFLE_ID = 1
TICKET_PRICE = 300
TICKET_CONVERT_RATE = 3000
DIOR_PALETTE_CONVERT_VALUE = 3000

def _today_key() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def _recalc_tier(user: User) -> None:
    p = int(user.points or 0)
    if p >= 500000:
        user.tier = "Gold VIP"
    elif p >= 100000:
        user.tier = "Premium"
    else:
        user.tier = "Base"

async def ensure_schema():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def notify_admin(text: str):
    if not BOT_TOKEN or not ADMIN_CHAT_ID:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": ADMIN_CHAT_ID, "text": text},
            )
    except Exception:
        logger.exception("notify_admin failed")

def tg_user_link(tid: int) -> str:
    return f"tg://user?id={int(tid)}"

def generate_claim_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(10))

async def get_ticket_row(session: AsyncSession, tid: int, raffle_id: int) -> RaffleTicket:
    row = (
        await session.execute(
            select(RaffleTicket).where(RaffleTicket.telegram_id == tid, RaffleTicket.raffle_id == raffle_id)
        )
    ).scalar_one_or_none()
    if not row:
        row = RaffleTicket(telegram_id=tid, raffle_id=raffle_id, count=0, updated_at=datetime.utcnow())
        session.add(row)
        await session.flush()
    return row

def _daily_tasks_map() -> dict[str, dict[str, Any]]:
    return {t["key"]: t for t in DAILY_TASKS}

async def _get_daily_logs(session: AsyncSession, tid: int, day: str) -> dict[str, DailyTaskLog]:
    rows = (
        await session.execute(
            select(DailyTaskLog).where(DailyTaskLog.telegram_id == tid, DailyTaskLog.day == day)
        )
    ).scalars().all()
    return {r.task_key: r for r in rows}

async def _daily_points_claimed(session: AsyncSession, tid: int, day: str) -> int:
    s = (
        await session.execute(
            select(func.coalesce(func.sum(DailyTaskLog.points), 0)).where(
                DailyTaskLog.telegram_id == tid,
                DailyTaskLog.day == day,
                DailyTaskLog.status == "claimed",
            )
        )
    ).scalar_one()
    return int(s or 0)

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
            done_at=now,
            points=0,
            meta=meta or {},
        )
    )

async def _can_unlock_bonus_day(task_map: dict[str, dict[str, Any]], logs: dict[str, DailyTaskLog]) -> bool:
    for key, cfg in task_map.items():
        if cfg.get("special"):
            continue
        lg = logs.get(key)
        if key == "open_post":
            cnt = int((lg.meta or {}).get("count", 0) if lg else 0)
            if cnt < int(cfg.get("need", 1)):
                return False
            if not lg or lg.status != "claimed":
                return False
        else:
            if not lg or lg.status != "claimed":
                return False
    return True

# -----------------------------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app_: FastAPI):
    await ensure_schema()
    yield

app.router.lifespan_context = lifespan

# -----------------------------------------------------------------------------
# WEBAPP (HTML/JS/CSS) - UI is intentionally kept stable
# -----------------------------------------------------------------------------
@app.get("/webapp", response_class=HTMLResponse)
async def webapp():
    html = r"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"/>
  <title>Natural Sense</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    :root{
      --bg:#0b1320;
      --card:rgba(255,255,255,0.06);
      --card2:rgba(255,255,255,0.08);
      --stroke:rgba(255,255,255,0.10);
      --txt:rgba(255,255,255,0.92);
      --muted:rgba(255,255,255,0.55);
      --gold:rgba(241,205,123,0.95);
      --shadow: 0 12px 32px rgba(0,0,0,0.45);
      --r:22px;
    }
    html,body{height:100%; background:var(--bg); margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;}
    body{ color:var(--txt); overflow-x:hidden; }
    .app{ max-width: 560px; margin:0 auto; padding: 14px 14px 110px; }
    .topCard{
      background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
      border: 1px solid var(--stroke);
      border-radius: var(--r);
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .row{ display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .h1{ font-size: 16px; font-weight: 700; letter-spacing: .2px; }
    .sub{ font-size: 13px; color: var(--muted); margin-top: 3px; }
    .badge{ display:inline-flex; align-items:center; gap:8px; padding: 8px 10px; background: rgba(255,255,255,0.08); border:1px solid var(--stroke); border-radius: 999px; font-size: 13px; }
    .grid{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
    .btn{
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--stroke);
      border-radius: 18px;
      padding: 12px;
      cursor: pointer;
      box-shadow: 0 8px 26px rgba(0,0,0,0.20);
      user-select:none;
      transition: transform .08s ease, background .12s ease;
    }
    .btn:active{ transform: scale(0.98); }
    .btnT{ font-weight:700; }
    .btnSub{ font-size: 12px; margin-top: 4px; color: var(--muted); }
    .sectionTitle{ display:flex; align-items:center; justify-content:space-between; margin: 16px 2px 8px; }
    .sectionTitle .t{ font-weight:800; }
    .sectionTitle .a{ font-size: 12px; color: var(--muted); cursor:pointer; }
    .cards{ display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .postCard{
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--stroke);
      border-radius: 18px;
      padding: 10px;
      cursor:pointer;
      overflow:hidden;
      position:relative;
    }
    .thumb{
      height: 92px;
      border-radius: 14px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      display:flex;
      align-items:center;
      justify-content:center;
      overflow:hidden;
    }
    .thumb img{ width:100%; height:100%; object-fit: cover; display:block; }
    .tagChip{
      position:absolute;
      left: 10px;
      top: 10px;
      padding: 6px 10px;
      font-size: 12px;
      border-radius: 999px;
      background: rgba(0,0,0,0.45);
      border: 1px solid rgba(255,255,255,0.12);
      backdrop-filter: blur(8px);
    }
    .pcTitle{ margin-top: 8px; font-weight: 800; font-size: 13px; line-height: 1.15; }
    .pcSub{ margin-top: 4px; font-size: 12px; color: var(--muted); line-height: 1.2; max-height: 3.8em; overflow:hidden; }
    .bottomNav{
      position: fixed;
      left:0; right:0; bottom:0;
      display:flex;
      justify-content:center;
      padding: 10px 0 16px;
      pointer-events:none;
    }
    .bottomNav .bar{
      pointer-events:auto;
      width: min(560px, calc(100% - 20px));
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--stroke);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 8px;
      display:flex;
      justify-content:space-around;
      gap: 6px;
    }
    .navBtn{
      flex:1;
      border-radius: 18px;
      padding: 10px 6px;
      text-align:center;
      color: var(--muted);
      font-size: 12px;
      cursor:pointer;
      user-select:none;
    }
    .navBtn.active{
      color: var(--txt);
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.10);
    }
    .sheetOverlay{
      position:fixed; inset:0;
      background: rgba(0,0,0,0.35);
      display:none;
      align-items:flex-end;
      justify-content:center;
      z-index: 999;
    }
    .sheetOverlay.open{ display:flex; }
    .sheet{
      width: min(560px, 100%);
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 22px 22px 0 0;
      padding: 14px 14px 20px;
      box-shadow: 0 -12px 36px rgba(0,0,0,0.55);
      backdrop-filter: blur(10px);
      max-height: 78vh;
      overflow:auto;
    }
    .taskRow{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 18px;
      padding: 12px;
      margin-top: 10px;
    }
    .taskTop{ display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .taskTitle{ font-weight:800; }
    .taskSub{ font-size: 12px; color: var(--muted); margin-top: 4px; }
    .pill{ display:inline-flex; align-items:center; gap:6px; padding: 6px 10px; border-radius: 999px; border:1px solid rgba(255,255,255,0.12); background: rgba(0,0,0,0.30); font-size: 12px; }
    .pill.ok{ border-color: rgba(120,255,190,0.25); }
    .claimBtn{
      margin-top: 10px;
      width:100%;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 16px;
      padding: 12px;
      font-weight:800;
      cursor:pointer;
      user-select:none;
      text-align:center;
      color: var(--txt);
    }
    .claimBtn[disabled]{ opacity: 0.45; cursor: default; }
    .smallMsg{ margin-top: 10px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="app" id="app"></div>

  <div class="sheetOverlay" id="dailyOverlay">
    <div class="sheet">
      <div id="dailyContent"></div>
    </div>
  </div>

  <div class="bottomNav">
    <div class="bar">
      <div class="navBtn active" id="navJournal">📰<div>Журнал</div></div>
      <div class="navBtn" id="navSearch">🔎<div>Поиск</div></div>
      <div class="navBtn" id="navBonus">🎁<div>Бонусы</div></div>
      <div class="navBtn" id="navProfile">👤<div>Профиль</div></div>
    </div>
  </div>

<script>
(function(){
    const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
    if(tg){ try{ tg.expand(); }catch(e){} }

    function haptic(type){ try{ tg && tg.HapticFeedback && tg.HapticFeedback.impactOccurred(type||"light"); }catch(e){} }
    function esc(s){ return (s==null?"":String(s)).replace(/[&<>'"]/g, c=>({ "&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;" }[c])); }
    function el(tag, cls, html){
      const e = document.createElement(tag);
      if(cls) e.className = cls;
      if(html != null) e.innerHTML = html;
      return e;
    }
    function openLink(url){
      try{
        if(tg && tg.openTelegramLink) return tg.openTelegramLink(url);
      }catch(e){}
      window.open(url, "_blank");
    }

    async function fetchJson(url, opts){
      const r = await fetch(url, opts||{});
      const t = await r.text();
      let j = null;
      try{ j = JSON.parse(t); }catch(e){}
      if(!r.ok){
        const msg = (j && (j.detail || j.message)) ? (j.detail || j.message) : (t || ("HTTP "+r.status));
        throw new Error(msg);
      }
      return j;
    }
    function apiGet(url){
      const sep = url.includes("?") ? "&" : "?";
      const bust = "t="+Date.now();
      return fetchJson(url + sep + bust, {method:"GET"});
    }
    function apiPost(url, body){
      const sep = url.includes("?") ? "&" : "?";
      const bust = "t="+Date.now();
      return fetchJson(url + sep + bust, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})});
    }

    const state = {
      tab: "journal",
      user: null,
      botUsername: null,
      posts: [],
      postsAll: [],
      searchQ: "",
      searchRes: [],
      dailyOpen: false,
      daily: null,
      dailyBusy: false,
      dailyMsg: "",
      __dailyRefreshT: null,
    };

    let tgUserId = null;

    // derive telegram_id
    try{
      if(tg && tg.initDataUnsafe && tg.initDataUnsafe.user && tg.initDataUnsafe.user.id){
        tgUserId = tg.initDataUnsafe.user.id;
      }
    }catch(e){}

    async function refreshUser(){
      if(!tgUserId) return;
      try{
        state.user = await apiGet("/api/user/"+encodeURIComponent(tgUserId));
      }catch(e){}
    }

    async function loadBotUsername(){
      try{
        const d = await apiGet("/api/bot/username");
        state.botUsername = d && d.username ? d.username : null;
      }catch(e){ state.botUsername = null; }
    }

    async function loadPosts(){
      try{
        state.posts = await apiGet("/api/posts?tag=%23Новинка&limit=6&offset=0");
      }catch(e){ state.posts = []; }
    }

    async function loadAllPosts(){
      try{
        state.postsAll = await apiGet("/api/posts?limit=50&offset=0");
      }catch(e){ state.postsAll = []; }
    }

    async function doSearch(){
      const q = (state.searchQ||"").trim();
      if(!q){ state.searchRes = []; render(); return; }
      try{
        // track event for daily
        try{ dailyEvent('use_search', {q:q}); }catch(e){}
        state.searchRes = await apiGet("/api/search?q="+encodeURIComponent(q)+"&limit=50&offset=0");
      }catch(e){ state.searchRes = []; }
      render();
    }

    function renderTopCard(root){
      const card = el("div","topCard");
      const r1 = el("div","row");
      const left = el("div");
      left.appendChild(el("div","h1","NS · Natural Sense"));
      left.appendChild(el("div","sub","Выпуск дня · люкс-журнал"));
      const right = el("div");
      const points = state.user ? (state.user.points||0) : 0;
      const tier = state.user ? (state.user.tier||"Base") : "Base";
      right.appendChild(el("div","badge", "💎 <b>"+esc(points)+"</b> · 🥇 "+esc(tier)));
      r1.appendChild(left);
      r1.appendChild(right);
      card.appendChild(r1);

      card.appendChild(el("div","sub","Подборки, факты и люкс-обзоры — прямо в Telegram."));

      const wrap = el("div","grid");
      const openCh = el("div","btn");
      openCh.appendChild(el("div","btnT","↩️ В канал"));
      openCh.appendChild(el("div","btnSub","Вернуться в ленту Natural Sense"));
      openCh.appendChild(el("div",null,'<div style="opacity:0.85">›</div>'));
      openCh.addEventListener("click", ()=>{
        haptic();
        try{ dailyEvent('open_channel'); }catch(e){}
        openLink("https://t.me/"""+CHANNEL_USERNAME+"""");
      });

      // other buttons preserved
      const categories = el("div","btn");
      categories.appendChild(el("div","btnT","📚 Категории"));
      categories.appendChild(el("div","btnSub","Темы и разделы журнала"));
      categories.addEventListener("click", ()=>{ haptic(); });

      const brands = el("div","btn");
      brands.appendChild(el("div","btnT","🏷️ Бренды"));
      brands.appendChild(el("div","btnSub","Все бренды и теги"));
      brands.addEventListener("click", ()=>{ haptic(); });

      const products = el("div","btn");
      products.appendChild(el("div","btnT","🧴 Продукты"));
      products.appendChild(el("div","btnSub","Типы продуктов"));
      products.addEventListener("click", ()=>{ haptic(); });

      const bag = el("div","btn");
      bag.appendChild(el("div","btnT","👜 Косметичка"));
      bag.appendChild(el("div","btnSub","Призы и билеты"));
      bag.addEventListener("click", async ()=>{
        haptic();
        try{ dailyEvent('open_inventory'); }catch(e){}
        // load inventory but keep UI stable
        try{ await apiGet("/api/inventory?telegram_id="+encodeURIComponent(tgUserId)); }catch(e){}
      });

      wrap.appendChild(openCh);
      wrap.appendChild(categories);
      wrap.appendChild(brands);
      wrap.appendChild(products);
      wrap.appendChild(bag);

      card.appendChild(wrap);
      root.appendChild(card);
    }

    function renderPosts(root){
      const head = el("div","sectionTitle");
      head.appendChild(el("div","t",'🆕 Новинки'));
      head.appendChild(el("div","a",'Смотреть всё ›'));
      root.appendChild(head);

      const grid = el("div","cards");
      (state.posts||[]).forEach(post=>{
        const c = el("div","postCard");
        const th = el("div","thumb");
        if(post.photo_url){
          const im = new Image();
          im.src = post.photo_url;
          th.appendChild(im);
        }else{
          th.innerHTML = "<div style='opacity:.65'>NS</div>";
        }
        c.appendChild(th);
        c.appendChild(el("div","tagChip","#Новинка"));
        c.appendChild(el("div","pcTitle", esc(post.title||"Пост")));
        c.appendChild(el("div","pcSub", esc(post.text||"")));
        c.addEventListener("click", ()=>{ haptic(); try{ dailyEvent('open_post', {message_id: post.message_id}); }catch(e){} openLink(post.url); });
        grid.appendChild(c);
      });
      root.appendChild(grid);
    }

    function renderSearch(root){
      root.appendChild(el("div","sectionTitle").appendChild(el("div","t","🔎 Поиск")) || el("div"));
      const inp = el("input");
      inp.style.width="100%";
      inp.style.padding="14px";
      inp.style.borderRadius="18px";
      inp.style.border="1px solid var(--stroke)";
      inp.style.background="rgba(255,255,255,0.06)";
      inp.style.color="var(--txt)";
      inp.style.outline="none";
      inp.placeholder="Найти пост...";
      inp.value = state.searchQ || "";
      inp.addEventListener("input", (e)=>{ state.searchQ = e.target.value; });
      inp.addEventListener("keydown", (e)=>{ if(e.key==="Enter"){ haptic(); doSearch(); } });
      root.appendChild(inp);

      const btn = el("div","btn");
      btn.style.marginTop="10px";
      btn.appendChild(el("div","btnT","Искать"));
      btn.appendChild(el("div","btnSub","Поиск по тексту постов"));
      btn.addEventListener("click", ()=>{ haptic(); doSearch(); });
      root.appendChild(btn);

      const grid = el("div","cards");
      grid.style.marginTop="12px";
      (state.searchRes||[]).forEach(post=>{
        const c = el("div","postCard");
        const th = el("div","thumb");
        if(post.photo_url){
          const im = new Image();
          im.src = post.photo_url;
          th.appendChild(im);
        }else{
          th.innerHTML = "<div style='opacity:.65'>NS</div>";
        }
        c.appendChild(th);
        c.appendChild(el("div","pcTitle", esc(post.title||"Пост")));
        c.appendChild(el("div","pcSub", esc(post.text||"")));
        c.addEventListener("click", ()=>{ haptic(); try{ dailyEvent('open_post', {message_id: post.message_id}); }catch(e){} openLink(post.url); });
        grid.appendChild(c);
      });
      root.appendChild(grid);
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

// Auto-track Mini App open once per session (does not affect UI)
try{
  if(typeof window !== 'undefined'){
    if(!window.__nsDailyOpenPinged){
      window.__nsDailyOpenPinged = true;
      if(tgUserId){ dailyEvent('open_miniapp'); }
    }
  }
}catch(e){}

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
  }

  if(state.dailyMsg){
    content.appendChild(el("div","smallMsg", esc(state.dailyMsg)));
  }

  if(!state.daily || !state.daily.items){
    content.appendChild(el("div","smallMsg","Загрузка..."));
    return;
  }

  state.daily.items.forEach(it=>{
    const row = el("div","taskRow");
    const top = el("div","taskTop");
    top.appendChild(el("div","taskTitle", esc(it.icon)+" "+esc(it.title)));
    top.appendChild(el("div","pill"+(it.done?" ok":""), "💎 +"+esc(it.points)));
    row.appendChild(top);
    const sub = el("div","taskSub", it.claimed ? "Получено" : (it.done ? "Выполнено" : "Не выполнено"));
    row.appendChild(sub);

    const btn = el("div","claimBtn", it.claimed ? "✅ Уже получено" : "🔒 Выполни чтобы забрать");
    if(it.done && !it.claimed){
      btn.innerHTML = "🎁 Забрать";
      btn.addEventListener("click", ()=>{ haptic(); claimDaily(it.key); });
    }else{
      btn.setAttribute("disabled","disabled");
    }
    row.appendChild(btn);

    if(it.need && it.need>1){
      row.appendChild(el("div","taskSub", esc(it.progress||0)+" / "+esc(it.need)));
    }

    content.appendChild(row);
  });
}

    function render(){
      const root = document.getElementById("app");
      if(!root) return;
      root.innerHTML = "";
      if(state.tab==="journal"){
        renderTopCard(root);
        renderPosts(root);
      }else if(state.tab==="search"){
        renderSearch(root);
      }else if(state.tab==="bonus"){
        // only daily button (safe)
        const c = el("div","topCard");
        c.appendChild(el("div","h1","🎁 Бонусы"));
        c.appendChild(el("div","sub","Ежедневные задания, чтобы набрать до 400 бонусов."));
        const b = el("div","btn");
        b.appendChild(el("div","btnT","🎯 Daily бонусы"));
        b.appendChild(el("div","btnSub","Открыть список заданий и забрать награды"));
        b.addEventListener("click", ()=>{ haptic(); openDaily(); });
        c.appendChild(el("div","grid").appendChild(b) || b);
        root.appendChild(c);
      }else if(state.tab==="profile"){
        const c = el("div","topCard");
        c.appendChild(el("div","h1","👤 Профиль"));
        const points = state.user ? (state.user.points||0) : 0;
        const tier = state.user ? (state.user.tier||"Base") : "Base";
        c.appendChild(el("div","sub","Баланс: "+esc(points)+" · Статус: "+esc(tier)));
        root.appendChild(c);
      }
      renderDailySheet();
    }

    function setTab(t){
      state.tab = t;
      document.getElementById("navJournal").classList.toggle("active", t==="journal");
      document.getElementById("navSearch").classList.toggle("active", t==="search");
      document.getElementById("navBonus").classList.toggle("active", t==="bonus");
      document.getElementById("navProfile").classList.toggle("active", t==="profile");
      render();
    }

    document.getElementById("navJournal").addEventListener("click", ()=>{ haptic(); setTab("journal"); });
    document.getElementById("navSearch").addEventListener("click", ()=>{ haptic(); setTab("search"); });
    document.getElementById("navBonus").addEventListener("click", ()=>{ haptic(); setTab("bonus"); });
    document.getElementById("navProfile").addEventListener("click", ()=>{ haptic(); setTab("profile"); try{ dailyEvent('open_profile'); }catch(e){} });

    (async function init(){
      await loadBotUsername();
      await refreshUser();
      await loadPosts();
      await loadAllPosts();
      render();
    })();
})();
</script>
</body>
</html>
"""
    return HTMLResponse(html)

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.get("/api/bot/username")
async def bot_username():
    if not BOT_TOKEN:
        return {"username": None}
    try:
        app_ = Application.builder().token(BOT_TOKEN).build()
        me = await app_.bot.get_me()
        return {"username": me.username}
    except Exception:
        return {"username": None}

@app.get("/api/user/{telegram_id}")
async def get_user(telegram_id: int):
    tid = int(telegram_id)
    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            user = User(telegram_id=tid, points=10)
            _recalc_tier(user)
            session.add(user)
            await session.commit()
        return {"telegram_id": tid, "points": int(user.points or 0), "tier": user.tier}

@app.get("/api/posts")
async def get_posts(tag: str | None = None, limit: int = 50, offset: int = 0):
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    async with async_session_maker() as session:
        q = select(Post).where(Post.deleted == False)
        if tag:
            # tags stored with leading #
            q = q.where(func.json_extract(Post.tags, "$").like(f'%"{tag}"%'))
        q = q.order_by(Post.message_id.desc()).limit(limit).offset(offset)
        rows = (await session.execute(q)).scalars().all()
        return [
            {
                "id": int(p.id),
                "message_id": int(p.message_id),
                "url": p.url,
                "tags": p.tags or [],
                "title": p.title,
                "text": p.text,
                "has_photo": bool(p.has_photo),
                "photo_url": p.photo_url,
            }
            for p in rows
        ]

@app.get("/api/search")
async def api_search(q: str, limit: int = 50, offset: int = 0):
    q = (q or "").strip()
    if not q:
        return []
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))

    async with async_session_maker() as session:
        rows = (
            await session.execute(
                select(Post)
                .where(Post.deleted == False)
                .where(func.lower(Post.text).like(f"%{q.lower()}%") | func.lower(Post.title).like(f"%{q.lower()}%"))
                .order_by(Post.message_id.desc())
                .limit(limit)
                .offset(offset)
            )
        ).scalars().all()
        return [
            {
                "id": int(p.id),
                "message_id": int(p.message_id),
                "url": p.url,
                "tags": p.tags or [],
                "title": p.title,
                "text": p.text,
                "has_photo": bool(p.has_photo),
                "photo_url": p.photo_url,
            }
            for p in rows
        ]

# -----------------------------------------------------------------------------
# INVENTORY
# -----------------------------------------------------------------------------
@app.get("/api/inventory")
async def inventory_api(telegram_id: int):
    tid = int(telegram_id)
    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Daily tracking: open inventory (cosmetic bag)
        try:
            await _mark_daily_done(session, tid, _today_key(), "open_inventory")
            await session.commit()
        except Exception:
            await session.rollback()

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
            )
        ).scalars().all()

        return {
            "telegram_id": tid,
            "ticket_count": int(t or 0),
            "claims": [
                {
                    "claim_id": int(c.id),
                    "claim_code": c.claim_code,
                    "prize_label": c.prize_label,
                    "status": c.status,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in claims
            ],
        }

# -----------------------------------------------------------------------------
# DAILY TASKS API
# -----------------------------------------------------------------------------
from pydantic import BaseModel, Field

class DailyEventReq(BaseModel):
    telegram_id: int
    event: str
    data: dict[str, Any] | None = None

class DailyClaimReq(BaseModel):
    telegram_id: int
    task_key: str

class DailyTaskItem(BaseModel):
    key: str
    title: str
    icon: str
    points: int
    done: bool
    claimed: bool
    need: int = 1
    progress: int = 0

class DailyTasksResp(BaseModel):
    telegram_id: int
    day: str
    max_points: int
    claimed_points: int
    remaining_points: int
    items: list[DailyTaskItem]

class DailyClaimResp(BaseModel):
    ok: bool = True
    task_key: str
    awarded: int
    claimed_points: int
    remaining_points: int
    user_points: int

@app.post("/api/daily/event")
async def daily_event_api(req: DailyEventReq):
    tid = int(req.telegram_id)
    day = _today_key()
    ev = (req.event or "").strip().lower()

    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            user = User(telegram_id=tid, points=10)
            session.add(user)
            await session.commit()

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
        elif ev == "reply_comment":
            await _mark_daily_done(session, tid, day, "reply_comment")
        elif ev == "spin_roulette":
            await _mark_daily_done(session, tid, day, "spin_roulette")
        elif ev == "convert_prize":
            await _mark_daily_done(session, tid, day, "convert_prize")
        elif ev == "open_post":
            logs = await _get_daily_logs(session, tid, day)
            lg = logs.get("open_post")
            cnt = int((lg.meta or {}).get("count", 0) if lg else 0)
            cnt = min(3, cnt + 1)
            await _mark_daily_done(session, tid, day, "open_post", {"count": cnt})
        else:
            return {"ok": True, "ignored": True}

        await session.commit()
        return {"ok": True}

@app.get("/api/daily/tasks", response_model=DailyTasksResp)
async def daily_tasks_api(telegram_id: int):
    tid = int(telegram_id)
    day = _today_key()
    task_map = _daily_tasks_map()

    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            user = User(telegram_id=tid, points=10)
            session.add(user)
            await session.commit()

        # Auto-mark 'open_miniapp' when tasks are requested (guaranteed tracking)
        await _mark_daily_done(session, tid, day, "open_miniapp")

        logs = await _get_daily_logs(session, tid, day)

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
                done = await _can_unlock_bonus_day(task_map, logs)
                claimed = bool(lg) and (lg.status == "claimed")
            items.append(
                DailyTaskItem(
                    key=key,
                    title=cfg["title"],
                    icon=cfg.get("icon", ""),
                    points=int(cfg["points"]),
                    done=bool(done),
                    claimed=bool(claimed),
                    need=need,
                    progress=progress,
                )
            )

        claimed_points = await _daily_points_claimed(session, tid, day)
        remaining = max(0, DAILY_MAX_POINTS_PER_DAY - claimed_points)
        return DailyTasksResp(
            telegram_id=tid,
            day=day,
            max_points=DAILY_MAX_POINTS_PER_DAY,
            claimed_points=claimed_points,
            remaining_points=remaining,
            items=items,
        )

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
            user = User(telegram_id=tid, points=10)
            session.add(user)
            await session.commit()

        logs = await _get_daily_logs(session, tid, day)
        lg = logs.get(key)

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
                done_at=now,
                claimed_at=now,
                points=award,
                meta={},
            )
        else:
            lg.status = "claimed"
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

# -----------------------------------------------------------------------------
# CONVERT / ROULETTE endpoints (only minimal additions: daily tracking)
# -----------------------------------------------------------------------------
class ConvertTicketsReq(BaseModel):
    telegram_id: int
    qty: int = 1

class ConvertTicketsResp(BaseModel):
    telegram_id: int
    points: int
    ticket_count: int
    converted_qty: int
    added_points: int

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

    # Daily tracking: any conversion counts as 'convert_prize'
    try:
        async with async_session_maker() as s:
            await _mark_daily_done(s, tid, _today_key(), "convert_prize")
            await s.commit()
    except Exception:
        pass

    return {
        "telegram_id": tid,
        "points": points_now,
        "ticket_count": tickets_now,
        "converted_qty": qty,
        "added_points": added,
    }

class ConvertPrizeReq(BaseModel):
    telegram_id: int
    claim_code: str

class ConvertPrizeResp(BaseModel):
    telegram_id: int
    points: int
    claim_code: str
    added_points: int

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

            if str(claim.status) not in {"draft", "awaiting_contact"}:
                raise HTTPException(status_code=400, detail="Заявка уже в обработке — конвертация недоступна")

            added = int(DIOR_PALETTE_CONVERT_VALUE)

            claim.status = "closed"
            note = "CONVERTED_TO_POINTS"
            if claim.contact_text:
                if note not in claim.contact_text:
                    claim.contact_text = (claim.contact_text + "\n" + note).strip()
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

    # Daily tracking: any conversion counts as 'convert_prize'
    try:
        async with async_session_maker() as s:
            await _mark_daily_done(s, tid, _today_key(), "convert_prize")
            await s.commit()
    except Exception:
        pass

    return {"telegram_id": tid, "points": points_now, "claim_code": code, "added_points": int(DIOR_PALETTE_CONVERT_VALUE)}

class SpinReq(BaseModel):
    telegram_id: int

@app.post("/api/roulette/spin")
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

            roll = random.randint(1, ROULETTE_TOTAL)
            acc = 0
            prize = ROULETTE_DISTRIBUTION[-1]
            for x in ROULETTE_DISTRIBUTION:
                acc += int(x["weight"])
                if roll <= acc:
                    prize = x
                    break

            prize_key = str(prize["key"])
            prize_type = str(prize["type"])
            prize_value = int(prize["value"])
            prize_label = str(prize["label"])

            spin = RouletteSpin(
                telegram_id=tid,
                created_at=now,
                prize_key=prize_key,
                prize_type=prize_type,
                prize_value=prize_value,
                prize_label=prize_label,
                roll=int(roll),
                resolution="pending",
                resolved_at=None,
                resolved_meta={},
            )
            session.add(spin)
            await session.flush()
            spin_id = int(spin.id)

            claim_id_for_resp = None
            claim_code_for_resp = None

            if prize_type == "points":
                user.points = int(user.points or 0) + int(prize_value)
                _recalc_tier(user)
                session.add(
                    PointTransaction(
                        telegram_id=tid,
                        type="roulette_win_points",
                        delta=int(prize_value),
                        meta={"spin_id": spin_id, "prize_key": prize_key},
                    )
                )
                spin.resolution = "converted"
                spin.resolved_at = now
                spin.resolved_meta = {"auto": "points"}

            elif prize_type == "raffle_ticket":
                ticket_row = await get_ticket_row(session, tid, DEFAULT_RAFFLE_ID)
                ticket_row.count = int(ticket_row.count or 0) + int(prize_value)
                ticket_row.updated_at = now
                spin.resolution = "converted"
                spin.resolved_at = now
                spin.resolved_meta = {"auto": "ticket"}
                session.add(
                    PointTransaction(
                        telegram_id=tid,
                        type="roulette_win_ticket",
                        delta=0,
                        meta={"spin_id": spin_id, "prize_key": prize_key, "tickets_added": int(prize_value)},
                    )
                )

            elif prize_type == "physical_dior_palette":
                claim_code = generate_claim_code()
                claim = RouletteClaim(
                    claim_code=claim_code,
                    telegram_id=tid,
                    spin_id=spin_id,
                    prize_type=prize_type,
                    prize_label=prize_label,
                    status="draft",
                    created_at=now,
                    updated_at=now,
                )
                session.add(claim)
                await session.flush()
                claim_id_for_resp = int(claim.id)
                claim_code_for_resp = str(claim.claim_code)

        await session.refresh(user)

    # Daily tracking: roulette spun successfully
    try:
        async with async_session_maker() as s:
            await _mark_daily_done(s, tid, _today_key(), "spin_roulette")
            await s.commit()
    except Exception:
        pass

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

class ConvertReq(BaseModel):
    telegram_id: int
    spin_id: int

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
                if str(existing_claim.status) not in {"draft", "awaiting_contact"}:
                    raise HTTPException(status_code=400, detail="Заявка уже в обработке — конвертация недоступна")
                await session.delete(existing_claim)

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

    # Daily tracking: any conversion counts as 'convert_prize'
    try:
        async with async_session_maker() as s:
            await _mark_daily_done(s, tid, _today_key(), "convert_prize")
            await s.commit()
    except Exception:
        pass

    return {"ok": True, "balance_after": int(user.points or 0), "converted_value": int(DIOR_PALETTE_CONVERT_VALUE)}

# -----------------------------------------------------------------------------
# Telegram Bot (kept as in base, minimal)
# -----------------------------------------------------------------------------
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

async def open_channel_clean(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CHANNEL_USERNAME:
        return
    try:
        await update.effective_chat.send_message(f"https://t.me/{CHANNEL_USERNAME}")
    except Exception:
        logger.exception("open_channel_clean failed")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tid = int(update.effective_user.id)
    async with async_session_maker() as session:
        user = (await session.execute(select(User).where(User.telegram_id == tid))).scalar_one_or_none()
        if not user:
            user = User(
                telegram_id=tid,
                username=update.effective_user.username,
                first_name=update.effective_user.first_name,
                points=10,
            )
            _recalc_tier(user)
            session.add(user)
            await session.commit()
    await update.message.reply_text("NS · Natural Sense", reply_markup=get_main_keyboard())

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (update.message.text or "").strip()
    if txt == "↩️ В канал":
        await open_channel_clean(update, context)
        return
    if txt == "👤 Профиль":
        await update.message.reply_text("Профиль доступен в Mini App.", reply_markup=get_main_keyboard())
        return
    if txt == "ℹ️ Помощь":
        await update.message.reply_text("Открой Mini App → Бонусы → Daily бонусы.", reply_markup=get_main_keyboard())
        return
    await update.message.reply_text("Открой Mini App кнопкой ниже.", reply_markup=get_main_keyboard())

def build_bot_app():
    app_ = Application.builder().token(BOT_TOKEN).build()
    app_.add_handler(CommandHandler("start", cmd_start))
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app_

bot_app: Application | None = None

@app.on_event("startup")
async def startup_event():
    global bot_app
    if BOT_TOKEN:
        bot_app = build_bot_app()
        asyncio.create_task(bot_app.initialize())
        asyncio.create_task(bot_app.start())
        logger.info("Bot started (polling inside FastAPI process)")

@app.on_event("shutdown")
async def shutdown_event():
    global bot_app
    if bot_app:
        try:
            await bot_app.stop()
            await bot_app.shutdown()
        except Exception:
            pass

@app.get("/health")
async def health():
    return {"ok": True}
