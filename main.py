import os
import re
import sqlite3
import logging
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4")  # фейковый
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # сюда НЕ вставляй ссылку в код
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "NaturalSense")
CHANNEL_URL = f"https://t.me/{CHANNEL_USERNAME}"

DB_PATH = "tags.db"
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")
PAGE_SIZE = 12

# =========================
# DB
# =========================
def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tag_posts (
            tag TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            PRIMARY KEY(tag, message_id)
        )
    """)
    con.commit()
    con.close()

def extract_tags(text: str) -> List[str]:
    if not text:
        return []
    return [f"#{m.group(1)}" for m in TAG_RE.finditer(text)]

def db_add(tag: str, message_id: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO tag_posts(tag, message_id) VALUES(?, ?)", (tag, message_id))
    con.commit()
    con.close()

def db_list(tag: str, limit: int, offset: int) -> List[int]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT message_id FROM tag_posts
        WHERE tag = ?
        ORDER BY message_id DESC
        LIMIT ? OFFSET ?
    """, (tag, limit, offset))
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]

def db_count(tag: str) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM tag_posts WHERE tag = ?", (tag,))
    n = cur.fetchone()[0]
    con.close()
    return int(n)

db_init()

# =========================
# MINI APP HTML
# =========================
MINIAPP_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>NS · Natural Sense</title>
  <style>
    :root{
      --bg:#0b0b0d; --line:#22222a; --panel:#101016;
      --text:#f2efe9; --muted:#b9b2a7; --btn:#16161f; --btn2:#0f0f15;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .wrap{max-width:520px;margin:0 auto;min-height:100vh;padding:18px;display:flex;flex-direction:column;gap:14px}
    .top{padding:10px 12px 0}
    .brand{font-weight:700;font-size:20px;letter-spacing:.2px}
    .sub{margin-top:6px;color:var(--muted);font-size:12px;letter-spacing:.8px;text-transform:lowercase}
    .screen{flex:1;border:1px solid var(--line);border-radius:18px;background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,0));
      padding:14px}
    .card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:16px;padding:14px}
    .h{font-size:16px;font-weight:700;margin:0 0 8px}
    .p{margin:0;color:var(--muted);font-size:13px;line-height:1.35}
    .grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:12px}
    .btn{width:100%;text-align:left;padding:14px 12px;border-radius:14px;border:1px solid var(--line);
      background:var(--btn);color:var(--text);font-size:14px;cursor:pointer}
    .btn:hover{background:var(--btn2)}
    .small{display:block;color:var(--muted);font-size:12px;margin-top:4px}
    .footer{display:flex;gap:10px;justify-content:space-between;align-items:center}
    .ghost{border:1px solid var(--line);background:transparent;color:var(--muted);
      padding:10px 12px;border-radius:12px;text-decoration:none;cursor:pointer;font-size:13px}
    .list{display:flex;flex-direction:column;gap:10px;margin-top:10px}
    .item{display:flex;justify-content:space-between;gap:10px;align-items:center;
      border:1px solid var(--line);border-radius:14px;padding:12px;background:rgba(255,255,255,.02)}
    .item a{color:#e9dcc7;text-decoration:none;font-weight:700}
    .pager{display:flex;gap:10px;margin-top:12px}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">NS · Natural Sense</div>
    <div class="sub">luxury beauty magazine</div>
  </div>

  <div class="screen" id="screen"></div>

  <div class="footer">
    <button class="ghost" id="back" style="display:none;">Back</button>
    <a class="ghost" id="channel" target="_blank" rel="noreferrer">Open channel</a>
  </div>
</div>

<script>
let cfg=null;
let stack=[];
const screen=document.getElementById("screen");
const back=document.getElementById("back");
const channel=document.getElementById("channel");

back.onclick=()=>{ stack.pop(); render(); };

function card(t,s){ return `<div class="card"><div class="h">${t}</div><div class="p">${s}</div></div>`; }
function btn(label, sub, onClick){
  const id="b_"+Math.random().toString(16).slice(2);
  setTimeout(()=>{ const el=document.getElementById(id); if(el) el.onclick=onClick; },0);
  return `<button class="btn" id="${id}">${label}${sub?`<span class="small">${sub}</span>`:""}</button>`;
}
function grid(html){ return `<div class="grid">${html}</div>`; }

function push(v){ stack.push(v); render(); }

async function loadCfg(){
  const r=await fetch("/api/config"); cfg=await r.json();
  channel.href=cfg.channel_url;
}

async function render(){
  back.style.display = stack.length>1 ? "inline-flex" : "none";
  const v=stack[stack.length-1];

  if(v.type==="home"){
    screen.innerHTML =
      card("Выберите раздел 👇","всё как в приложении") +
      grid(
        btn("📂 Категории","sections", ()=>push({type:"list", key:"categories", title:"Категории"})) +
        btn("🏷 Бренды","brands", ()=>push({type:"list", key:"brands", title:"Бренды"})) +
        btn("💸 Sephora","prices & picks", ()=>push({type:"list", key:"sephora", title:"Sephora"})) +
        btn("↩ В канал","", ()=>window.open(cfg.channel_url,"_blank"))
      );
    return;
  }

  if(v.type==="list"){
    const arr=cfg[v.key]||[];
    let html=card(v.title,"выберите пункт");
    let b="";
    for(const it of arr){
      b += btn(it.title, `tag: ${it.tag}`, ()=>push({type:"posts", title:it.title, tag:it.tag, offset:0}));
    }
    screen.innerHTML = html + grid(b);
    return;
  }

  if(v.type==="posts"){
    const r=await fetch(`/api/posts?tag=${encodeURIComponent(v.tag)}&offset=${v.offset}`);
    const data=await r.json();

    let html = card(v.title, `Материалов: ${data.total} · ${v.tag}`);

    if(data.total===0){
      html += `<div class="card"><div class="p">Пока нет постов с тегом ${v.tag}. Добавь тег в посты канала.</div></div>`;
      screen.innerHTML = html;
      return;
    }

    html += `<div class="list">` + data.posts.map(p=>`
      <div class="item">
        <div class="p">Пост #${p.message_id}</div>
        <a href="${p.url}" target="_blank" rel="noreferrer">Открыть</a>
      </div>
    `).join("") + `</div>`;

    const prev=Math.max(0, v.offset - data.limit);
    const next=v.offset + data.limit;

    html += `
      <div class="pager">
        <button class="ghost" id="prev" ${v.offset<=0?"disabled":""}>Prev</button>
        <button class="ghost" id="next" ${(next>=data.total)?"disabled":""}>Next</button>
      </div>
    `;

    screen.innerHTML = html;

    setTimeout(()=>{
      const p=document.getElementById("prev");
      const n=document.getElementById("next");
      if(p) p.onclick=()=>{ v.offset=prev; render(); };
      if(n) n.onclick=()=>{ v.offset=next; render(); };
    },0);

    return;
  }
}

(async function(){
  await loadCfg();
  stack=[{type:"home"}];
  render();
})();
</script>
</body>
</html>
"""

# =========================
# FASTAPI
# =========================
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse("<h3>OK</h3><p>Mini App: <a href='/webapp'>/webapp</a></p>")

@app.get("/webapp", response_class=HTMLResponse)
def webapp():
    return HTMLResponse(MINIAPP_HTML)

@app.get("/api/config")
def api_config():
    return {
        "channel_url": CHANNEL_URL,
        "categories": [
            {"title": "🆕 Новинка", "tag": "#Новинка"},
            {"title": "💎 Люкс", "tag": "#Люкс"},
            {"title": "🔥 Тренд", "tag": "#Тренд"},
            {"title": "⭐ Оценка", "tag": "#Оценка"},
            {"title": "🧪 Факты/Состав", "tag": "#Факты"},
        ],
        "brands": [
            {"title": "Dior", "tag": "#Dior"},
            {"title": "Chanel", "tag": "#Chanel"},
            {"title": "YSL", "tag": "#YSL"},
            {"title": "Charlotte", "tag": "#Charlotte"},
        ],
        "sephora": [
            {"title": "Новинки", "tag": "#SephoraNew"},
            {"title": "Топ", "tag": "#SephoraTop"},
            {"title": "Скидки", "tag": "#SephoraSale"},
        ],
    }

@app.get("/api/posts")
def api_posts(tag: str, offset: int = 0, limit: int = PAGE_SIZE):
    total = db_count(tag)
    ids = db_list(tag, limit, offset)
    posts = [{"message_id": mid, "url": f"{CHANNEL_URL}/{mid}"} for mid in ids]
    return {"tag": tag, "total": total, "offset": offset, "limit": limit, "posts": posts}

# =========================
# TELEGRAM BOT
# =========================
tg_app: Optional[Application] = None

def start_keyboard() -> InlineKeyboardMarkup:
    if PUBLIC_BASE_URL:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✦ Open Journal", web_app=WebAppInfo(url=f"{PUBLIC_BASE_URL}/webapp"))],
            [InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)],
        ])
    return InlineKeyboardMarkup([[InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)]])

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("NS · Natural Sense\nluxury beauty magazine", reply_markup=start_keyboard())

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ OK")

@app.on_event("startup")
async def on_startup():
    global tg_app
    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("ping", cmd_ping))

    await tg_app.initialize()
    await tg_app.start()

    # webhook ставим только если есть домен
    if PUBLIC_BASE_URL:
        try:
            await tg_app.bot.set_webhook(url=f"{PUBLIC_BASE_URL}/telegram/webhook")
            logging.info("Webhook set")
        except Exception as e:
            logging.error("Webhook error: %s", e)

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()

@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)

    # индексируем теги из постов канала
    if update and update.channel_post:
        text = update.channel_post.text or update.channel_post.caption or ""
        for t in extract_tags(text):
            db_add(t, update.channel_post.message_id)

    return JSONResponse({"ok": True})
