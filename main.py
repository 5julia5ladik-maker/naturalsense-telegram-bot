import os
import re
import sqlite3
import logging
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4")  # фейковый
PUBLIC_BASE_URL = os.getenv("https://naturalsense-telegram-bot-production.up.railway.app/
", "").rstrip("/")  # https://xxx.up.railway.app
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "NaturalSense")
CHANNEL_URL = f"https://t.me/NaturalSense"
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

# =========================
# FASTAPI APP
# =========================
app = FastAPI()
db_init()

# =========================
# MINI APP (HTML/CSS/JS) — В ОДНОМ ОТВЕТЕ
# =========================
MINIAPP_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Natural Sense · Journal</title>
  <style>
    :root{
      --bg:#0b0b0d;
      --panel:#111116;
      --text:#f3f1ed;
      --muted:#b8b2a8;
      --line:#22222a;
      --btn:#171720;
      --btn2:#0f0f14;
      --accent:#e7dcc7;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .app{max-width:520px;margin:0 auto;min-height:100vh;display:flex;flex-direction:column;
      padding:18px;gap:14px}
    .top{padding:12px 12px 0 12px}
    .brand{font-weight:600;letter-spacing:.2px;font-size:20px}
    .subtitle{margin-top:6px;color:var(--muted);font-size:13px;letter-spacing:.6px;text-transform:lowercase}
    .screen{flex:1;border:1px solid var(--line);border-radius:18px;padding:14px;
      background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,0))}
    .card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:16px;padding:14px;margin-bottom:12px}
    .h1{font-size:18px;font-weight:600;margin:0 0 8px 0}
    .p{margin:0;color:var(--muted);font-size:13px;line-height:1.35}
    .grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}
    .btn{width:100%;padding:14px 12px;border-radius:14px;border:1px solid var(--line);
      background:var(--btn);color:var(--text);font-size:14px;text-align:left;cursor:pointer}
    .btn:hover{background:var(--btn2)}
    .btn .small{display:block;color:var(--muted);font-size:12px;margin-top:4px}
    .footer{display:flex;gap:10px;justify-content:space-between;align-items:center}
    .ghost{border:1px solid var(--line);background:transparent;color:var(--muted);
      padding:10px 12px;border-radius:12px;text-decoration:none;cursor:pointer;font-size:13px}
    .list{display:flex;flex-direction:column;gap:10px;margin-top:10px}
    .item{display:flex;justify-content:space-between;gap:10px;align-items:center;
      border:1px solid var(--line);border-radius:14px;padding:12px;background:rgba(255,255,255,.02)}
    .item a{color:var(--accent);text-decoration:none;font-weight:600}
    .pager{display:flex;gap:10px;margin-top:12px}
  </style>
</head>
<body>
  <div class="app">
    <div class="top">
      <div class="brand">NS · Natural Sense</div>
      <div class="subtitle">luxury beauty journal</div>
    </div>

    <div id="screen" class="screen"></div>

    <div class="footer">
      <button class="ghost" id="btnBack" style="display:none;">Back</button>
      <a class="ghost" id="btnChannel" target="_blank" rel="noreferrer">Open channel</a>
    </div>
  </div>

<script>
let config=null;
let stack=[];
const screen=document.getElementById("screen");
const btnBack=document.getElementById("btnBack");
const btnChannel=document.getElementById("btnChannel");

btnBack.addEventListener("click", ()=>{ stack.pop(); render(); });

function push(view){ stack.push(view); render(); }
function card(title, subtitle){
  return `<div class="card"><div class="h1">${title}</div><div class="p">${subtitle}</div></div>`;
}
function gridButtons(html){ return `<div class="grid">${html}</div>`; }

function button(label, sub, onClick){
  const id="b_"+Math.random().toString(16).slice(2);
  setTimeout(()=>{ const el=document.getElementById(id); if(el) el.onclick=onClick; },0);
  return `<button class="btn" id="${id}">${label}${sub?`<span class="small">${sub}</span>`:""}</button>`;
}

function homeView(){ return {type:"home"}; }
function sectionCoverView(sectionKey,title){ return {type:"sectionCover", sectionKey, title}; }
function sectionListView(sectionKey,title){ return {type:"sectionList", sectionKey, title}; }
function tagCoverView(tag,title){ return {type:"tagCover", tag, title}; }
function postsView(tag,title,offset=0){ return {type:"posts", tag, title, offset}; }

async function loadConfig(){
  const r=await fetch("/api/config");
  config=await r.json();
  btnChannel.href=config.channel_url;
}

async function render(){
  if(!config) return;
  btnBack.style.display = stack.length>1 ? "inline-flex" : "none";
  const view=stack[stack.length-1];

  if(view.type==="home"){
    screen.innerHTML =
      card("NS · Natural Sense","luxury beauty journal") +
      gridButtons(
        button("📂 Categories","Editorial sections", ()=>push(sectionCoverView("categories","Categories"))) +
        button("🏷 Brands","Houses & icons", ()=>push(sectionCoverView("brands","Brands"))) +
        button("💸 Sephora","Curated picks & updates", ()=>push(sectionCoverView("sephora","Sephora"))) +
        button("💎 Beauty Challenges","Editorial events", ()=>alert("MVP: позже добавим")) +
        button("↩ Open channel", config.channel_url, ()=>window.open(config.channel_url,"_blank"))
      );
    return;
  }

  if(view.type==="sectionCover"){
    screen.innerHTML =
      card(`Natural Sense · ${view.title}`, "luxury beauty journal") +
      gridButtons(
        button("✦ Open","Continue", ()=>push(sectionListView(view.sectionKey, view.title))) +
        button("Back","", ()=>{stack.pop(); render();})
      );
    return;
  }

  if(view.type==="sectionList"){
    const arr=config[view.sectionKey]||[];
    let btns="";
    for(const it of arr){
      btns += button(it.title, `tag: ${it.tag}`, ()=>push(tagCoverView(it.tag, it.title)));
    }
    screen.innerHTML = card(view.title,"Choose a section") + gridButtons(btns);
    return;
  }

  if(view.type==="tagCover"){
    screen.innerHTML =
      card(view.title, `tag: ${view.tag}`) +
      gridButtons(
        button("✦ Open materials","Articles in channel", ()=>push(postsView(view.tag, view.title, 0))) +
        button("Back","", ()=>{stack.pop(); render();})
      );
    return;
  }

  if(view.type==="posts"){
    const r=await fetch(`/api/posts?tag=${encodeURIComponent(view.tag)}&offset=${view.offset}`);
    const data=await r.json();

    let list="";
    if(data.total===0){
      list = `<div class="card"><div class="p">No materials yet for ${view.tag}. Publish new posts with this tag.</div></div>`;
    } else {
      list = `<div class="card"><div class="p">${view.title} · materials: ${data.total}</div></div>`;
      list += `<div class="list">` + data.posts.map(p => `
        <div class="item">
          <div class="p">Material #${p.message_id}</div>
          <a href="${p.url}" target="_blank" rel="noreferrer">Open</a>
        </div>
      `).join("") + `</div>`;
    }

    const prev=Math.max(0, view.offset - data.limit);
    const next=view.offset + data.limit;

    list += `
      <div class="pager">
        <button class="ghost" id="prevBtn" ${view.offset<=0?"disabled":""}>Prev</button>
        <button class="ghost" id="nextBtn" ${(next>=data.total)?"disabled":""}>Next</button>
      </div>
    `;

    screen.innerHTML=list;

    setTimeout(()=>{
      const p=document.getElementById("prevBtn");
      const n=document.getElementById("nextBtn");
      if(p) p.onclick=()=>{ view.offset=prev; render(); };
      if(n) n.onclick=()=>{ view.offset=next; render(); };
    },0);

    return;
  }
}

(async function init(){
  await loadConfig();
  stack=[homeView()];
  render();
})();
</script>
</body>
</html>
"""

# =========================
# MINI APP ROUTE
# =========================
@app.get("/webapp", response_class=HTMLResponse)
def webapp():
    return HTMLResponse(MINIAPP_HTML)

# =========================
# API
# =========================
@app.get("/api/config")
def api_config():
    return {
        "channel_url": CHANNEL_URL,
        "categories": [
            {"title": "Новинка", "tag": "#Новинка"},
            {"title": "Люкс", "tag": "#Люкс"},
            {"title": "Тренд", "tag": "#Тренд"},
            {"title": "Оценка", "tag": "#Оценка"},
            {"title": "Факты / состав", "tag": "#Факты"},
        ],
        "brands": [
            {"title": "Dior", "tag": "#Dior"},
            {"title": "Chanel", "tag": "#Chanel"},
            {"title": "Charlotte", "tag": "#Charlotte"},
            {"title": "YSL", "tag": "#YSL"},
        ],
        "sephora": [
            {"title": "Новинки", "tag": "#SephoraNew"},
            {"title": "Best sellers", "tag": "#SephoraTop"},
            {"title": "Выгодно сейчас", "tag": "#SephoraSale"},
        ],
    }

@app.get("/api/posts")
def api_posts(tag: str, offset: int = 0, limit: int = PAGE_SIZE):
    total = db_count(tag)
    ids = db_list(tag, limit, offset)
    posts = [{"message_id": mid, "url": f"{CHANNEL_URL}/{mid}"} for mid in ids]
    return {"tag": tag, "total": total, "offset": offset, "limit": limit, "posts": posts}

@app.get("/", response_class=HTMLResponse)
def root():
    return f"""
    <html>
      <body style="font-family:Arial">
        <h2>NS Mini App is running</h2>
        <p>Open mini app: <a href="/webapp">/webapp</a></p>
      </body>
    </html>
    """

# =========================
# TELEGRAM BOT (WEBHOOK)
# =========================
tg_app: Optional[Application] = None

def home_kb():
    if not PUBLIC_BASE_URL:
        # мини апп не откроется без публичного домена
        return InlineKeyboardMarkup([[InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)]])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✦ Open Journal", web_app=WebAppInfo(url=f"{PUBLIC_BASE_URL}/webapp"))],
        [InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)],
    ])

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("NS · Natural Sense\nluxury beauty journal", reply_markup=home_kb())

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ OK")

async def init_telegram():
    global tg_app
    if tg_app is not None:
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("ping", cmd_ping))

    await tg_app.initialize()
    await tg_app.start()

@app.on_event("startup")
async def on_startup():
    await init_telegram()

    if PUBLIC_BASE_URL:
        webhook_url = f"{PUBLIC_BASE_URL}/telegram/webhook"
        try:
            await tg_app.bot.set_webhook(url=webhook_url)
            logging.info("Webhook set: %s", webhook_url)
        except Exception as e:
            logging.error("Webhook set failed: %s", e)

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()

@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    if not tg_app:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    data = await req.json()
    update = Update.de_json(data, tg_app.bot)

    # обработка апдейта
    await tg_app.process_update(update)

    # индексируем посты из канала
    if update and update.channel_post:
        text = update.channel_post.text or update.channel_post.caption or ""
        tags = extract_tags(text)
        for t in tags:
            db_add(t, update.channel_post.message_id)

    return JSONResponse({"ok": True})
