import os
import re
import sqlite3
import logging
from typing import Optional, List, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG (можешь менять здесь, но лучше через Railway Variables)
# =========================
BOT_TOKEN = os.getenv("8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4", "").strip()  # обязательно реальный
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")  # https://xxx.up.railway.app
BOT_USERNAME = os.getenv("BOT_USERNAME", "naturalsense_assistant_bot").lstrip("@")  # юзернейм бота
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "NaturalSense").lstrip("@")  # юзернейм канала
CHANNEL_URL = f"https://t.me/{CHANNEL_USERNAME}"

DB_PATH = "tags.db"
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")
PAGE_SIZE = 8

# =========================
# TAG LIST (как ты просил)
# =========================
CATEGORIES = [
    ("🆕 Новинка", "#Новинка"),
    ("💎 Кратко о люкс продукте", "#Люкс"),
    ("🔥 Тренд", "#Тренд"),
    ("🏛 История бренда", "#История"),
    ("⭐ Личная оценка продукта", "#Оценка"),
    ("🧴 Тип продукта / факты", "#Факты"),
    ("🧪 Составы продуктов", "#Состав"),
]

BRANDS = [
    ("Dior", "#Dior"),
    ("Chanel", "#Chanel"),
    ("YSL", "#YSL"),
    ("Charlotte Tilbury", "#Charlotte"),
]

SEPHORA = [
    ("💸 Актуальные цены", "#SephoraPrice"),
    ("🆕 Новинки Sephora", "#SephoraNew"),
    ("🏷 Скидки / находки", "#SephoraDeals"),
]

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

def db_count(tag: str) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM tag_posts WHERE tag = ?", (tag,))
    n = cur.fetchone()[0]
    con.close()
    return int(n)

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

db_init()

# =========================
# MINI APP (как "приложение")
# =========================
MINIAPP_HTML = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NS · Natural Sense</title>
<style>
:root{
  --bg:#0b0b0d;
  --panel:#101016;
  --line:#22222a;
  --text:#f2efe9;
  --muted:#b9b2a7;
  --btn:#16161f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
.wrap{max-width:540px;margin:0 auto;min-height:100vh;padding:18px;display:flex;flex-direction:column;gap:14px}
.head{padding:6px 2px 0}
.brand{font-weight:800;font-size:20px;letter-spacing:.2px}
.sub{margin-top:6px;color:var(--muted);font-size:12px;letter-spacing:.9px;text-transform:lowercase}
.card{border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:18px;padding:14px}
.h{font-size:16px;font-weight:800;margin:0 0 8px}
.p{margin:0;color:var(--muted);font-size:13px;line-height:1.35}
.grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:12px}
.btn{width:100%;text-align:left;padding:14px 12px;border-radius:16px;border:1px solid var(--line);
  background:var(--btn);color:var(--text);font-size:14px;cursor:pointer}
.small{display:block;color:var(--muted);font-size:12px;margin-top:4px}
.list{display:flex;flex-direction:column;gap:10px;margin-top:10px}
.item{display:flex;justify-content:space-between;gap:10px;align-items:center;
  border:1px solid var(--line);border-radius:16px;padding:12px;background:rgba(255,255,255,.02)}
.item a{color:#e9dcc7;text-decoration:none;font-weight:800}
.footer{display:flex;gap:10px;justify-content:space-between}
.ghost{border:1px solid var(--line);background:transparent;color:var(--muted);
  padding:10px 12px;border-radius:14px;text-decoration:none;cursor:pointer;font-size:13px}
.pager{display:flex;gap:10px;margin-top:12px}
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div class="brand">NS · Natural Sense</div>
    <div class="sub">luxury beauty magazine</div>
  </div>

  <div class="card" id="screen"></div>

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

function btn(label, sub, onClick){
  const id="b_"+Math.random().toString(16).slice(2);
  setTimeout(()=>{ const el=document.getElementById(id); if(el) el.onclick=onClick; },0);
  return <button class="btn" id="${id}">${label}${sub?<span class="small">${sub}</span>:""}</button>;
}
function grid(html){ return <div class="grid">${html}</div>; }

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
      <div class="h">Выберите раздел 👇</div><div class="p">всё как в приложении</div> +
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
    let html=<div class="h">${v.title}</div><div class="p">выберите пункт</div>;
    let b="";
    for(const it of arr){
      b += btn(it.title, tag: ${it.tag}, ()=>push({type:"posts", title:it.title, tag:it.tag, offset:0}));
    }
    screen.innerHTML = html + grid(b);
    return;
  }

  if(v.type==="posts"){
    const r=await fetch(/api/posts?tag=${encodeURIComponent(v.tag)}&offset=${v.offset});
    const data=await r.json();

    let html = <div class="h">${v.title}</div><div class="p">Материалов: ${data.total} · ${v.tag}</div>;

    if(data.total===0){
      html += <div class="p" style="margin-top:10px">Пока нет постов с тегом ${v.tag}. Добавь тег в посты канала.</div>;
      screen.innerHTML = html;
      return;
    }

    html += <div class="list"> + data.posts.map(p=>
      <div class="item">
        <div class="p">Пост #${p.message_id}</div>
        <a href="${p.url}" target="_blank" rel="noreferrer">Открыть</a>
      </div>
    ).join("") + </div>;

    const prev=Math.max(0, v.offset - data.limit);
    const next=v.offset + data.limit;
    html += 
      <div class="pager">
        <button class="ghost" id="prev" ${v.offset<=0?"disabled":""}>Prev</button>
        <button class="ghost" id="next" ${(next>=data.total)?"disabled":""}>Next</button>
      </div>
    ;

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
# FastAPI
# =========================
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse("<h3>OK</h3><p>MiniApp: <a href='/webapp'>/webapp</a></p>")

@app.get("/webapp", response_class=HTMLResponse)
def webapp():
    return HTMLResponse(MINIAPP_HTML)

@app.get("/api/config")
def api_config():
    return {
        "channel_url": CHANNEL_URL,
        "categories": [{"title": t, "tag": tag} for (t, tag) in CATEGORIES],
        "brands": [{"title": t, "tag": tag} for (t, tag) in BRANDS],
        "sephora": [{"title": t, "tag": tag} for (t, tag) in SEPHORA],
    }

@app.get("/api/posts")
def api_posts(tag: str, offset: int = 0, limit: int = PAGE_SIZE):
    total = db_count(tag)
    ids = db_list(tag, limit, offset)
    posts = [{"message_id": mid, "url": f"{CHANNEL_URL}/{mid}"} for mid in ids]
    return {"tag": tag, "total": total, "offset": offset, "limit": limit, "posts": posts}

# =========================
# Telegram Bot (python-telegram-bot)
# =========================
tg_app: Optional[Application] = None

def deep_link(payload: str) -> str:
    # открывает бота по кнопке из канала
    return f"https://t.me/{BOT_USERNAME}?start={payload}"

def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", callback_data="menu:categories")],
        [InlineKeyboardButton("🏷 Бренды", callback_data="menu:brands")],
        [InlineKeyboardButton("💸 Sephora", callback_data="menu:sephora")],
        [InlineKeyboardButton("✦ Открыть журнал (app)", web_app=WebAppInfo(url=f"{PUBLIC_BASE_URL}/webapp"))] if PUBLIC_BASE_URL else [],
        [InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)],
    ])

def list_kb(kind: str, items: List[Tuple[str, str]]) -> InlineKeyboardMarkup:
    rows = []
    for title, tag in items:
        rows.append([InlineKeyboardButton(title, callback_data=f"tag:{kind}:{tag}:0")])
    rows.append([InlineKeyboardButton("⬅ Назад", callback_data="back:main")])
    return InlineKeyboardMarkup(rows)

def posts_kb(kind: str, tag: str, offset: int, total: int, ids: List[int]) -> InlineKeyboardMarkup:
    rows = []
    for mid in ids:
        rows.append([InlineKeyboardButton(f"Открыть пост #{mid}", url=f"{CHANNEL_URL}/{mid}")])

    nav = []
    if offset > 0:
        nav.append(InlineKeyboardButton("⬅ Prev", callback_data=f"tag:{kind}:{tag}:{max(0, offset-PAGE_SIZE)}"))
    if offset + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("Next ➡", callback_data=f"tag:{kind}:{tag}:{offset+PAGE_SIZE}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("⬅ Назад", callback_data=f"back:{kind}")])
    rows.append([InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)])
    return InlineKeyboardMarkup(rows)

def kind_items(kind: str) -> List[Tuple[str, str]]:
    if kind == "categories":
        return CATEGORIES
    if kind == "brands":
        return BRANDS
    return SEPHORA

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ OK")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /start или /start categories /start brands /start sephora
    payload = ""
    if context.args:
        payload = context.args[0].strip().lower()
        if payload in ("categories", "brands", "sephora"):
        items = kind_items(payload)
        await update.message.reply_text("Выберите пункт 👇", reply_markup=list_kb(payload, items))
        return

    await update.message.reply_text(
        "NS · Natural Sense\nluxury beauty journal",
        reply_markup=main_menu_kb()
    )

async def cmd_pinmenu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Шлёт пост в канал и закрепляет его.
    ВАЖНО: бот должен быть админом канала с правом Manage Messages.
    """
    bot = context.bot

    text = "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", url=deep_link("categories"))],
        [InlineKeyboardButton("🏷 Бренды", url=deep_link("brands"))],
        [InlineKeyboardButton("💸 Sephora", url=deep_link("sephora"))],
        [InlineKeyboardButton("✦ Открыть журнал (app)", url=f"{PUBLIC_BASE_URL}/webapp")] if PUBLIC_BASE_URL else [],
        [InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)],
    ])

    msg = await bot.send_message(chat_id=f"@{CHANNEL_USERNAME}", text=text, reply_markup=kb, disable_web_page_preview=True)
    try:
        await bot.pin_chat_message(chat_id=f"@{CHANNEL_USERNAME}", message_id=msg.message_id, disable_notification=True)
    except Exception as e:
        logging.error("Pin failed: %s", e)
        await update.message.reply_text("Меню отправил, но закрепить не смог — проверь права бота в канале (Manage Messages).")
        return

    await update.message.reply_text("✅ Меню отправлено и закреплено.")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    data = q.data or ""

    if data == "back:main":
        await q.edit_message_text("NS · Natural Sense\nluxury beauty journal", reply_markup=main_menu_kb())
        return

    if data.startswith("menu:"):
        kind = data.split(":", 1)[1]
        items = kind_items(kind)
        await q.edit_message_text("Выберите пункт 👇", reply_markup=list_kb(kind, items))
        return

    if data.startswith("back:"):
        kind = data.split(":", 1)[1]
        items = kind_items(kind)
        await q.edit_message_text("Выберите пункт 👇", reply_markup=list_kb(kind, items))
        return

    if data.startswith("tag:"):
        # tag:<kind>:<tag>:<offset>
        parts = data.split(":")
        if len(parts) != 4:
            return
        kind, tag, offset_s = parts[1], parts[2], parts[3]
        try:
            offset = int(offset_s)
        except:
            offset = 0

        total = db_count(tag)
        ids = db_list(tag, PAGE_SIZE, offset)
        title = tag

        if total == 0:
            await q.edit_message_text(
                f"Пока нет постов с тегом {tag}.\nДобавь тег в посты канала — и бот начнет показывать их.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⬅ Назад", callback_data=f"back:{kind}")],
                    [InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)]
                ])
            )
            return

        await q.edit_message_text(
            f"{title}\nМатериалов: {total}",
            reply_markup=posts_kb(kind, tag, offset, total, ids),
            disable_web_page_preview=True
        )
        return

async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # индексируем теги из постов канала
    post = update.channel_post
    if not post:
        return
    text = post.text or post.caption or ""
    tags = extract_tags(text)
    if not tags:
        return
    for t in tags:
        db_add(t, post.message_id)
    logging.info("Indexed tags %s for msg %s", tags, post.message_id)

@app.on_event("startup")
async def startup():
    global tg_app

    if not BOT_TOKEN:
        logging.error("BOT_TOKEN is empty. Set Railway Variable BOT_TOKEN.")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("ping", cmd_ping))
    tg_app.add_handler(CommandHandler("pinmenu", cmd_pinmenu))
    tg_app.add_handler(CallbackQueryHandler(on_callback))
    tg_app.add_handler(CommandHandler("help", cmd_start))

    # channel posts (для индекса тегов)
    tg_app.add_handler(
        # это сработает на channel_post апдейты
        # (в PTB channel_post приходит как Update.channel_post, handler можно не отдельный, но так стабильнее)
        # используем MessageHandler не надо — обработаем через application.process_update ниже в webhook
        # поэтому просто оставим функцию ниже, а тег-индексация будет в /telegram/webhook
        CommandHandler("_noop", cmd_ping)
    )

    await tg_app.initialize()
    await tg_app.start()

    # ставим webhook
    if PUBLIC_BASE_URL:
        webhook_url = f"{PUBLIC_BASE_URL}/telegram/webhook"
        try:
            await tg_app.bot.set_webhook(webhook_url)
            logging.info("Webhook set: %s", webhook_url)
        except Exception as e:
            logging.error("Webhook set failed: %s", e)
    else:
        logging.warning("PUBLIC_BASE_URL empty. Webhook not set.")

@app.on_event("shutdown")
async def shutdown():
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()

@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    if not tg_app:
        return JSONResponse({"ok": False, "error": "tg_app not ready"})

    data = await req.json()

    upd = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(upd)

    # индексируем теги из канала прямо здесь
    if upd and upd.channel_post:
        post = upd.channel_post
        text = post.text or post.caption or ""
        for t in extract_tags(text):
            db_add(t, post.message_id)

    return JSONResponse({"ok": True})
