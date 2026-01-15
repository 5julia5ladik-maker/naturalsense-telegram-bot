import os
import re
import sqlite3
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    WebAppInfo,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ns")

# =========================
# ENV / CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")  # ОБЯЗАТЕЛЬНО реальный из BotFather
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # домен Railway, без /
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "NaturalSense").lstrip("@")
CHANNEL_URL = f"https://t.me/{CHANNEL_USERNAME}"
CHANNEL_CHAT_ID = os.getenv("CHANNEL_CHAT_ID", "").strip()  # optional: -100xxxxxxxxxx (если хочешь pin в канале)
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0") or "0")  # optional

DB_PATH = "tags.db"
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")
PAGE_SIZE = 12

# =========================
# MENU DATA (минимум как ты хотел)
# =========================
CATEGORIES = [
    ("🆕 Новинка", "#Новинка"),
    ("💎 Люкс", "#Люкс"),
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
    ("🏷 Скидки / находки", "#SephoraSale"),
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
# FASTAPI (Mini App / API)
# =========================
app = FastAPI()

MINIAPP_HTML = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>NS · Natural Sense</title>
<style>
  :root{--bg:#0c0c10;--line:#22222b;--panel:#101018;--text:#f2ede4;--muted:#b9b2a7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
  .wrap{max-width:520px;margin:0 auto;min-height:100vh;padding:18px;display:flex;flex-direction:column;gap:14px}
  .title{font-weight:800;font-size:20px}
  .sub{color:var(--muted);font-size:12px;letter-spacing:.8px;text-transform:lowercase;margin-top:4px}
  .panel{border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.02);padding:14px}
  .btn{width:100%;text-align:left;padding:14px 12px;border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.03);color:var(--text);font-size:14px;cursor:pointer;margin-top:10px}
  .btn:hover{background:rgba(255,255,255,.06)}
  .small{display:block;color:var(--muted);font-size:12px;margin-top:4px}
</style>
</head>
<body>
<div class="wrap">
  <div>
    <div class="title">NS · Natural Sense</div>
    <div class="sub">luxury beauty magazine</div>
  </div>

  <div class="panel">
    <div style="font-weight:700;margin-bottom:8px;">Выберите раздел 👇</div>
    <button class="btn" onclick="go('cat')">📂 Категории<span class="small">по тегам</span></button>
    <button class="btn" onclick="go('brand')">🏷 Бренды<span class="small">по тегам</span></button>
    <button class="btn" onclick="go('seph')">💸 Sephora<span class="small">цены / находки</span></button>
    <button class="btn" onclick="openChannel()">↩ В канал<span class="small">открыть @NaturalSense</span></button>
  </div>
</div>

<script>
async function cfg(){ return (await fetch("/api/config")).json(); }
async function go(which){
  const c = await cfg();
  const map = {cat:c.categories, brand:c.brands, seph:c.sephora};
  const list = map[which] || [];
  let html = '<div class="wrap"><div><div class="title">NS · Natural Sense</div><div class="sub">luxury beauty magazine</div></div>';
  html += '<div class="panel"><div style="font-weight:700;margin-bottom:8px;">Выберите пункт</div>';
  list.forEach(it=>{
    html += `<button class="btn" onclick="openTag('${it.tag}')">${it.title}<span class="small">${it.tag}</span></button>`;
  });
  html += `<button class="btn" onclick="location.href='/webapp'">← Назад</button>`;
  html += '</div></div>';
  document.body.innerHTML = html;
}
async function openTag(tag){
  const c = await cfg();
  const url = `/api/posts?tag=${encodeURIComponent(tag)}&offset=0`;
  const data = await (await fetch(url)).json();
  let html = '<div class="wrap"><div><div class="title">NS · Natural Sense</div><div class="sub">luxury beauty magazine</div></div>';
  html += `<div class="panel"><div style="font-weight:700;margin-bottom:8px;">${tag} · материалов: ${data.total}</div>`;
  if(data.total === 0){
    html += `<div style="color:var(--muted)">Пока нет постов с этим тегом. Добавь тег в посты канала.</div>`;
  } else {
    data.posts.forEach(p=>{
      html += `<button class="btn" onclick="window.open('${p.url}','_blank')">Открыть пост #${p.message_id}<span class="small">${p.url}</span></button>`;
    });
  }
  html += `<button class="btn" onclick="location.href='/webapp'">← Назад</button>`;
  html += '</div></div>';
  document.body.innerHTML = html;
}
async function openChannel(){
  const c = await cfg();
  window.open(c.channel_url, "_blank");
}
</script>
</body>
</html>
"""

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
# TELEGRAM BOT
# =========================
tg_app: Optional[Application] = None

def kb_main() -> InlineKeyboardMarkup:
    # Главное меню в боте (не в канале)
    rows = [
        [InlineKeyboardButton("📂 Категории", callback_data="m:cat")],
        [InlineKeyboardButton("🏷 Бренды", callback_data="m:brand")],
        [InlineKeyboardButton("💸 Sephora", callback_data="m:seph")],
    ]
    # Mini App кнопка (как приложение)
    if PUBLIC_BASE_URL:
        rows.append([InlineKeyboardButton("✦ Open Journal", web_app=WebAppInfo(url=f"{PUBLIC_BASE_URL}/webapp"))])
    rows.append([InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)])
    return InlineKeyboardMarkup(rows)

def kb_list(kind: str) -> InlineKeyboardMarkup:
    if kind == "cat":
        items = CATEGORIES
        back = "m:home"
    elif kind == "brand":
        items = BRANDS
        back = "m:home"
    else:
        items = SEPHORA
        back = "m:home"

    rows = []
    for title, tag in items:
        rows.append([InlineKeyboardButton(title, callback_data=f"t:{tag}:0")])
    rows.append([InlineKeyboardButton("← Назад", callback_data=back)])
    return InlineKeyboardMarkup(rows)

def kb_posts(tag: str, offset: int, total: int) -> InlineKeyboardMarkup:
    rows = []
    # пагинация
    prev_off = max(0, offset - PAGE_SIZE)
    next_off = offset + PAGE_SIZE
    nav = []
    if offset > 0:
        nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"t:{tag}:{prev_off}"))
    if next_off < total:
        nav.append(InlineKeyboardButton("Next ▶", callback_data=f"t:{tag}:{next_off}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("← В меню", callback_data="m:home")])
    rows.append([InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)])
    return InlineKeyboardMarkup(rows)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "NS · Natural Sense\nluxury beauty journal\n\nВыберите раздел 👇",
        reply_markup=kb_main()
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ OK")

async def cmd_pinmenu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Закрепить меню в канале (если заданы CHANNEL_CHAT_ID и права админа)
    if not CHANNEL_CHAT_ID:
        await update.message.reply_text("❌ CHANNEL_CHAT_ID не задан в Variables (нужно -100xxxxxxxxxx).")
        return

    text = "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇"
    msg = await context.bot.send_message(chat_id=CHANNEL_CHAT_ID, text=text, reply_markup=kb_main())
    try:
        await context.bot.pin_chat_message(chat_id=CHANNEL_CHAT_ID, message_id=msg.message_id, disable_notification=True)
        await update.message.reply_text("✅ Меню отправлено и закреплено.")
    except Exception as e:
        await update.message.reply_text(f"❌ Не смог закрепить: {e}")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    # Меню
    if data == "m:home":
        await q.edit_message_text("NS · Natural Sense\nluxury beauty journal\n\nВыберите раздел 👇", reply_markup=kb_main())
        return
    if data == "m:cat":
        await q.edit_message_text("📂 Категории\nВыберите пункт:", reply_markup=kb_list("cat"))
        return
    if data == "m:brand":
        await q.edit_message_text("🏷 Бренды\nВыберите пункт:", reply_markup=kb_list("brand"))
        return
    if data == "m:seph":
        await q.edit_message_text("💸 Sephora\nВыберите пункт:", reply_markup=kb_list("seph"))
        return

    # Теги: t:#Dior:0
    if data.startswith("t:"):
        try:
            _, tag, off = data.split(":", 2)
            offset = int(off)
        except Exception:
            await q.edit_message_text("❌ Ошибка кнопки. Вернись в меню.", reply_markup=kb_main())
            return

        total = db_count(tag)
        ids = db_list(tag, PAGE_SIZE, offset)

        if total == 0:
            text = f"{tag}\n\nПока нет постов с этим тегом.\nДобавь тег в посты канала и попробуй снова."
            await q.edit_message_text(text, reply_markup=kb_posts(tag, offset, total))
            return

        # формируем список ссылок
        lines = [f"{tag} · материалов: {total}\n"]
        for mid in ids:
            lines.append(f"• {CHANNEL_URL}/{mid}")
        text = "\n".join(lines)

        await q.edit_message_text(text, disable_web_page_preview=True, reply_markup=kb_posts(tag, offset, total))
        return

    await q.edit_message_text("Меню:", reply_markup=kb_main())

# =========================
# WEBHOOK ENDPOINT
# =========================
@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    if not tg_app:
        return JSONResponse({"ok": False, "error": "tg_app not ready"}, status_code=503)

    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)

    # Индексация тегов из канала (важно: бот должен быть админом канала, чтобы получать channel_post)
    if update and update.channel_post:
        text = update.channel_post.text or update.channel_post.caption or ""
        tags = extract_tags(text)
        if tags:
            for t in tags:
                db_add(t, update.channel_post.message_id)

    return JSONResponse({"ok": True})

# =========================
# STARTUP / SHUTDOWN
# =========================
@app.on_event("startup")
async def on_startup():
    global tg_app

    if not BOT_TOKEN:
        log.error("BOT_TOKEN is empty. Set it in Railway Variables.")
        return

    tg_app = Application.builder().token(BOT_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(CommandHandler("ping", cmd_ping))
    tg_app.add_handler(CommandHandler("pinmenu", cmd_pinmenu))
    tg_app.add_handler(CallbackQueryHandler(on_callback))

    await tg_app.initialize()
    await tg_app.start()

    # Ставим webhook (обязательно PUBLIC_BASE_URL)
    if PUBLIC_BASE_URL:
        wh = f"{PUBLIC_BASE_URL}/telegram/webhook"
        try:
            await tg_app.bot.set_webhook(url=wh, drop_pending_updates=True)
            log.info("Webhook set to %s", wh)
        except Exception as e:
            log.error("Webhook set failed: %s", e)
    else:
        log.warning("PUBLIC_BASE_URL is empty. Webhook can't be set.")

@app.on_event("shutdown")
async def on_shutdown():
    if tg_app:
        await tg_app.stop()
        await tg_app.shutdown()
