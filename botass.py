import os
import re
import sqlite3
import logging
from typing import List, Tuple

from telegram import (
    Update,
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
from telegram.error import TelegramError
from telegram.constants import ChatType


logging.basicConfig(level=logging.INFO)

# =========================
# НАСТРОЙКИ (МИНИМУМ)
# =========================
TOKEN = "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4"  # фейковый как ты просил

BOT_USERNAME = "naturalsense_assistant_bot"  # без @
CHANNEL_USERNAME = "NaturalSense"
CHANNEL_URL = "https://t.me/NaturalSense"
CHANNEL_ID = "@NaturalSense"  # если приватный — будет -100...

# Опционально: картинка обложки (URL на картинку). Можно оставить пустым.
COVER_IMAGE_URL = os.getenv("COVER_IMAGE_URL", "").strip()

# =========================
# БАЗА ТЕГОВ (SQLite MVP)
# =========================
DB_PATH = "tags.db"

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
# ТЕГИ
# =========================
TAG_RE = re.compile(r"#([A-Za-zА-Яа-я0-9_]+)")

def extract_tags(text: str) -> List[str]:
    if not text:
        return []
    return [f"#{m.group(1)}" for m in TAG_RE.finditer(text)]

# =========================
# СТРУКТУРА MVP (минимал)
# =========================
PAGE_SIZE = 10

CATEGORIES = [
    ("🆕 Новинка", "#Новинка"),
    ("💎 Люкс", "#Люкс"),
    ("🔥 Тренд", "#Тренд"),
    ("⭐ Оценка", "#Оценка"),
    ("🧠 Факты / состав", "#Факты"),  # можно позже разделить
]

BRANDS = [
    ("Dior", "#Dior"),
    ("Chanel", "#Chanel"),
    ("Charlotte", "#Charlotte"),
    ("YSL", "#YSL"),
]

SEPHORA = [
    ("🆕 Новинки", "#SephoraNew"),
    ("⭐ Best sellers", "#SephoraTop"),
    ("🔻 Выгодно сейчас", "#SephoraSale"),
]

# =========================
# UI: ВСПОМОГАТЕЛЬНОЕ
# =========================
def kb_home():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", callback_data="go:categories")],
        [InlineKeyboardButton("🏷 Бренды", callback_data="go:brands")],
        [InlineKeyboardButton("💸 Sephora", callback_data="go:sephora")],
        [InlineKeyboardButton("💎 Beauty Challenges", callback_data="go:challenges")],
        [InlineKeyboardButton("↩ В канал", url=CHANNEL_URL)],
    ])

def kb_cover(open_cb: str):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✦ Открыть", callback_data=open_cb)],
        [InlineKeyboardButton("⬅ Назад", callback_data="go:home")],
    ])

def kb_list(items: List[Tuple[str, str]], back_cb: str):
    rows = [[InlineKeyboardButton(title, callback_data=f"cover:{tag}")] for title, tag in items]
    rows.append([InlineKeyboardButton("⬅ Назад", callback_data=back_cb)])
    return InlineKeyboardMarkup(rows)

def kb_posts(tag: str, offset: int):
    ids = db_list(tag, PAGE_SIZE, offset)
    total = db_count(tag)

    rows = []
    for mid in ids:
        rows.append([InlineKeyboardButton("📌 Открыть материал", url=f"{CHANNEL_URL}/{mid}")])

    nav = []
    if offset > 0:
        nav.append(InlineKeyboardButton("⬅", callback_data=f"posts:{tag}:{max(0, offset - PAGE_SIZE)}"))
    if offset + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("➡", callback_data=f"posts:{tag}:{offset + PAGE_SIZE}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("🏠 На главную", callback_data="go:home")])

    if total == 0:
        rows = [
            [InlineKeyboardButton("🏠 На главную", callback_data="go:home")]
        ]

    return InlineKeyboardMarkup(rows), total

def text_home():
    return "NS · Natural Sense\nluxury beauty journal"

def text_section(title: str):
    return f"{title}\n\nNS · Natural Sense\nluxury beauty journal"

def text_tag_cover(tag: str):
    # “обложка” конкретной рубрики/бренда
    return f"{tag}\n\nNS · Natural Sense\nluxury beauty journal\n\n✦ Откройте материалы по этому разделу."

def text_challenges():
    return "Beauty Challenges\n\nNS · Natural Sense\nluxury beauty journal\n\n(Раздел MVP — позже добавим текущий челлендж, архив и участие.)"

# =========================
# SEND: аккуратно (фото или текст)
# =========================
async def send_cover(update_or_query, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup: InlineKeyboardMarkup):
    """
    Если задан COVER_IMAGE_URL — отправляем/редактируем фото-обложку.
    Иначе — просто текст.
    """
    # 1) если это query — пытаемся редактировать сообщение
    q = getattr(update_or_query, "callback_query", None)
    if q:
        try:
            # Если обложка без фото — редактируем текст
            if not COVER_IMAGE_URL:
                await q.edit_message_text(text, reply_markup=reply_markup)
                return
            # Если с фото — редактирование медиа сложнее/ломкое → проще переслать новое
            await q.message.delete()
            await context.bot.send_photo(chat_id=q.message.chat_id, photo=COVER_IMAGE_URL, caption=text, reply_markup=reply_markup)
            return
        except TelegramError:
            # fallback
            await context.bot.send_message(chat_id=q.message.chat_id, text=text, reply_markup=reply_markup)
            return

    # 2) если это обычное сообщение (/start)
    msg = update_or_query.message
    if not msg:
        return
    if COVER_IMAGE_URL:
        await msg.reply_photo(photo=COVER_IMAGE_URL, caption=text, reply_markup=reply_markup)
    else:
        await msg.reply_text(text, reply_markup=reply_markup)

# =========================
# COMMANDS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # deeplink: /start home | categories | brands | sephora | challenges
    arg = context.args[0] if context.args else "home"

    if arg == "categories":
        await show_categories_cover(update, context); return
    if arg == "brands":
        await show_brands_cover(update, context); return
    if arg == "sephora":
        await show_sephora_cover(update, context); return
    if arg == "challenges":
        await show_challenges_cover(update, context); return

    await send_cover(update, context, text_home(), kb_home())

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ OK")

# Пин “входа в журнал” в канал (кнопки ведут в бот)
async def pinmenu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != ChatType.PRIVATE:
        return

    text = "NS · Natural Sense\nluxury beauty journal\n\nOpen the journal 👇"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", url=f"https://t.me/{BOT_USERNAME}?start=categories")],
        [InlineKeyboardButton("🏷 Бренды", url=f"https://t.me/{BOT_USERNAME}?start=brands")],
        [InlineKeyboardButton("💸 Sephora", url=f"https://t.me/{BOT_USERNAME}?start=sephora")],
        [InlineKeyboardButton("💎 Beauty Challenges", url=f"https://t.me/{BOT_USERNAME}?start=challenges")],
    ])

    try:
        msg = await context.bot.send_message(chat_id=CHANNEL_ID, text=text, reply_markup=kb)
        await context.bot.pin_chat_message(chat_id=CHANNEL_ID, message_id=msg.message_id)
        await update.message.reply_text("✅ Закреп создан и закреплён.")
    except TelegramError as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

# =========================
# SCREENS: COVER PAGES (как на референсе)
# =========================
async def show_categories_cover(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_cover(update, context, text_section("Categories"), kb_cover("open:categories"))

async def show_brands_cover(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_cover(update, context, text_section("Brands"), kb_cover("open:brands"))

async def show_sephora_cover(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_cover(update, context, text_section("Sephora"), kb_cover("open:sephora"))

async def show_challenges_cover(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_cover(update, context, text_challenges(), InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅ Назад", callback_data="go:home")],
    ]))

# =========================
# CALLBACKS
# =========================
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "go:home":
        await send_cover(update, context, text_home(), kb_home()); return

    # Home → section cover pages
    if data == "go:categories":
        await q.edit_message_text(text_section("Categories"), reply_markup=kb_cover("open:categories")); return
    if data == "go:brands":
        await q.edit_message_text(text_section("Brands"), reply_markup=kb_cover("open:brands")); return
    if data == "go:sephora":
        await q.edit_message_text(text_section("Sephora"), reply_markup=kb_cover("open:sephora")); return
    if data == "go:challenges":
        await q.edit_message_text(text_challenges(), reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("⬅ Назад", callback_data="go:home")]
        ])); return

    # cover → list
    if data == "open:categories":
        await q.edit_message_text("Categories", reply_markup=kb_list(CATEGORIES, "go:home")); return
    if data == "open:brands":
        await q.edit_message_text("Brands", reply_markup=kb_list(BRANDS, "go:home")); return
    if data == "open:sephora":
        await q.edit_message_text("Sephora", reply_markup=kb_list(SEPHORA, "go:home")); return

    # tag cover page
    if data.startswith("cover:"):
        tag = data.split(":", 1)[1]  # "#Dior"
        await q.edit_message_text(text_tag_cover(tag), reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("✦ Открыть материалы", callback_data=f"posts:{tag}:0")],
            [InlineKeyboardButton("⬅ Назад", callback_data="go:home")],
        ]))
        return

    # posts list
    if data.startswith("posts:"):
        _, tag, offset_str = data.split(":", 2)
        offset = int(offset_str)
        kb, total = kb_posts(tag, offset)
        if total == 0:
            await q.edit_message_text(
                f"{tag}\n\nПока нет материалов с этим тегом.\n"
                "Важно: бот начнёт собирать посты, когда он админ канала, и ты публикуешь новые посты с тегами.",
                reply_markup=kb
            )
        else:
            await q.edit_message_text(f"{tag} · materials: {total}", reply_markup=kb)
        return

# =========================
# INDEX: новые посты канала
# =========================
async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return

    text = msg.text or msg.caption or ""
    tags = extract_tags(text)
    if not tags:
        return

    for t in tags:
        db_add(t, msg.message_id)

    logging.info("Indexed %s tags=%s", msg.message_id, tags)

# =========================
# MAIN
# =========================
def main():
    db_init()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("pinmenu", pinmenu))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.ChatType.CHANNEL, on_channel_post))

    app.run_polling()

if __name__ == "__main__":
    main()
