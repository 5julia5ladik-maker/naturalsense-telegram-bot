import re
import sqlite3
import logging
from typing import List, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ChatType
from telegram.error import TelegramError

logging.basicConfig(level=logging.INFO)

# =========================
# НАСТРОЙКИ
# =========================
TOKEN = "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4"

BOT_USERNAME = "naturalsense_assistant_bot"   # username бота без @
CHANNEL_USERNAME = "NaturalSense"
CHANNEL_URL = "https://t.me/NaturalSense"
CHANNEL_ID = "@NaturalSense"   # если приватный → -100...

# =========================
# БАЗА ДЛЯ ТЕГОВ
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

def db_list(tag: str, limit: int, offset: int):
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

def extract_tags(text: str):
    if not text:
        return []
    return [f"#{m.group(1)}" for m in TAG_RE.finditer(text)]

# =========================
# СТРУКТУРА
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
    ("Charlotte Tilbury", "#Charlotte"),
    ("Chanel", "#Chanel"),
    ("Yves Saint Laurent", "#YSL"),
]

SEPHORA = [
    ("🔻 Скидки", "#SephoraSale"),
    ("🎁 Подарки", "#SephoraGift"),
    ("🆕 Новинки", "#SephoraNew"),
    ("⭐ Best sellers", "#SephoraTop"),
]

PAGE_SIZE = 10

# =========================
# КЛАВИАТУРЫ
# =========================
def main_menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", callback_data="menu:categories")],
        [InlineKeyboardButton("🏷 Бренды", callback_data="menu:brands")],
        [InlineKeyboardButton("💸 Sephora", callback_data="menu:sephora")],
        [InlineKeyboardButton("↩️ В канал", url=CHANNEL_URL)],
    ])

def section_kb(items):
    rows = [[InlineKeyboardButton(title, callback_data=f"tag:{tag}:0")] for title, tag in items]
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:home")])
    return InlineKeyboardMarkup(rows)

def posts_kb(tag: str, offset: int):
    ids = db_list(tag, PAGE_SIZE, offset)
    total = db_count(tag)
    rows = []

    for mid in ids:
        rows.append([InlineKeyboardButton("📌 Открыть пост", url=f"{CHANNEL_URL}/{mid}")])

    nav = []
    if offset > 0:
        nav.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"tag:{tag}:{max(0, offset-PAGE_SIZE)}"))
    if offset + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("➡️ Ещё", callback_data=f"tag:{tag}:{offset+PAGE_SIZE}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("🏠 Меню", callback_data="menu:home")])

    if total == 0:
        rows = [[InlineKeyboardButton("🏠 Меню", callback_data="menu:home")]]

    return InlineKeyboardMarkup(rows), total

# =========================
# START С DEEPLINK
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = context.args[0] if context.args else "menu"

    if arg == "categories":
        await update.message.reply_text("📂 Категории", reply_markup=section_kb(CATEGORIES))
        return

    if arg == "brands":
        await update.message.reply_text("🏷 Бренды", reply_markup=section_kb(BRANDS))
        return

    if arg == "sephora":
        await update.message.reply_text("💸 Sephora", reply_markup=section_kb(SEPHORA))
        return

    await update.message.reply_text(
        "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇",
        reply_markup=main_menu_kb()
    )

# =========================
# CALLBACK
# =========================
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "menu:home":
        await q.edit_message_text(
            "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇",
            reply_markup=main_menu_kb()
        )
        return

    if data == "menu:categories":
        await q.edit_message_text("📂 Категории", reply_markup=section_kb(CATEGORIES))
        return

    if data == "menu:brands":
        await q.edit_message_text("🏷 Бренды", reply_markup=section_kb(BRANDS))
        return

    if data == "menu:sephora":
        await q.edit_message_text("💸 Sephora", reply_markup=section_kb(SEPHORA))
        return

    if data.startswith("tag:"):
        _, tag, offset_str = data.split(":", 2)
        offset = int(offset_str)
        kb, total = posts_kb(tag, offset)

        if total == 0:
            await q.edit_message_text(
                f"{tag}\n\nПока нет постов с этим тегом.",
                reply_markup=kb
            )
        else:
            await q.edit_message_text(f"{tag} — найдено: {total}", reply_markup=kb)

# =========================
# ИНДЕКСАЦИЯ ПОСТОВ КАНАЛА
# =========================
async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return

    text = msg.text or msg.caption or ""
    tags = extract_tags(text)
    for t in tags:
        db_add(t, msg.message_id)

# =========================
# ЗАКРЕП
# =========================
async def pinmenu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != ChatType.PRIVATE:
        return

    text = "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇"

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", url=f"https://t.me/{BOT_USERNAME}?start=categories")],
        [InlineKeyboardButton("🏷 Бренды", url=f"https://t.me/{BOT_USERNAME}?start=brands")],
        [InlineKeyboardButton("💸 Sephora", url=f"https://t.me/{BOT_USERNAME}?start=sephora")],
    ])

    try:
        msg = await context.bot.send_message(chat_id=CHANNEL_ID, text=text, reply_markup=kb)
        await context.bot.pin_chat_message(chat_id=CHANNEL_ID, message_id=msg.message_id)
        await update.message.reply_text("✅ Меню отправлено и закреплено.")
    except TelegramError as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

# =========================
# MAIN
# =========================
def main():
    db_init()
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("pinmenu", pinmenu))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.ChatType.CHANNEL, on_channel_post))

    app.run_polling()

if __name__ == "__main__":
    main()
