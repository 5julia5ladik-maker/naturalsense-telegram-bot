import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ChatType
from telegram.error import TelegramError

logging.basicConfig(level=logging.INFO)

# =========================
# ВСТАВЬ ТОКЕН ОДИН РАЗ
# =========================
TOKEN = "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4"

# =========================
# НАСТРОЙКИ
# =========================
BOT_USERNAME = "naturalsense_assistant_bot"

# ВАЖНО:
# 1) Если канал публичный и ссылка t.me/XXXX -> тут должно быть "@XXXX"
# 2) Если канал приватный -> тут будет "-1001234567890"
CHANNEL_ID = "@NaturalSense"

CHANNEL_URL = "https://t.me/NaturalSense"


# -------------------------
# Keyboards
# -------------------------
def menu_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎨 Тон кожи", url=f"https://t.me/{BOT_USERNAME}?start=tone"),
            InlineKeyboardButton("💧 Тип кожи", url=f"https://t.me/{BOT_USERNAME}?start=skin"),
        ],
        [
            InlineKeyboardButton("📰 Новости", url=f"https://t.me/{BOT_USERNAME}?start=news"),
            InlineKeyboardButton("🧴 Обзоры", url=f"https://t.me/{BOT_USERNAME}?start=reviews"),
        ],
        [
            InlineKeyboardButton("🔍 Теги", url=f"https://t.me/{BOT_USERNAME}?start=tags"),
        ],
        [
            InlineKeyboardButton("↩️ Вернуться в канал", url=CHANNEL_URL),
        ],
    ])


def tone_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🤍 Очень светлый", callback_data="tone:very_light")],
        [InlineKeyboardButton("🌤 Светлый", callback_data="tone:light")],
        [InlineKeyboardButton("🌼 Средний", callback_data="tone:medium")],
        [InlineKeyboardButton("🌰 Тёмный", callback_data="tone:deep")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:menu")],
    ])


def skin_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💧 Сухая", callback_data="skin:dry")],
        [InlineKeyboardButton("🌿 Нормальная", callback_data="skin:normal")],
        [InlineKeyboardButton("⚖️ Комбинированная", callback_data="skin:combo")],
        [InlineKeyboardButton("💎 Жирная", callback_data="skin:oily")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:menu")],
    ])


def news_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔥 Новинки недели", url=CHANNEL_URL)],
        [InlineKeyboardButton("💄 Запуски брендов", url=CHANNEL_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:menu")],
    ])


def reviews_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⭐ Топ продукты", url=CHANNEL_URL)],
        [InlineKeyboardButton("🧴 Уход", url=CHANNEL_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:menu")],
    ])


def tags_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("#news", callback_data="tag:news")],
        [InlineKeyboardButton("#reviews", callback_data="tag:reviews")],
        [InlineKeyboardButton("#compare", callback_data="tag:compare")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:menu")],
    ])


# -------------------------
# /start
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0] if context.args else "menu").lower().strip()

    if arg == "tone":
        await update.message.reply_text("🎨 Выбери тон кожи:", reply_markup=tone_kb()); return
    if arg == "skin":
        await update.message.reply_text("💧 Выбери тип кожи:", reply_markup=skin_kb()); return
    if arg == "news":
        await update.message.reply_text("📰 Новости:", reply_markup=news_kb()); return
    if arg == "reviews":
        await update.message.reply_text("🧴 Обзоры:", reply_markup=reviews_kb()); return
    if arg == "tags":
        await update.message.reply_text("🔍 Выбери тег:", reply_markup=tags_kb()); return

    await update.message.reply_text(
        "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇",
        reply_markup=menu_kb()
    )


# -------------------------
# Callbacks
# -------------------------
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "back:menu":
        await q.edit_message_text(
            "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇",
            reply_markup=menu_kb()
        )
        return

    if data.startswith("tone:"):
        context.user_data["tone"] = data.split(":", 1)[1]
        await q.edit_message_text("🤍 Тон кожи сохранён", reply_markup=tone_kb())
        return

    if data.startswith("skin:"):
        context.user_data["skin"] = data.split(":", 1)[1]
        await q.edit_message_text("💧 Тип кожи сохранён", reply_markup=skin_kb())
        return

    if data.startswith("tag:"):
        tag = data.split(":", 1)[1]
        await q.edit_message_text(f"🔍 Тег выбран: #{tag}", reply_markup=tags_kb())
        return

    await q.edit_message_text("Ок ✅", reply_markup=menu_kb())


# -------------------------
# /pinmenu  (публикует пост-меню в канал и закрепляет)
# -------------------------
async def pinmenu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # чтобы никто в канале/группе случайно не запускал
    if update.effective_chat.type != ChatType.PRIVATE:
        return

    text = "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇"
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎨 Тон кожи", url=f"https://t.me/{BOT_USERNAME}?start=tone"),
            InlineKeyboardButton("💧 Тип кожи", url=f"https://t.me/{BOT_USERNAME}?start=skin"),
        ],
        [
            InlineKeyboardButton("📰 Новости", url=f"https://t.me/{BOT_USERNAME}?start=news"),
            InlineKeyboardButton("🧴 Обзоры", url=f"https://t.me/{BOT_USERNAME}?start=reviews"),
        ],
        [
            InlineKeyboardButton("🔍 Теги", url=f"https://t.me/{BOT_USERNAME}?start=tags"),
        ],
    ])

    # 1) отправка
    try:
        msg = await context.bot.send_message(chat_id=CHANNEL_ID, text=text, reply_markup=kb)
    except TelegramError as e:
        await update.message.reply_text(
            "❌ НЕ смог отправить меню в канал.\n\n"
            f"Причина: {e}\n\n"
            "Проверь: бот — админ канала и CHANNEL_ID указан верно."
        )
        return

    # 2) закреп
    try:
        await context.bot.pin_chat_message(chat_id=CHANNEL_ID, message_id=msg.message_id)
        await update.message.reply_text("✅ Меню отправлено и ЗАКРЕПЛЕНО в канале.")
    except TelegramError as e:
        await update.message.reply_text(
            "⚠️ Меню отправлено, но НЕ закрепилось.\n\n"
            f"Причина: {e}\n\n"
            "Проверь права бота в канале: 'Управление сообщениями канала' (закреп)."
        )


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("pinmenu", pinmenu))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
