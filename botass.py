import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

# =========================
# 1) ВСТАВЬ ТОКЕН В ЭТУ СТРОКУ (1 раз)
# =========================
TOKEN = "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4"

# =========================
# 2) НАСТРОЙКИ КАНАЛА/БОТА
# =========================
BOT_USERNAME = "naturalsense_assistant_bot"
CHANNEL_USERNAME = "NaturalSense"     # то, что после t.me/
CHANNEL_URL = f"https://t.me/{CHANNEL_USERNAME}"

# =========================
# UI (клавиатуры)
# =========================
def menu_kb():
    # ВАЖНО: из закрепа люди будут попадать сразу в tone/skin/news/reviews/tags
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
    # Пока заглушки: потом поменяем на ссылки на КОНКРЕТНЫЕ посты
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔥 Новинки недели", url=CHANNEL_URL)],
        [InlineKeyboardButton("💄 Запуски брендов", url=CHANNEL_URL)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:menu")],
    ])


def reviews_kb():
    # Пока заглушки: потом поменяем на ссылки на КОНКРЕТНЫЕ посты
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


# =========================
# /start с deep-link параметрами
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0] if context.args else "menu").lower().strip()

    if arg == "tone":
        await update.message.reply_text("🎨 Выбери тон кожи:", reply_markup=tone_kb())
        return

    if arg == "skin":
        await update.message.reply_text("💧 Выбери тип кожи:", reply_markup=skin_kb())
        return

    if arg == "news":
        await update.message.reply_text("📰 Новости:", reply_markup=news_kb())
        return

    if arg == "reviews":
        await update.message.reply_text("🧴 Обзоры:", reply_markup=reviews_kb())
        return

    if arg == "tags":
        await update.message.reply_text("🔍 Выбери тег:", reply_markup=tags_kb())
        return

    await update.message.reply_text(
        "NS · Natural Sense\nprivate beauty space\n\nВыберите раздел 👇",
        reply_markup=menu_kb()
    )


# =========================
# Callback-обработка (нажатия внутри tone/skin/tags)
# =========================
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

    # Сохраняем выбор (пока в памяти пользователя)
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


# =========================
# Запуск
# =========================
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
