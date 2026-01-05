import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# =========================
# ВСТАВЬ СВОЙ ТОКЕН ЗДЕСЬ
# =========================
TOKEN = "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4"  # <-- вставь токен между кавычками

CHANNEL_URL = "https://t.me/NaturalSense"

logging.basicConfig(level=logging.INFO)


def menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎨 Выбрать тон кожи", callback_data="tone")],
        [InlineKeyboardButton("💧 Тип кожи", callback_data="skin")],
        [InlineKeyboardButton("📰 Новости", url=CHANNEL_URL)],
        [InlineKeyboardButton("🧴 Обзоры", url=CHANNEL_URL)],
        [InlineKeyboardButton("🔍 Поиск по тегам", callback_data="tags")],
        [InlineKeyboardButton("↩️ Вернуться в канал", url=CHANNEL_URL)],
    ])


def tone_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🤍 Очень светлый", callback_data="tone:very_light")],
        [InlineKeyboardButton("🌤 Светлый", callback_data="tone:light")],
        [InlineKeyboardButton("🌼 Средний", callback_data="tone:medium")],
        [InlineKeyboardButton("🌰 Тёмный", callback_data="tone:deep")],
        [InlineKeyboardButton("✅ Готово → меню", callback_data="go:menu")],
    ])


def skin_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💧 Сухая", callback_data="skin:dry")],
        [InlineKeyboardButton("🌿 Нормальная", callback_data="skin:normal")],
        [InlineKeyboardButton("⚖️ Комбинированная", callback_data="skin:combo")],
        [InlineKeyboardButton("💎 Жирная", callback_data="skin:oily")],
        [InlineKeyboardButton("✅ Готово → меню", callback_data="go:menu")],
    ])


def tags_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("#news", callback_data="tag:news")],
        [InlineKeyboardButton("#reviews", callback_data="tag:reviews")],
        [InlineKeyboardButton("#compare", callback_data="tag:compare")],
        [InlineKeyboardButton("↩️ Назад в меню", callback_data="go:menu")],
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0] if context.args else "menu").lower()

    if arg == "tone":
        await update.message.reply_text("Выбери тон кожи:", reply_markup=tone_kb())
        return

    if arg == "skin":
        await update.message.reply_text("Выбери тип кожи:", reply_markup=skin_kb())
        return

    if arg == "tags":
        await update.message.reply_text("Выбери тег:", reply_markup=tags_kb())
        return

    await update.message.reply_text("✅ Меню Natural Sense", reply_markup=menu_kb())


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data

    if data == "go:menu":
        await q.edit_message_text("✅ Меню Natural Sense", reply_markup=menu_kb())
        return

    if data.startswith("tone:"):
        await q.edit_message_text("Тон кожи сохранён 🤍", reply_markup=tone_kb())
        return

    if data.startswith("skin:"):
        await q.edit_message_text("Тип кожи сохранён 🤍", reply_markup=skin_kb())
        return

    if data.startswith("tag:"):
        tag = data.split(":", 1)[1]
        await q.edit_message_text(
            f"🔍 Тег выбран: #{tag}\n\n(Пока заглушка — дальше подключим выдачу постов)",
            reply_markup=tags_kb()
        )
        return

    await q.edit_message_text("Ок", reply_markup=menu_kb())


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
