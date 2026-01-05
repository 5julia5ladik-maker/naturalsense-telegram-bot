import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO)

CHANNEL_URL = "https://t.me/NaturalSense"


def must_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}. Add it in Railway → Variables.")
    return v


TOKEN = must_env("TOKEN")


def menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎨 Выбрать тон кожи", url="https://t.me/naturalsense_assistant_bot?start=tone")],
        [InlineKeyboardButton("💧 Тип кожи", url="https://t.me/naturalsense_assistant_bot?start=skin")],
        [InlineKeyboardButton("📰 Новости", url=CHANNEL_URL)],
        [InlineKeyboardButton("🧴 Обзоры", url=CHANNEL_URL)],
        [InlineKeyboardButton("🔍 Поиск по тегам", url="https://t.me/naturalsense_assistant_bot?start=tags")],
        [InlineKeyboardButton("↩️ Вернуться в канал", url=CHANNEL_URL)],
    ])


def tone_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🤍 Очень светлый", callback_data="tone:very_light")],
        [InlineKeyboardButton("🌤 Светлый", callback_data="tone:light")],
        [InlineKeyboardButton("🌼 Средний", callback_data="tone:medium")],
        [InlineKeyboardButton("🌰 Тёмный", callback_data="tone:deep")],
        [InlineKeyboardButton("✅ Готово → меню", callback_data="go:menu")],
    ])


def skin_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💧 Сухая", callback_data="skin:dry")],
        [InlineKeyboardButton("🌿 Нормальная", callback_data="skin:normal")],
        [InlineKeyboardButton("⚖️ Комбинированная", callback_data="skin:combo")],
        [InlineKeyboardButton("💎 Жирная", callback_data="skin:oily")],
        [InlineKeyboardButton("✅ Готово → меню", callback_data="go:menu")],
    ])


def tags_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("#news", callback_data="tag:news")],
        [InlineKeyboardButton("#reviews", callback_data="tag:reviews")],
        [InlineKeyboardButton("#compare", callback_data="tag:compare")],
        [InlineKeyboardButton("↩️ Назад в меню", callback_data="go:menu")],
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0] if context.args else "menu").lower().strip()

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
    data = (q.data or "").strip()

    if data == "go:menu":
        await q.edit_message_text("✅ Меню Natural Sense", reply_markup=menu_kb())
        return

    # Сохраняем в память (пока без базы) — для будущего
    if data.startswith("tone:"):
        context.user_data["tone"] = data.split(":", 1)[1]
        await q.edit_message_text("Тон кожи сохранён 🤍", reply_markup=tone_kb())
        return

    if data.startswith("skin:"):
        context.user_data["skin"] = data.split(":", 1)[1]
        await q.edit_message_text("Тип кожи сохранён 🤍", reply_markup=skin_kb())
        return

    if data.startswith("tag:"):
        tag = data.split(":", 1)[1]
        # Здесь можно позже сделать поиск по каналу/список постов.
        await q.edit_message_text(
            f"🔍 Тег выбран: #{tag}\n\nПока это заглушка. Дальше подключим выдачу постов по тегу.",
            reply_markup=tags_kb()
        )
        return

    await q.edit_message_text("Ок ✅", reply_markup=menu_kb())


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
