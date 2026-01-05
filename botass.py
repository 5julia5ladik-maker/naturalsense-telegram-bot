import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = os.getenv("8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = context.args[0] if context.args else "menu"

    if arg == "menu":
        text = "Personal menu 🤍"
        keyboard = [
            [InlineKeyboardButton("🎨 Skin tone", callback_data="tone")],
            [InlineKeyboardButton("💧 Skin type", callback_data="skin")],
            [InlineKeyboardButton("↩️ Back to channel", url="https://t.me/NaturalSense")]
        ]

    elif arg == "tone":
        text = "Select your skin tone:"
        keyboard = [
            [InlineKeyboardButton("Very light", callback_data="tone_light")],
            [InlineKeyboardButton("Light", callback_data="tone_light2")],
            [InlineKeyboardButton("Medium", callback_data="tone_medium")],
            [InlineKeyboardButton("Deep", callback_data="tone_deep")]
        ]

    elif arg == "skin":
        text = "Select your skin type:"
        keyboard = [
            [InlineKeyboardButton("Dry", callback_data="skin_dry")],
            [InlineKeyboardButton("Normal", callback_data="skin_normal")],
            [InlineKeyboardButton("Combination", callback_data="skin_combo")],
            [InlineKeyboardButton("Oily", callback_data="skin_oily")]
        ]

    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text="Saved 🤍")


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(buttons))
    app.run_polling()


if __name__ == "__main__":
    main()
