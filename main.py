# botass.py
import os
import logging
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
)

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG (всё тут)
# =========================
BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    "8591165656:AAFvwMeza7LXruoId7sHqQ_FEeTgmBgqqi4"  # фейковый как ты просил
)

CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "NaturalSense")
CHANNEL_URL = f"https://t.me/{CHANNEL_USERNAME}"

# =========================
# helpers
# =========================
def channel_search_link(tag: str) -> str:
    """
    Открывает канал и сразу поиск по #tag (работает для публичных каналов).
    tag можно давать с решёткой или без.
    """
    tag = tag.strip()
    if not tag.startswith("#"):
        tag = "#" + tag
    # Telegram понимает q=... как поиск по чату/каналу
    return f"https://t.me/{CHANNEL_USERNAME}?q={tag.replace('#', '%23')}"

def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Категории", callback_data="main:categories")],
        [InlineKeyboardButton("🏷 Бренды", callback_data="main:brands")],
        [InlineKeyboardButton("💸 Sephora", callback_data="main:sephora")],
        [InlineKeyboardButton("💎 Beauty Challenges", callback_data="main:challenges")],
        [InlineKeyboardButton("↩️ В канал", url=CHANNEL_URL)],
    ])

def kb_back_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Назад", callback_data="nav:back_main")],
        [InlineKeyboardButton("↩️ В канал", url=CHANNEL_URL)],
    ])

def kb_categories() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🆕 Новинка", url=channel_search_link("Новинка"))],
        [InlineKeyboardButton("💎 Кратко о люкс продукте", url=channel_search_link("Люкс"))],
        [InlineKeyboardButton("🔥 Тренд", url=channel_search_link("Тренд"))],
        [InlineKeyboardButton("🏛 История бренда", url=channel_search_link("История"))],
        [InlineKeyboardButton("⭐ Личная оценка продукта", url=channel_search_link("Оценка"))],
        [InlineKeyboardButton("🧴 Тип продукта / факты", url=channel_search_link("Факты"))],
        [InlineKeyboardButton("🧪 Составы продуктов", url=channel_search_link("Состав"))],
        [InlineKeyboardButton("⬅️ Назад", callback_data="nav:back_main")],
    ])

def kb_brands() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ Dior", url=channel_search_link("Dior"))],
        [InlineKeyboardButton("✨ Chanel", url=channel_search_link("Chanel"))],
        [InlineKeyboardButton("✨ Charlotte Tilbury", url=channel_search_link("CharlotteTilbury"))],
        [InlineKeyboardButton("⬅️ Назад", callback_data="nav:back_main")],
    ])

def kb_sephora() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🇹🇷 Актуальные цены (TR)", url=channel_search_link("SephoraTR"))],
        [InlineKeyboardButton("🎁 Подарки / акции", url=channel_search_link("SephoraPromo"))],
        [InlineKeyboardButton("🧾 Гайды / как покупать", url=channel_search_link("SephoraGuide"))],
        [InlineKeyboardButton("⬅️ Назад", callback_data="nav:back_main")],
    ])

def kb_challenges() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📸 Фото косметического места", url=channel_search_link("Challenge"))],
        [InlineKeyboardButton("🛍 Лучшие покупки месяца", url=channel_search_link("Challenge"))],
        [InlineKeyboardButton("💄 Самый странный дизайн помады", url=channel_search_link("Challenge"))],
        [InlineKeyboardButton("⬅️ Назад", callback_data="nav:back_main")],
    ])


# =========================
# handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "NS · Natural Sense\n"
        "luxury beauty journal\n\n"
        "Выберите раздел 👇"
    )
    await update.message.reply_text(text, reply_markup=kb_main())

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    data = q.data or ""

    if data == "nav:back_main":
        text = (
            "NS · Natural Sense\n"
            "luxury beauty journal\n\n"
            "Выберите раздел 👇"
        )
        await q.edit_message_text(text, reply_markup=kb_main())
        return

    if data == "main:categories":
        await q.edit_message_text("📂 Категории — выберите:", reply_markup=kb_categories())
        return

    if data == "main:brands":
        await q.edit_message_text("🏷 Бренды — выберите:", reply_markup=kb_brands())
        return

    if data == "main:sephora":
        await q.edit_message_text("💸 Sephora — выберите:", reply_markup=kb_sephora())
        return

    if data == "main:challenges":
        await q.edit_message_text("💎 Beauty Challenges — выберите:", reply_markup=kb_challenges())
        return


def main():
    if not BOT_TOKEN or ":" not in BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN пустой или не похож на токен. Проверь переменную или строку в коде.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CallbackQueryHandler(on_callback))

    logging.info("Bot started (POLLING).")
    # ВАЖНО: drop_pending_updates=True чтобы старые апдейты не мешали
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
