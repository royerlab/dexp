from arbol import aprint
from skimage.data import camera
from telegram import Update
from telegram.ext import MessageHandler, Filters, CallbackContext, CommandHandler

from dexp.utils.telegram_bot.telegram_bot import TelegramBot


def demo_telegram_bot():
    bot = TelegramBot()

    def start(update: Update, context: CallbackContext) -> None:
        """Send a message when the command /start is issued."""
        update.message.reply_text('Hi!')

    def help_command(update: Update, context: CallbackContext) -> None:
        """Send a message when the command /help is issued."""
        update.message.reply_text('Help!')

    def echo(update: Update, context: CallbackContext) -> None:
        """Echo the user message."""
        aprint(update.message.text)
        update.message.reply_text(update.message.text)

    def image(message: str):
        if 'camera' in message:
            return camera(), 'viridis'

    bot.start()

    bot._add_handler(CommandHandler("start", start))
    bot._add_handler(CommandHandler("help", help_command))
    bot.add_image_handler(image)
    bot._add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    bot.idle()


demo_telegram_bot()
