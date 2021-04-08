from typing import Callable

import numpy
from arbol import aprint
from numpy import ndarray
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, Handler
# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
from telegram.ext.dispatcher import DEFAULT_GROUP

from dexp.cli.config import get_telegram_bot_token, get_telegram_bot_password
from dexp.processing.backends.backend import Backend


# import logging
# # Enable logging
# from dexp.cli.config import get_telegram_bot_token
#
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
# )
#
# logger = logging.getLogger(__name__)


class TelegramBot:

    def __init__(self, token: str = get_telegram_bot_token(), password: str = get_telegram_bot_password()):
        """
        Constructs a Telegram Bot

        Parameters
        ----------
        blocking : if True then this call will block until the Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
        """

        super().__init__()

        self.updater = Updater(token=token, use_context=True)
        self.password: str = password
        self.authorised: bool = False

        def password_command(update: Update, context: CallbackContext) -> None:
            """Send a message when the command /help is issued."""
            if self.password in update.message.text:
                update.message.reply_text('✅ Authentication successful!')
                self.authorised = True
            else:
                update.message.reply_text('❌ Authentication failed!')

        self._add_handler(CommandHandler("password", password_command))

    def _add_handler(self, handler: Handler, group: int = DEFAULT_GROUP):
        self.updater.dispatcher.add_handler(handler=handler, group=group)

    def add_text_handler(self, function: Callable[[str], str]):
        """
        Adds a simple handler that takes a text message and returns a text message in return.

        Parameters
        ----------
        function : function that takes a string and returns a string
        """

        def _fun(update: Update, context: CallbackContext) -> None:
            if self.authorised:
                message = update.message.text
                reply = function(message)
                aprint(f"Telegram Bot: received '{message}', replying: '{reply}'")
                update.message.reply_text(reply)

        self._add_handler(MessageHandler(Filters.text & ~Filters.command, _fun))

    def add_image_handler(self, function: Callable[[str], ndarray]):
        """
        Adds a simple handler that takes a text message as input and returns an image in return.
        The image is passed as a 2D numpy array normalised within [0, 1]

        Parameters
        ----------
        function : function that takes a string and returns an image (2D numpy array)
        """

        def _fun(update: Update, context: CallbackContext) -> None:
            if self.authorised:
                message = update.message.text
                array, cmap_name = function(message)
                array = Backend.to_numpy(array)

                from matplotlib.cm import get_cmap
                cmap = get_cmap(cmap_name)

                from PIL import Image
                image = Image.fromarray(numpy.uint8(cmap(array)[..., 0:3] * 255), mode='RGB')
                from io import BytesIO
                image_bytes = BytesIO()
                image_bytes.name = 'image.png'
                image.save(image_bytes, 'PNG')
                image_bytes.seek(0)

                aprint(f"Telegram Bot: received '{message}', replying with image: '{image}'")
                update.message.reply_photo(image_bytes)

        self._add_handler(MessageHandler(Filters.text & ~Filters.command, _fun))

    def start(self, blocking=False, clean=True):
        """
        Starts the bot

        Parameters
        ----------
        blocking : if True then this call will block until the Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
        """
        self.updater.start_polling(clean=clean)

        if blocking:
            self.idle()

    def idle(self):
        """
        Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT. This should be used most of the time, since
        start_polling() is non-blocking and will stop the bot gracefully.

        """

        self.updater.idle()

    def close(self):
        """
        Close (stops) the bot

        """
        self.updater.stop()
