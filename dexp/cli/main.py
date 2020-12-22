import click
from arbol.arbol import aprint, asection

from dexp.processing.utils.mkl_util import set_mkl_threads

_default_store = 'dir'
_default_clevel = 3
_default_codec = 'zstd'
_default_workers_backend = 'threading'

set_mkl_threads()


def log_uncaught_exceptions(exception_type, exception, tb):
    import traceback
    print(''.join(traceback.format_tb(tb)))
    print('{0}: {1}'.format(exception_type, exception))


import sys

sys.excepthook = log_uncaught_exceptions

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
def cli():
    aprint("__________________________________________")
    aprint("  DEXP -- Data EXploration & Processing   ")
    aprint("  Royer lab                               ")
    aprint("__________________________________________")
    aprint("")

    try:
        from dexp.processing.backends.cupy_backend import CupyBackend
        available = CupyBackend.available_devices()
        with asection(f"Available GPU devices: {available}"):
            for device_id in available:
                backend = CupyBackend(device_id)
                aprint(backend)

    except (ModuleNotFoundError, NotImplementedError):
        aprint("'cupy' module not found! ignored!")

    pass


from dexp.cli.commands.info import info
from dexp.cli.commands.check import check
from dexp.cli.commands.copy import copy
from dexp.cli.commands.add import add
from dexp.cli.commands.concat import concat
from dexp.cli.commands.tiff import tiff
from dexp.cli.commands.view import view
from dexp.cli.commands.serve import serve

from dexp.cli.commands.fuse import fuse
from dexp.cli.commands.deconv import deconv
from dexp.cli.commands.isonet import isonet

from dexp.cli.commands.render import render
from dexp.cli.commands.blend import blend
from dexp.cli.commands.stack import stack
from dexp.cli.commands.mp4 import mp4

from dexp.cli.commands.speedtest import speedtest

cli.add_command(info)
cli.add_command(check)
cli.add_command(copy)
cli.add_command(add)
cli.add_command(concat)
cli.add_command(tiff)
cli.add_command(view)
cli.add_command(serve)

cli.add_command(fuse)
cli.add_command(deconv)
cli.add_command(isonet)

cli.add_command(render)
cli.add_command(blend)
cli.add_command(stack)
cli.add_command(mp4)

cli.add_command(speedtest)
