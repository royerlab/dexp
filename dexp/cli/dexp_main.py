import click
from arbol.arbol import aprint, asection

from dexp.processing.utils.mkl_util import set_mkl_threads

set_mkl_threads()

# def log_uncaught_exceptions(exception_type, exception, tb):
#     import traceback
#     print(''.join(traceback.format_tb(tb)))
#     print('{0}: {1}'.format(exception_type, exception))


# import sys

# sys.excepthook = log_uncaught_exceptions

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
def cli():
    aprint("__________________________________________")
    aprint("  DEXP -- Data EXploration & Processing   ")
    aprint("  Dataset processing dexp_commands             ")
    aprint("  Royer lab                               ")
    aprint("__________________________________________")
    aprint("")

    try:
        from dexp.processing.backends.cupy_backend import CupyBackend
        available = CupyBackend.available_devices()
        with asection(f"Available GPU devices: {available}"):
            for device_id in available:
                from cupy.cuda.runtime import CUDARuntimeError
                try:
                    backend = CupyBackend(device_id)
                    aprint(backend)
                except CUDARuntimeError as e:
                    aprint(f"Error while querying available devices: {e}")

    except (ModuleNotFoundError, NotImplementedError):
        aprint("'cupy' module not found! ignored!")

    pass


from dexp.cli.dexp_commands.info import info
from dexp.cli.dexp_commands.check import check
from dexp.cli.dexp_commands.copy import copy
from dexp.cli.dexp_commands.crop import crop
from dexp.cli.dexp_commands.fastcopy import fastcopy
from dexp.cli.dexp_commands.add import add
from dexp.cli.dexp_commands.tiff import tiff
from dexp.cli.dexp_commands.view import view
from dexp.cli.dexp_commands.serve import serve
from dexp.cli.dexp_commands.extract_psf import extract_psf

from dexp.cli.dexp_commands.deskew import deskew
from dexp.cli.dexp_commands.fuse import fuse
from dexp.cli.dexp_commands.register import register
from dexp.cli.dexp_commands.deconv import deconv
from dexp.cli.dexp_commands.stabilize import stabilize
from dexp.cli.dexp_commands.isonet import isonet

from dexp.cli.dexp_commands.speedtest import speedtest

from dexp.cli.video_commands.volrender import volrender
from dexp.cli.dexp_commands.projrender import projrender

cli.add_command(info)
cli.add_command(check)
cli.add_command(copy)
cli.add_command(crop)
cli.add_command(fastcopy)
cli.add_command(add)
cli.add_command(tiff)
cli.add_command(view)
cli.add_command(serve)
cli.add_command(extract_psf)

cli.add_command(deskew)
cli.add_command(fuse)
cli.add_command(register)
cli.add_command(stabilize)
cli.add_command(deconv)
cli.add_command(isonet)

cli.add_command(speedtest)

cli.add_command(volrender)
cli.add_command(projrender)
