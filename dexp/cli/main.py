from os.path import join, exists

import click
from numpy import s_

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

_default_store = 'dir'
_default_clevel = 3
_default_codec = 'zstd'

import sys


def log_uncaught_exceptions(exception_type, exception, tb):
    import traceback
    print(''.join(traceback.format_tb(tb)))
    print('{0}: {1}'.format(exception_type, exception))


sys.excepthook = log_uncaught_exceptions


def _get_dataset_from_path(input_path):
    if exists(join(input_path, 'stacks')):
        input_dataset = CCDataset(input_path)
    else:
        input_dataset = ZDataset(input_path)
    return input_dataset


def _get_folder_name_without_end_slash(input_path):
    if input_path.endswith('/') or input_path.endswith('\\'):
        input_path = input_path[:-1]
    return input_path


def _parse_slicing(slicing: str):
    if slicing is not None:
        print(f"Slicing: {slicing}")
        dummy = s_[1, 2]
        slicing = eval(f"s_{slicing}")
    return slicing


@click.group()
def cli():
    print("------------------------------------------")
    print("  DEXP -- Data EXploration & Processing   ")
    print("  Royer lab                               ")
    print("------------------------------------------")
    print("")
    pass


from dexp.cli.commands.check import check
from dexp.cli.commands.copy import copy
from dexp.cli.commands.add import add
from dexp.cli.commands.fuse import fuse
from dexp.cli.commands.deconv import deconv
from dexp.cli.commands.isonet import isonet
from dexp.cli.commands.info import info
from dexp.cli.commands.tiff import tiff
from dexp.cli.commands.view import view
from dexp.cli.commands.render import render
from dexp.cli.commands.blend import blend
from dexp.cli.commands.stack import stack
from dexp.cli.commands.mp4 import mp4

cli.add_command(check)
cli.add_command(copy)
cli.add_command(add)
cli.add_command(fuse)
cli.add_command(deconv)
cli.add_command(isonet)
cli.add_command(info)
cli.add_command(tiff)
cli.add_command(view)
cli.add_command(render)
cli.add_command(blend)
cli.add_command(stack)
cli.add_command(mp4)
