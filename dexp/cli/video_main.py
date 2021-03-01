import click
from arbol.arbol import aprint


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
    aprint("  Video editing commands                  ")
    aprint("  Royer lab                               ")
    aprint("__________________________________________")
    aprint("")


from dexp.cli.video_commands.blend import overlay
from dexp.cli.video_commands.blend import blend
from dexp.cli.video_commands.stack import stack
from dexp.cli.video_commands.mp4 import mp4

cli.add_command(overlay)
cli.add_command(blend)
cli.add_command(stack)
cli.add_command(mp4)
