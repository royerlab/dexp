import click
from arbol.arbol import aprint

from dexp.cli.video_commands.blend import blend
from dexp.cli.video_commands.mp4 import mp4
from dexp.cli.video_commands.overlay import overlay
from dexp.cli.video_commands.resize import resize
from dexp.cli.video_commands.stack import stack

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group()
def cli():
    aprint("__________________________________________")
    aprint("  DEXP -- Data EXploration & Processing   ")
    aprint("  Video editing dexp_commands                  ")
    aprint("  Royer lab                               ")
    aprint("__________________________________________")
    aprint("")
    aprint(
        "Note: videos are folders of images (typically PNGs), final step is to convert such image sequences into MP4s"
    )
    # input("Press Enter to continue...")


cli.add_command(overlay)
cli.add_command(blend)
cli.add_command(stack)
cli.add_command(resize)
cli.add_command(mp4)
