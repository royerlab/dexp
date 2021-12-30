from os import system
from os.path import exists, join

import click
from arbol.arbol import aprint, asection

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group()
def cli():
    aprint("__________________________________________")
    aprint("  DEXP -- Data EXploration & Processing   ")
    aprint("  System installation commands            ")
    aprint("  Royer lab                               ")
    aprint("__________________________________________")
    aprint("")
    # input("Press Enter to continue...")


@click.command()
@click.argument("cuda", nargs=1)
def cudalibs(cuda: str):
    """Install additional CUDA libraries.

    CUDA is the cuda version: 11.2, 11.1, 11.0, ...
    """
    with asection("Install additional CUDA libraries"):

        from os.path import expanduser

        home = expanduser("~")
        if not exists(join(home, f".cupy/cuda_lib/{cuda}")):
            aprint(f"Installing CUDNN for CUDA {cuda}")
            system(f"python -m cupyx.tools.install_library --library cudnn --cuda {cuda}")
            aprint(f"Installing CUTENSOR for CUDA {cuda}")
            system(f"python -m cupyx.tools.install_library --library cutensor --cuda {cuda}")
        else:
            aprint("Libraries already installed")


cli.add_command(cudalibs)
