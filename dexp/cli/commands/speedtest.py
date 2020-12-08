import click
from arbol.arbol import aprint, asection


@click.command()
def speedtest():

    with asection(f"Measuring write/read at current path:"):

        import os
        import subprocess
        cwd = os.getcwd()

        with asection("Write speed:"):
            result = subprocess.run(['time', 'sh', '-c', f"dd if=/dev/zero of={cwd}/__speedtestfile__ bs=4M count=256 && sync" ], capture_output=True)
            #aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        with asection("Read speed:"):
            result = subprocess.run(['time', 'sh', '-c', f"dd if={cwd}/__speedtestfile__ of=/dev/null bs=4M count=256 && sync"], capture_output=True)
            #aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        os.remove(f"{cwd}/__speedtestfile__")


