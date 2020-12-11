import click
from arbol.arbol import aprint, asection


@click.command()
def speedtest():
    import os
    cwd = os.getcwd()

    with asection(f"Measuring write/read at current path: {cwd}"):
        import subprocess

        filename = 'zzspeedtestfilezz'

        with asection("Write speed:"):
            result = subprocess.run(['time', 'sh', '-c', f'dd if=/dev/zero of="{cwd}/{filename}" bs=4M count=256 && sync'], capture_output=True)
            # aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        with asection("Read speed:"):
            result = subprocess.run(['time', 'sh', '-c', f'dd if="{cwd}/{filename}" of=/dev/null bs=4M count=256 && sync'], capture_output=True)
            # aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        os.remove(f"{cwd}/{filename}")
