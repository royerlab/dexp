import click
from arbol.arbol import aprint, asection


@click.command()
def speedtest():
    """ Estimates storage medium speed.
    """

    import os
    cwd = os.getcwd()

    with asection(f"Measuring write/read at current path: {cwd}"):
        import subprocess

        filename = 'zz__speedtestfile__zz'

        with asection("Write speed:"):
            result = subprocess.run(['time', 'sh', '-c', f'dd if=/dev/zero of="{cwd}/{filename}" bs=4M count=256 && sync'], capture_output=True)
            # aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        with asection("Read speed:"):
            result = subprocess.run(['time', 'sh', '-c', f'dd if="{cwd}/{filename}" of=/dev/null bs=4M count=256 && sync'], capture_output=True)
            # aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        with asection("Read speed with cache clearing:"):
            # clear cache:
            return_code = os.system('sync; echo 3 > /proc/sys/vm/drop_caches')
            if return_code != 0:
                aprint("you must be root to clear the cache!")

            result = subprocess.run(['time', 'sh', '-c', f'dd if="{cwd}/{filename}" of=/dev/null bs=4M count=256 && sync'], capture_output=True)
            # aprint(result.stdout.decode("utf-8").split('\n'))
            aprint(result.stderr.decode("utf-8").split('\n')[2])

        os.remove(f"{cwd}/{filename}")
