from os.path import join

from arbol import aprint, asection


def perform_speed_test():

    import os

    cwd = os.getcwd()
    with asection(f"Measuring write/read at current path: {cwd}"):
        import subprocess

        filename = "zz__speedtestfile__zz"
        filepath = join(cwd, filename)

        from sys import platform

        if platform == "linux" or platform == "linux2" or platform == "darwin":
            with asection("Write speed:"):
                result = subprocess.run(
                    ["time", "sh", "-c", f'dd if=/dev/zero of="{filepath}" bs=4194304 count=256 && sync'],
                    capture_output=True,
                )
                # aprint(result.stdout.decode("utf-8").split('\n'))
                aprint(result.stderr.decode("utf-8").split("\n")[2])

            with asection("Read speed:"):
                result = subprocess.run(
                    ["time", "sh", "-c", f'dd if="{filepath}" of=/dev/null bs=4194304 count=256 && sync'],
                    capture_output=True,
                )
                # aprint(result.stdout.decode("utf-8").split('\n'))
                aprint(result.stderr.decode("utf-8").split("\n")[2])

            with asection("Read speed with cache clearing:"):
                # clear cache:
                return_code = os.system("sync; echo 3 > /proc/sys/vm/drop_caches")
                if return_code != 0:
                    aprint("you must be root to clear the cache!")

                result = subprocess.run(
                    ["time", "sh", "-c", f'dd if="{filepath}" of=/dev/null bs=4194304 count=256 && sync'],
                    capture_output=True,
                )
                # aprint(result.stdout.decode("utf-8").split('\n'))
                aprint(result.stderr.decode("utf-8").split("\n")[2])

        elif platform == "win32":
            aprint(f"Not implemented yet for operating system: {platform}")
        else:
            aprint(f"Unknown operating system: {platform}")

        try:
            os.remove(filepath)
        except:  # noqa
            aprint(f"Could not delete file: {filepath}")
