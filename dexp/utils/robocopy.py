import os
import subprocess

from arbol import aprint, asection


def robocopy(
    source_folder: str,
    dest_folder: str,
    move_files: bool = False,
    nb_threads: int = 8,
    large_files: bool = False,
    wait_to_finish: bool = True,
):
    """
    Start a generic robocopy job on Windows to copy files from one folder to another

    Args
    ---------

    Returns
    ---------
    copyProcess (Popen Object) :
    > this is the Popen object that's running the copy process. This can be
    > used to check or kill the process if necessary.
    """

    with asection(
        f"Starting a Windows Robocopy job: copying all files and folders from {source_folder} "
        + f"to {dest_folder} with {nb_threads} threads."
    ):
        # checks for network file paths in source and destination paths
        if "\\" in source_folder[:1]:
            source_folder = "\\" + source_folder

        if "\\" in dest_folder[:1]:
            dest_folder = "\\" + dest_folder

        # replaces slashes:
        source = (source_folder).replace("/", "\\")
        dest = (dest_folder).replace("/", "\\")

        # logfile:
        log_file = "robocopy_log.txt"
        if os.path.exists(log_file):
            os.remove(log_file)

        # format the command list for Popen
        copyCommand = [
            "ROBOCOPY",
            source,
            dest,
            "/e",
            "/R:10",
            "/W:5",
            "/TBD",
            "/NP",
            # '/V',
            "/eta",
            "/tee",
            f"/MT:{nb_threads}",
            f"/log:{log_file}",
        ]

        if move_files:
            copyCommand.append("/move")

        if large_files:
            copyCommand.append("/j")

        # use subprocess to start a copy command
        aprint(f"Robocopy command: {copyCommand}")
        copy_process = subprocess.Popen(copyCommand)

        if wait_to_finish:
            copy_process.wait()

    return copy_process
