import os
from os.path import exists

import click
from arbol.arbol import aprint, asection


@click.command()
@click.argument('input_path')
@click.option('--prefix', '-p', type=str, default='frame_', help='prefix for image filenames')
@click.option('--extension', '-e', type=str, default='png', help='extension for image filenames')
@click.option('--output_path', '-o', type=str, default=None, help='Output file path for movie')
@click.option('--framerate', '-fps', type=int, default=30, help='Sets the framerate in frames per second', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
def mp4(input_path, prefix, extension, output_path, framerate, overwrite=False):
    if output_path is None:
        videofilepath = os.path.basename(os.path.normpath(input_path)) + '.mp4'
    else:
        videofilepath = output_path

    videofilepath = videofilepath.replace('frames_', '')

    if overwrite or not exists(videofilepath):
        with asection(f"Converting PNG files at: {input_path}, into MP4 file: {videofilepath}, framerate: {framerate} "):
            ffmpeg_command = f"ffmpeg -framerate {framerate} -start_number 0 -i '{input_path}/{prefix}%d.{extension}'  " \
                             f"-f mp4 -vcodec libx264 -preset slow -pix_fmt yuv420p -y {videofilepath}"
            # -vf  \"crop=576:1240:320:0\"
            os.system(ffmpeg_command)
    else:
        aprint(f"Video file: {videofilepath} already exists! use -w option to force overwrite...")
