import os
from os.path import exists

import click
from arbol.arbol import aprint, asection


@click.command()
@click.argument('input_path')
@click.option('--prefix', '-p', type=str, default='frame_', help='Prefix for image filenames')
@click.option('--leading', '-l', type=int, default=5, help='Number of leading zeros for frame indices.')
@click.option('--extension', '-e', type=str, default='png', help='Extension for image filenames')
@click.option('--output_path', '-o', type=str, default=None, help='Output file path for movie')
@click.option('--framerate', '-fps', type=int, default=30, help='Sets the framerate in frames per second', show_default=True)
@click.option('--preset', '-ps', type=str, default='slow',
              help='A preset is a collection of options that will provide a certain encoding speed to compression ratio.'
                   'A slower preset will provide better compression (compression is quality per filesize).'
                   'This means that, for example, if you target a certain file size or constant bit rate,'
                   'you will achieve better quality with a slower preset.'
                   'Similarly, for constant quality encoding, you will simply save bitrate by choosing a slower preset.',
              show_default=True)
@click.option('--constantratefactor', '-crf', type=int, default=20,
              help='The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible.'
                   'A lower value generally leads to higher quality, and a subjectively sane range is 17–28.'
                   'Consider 17 or 18 to be visually lossless or nearly so; it should look the same or nearly the same '
                   'as the input but it is not technically lossless.'
                   'The range is exponential, so increasing the CRF value +6 results in roughly half the bitrate / file size,'
                   ' while -6 leads to roughly twice the bitrate.',
              show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
def mp4(input_path,
        prefix,
        leading,
        extension,
        output_path,
        framerate,
        preset,
        constantratefactor,
        overwrite):

    if output_path is None:
        videofilepath = os.path.basename(os.path.normpath(input_path)) + '.mp4'
    else:
        videofilepath = output_path

    videofilepath = videofilepath.replace('frames_', '')

    if overwrite or not exists(videofilepath):
        with asection(f"Converting PNG files at: {input_path}, into MP4 file: {videofilepath}, framerate: {framerate} "):
            ffmpeg_command = f"ffmpeg -hwaccel auto -framerate {framerate} -start_number 0 -i '{input_path}/{prefix}%0{leading}d.{extension}'  " \
                             f"-f mp4 -vcodec libx264 -preset {preset} -crf {constantratefactor} -pix_fmt yuv420p  -y {videofilepath}"
            # -vf  \"crop=576:1240:320:0\"
            os.system(ffmpeg_command)
    else:
        aprint(f"Video file: {videofilepath} already exists! use -w option to force overwrite...")
#
# #low quality
# #ffmpeg -r 4 -i /data_fish_TL100_range1200um_step0.31_4um_15ms_dualv_fused_colormax_projection/frame_%05d.png -vcodec mpeg4 -y movie.mp4
#
# #high quality
# ffmpeg -framerate 30 -i data_fish_TL100_range1200um_step0.31_4um_15ms_dualv_fused_stabilized_colormax_projection/frame_%05d.png -f mp4 -vcodec libx264 -preset slow -pix_fmt yuv420p -y movie.mp4
#
# #change resolution
# ffmpeg -i movie.mp4 -s hd1080 movie_1080.mp4
# # or use -vf scale=1920x1080 to set to other scales
