import os
from os.path import exists

import click
from arbol.arbol import aprint, asection


@click.command()
@click.argument("input_path")
@click.option("--prefix", "-p", type=str, default="frame_", help="Prefix for image filenames")
@click.option("--leading", "-l", type=int, default=5, help="Number of leading zeros for frame indices.")
@click.option("--extension", "-e", type=str, default="png", help="Extension for image filenames")
@click.option("--output_path", "-o", type=str, default=None, help="Output file path for movie")
@click.option(
    "--framerate", "-fps", type=int, default=30, help="Sets the framerate in frames per second", show_default=True
)
@click.option(
    "--codec",
    "-c",
    type=str,
    default="h264_nvenc",
    help="Encoding codec, For x264 class codecs: libx264 (reference but slow), or: h264_nvenc (faster, GPU). For x265 class codecs: libx265 (reference but slow), or: hevc_nvenc(faster, GPU",
    show_default=True,
)
@click.option(
    "--preset",
    "-ps",
    type=str,
    default="slow",
    help="Possible values: ultrafast, superfast, veryfast, faster, fast, medium (default preset), slow, slower, veryslow."
    " A preset is a collection of options that will provide a certain encoding speed to compression ratio."
    " A slower preset will provide better compression (compression is quality per filesize)."
    " This means that, for example, if you target a certain file size or constant bit rate,"
    " you will achieve better quality with a slower preset."
    " Similarly, for constant quality encoding, you will simply save bitrate by choosing a slower preset.",
    show_default=True,
)
@click.option(
    "--quality",
    "-q",
    type=int,
    default=0,
    help="Sets the quality of the video by modulating the bit rate. Set to a higher value for higher quality, or lower (negative ok) value for lower quality. Means different things for different codecs.",
    show_default=True,
)
@click.option(
    "--width",
    "-wi",
    type=int,
    default=0,
    help="Video frame width, recommended values: 1280(HD720), 1920(HD1080), 2048(2K), 3840(UHD-1), 4096(4K), 7680(8K). If zero then original size preserved, "
    "if -1 the best fit is found while allowing some downscaling. Height is automatically determined to preserve aspect ratio,"
    " and is forced to be a multiple of 32.",
    show_default=True,
)
@click.option(
    "--pixelformat",
    "-pf",
    type=str,
    default="yuv420p",
    help="Pixel format, can be yuv420p (default, recommended) or yuv444p (no chrominance downsampling but better compatibility),"
    " or any other allowed ffmpeg & codec pixel format.",
    show_default=True,
)
@click.option(
    "--overwrite", "-w", is_flag=True, help="to force overwrite of target", show_default=True
)  # , help='dataset slice'
def mp4(
    input_path,
    prefix,
    leading,
    extension,
    output_path,
    framerate,
    codec,
    preset,
    quality,
    width,
    pixelformat,
    overwrite,
):
    """Converts a folder of images into an MP4 video file."""

    if output_path is None:
        videofilepath = os.path.basename(os.path.normpath(input_path)) + ".mp4"
    else:
        videofilepath = output_path

    videofilepath = videofilepath.replace("frames_", "")

    if overwrite or not exists(videofilepath):
        with asection(
            f"Converting PNG files at: {input_path}, into MP4 file: {videofilepath}, framerate: {framerate} "
        ):

            scale_option = f"-vf scale={width}:-8:flags=bicubic" if width > 0 else ""
            # black_background_filter = f'-filter_complex "{scale_option};color=black,format={pixelformat}[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1"'

            # some codec wizardry to be able to effectively modulate quality
            quality = (
                f"-rc vbr -cq {26 - quality} -qmin {26 - quality} -qmax {26 - quality} -b:v 0 "
                if "nvenc" in codec
                else f"-crf {21 - quality}"
            )

            ffmpeg_command = (
                f"ffmpeg -hwaccel auto -framerate {framerate} -start_number 0 -i '{input_path}/{prefix}%0{leading}d.{extension}'  "
                f"-f mp4 -vcodec {codec} -preset {preset}  {quality} -pix_fmt {pixelformat} {scale_option} -y {videofilepath}"
            )
            # -vf  \"crop=576:1240:320:0\"  ""  ,setsar=1:1
            # -pix_fmt {pixelformat} {scale_option}
            # ,setsar=1:1
            # h264_nvenc nvenc_h264 libx264

            aprint(f"Executing command: '{ffmpeg_command}'")

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
