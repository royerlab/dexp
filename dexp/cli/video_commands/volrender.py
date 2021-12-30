import math
import os
from os.path import exists, join

import click
import numpy
from arbol.arbol import aprint, asection, section

from dexp.cli.parsing import _parse_channels, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets


@click.command()
@click.argument("input_paths", nargs=-1)
@click.option(
    "--output_path", "-o", default=None, help="Output folder to store rendered PNGs. Default is: frames_<channel_name>"
)
@click.option("--channels", "-c", default=None, help="list of channels to color, all channels when ommited.")
@click.option(
    "--slicing",
    "-s",
    default=None,
    help="dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).",
)
@click.option(
    "--overwrite", "-w", is_flag=True, help="to force overwrite of target", show_default=True
)  # , help='dataset slice'
@click.option("--aspect", "-a", type=float, default=4, help="sets aspect ratio e.g. 4", show_default=True)
@click.option(
    "--colormap",
    "-cm",
    type=str,
    default="magma",
    help="sets colormap, e.g. viridis, gray, magma, plasma, inferno ",
    show_default=True,
)
@click.option(
    "--rendersize",
    "-rs",
    type=int,
    default=2048,
    help="Sets the frame color size. i.e. -ws 400 sets the window to 400x400",
    show_default=True,
)
@click.option(
    "--clim",
    "-cl",
    type=str,
    default="0,1024",
    help="Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]",
)
@click.option(
    "--options",
    "-opt",
    type=str,
    default=None,
    help="Render options, e.g: 'gamma=1.2,box=True'. Important: no white spaces!!! Complete list with defaults will be displayed on first run",
)
def volrender(
    input_paths,
    output_path,
    channels=None,
    slicing=None,
    overwrite=False,
    aspect=None,
    colormap="viridis",
    rendersize=1536,
    clim=None,
    options=None,
):
    """Renders 3D volumetric video"""

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, channels)
    slicing = _parse_slicing(slicing)

    aprint(f"Volumetric rendering of: {input_paths} to {output_path} for channels: {channels}, slicing: {slicing} ")

    for channel in channels:

        with asection(f"Channel '{channel}' shape: {input_dataset.shape(channel)}:"):
            aprint(input_dataset.info(channel))

        array = input_dataset.get_array(channel, wrap_with_dask=True)
        dtype = array.dtype

        if slicing:
            array = array[slicing]

        aprint(f"Rendering array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

        if clim is None:
            aprint("Computing min and max from first stack...")
            first_stack = numpy.array(input_dataset.get_stack(channel, 0, per_z_slice=False))
            min_value = max(0, first_stack.min() - 100)
            max_value = first_stack.max() + 100
            aprint(f"min={min_value} and max={max_value}.")
        else:
            aprint(f"provided min and max for contrast limits: {clim}")
            min_value, max_value = (float(strvalue) for strvalue in clim.split(","))

        aprint(f"Provided rendering options: {options}")
        options = dict(item.split("=") for item in options.split(",")) if options is not None else dict()

        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")

        nbtp = array.shape[0]
        nbframes = int(options["nbframes"]) if "nbframes" in options else 1010

        skip = int(options["skip"]) if "skip" in options else 1
        cut = str(options["cut"]) if "cut" in options else "nocut"
        cutpos = float(options["cutpos"]) if "cutpos" in options else 0
        cutspeed = float(options["cutspeed"]) if "cutspeed" in options else 0
        irot = options["irot"] if "irot" in options else ""
        raxis = options["raxis"] if "raxis" in options else "y"
        rstart = float(options["rstart"]) if "rstart" in options else 0
        rspeed = float(options["rspeed"]) if "rspeed" in options else 0.15
        tstart = int(options["tstart"]) if "tstart" in options else 0
        tspeed = float(options["tspeed"]) if "tspeed" in options else 0.5
        gamma = float(options["gamma"]) if "gamma" in options else 0.9
        zoom = float(options["zoom"]) if "zoom" in options else 1.45
        alpha = float(options["alpha"]) if "alpha" in options else 0.3
        box = str2bool(options["box"]) if "box" in options else False
        maxsteps = int(options["maxsteps"]) if "maxsteps" in options else 1024
        norm = str2bool(options["norm"]) if "norm" in options else True
        normrange = float(options["normrange"]) if "normrange" in options else 1024
        videofilename = options["name"] if "name" in options else "video.mp4"

        with asection("Options:"):
            aprint(f"Video filename          : {videofilename}")
            aprint(f"Number of time points   : {nbtp}")
            aprint(f"Number of frames        : {nbframes}  \toption: nbframes: \tint")
            aprint(f"Render one frame every  : {skip}      \toption: skip:     \tint")
            aprint(
                f"Volume cutting          : {cut}       \toption: cut:      \t[left, right, top, bottom, front, back, none]"
            )
            aprint(f"Volume cutting position : {cutpos}    \toption: cutpos:   \tfloat")
            aprint(f"Volume cutting speed    : {cutspeed}  \toption: cutspeed: \tfloat")
            aprint(f"Initial time point      : {tstart}    \toption: tstart:   \tint")
            aprint(f"Time    speed           : {tspeed}    \toption: tspeed:   \tfloat")
            aprint(
                f"Initial rotation        : {irot}      \toption: irot:     \t xxyzzz -> 90deg along x, 45deg along y, 135deg along z "
            )
            aprint(f"Rotation axis           : {raxis}     \toption: raxis:    \t[x,y,z]")
            aprint(f"Initial rotation angle  : {rstart}    \toption: rstart:   \tfloat")
            aprint(f"Rotation speed          : {rspeed}    \toption: rspeed:   \tfloat")
            aprint(f"Gamma                   : {gamma}     \toption: gamma:    \tfloat")
            aprint(f"Zoom                    : {zoom}      \toption: zoom:     \tfloat")
            aprint(f"Alpha                   : {alpha}     \toption: alpha:    \tfloat")
            aprint(f"box?                    : {box}       \toption: box:      \tbool (true/false)")
            aprint(f"normalisation           : {norm}      \toption: norm:     \tbool (true/false)")
            aprint(f"normalisation range     : {normrange} \toption: normrange:\tfloat")
            aprint(f"Max steps for vol color: {maxsteps}  \toption: maxsteps: \tint")

        if output_path is None:
            output_path = f"frames_{channel}"
        os.makedirs(output_path, exist_ok=True)
        from spimagine import DataModel, NumpyData, volshow

        aprint("Opening spimagine...")
        import spimagine

        spimagine.config.__DEFAULTMAXSTEPS__ = maxsteps
        spimagine.config.__DEFAULT_TEXTURE_WIDTH__ = rendersize

        datamodel = DataModel(NumpyData(array[0].compute()))
        aprint("Creating Spimagine window... (you can minimise but don't close!)")
        win = volshow(datamodel, stackUnits=(1.0, 1.0, aspect), autoscale=False, show_window=True)
        win.resize(rendersize + 32, rendersize + 32)
        win.showMinimized()

        while section("Rendering:"):
            for i in range(0, nbframes, skip):
                aprint("______________________________________________________________________________")
                aprint(f"Frame     : {i}")

                tp = tstart + int(tspeed * i)
                if tp >= nbtp:
                    break
                aprint(f"Time point: {tp}")

                angle = rstart + rspeed * i
                aprint(f"Angle     : {angle}")

                effcutpos = cutpos + cutspeed

                filename = join(output_path, f"frame_{i:05}.png")

                if overwrite or not exists(filename):

                    aprint("Loading stack...")
                    stack = array[int(tp)].compute()

                    if norm:
                        aprint("Computing percentile...")
                        rmax = numpy.percentile(stack[::8].astype(numpy.float32), q=99.99).astype(numpy.float32)
                        # rmax = numpy.max(stack[::8]).astype(numpy.float32)
                        aprint(f"rmax={rmax}")

                        aprint("Normalising...")
                        norm_max_value = normrange
                        norm_min_value = 64.0
                        # stack = norm_min_value+stack*((norm_max_value-norm_min_value)/rmax)
                        stack = stack * numpy.array((norm_max_value - norm_min_value) / rmax, dtype=numpy.float32)
                        stack += numpy.array(norm_min_value, dtype=dtype)
                        stack = stack.astype(dtype)

                    # print("Opening spimagine...")
                    # win = volshow(stack, stackUnits=(1., 1., aspect), autoscale=False, show_window=True)

                    aprint("Loading stack into Spimagine...")
                    datamodel.setContainer(NumpyData(stack))
                    win.setModel(datamodel)

                    aprint("Setting rendering parameters...")
                    if colormap in spimagine.config.__COLORMAPDICT__:
                        win.set_colormap(colormap)
                    else:
                        from matplotlib import colors

                        rgb = colors.to_rgba(colormap)[:3]
                        aprint(f"Turning the provided color: {colormap} = {rgb} into a colormap.")
                        win.set_colormap_rgb(rgb)

                    win.transform.setStackUnits(1.0, 1.0, aspect)
                    win.transform.setGamma(gamma)
                    win.transform.setMin(min_value)
                    win.transform.setMax(max_value)
                    win.transform.setZoom(zoom)
                    win.transform.setAlphaPow(alpha)
                    win.transform.setBox(box)

                    if cut == "left":
                        win.transform.setBounds(effcutpos, 1, -1, 1, -1, 1)
                    elif cut == "right":
                        win.transform.setBounds(-1, effcutpos, -1, 1, -1, 1)
                    elif cut == "top":
                        win.transform.setBounds(-1, 1, effcutpos, 1, -1, 1)
                    elif cut == "bottom":
                        win.transform.setBounds(-1, 1, -1, effcutpos, -1, 1)
                    elif cut == "front":
                        win.transform.setBounds(-1, 1, -1, 1, effcutpos, 1)
                    elif cut == "back":
                        win.transform.setBounds(-1, 1, -1, 1, -1, effcutpos)
                    elif cut == "centerx":
                        win.transform.setBounds(-0.25 - effcutpos, 0.25 + effcutpos, -1, 1, -1, 1)
                    elif cut == "centery":
                        win.transform.setBounds(-1, 1, -0.25 - effcutpos, 0.25 + effcutpos, -1, 1)
                    elif cut == "centerz":
                        win.transform.setBounds(-1, 1, -1, 1, -0.25 - effcutpos, 0.25 + effcutpos)
                    elif cut == "none":
                        win.transform.setBounds(-1, 1, -1, 1, -1, 1)

                    win.transform.setRotation(0, 1, 0, 0)

                    for character in irot:
                        if character == "x":
                            aprint(f"Rotating along x axis by 45 deg (prev quatRot={win.transform.quatRot})")
                            win.transform.addRotation(0.5 * math.pi / 4, 1, 0, 0)
                        elif character == "y":
                            aprint(f"Rotating along y axis by 45 deg (prev quatRot={win.transform.quatRot})")
                            win.transform.addRotation(0.5 * math.pi / 4, 0, 1, 0)
                        elif character == "z":
                            aprint(f"Rotating along z axis by 45 deg (prev quatRot={win.transform.quatRot})")
                            win.transform.addRotation(0.5 * math.pi / 4, 0, 0, 1)
                    aprint(f"Rotation after initial axis rotation: {win.transform.quatRot}")

                    if "x" in raxis:
                        aprint(f"Rotating along x axis by {angle} deg (prev quatRot={win.transform.quatRot})")
                        win.transform.addRotation(0.5 * angle * (math.pi / 180), 1, 0, 0)
                    if "y" in raxis:
                        aprint(f"Rotating along y axis by {angle} deg (prev quatRot={win.transform.quatRot})")
                        win.transform.addRotation(0.5 * angle * (math.pi / 180), 0, 1, 0)
                    if "z" in raxis:
                        aprint(f"Rotating along z axis by {angle} deg (prev quatRot={win.transform.quatRot})")
                        win.transform.addRotation(0.5 * angle * (math.pi / 180), 0, 0, 1)

                    aprint(f"Final rotation: {win.transform.quatRot}")

                    aprint(f"Saving frame: {filename}")
                    win.saveFrame(filename)

    win.closeMe()
    input_dataset.close()
    aprint("Done!")

    raise SystemExit
    import sys

    sys.exit()
