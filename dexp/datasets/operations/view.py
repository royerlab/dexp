import dask
import numpy
from arbol import aprint
from napari import gui_qt, Viewer
from napari._qt.qthreading import thread_worker


def dataset_view(aspect,
                 channels,
                 clim,
                 colormap,
                 input_dataset,
                 input_path,
                 slicing,
                 volume,
                 windowsize):

    # Annoying napari induced warnings:
    import warnings
    warnings.filterwarnings("ignore")
    with gui_qt():
        viewer = Viewer(title=f"DEXP | viewing with napari: {input_path} ", ndisplay=3 if volume else 2)

        viewer.window.resize(windowsize + 256, windowsize)

        for channel in channels:
            aprint(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
            aprint(input_dataset.info(channel))

            array = input_dataset.get_array(channel, wrap_with_dask=True)

            if slicing:
                array = array[slicing]

            aprint(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

            if clim is None:
                aprint(f"Computing min and max from first stack...")
                first_stack = numpy.array(input_dataset.get_stack(channel, 0, per_z_slice=False))[::8]
                first_stack_proj = numpy.max(first_stack, axis=0)
                min_value = numpy.percentile(first_stack_proj[::4], q=0.1)
                max_value = numpy.percentile(first_stack_proj[::4], q=99.99)
                aprint(f"min={min_value} and max={max_value}.")
                contrast_limits = [max(0, min_value - 32), max_value + 32]
            else:
                aprint(f"provided min and max for contrast limits: {clim}")
                min_value, max_value = (float(strvalue) for strvalue in clim.split(','))
                contrast_limits = [min_value, max_value]

            # flip x for second camera:
            if 'C1' in channel:
                array = dask.array.flip(array, -1)

            layer = viewer.add_image(array,
                                     name=channel,
                                     contrast_limits=contrast_limits,
                                     blending='additive',
                                     colormap=colormap,
                                     attenuation=0.04,
                                     rendering='attenuated_mip')

            if not aspect is None:
                layer.scale = (1, aspect, 1, 1) if array.ndim == 4 else (aspect, 1, 1)
                aprint(f"Setting aspect ratio to {aspect} (layer.scale={layer.scale})")

            # For some reason spome parameters refuse to be set, this solves it:
            @thread_worker
            def workaround_for_recalcitrant_parameters():
                aprint("Setting 3D rendering parameters")
                layer.attenuation = 0.02
                layer.rendering = 'attenuated_mip'

            worker = workaround_for_recalcitrant_parameters()
            worker.start()
