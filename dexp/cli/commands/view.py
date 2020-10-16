import click
import dask
import numpy

from dexp.cli.main import _get_dataset_from_path, _parse_slicing


@click.command()
@click.argument('input_path')
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--volume', '-v', is_flag=True, help='to view with volume rendering (3D ray casting)', show_default=True)
@click.option('--aspect', '-a', type=float, default=4, help='sets aspect ratio e.g. 4', show_default=True)
@click.option('--colormap', '-cm', type=str, default='viridis', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--render', '-r', type=str, default=None, help='Renders video using napari movie script (not great, prefer the render command instead)')
@click.option('--windowsize', '-ws', type=int, default=1536, help='Sets the napari window size. i.e. -ws 400 sets the window to 400x400', show_default=True)
@click.option('--clim', '-cl', type=str, default=None, help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]', show_default=True)
def view(input_path, channels=None, slicing=None, volume=False, aspect=None, colormap='viridis', render=None, windowsize=1536, clim=None):

    from napari import Viewer, gui_qt
    from napari._qt.threading import thread_worker

    input_dataset = _get_dataset_from_path(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    slicing = _parse_slicing(slicing)
    print(f"Requested slicing: {slicing} ")

    print(f"Available channel(s): {input_dataset.channels()}")
    print(f"Requested channel(s): {channels}")
    print(f"Selected channel(s):  {selected_channels}")

    # Annoying napari induced warnings:
    import warnings
    warnings.filterwarnings("ignore")

    with gui_qt():
        viewer = Viewer(title=f"DEXP | viewing with napari: {input_path} ", ndisplay=3 if volume else 2)

        viewer.window.resize(windowsize+256, windowsize)

        for channel in selected_channels:
            print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
            print(input_dataset.info(channel))

            array = input_dataset.get_array(channel, wrap_with_dask=True)

            if slicing:
                array = array[slicing]

            print(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

            if clim is None:
                print(f"Computing min and max from first stack...")
                first_stack = numpy.array(input_dataset.get_stack(channel, 0, per_z_slice=False))[::8]
                min_value = numpy.percentile(first_stack[::16], q=0.1)
                max_value = numpy.percentile(first_stack[::16], q=99.99)
                print(f"min={min_value} and max={max_value}.")
                contrast_limits = [max(0, min_value - 32), max_value + 32]
            else:
                print(f"provided min and max for contrast limits: {clim}")
                min_value, max_value = ( float(strvalue) for strvalue in clim.split(','))
                contrast_limits = [min_value, max_value]


            # flip x for second camera:
            if 'C1' in channel:
                array = dask.array.flip(array,-1)

            layer = viewer.add_image(array,
                                     name=channel,
                                     contrast_limits=contrast_limits,
                                     blending='additive',
                                     colormap=colormap,
                                     attenuation=0.04,
                                     rendering='attenuated_mip')

            if not aspect is None:
                layer.scale[-3] = aspect
                print(f"Setting aspect ratio to {aspect} (layer.scale={layer.scale})")

            #For some reason spome parameters refuse to be set, this solves it:
            @thread_worker
            def workaround_for_recalcitrant_parameters():
                print("Setting 3D rendering parameters")
                layer.attenuation=0.02
                layer.rendering='attenuated_mip'

            worker = workaround_for_recalcitrant_parameters()
            worker.start()


            if render is not None:

                render = render.strip()
                parameters = dict(item.split("=") for item in render.split(",")) if render != 'defaults' else dict()

                backend = parameters['backend'] if 'backend' in parameters else 'naparimovie'

                if backend == 'naparimovie':
                    from naparimovie import Movie

                    script = parameters['script'] if 'script' in parameters else 'script.txt'
                    steps = int(parameters['steps']) if 'steps' in parameters else 60
                    res = int(parameters['res']) if 'res' in parameters else 1024
                    fps = int(parameters['fps']) if 'fps' in parameters else 60
                    name = parameters['name'] if 'name' in parameters else 'movie.mp4'

                    print(f"Movie Parameters provided: {parameters}")
                    print(f"Movie script: {script}, steps={steps}, res={res}, fps={fps}, name={name}")

                    #time.sleep(1)
                    movie = Movie(myviewer=viewer)
                    movie.inter_steps = steps
                    movie.create_state_dict_from_script(script)
                    movie.make_movie(name=name, resolution=res, fps=fps, show=False)