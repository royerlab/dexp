from arbol import aprint

from dexp.utils.backends import CupyBackend, NumpyBackend
from dexp.video.demo.demo_crop_resize_pad import demo_video_crop_resize_pad


def test_overlay_numpy():
    with NumpyBackend():
        demo_video_crop_resize_pad(display=False)


def test_overlay_cupy():
    try:
        with CupyBackend():
            demo_video_crop_resize_pad(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
