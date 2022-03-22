# flake8: noqa
from arbol import Arbol, asection

from dexp.processing.crop.representative_crop import representative_crop
from dexp.utils.backends import CupyBackend, NumpyBackend


def demo_rep_crop_numpy():
    with NumpyBackend():
        _demo_representative_crop(fast_mode=True)
        _demo_representative_crop(fast_mode=False)


def demo_rep_crop_cupy():
    try:
        with CupyBackend():
            _demo_representative_crop(fast_mode=True)
            _demo_representative_crop(fast_mode=False)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_representative_crop(fast_mode: bool = True, display: bool = True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Arbol.enable_output = True
    Arbol.set_log_max_depth(5)

    from skimage import data
    from skimage.color import rgb2gray

    image = data.astronaut()
    image = rgb2gray(image)

    with asection(f"Computing crop for image of shape: {image.shape}"):
        crop_size = 16000
        crop = representative_crop(image, crop_size=crop_size, fast_mode=fast_mode, display=False)

    if display:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name="image")
            viewer.add_image(crop, name="crop")

    assert crop.size <= int(crop_size * 1.05)


if __name__ == "__main__":
    demo_rep_crop_cupy()
    demo_rep_crop_numpy()
