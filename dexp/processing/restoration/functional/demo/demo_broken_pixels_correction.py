from scipy.ndimage import median_filter
from skimage.util import random_noise

from aydin.corrections.broken_pixels.broken_pixels import BrokenPixelsCorrection
from aydin.io.datasets import camera, normalise
from aydin.util.log.log import Log


def demo_suppress_fixed_background_real():
    Log.override_test_exclusion = True

    image = normalise(camera())
    image = random_noise(image, mode="s&p", amount=0.03, seed=0, clip=False)

    bpc = BrokenPixelsCorrection(lipschitz=0.15)

    corrected = bpc.correct(image)

    median = median_filter(image, size=3)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(median, name='median_filtered')
        viewer.add_image(corrected, name='corrected')


demo_suppress_fixed_background_real()
