from aydin.corrections.background.background import BackgroundCorrection
from aydin.io.datasets import examples_single
from aydin.util.log.log import Log


def demo_suppress_fixed_background_real():
    Log.override_test_exclusion = True

    image = examples_single.huang_fixed_pattern_noise.get_array()  # [:, 0:64, 0:64]

    bs = BackgroundCorrection()

    pre_processed = bs.correct(image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(pre_processed, name='pre_processed')


demo_suppress_fixed_background_real()
