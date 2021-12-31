from dexp.datasets.synthetic_datasets import generate_fusion_test_data
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_translation_nD_numpy():
    with NumpyBackend():
        register_translation_nD()


def demo_register_translation_nD_cupy():
    try:
        with CupyBackend():
            register_translation_nD()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def register_translation_nD(length_xy=320):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(
        add_noise=False, shift=(1, 5, -13), volume_fraction=0.5, length_xy=length_xy, length_z_factor=2
    )

    model = TranslationRegistrationModel(shift_vector=(-1, -5, 13))

    image1_reg, image2_reg = model.apply_pair(image1, image2, pad=False)
    image1_reg_pad, image2_reg_pad = model.apply_pair(image1, image2, pad=True)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name="image_gt", visible=False)
        viewer.add_image(_c(image1), name="image1", colormap="bop blue", blending="additive")
        viewer.add_image(_c(image2), name="image2", colormap="bop orange", blending="additive", visible=False)
        viewer.add_image(_c(image1_reg), name="image1_reg", colormap="bop blue", blending="additive", visible=False)
        viewer.add_image(_c(image2_reg), name="image2_reg", colormap="bop orange", blending="additive")
        viewer.add_image(
            _c(image1_reg_pad), name="image1_reg_pad", colormap="bop blue", blending="additive", visible=False
        )
        viewer.add_image(_c(image2_reg_pad), name="image2_reg_pad", colormap="bop orange", blending="additive")


if __name__ == "__main__":
    demo_register_translation_nD_cupy()
    demo_register_translation_nD_numpy()
