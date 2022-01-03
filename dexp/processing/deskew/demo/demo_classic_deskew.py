from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deskew.classic_deskew import classic_deskew_dimensionless
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_classic_deskew_numpy():
    with NumpyBackend():
        _classic_deskew(length=64)


def demo_classic_deskew_cupy():
    try:
        with CupyBackend():
            _classic_deskew()
            return True
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")
        return False


def _classic_deskew(length: int = 128, zoom: int = 4, display: bool = True):

    _deskew(1, length, zoom, display)
    _deskew(0.5, length, zoom, display)


def _deskew(shift, length, zoom, display):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()
    with timeit("generate demo image"):
        # generate nuclei image:
        _, _, image = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length,
            length_z_factor=1,
            independent_haze=True,
            sphere=True,
            zoom=zoom,
            add_offset=False,
            background_strength=0.07,
            dtype=xp.float32,
        )

        # Pad:
        pad_width = (
            (int(shift * zoom * length // 2), int(shift * zoom * length // 2)),
            (0, 0),
            (0, 0),
        )
        image = xp.pad(image, pad_width=pad_width)

        # Add blur:
        psf = nikon16x08na()
        psf = psf.astype(dtype=image.dtype, copy=False)
        image = fft_convolve(image, psf)

        # apply skew:
        matrix = xp.asarray([[1, shift, 0], [0, 1, 0], [0, 0, 1]])
        offset = 0 * xp.asarray([image.shape[0] // 2, 0, 0])
        # matrix = xp.linalg.inv(matrix)
        skewed = sp.ndimage.affine_transform(image, matrix, offset=offset)

        # Add noise and clip
        # skewed += xp.random.uniform(-1, 1)
        skewed = xp.clip(skewed, a_min=0, a_max=None)

        # cast to uint16:
        skewed = skewed.astype(dtype=xp.uint16)
    with timeit("deskew image"):
        # apply deskewing:
        deskewed = classic_deskew_dimensionless(
            skewed,
            depth_axis=0,
            lateral_axis=1,
            shift=shift,
        )
    if display:
        import napari
        from napari import Viewer

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer(ndisplay=3)
        viewer.add_image(
            _c(image),
            name="image",
            colormap="bop orange",
            blending="additive",
            rendering="attenuated_mip",
            attenuation=0.01,
        )
        viewer.add_image(
            _c(skewed),
            name="skewed_image",
            colormap="bop blue",
            blending="additive",
            visible=False,
            rendering="attenuated_mip",
            attenuation=0.01,
        )
        viewer.add_image(
            _c(deskewed),
            name="deskewed_image",
            colormap="bop purple",
            blending="additive",
            rendering="attenuated_mip",
            attenuation=0.01,
        )

        napari.run()
    # compute mean absolute errors:
    error_skewed = xp.mean(xp.absolute(image - skewed))
    error_deskewed = xp.mean(xp.absolute(image - deskewed))
    print(f"error_skewed = {error_skewed}")
    print(f"error_deskewed = {error_deskewed}")
    # Asserts that check if things behave as expected:
    assert error_deskewed < error_skewed
    assert error_skewed > 10
    assert error_deskewed < 2


if __name__ == "__main__":
    if not demo_classic_deskew_cupy():
        demo_classic_deskew_numpy()
