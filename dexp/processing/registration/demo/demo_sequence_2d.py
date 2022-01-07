import random

from arbol import aprint, asection
from skimage.data import camera

from dexp.datasets.synthetic_datasets import (
    binary_blobs,
    generate_nuclei_background_data,
)
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.sequence import image_stabilisation
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_sequence_2d_numpy():
    with NumpyBackend():
        _register_sequence_2d()


def demo_register_sequence_2d_cupy():
    try:
        with CupyBackend():
            _register_sequence_2d()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_sequence_2d(
    length_xy=512,
    n=512,
    drift_strength=1.5,
    warp_grid_size=6,
    warp_strength=2.5,
    ratio_bad_frames=0.05,
    additive_noise=0.05,
    multiplicative_noise=0.1,
    run_test=False,
    display=True,
):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # used for simulation:
    shifts = []
    vector_fields = []
    x, y = 0, 0
    vector_field = xp.zeros((warp_grid_size,) * 2 + (2,), dtype=xp.float32)

    with asection("generate dataset"):

        random.seed(0)
        xp.random.seed(0)

        # prepare simulation data:
        for i in range(n):
            # drift:
            if i < (2 * n // 3):
                x += drift_strength * random.uniform(-1, 1.7)
                y += drift_strength * random.uniform(-1.3, 1)
            else:
                x += drift_strength * random.uniform(-1.5, 1)
                y += drift_strength * random.uniform(-1, 1.5)

            # deformation over time:
            vector_field += warp_strength * xp.random.uniform(low=-1, high=+1, size=(warp_grid_size,) * 2 + (2,))

            # simulate sudden imaging jumps:
            if i == n // 2 - n // 3:
                x += 38
                y += -35
                vector_field += (
                    16 * warp_strength * xp.random.uniform(low=-1, high=+1, size=(warp_grid_size,) * 2 + (2,))
                )
            if i == n // 2 + n // 3:
                x += 37
                y += -29
                vector_field += (
                    16 * warp_strength * xp.random.uniform(low=-1, high=+1, size=(warp_grid_size,) * 2 + (2,))
                )

            # keep for later:
            shifts.append((x, y))
            vector_fields.append(xp.copy(vector_field))

        # blobs:
        image1 = binary_blobs(length=length_xy, n_dim=2, blob_size_fraction=0.04, volume_fraction=0.05)
        image1 = image1.astype(xp.float32)
        image1 = sp.ndimage.gaussian_filter(image1, sigma=4)

        # camera:
        _ = camera().astype(xp.float32)  # image2

        # nuclei:
        _, _, image3 = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length_xy // 4,
            length_z_factor=1,
            independent_haze=True,
            sphere=True,
            add_offset=False,
            zoom=4,
            dtype=xp.float32,
        )
        image3 = image3[image3.shape[0] // 2].astype(xp.float32)

        # choice between 3 images:
        image = Backend.to_backend(image3)

        # crop so we test weird non-square image shape:
        image = image[0:511, 0:501]

        # generate reference 'ground truth' timelapse
        image = xp.stack(image for _ in range(n))

        # generate shifted, distorted, degraded timelapse:
        shifted = xp.stack(
            (
                sp.ndimage.shift(warp(i, vf, vector_field_upsampling=4), shift=s)
                for i, s, vf in zip(image, shifts, vector_fields)
            )
        )
        shifted *= xp.clip(
            xp.random.normal(loc=1, scale=multiplicative_noise, size=shifted.shape, dtype=xp.float32), 0.1, 10
        )
        shifted += additive_noise * xp.random.rand(*shifted.shape, dtype=xp.float32)
        shifted = xp.clip(shifted - 50, 0)

        # simulate dropped, highly corrupted frames:
        for _ in range(int(shifted.shape[0] * ratio_bad_frames)):
            index = random.randint(0, n - 1)
            shifted[index] = sp.ndimage.shift(shifted[index], shift=(random.uniform(-27, 27), random.uniform(-27, 27)))
            shifted[index] += xp.random.rand(*shifted[index].shape)
            shifted[index] *= random.uniform(0.001, 0.5)
            shifted[index] += random.uniform(0.001, 0.1) * xp.random.rand(*shifted[index].shape)

    with asection("register_translation_2d"):
        # compute image sequence stabilisation model:
        model = image_stabilisation(
            shifted, axis=0, sigma=3, edge_filter=False, denoise_input_sigma=1, debug_output="stabilisation"
        )
        aprint(f"model: {model}")

    with asection("shift back"):

        # apply stabilisation:
        stabilised_seq = model.apply_sequence(shifted, axis=0)

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.grid.enabled = True
            viewer.add_image(_c(shifted), name="shifted", colormap="bop purple", blending="additive", visible=True)
            viewer.add_image(
                _c(stabilised_seq), name="stabilised_seq", colormap="bop blue", blending="additive", visible=True
            )

    return image, shifted, stabilised_seq, model


if __name__ == "__main__":
    demo_register_sequence_2d_cupy()
