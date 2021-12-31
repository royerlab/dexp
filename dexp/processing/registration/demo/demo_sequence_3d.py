import random

from arbol import aprint, asection

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.sequence import image_stabilisation
from dexp.processing.registration.sequence_proj import image_stabilisation_proj
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_sequence_3d_numpy():
    with NumpyBackend():
        _register_sequence_3d(use_projections=True)
        # _register_sequence_3d(use_projections=False)


def demo_register_sequence_3d_cupy():
    try:
        with CupyBackend():
            _register_sequence_3d(use_projections=True)
            # _register_sequence_3d(use_projections=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_sequence_3d(
    length_xy=320,
    n=128,
    drift_strength=1.5,
    warp_grid_size=8,
    warp_strength=2.5,
    ratio_bad_frames=0.05,
    additive_noise=0.05,
    multiplicative_noise=0.1,
    use_projections=False,
    display=True,
):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # used for simulation:
    shifts = []
    vector_fields = []
    x, y, z = 0, 0, 0
    vector_field = xp.zeros((warp_grid_size,) * 3 + (3,), dtype=xp.float32)

    with asection("generate dataset"):

        random.seed(0)
        xp.random.seed(0)

        # prepare simulation data:
        for i in range(n):
            # drift:
            if i < (2 * n // 3):
                x += drift_strength * random.uniform(-1, 1.7)
                y += drift_strength * random.uniform(-1.3, 1)
                z += drift_strength * random.uniform(-1, 1)
            else:
                x += drift_strength * random.uniform(-1.5, 1)
                y += drift_strength * random.uniform(-1, 1.5)
                z += drift_strength * random.uniform(-1, 1)

            # deformation over time:
            vector_field += warp_strength * xp.random.uniform(low=-1, high=+1, size=(warp_grid_size,) * 3 + (3,))

            # simulate sudden imaging jumps:
            if i == n // 2 - n // 3:
                x += 17
                y += -25
                z += 18
                vector_field += (
                    10 * warp_strength * xp.random.uniform(low=-1, high=+1, size=(warp_grid_size,) * 3 + (3,))
                )
            if i == n // 2 + n // 3:
                x += -27
                y += +19
                z += -11
                vector_field += (
                    10 * warp_strength * xp.random.uniform(low=-1, high=+1, size=(warp_grid_size,) * 3 + (3,))
                )

            # keep for later:
            shifts.append((x, y, z))
            vector_fields.append(xp.copy(vector_field))

        # nuclei:
        _, _, image = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length_xy // 4,
            length_z_factor=1,
            independent_haze=True,
            sphere=True,
            add_offset=False,
            zoom=2,
            dtype=xp.float32,
        )
        image = Backend.to_backend(image)

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
            shifted[index] = sp.ndimage.shift(
                shifted[index], shift=(random.uniform(-27, 27), random.uniform(-27, 27), random.uniform(-27, 27))
            )
            shifted[index] += xp.random.rand(*shifted[index].shape)
            shifted[index] *= random.uniform(0.001, 0.5)
            shifted[index] += random.uniform(0.001, 0.1) * xp.random.rand(*shifted[index].shape)

    with asection("register_translation_2d"):
        # compute image sequence stabilisation model:
        if use_projections:
            model = image_stabilisation_proj(
                shifted,
                axis=0,
                sigma=3,
                min_confidence=0.5,
                edge_filter=False,
                enable_com=True,
                denoise_input_sigma=1,
                debug_output="stabilisation" if display else None,
            )
        else:
            model = image_stabilisation(
                shifted,
                axis=0,
                sigma=3,
                edge_filter=False,
                denoise_input_sigma=1,
                debug_output="stabilisation" if display else None,
            )
        aprint(f"model: {model}")

    with asection("shift back"):

        # apply stabilisation:
        stabilised_seq = model.apply_sequence(shifted, axis=0, pad_width=None)

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer(ndisplay=3)
            viewer.grid.enabled = True
            viewer.add_image(_c(shifted), name="shifted", colormap="bop purple", blending="additive", visible=True)
            viewer.add_image(
                _c(stabilised_seq), name="stabilised_seq", colormap="bop blue", blending="additive", visible=True
            )

    return image, shifted, stabilised_seq, model


if __name__ == "__main__":
    demo_register_sequence_3d_cupy()
    # demo_register_translation_3d_numpy()
