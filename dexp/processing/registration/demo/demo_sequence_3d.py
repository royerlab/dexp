import random

from arbol import aprint, asection

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.sequence import sequence_stabilisation
from dexp.processing.registration.sequence_proj import sequence_stabilisation_proj
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data


def demo_register_sequence_3d_numpy():
    with NumpyBackend():
        _register_sequence_3d(use_projections=True)
        _register_sequence_3d(use_projections=False)


def demo_register_sequence_3d_cupy():
    try:
        with CupyBackend():
            _register_sequence_3d(use_projections=True)
            _register_sequence_3d(use_projections=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_sequence_3d(length_xy=256,
                          n=64,
                          drift_strength=0.8,
                          warp_grid_size=8,
                          warp_strength=2.5,
                          ratio_bad_frames=0.05,
                          additive_noise=0.05,
                          multiplicative_noise=0.1,
                          use_projections=False,
                          run_test=False,
                          display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # used for simulation:
    shifts = []
    vector_fields = []
    x, y, z = 0, 0, 0
    vector_field = xp.zeros((warp_grid_size,) * 3 + (3,), dtype=xp.float32)

    with asection("generate dataset"):
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
                x += 7
                y += -5
                z += 8
            if i == n // 2 + n // 3:
                x += -7
                y += +9
                z += -11

            # keep for later:
            shifts.append((x, y, z))
            vector_fields.append(xp.copy(vector_field))

        # nuclei:
        _, _, image = generate_nuclei_background_data(add_noise=False,
                                                      length_xy=length_xy // 4,
                                                      length_z_factor=1,
                                                      independent_haze=True,
                                                      sphere=True,
                                                      zoom=2,
                                                      dtype=xp.float32)
        image = Backend.to_backend(image)

        # generate reference 'ground truth' timelapse
        image = xp.stack(image for _ in range(n))

        # generate shifted, distorted, degraded timelapse:
        shifted = xp.stack((sp.ndimage.shift(warp(i, vf, vector_field_upsampling=4), shift=s) for i, s, vf in zip(image, shifts, vector_fields)))
        shifted *= xp.clip(xp.random.normal(loc=1, scale=multiplicative_noise, size=shifted.shape, dtype=xp.float32), 0.1, 10)
        shifted += additive_noise * xp.random.rand(*shifted.shape, dtype=xp.float32)

        # simulate dropped, highly corrupted frames:
        for _ in range(int(shifted.shape[0] * ratio_bad_frames)):
            index = random.randint(0, n - 1)
            shifted[index] = sp.ndimage.shift(shifted[index], shift=(random.uniform(-27, 27), random.uniform(-27, 27), random.uniform(-27, 27)))
            shifted[index] += xp.random.rand(*shifted[index].shape)
            shifted[index] *= random.uniform(0.001, 0.5)
            shifted[index] += random.uniform(0.001, 0.1) * xp.random.rand(*shifted[index].shape)

    with asection("register_translation_2d"):
        # compute image sequence stabilisation model:
        if use_projections:
            model = sequence_stabilisation_proj(shifted, axis=0)
        else:
            model = sequence_stabilisation(shifted, axis=0)
        aprint(f"model: {model}")

    with asection("shift back"):

        # how much to pad:
        padding = 64
        pad_width = None  # ((padding, padding), (padding, padding))

        # apply stabilisation:
        stabilised = model.apply_sequence(shifted, axis=0, pad_width=pad_width)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', visible=True)
            viewer.add_image(_c(shifted), name='shifted', colormap='bop purple', blending='additive', visible=False)
            viewer.add_image(_c(stabilised), name='stabilised', colormap='bop blue', blending='additive', visible=True)

    return image, shifted, stabilised, model


if __name__ == "__main__":
    demo_register_sequence_3d_cupy()
    # demo_register_translation_3d_numpy()
