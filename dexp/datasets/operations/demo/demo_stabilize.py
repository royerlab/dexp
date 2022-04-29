import random
import tempfile
from os.path import join

import numpy as np
from arbol import aprint, asection

from dexp.datasets import ZDataset
from dexp.datasets.operations.stabilize import dataset_stabilize
from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.sequence_proj import image_stabilisation_proj
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_stabilize_numpy():
    with NumpyBackend():
        _demo_stabilize()


def demo_stabilize_cupy():
    try:
        with CupyBackend():
            _demo_stabilize()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _demo_stabilize(
    length_xy=256,
    n=64,
    drift_strength=0.8,
    warp_grid_size=8,
    warp_strength=2.5,
    ratio_bad_frames=0.0,
    additive_noise=0.05,
    multiplicative_noise=0.1,
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
        _, _, image = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length_xy // 4,
            length_z_factor=1,
            independent_haze=True,
            sphere=True,
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

        # simulate dropped, highly corrupted frames:
        for _ in range(int(shifted.shape[0] * ratio_bad_frames)):
            index = random.randint(0, n - 1)
            shifted[index] = sp.ndimage.shift(
                shifted[index], shift=(random.uniform(-27, 27), random.uniform(-27, 27), random.uniform(-27, 27))
            )
            shifted[index] += xp.random.rand(*shifted[index].shape)
            shifted[index] *= random.uniform(0.001, 0.5)
            shifted[index] += random.uniform(0.001, 0.1) * xp.random.rand(*shifted[index].shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint("created temporary directory", tmpdir)

        input_path = join(tmpdir, "input.zarr")
        output_path = join(tmpdir, "output.zarr")
        input_dataset = ZDataset(path=input_path, mode="w", store="dir")
        output_dataset = ZDataset(path=output_path, mode="w")

        zarr_input_array = input_dataset.add_channel(
            name="channel", shape=shifted.shape, chunks=(1, 50, 50, 50), dtype=shifted.dtype, codec="zstd", clevel=3
        )
        input_dataset.write_array(channel="channel", array=Backend.to_numpy(shifted))

        dataset_stabilize(
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            model_output_path=join(tmpdir, "model.json"),
            channels=["channel"],
        )

        zarr_output_array = np.asarray(output_dataset.get_array("channel"))

        # stabilise with lower level API:
        model = image_stabilisation_proj(shifted, axis=0, projection_type="max")
        aprint(f"model: {model}")

        # apply stabilisation:
        stabilised_seq_ll = Backend.to_numpy(model.apply_sequence(shifted, axis=0))

        aprint(f"stabilised_seq_ll.shape={stabilised_seq_ll.shape}")
        aprint(f"zarr_output_array.shape={zarr_output_array.shape}")
        np.testing.assert_allclose(stabilised_seq_ll.shape[1], zarr_output_array.shape[1], atol=10)
        np.testing.assert_allclose(stabilised_seq_ll.shape[2], zarr_output_array.shape[2], atol=10)
        np.testing.assert_allclose(stabilised_seq_ll.shape[3], zarr_output_array.shape[3], atol=10)
        error = np.mean(np.abs(stabilised_seq_ll[:, 0:64, 0:64, 0:64] - zarr_output_array[:, 0:64, 0:64, 0:64]))
        error_null = np.mean(
            np.abs(stabilised_seq_ll[:, 0:64, 0:64, 0:64] - Backend.to_numpy(shifted)[:, 0:64, 0:64, 0:64])
        )
        aprint(f"error={error} versus error_null={error_null}")
        assert error < error_null * 0.75

        if display:
            import napari

            def _c(array):
                return Backend.to_numpy(array)

            viewer = napari.Viewer(ndisplay=2)
            viewer.add_image(_c(image), name="image", colormap="bop orange", blending="additive", visible=True)
            viewer.add_image(_c(shifted), name="shifted", colormap="bop purple", blending="additive", visible=True)
            viewer.add_image(
                _c(zarr_input_array),
                name="zarr_input_array",
                colormap="bop purple",
                blending="additive",
                visible=True,
            )
            viewer.add_image(
                _c(zarr_output_array),
                name="zarr_output_array",
                colormap="bop blue",
                blending="additive",
                visible=True,
            )
            viewer.add_image(
                _c(stabilised_seq_ll),
                name="stabilised_seq_ll",
                colormap="bop blue",
                blending="additive",
                visible=True,
            )
            viewer.grid.enabled = True

            napari.run()


if __name__ == "__main__":
    demo_stabilize_cupy()
    # demo_stabilize_numpy()
