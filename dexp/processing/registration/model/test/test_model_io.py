from arbol import aprint

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.model.model_io import from_json, model_list_to_file, model_list_from_file
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel


def test_model_io_numpy():
    with NumpyBackend():
        _test_model_io_to_json()
        _test_model_io_to_file()


def test_model_io_cupy():
    try:
        with CupyBackend():
            _test_model_io_to_json()
            _test_model_io_to_file()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _test_model_io_to_json():
    xp = Backend.get_xp_module()

    # Testing read/write translation model to json:
    model = TranslationRegistrationModel(shift_vector=[-1, -5, 13], confidence=0.6)
    json_str = model.to_json()
    new_model = from_json(json_str)
    assert (new_model.shift_vector == model.shift_vector).all()
    assert (new_model.confidence == model.confidence).all()

    # Testing read/write warp model to json:
    magnitude = 15
    warp_grid_size = 4
    vector_field = xp.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    confidence = xp.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3)
    model = WarpRegistrationModel(vector_field=vector_field, confidence=confidence)
    json_str = model.to_json()
    new_model = from_json(json_str)
    assert (new_model.vector_field == model.vector_field).all()
    assert (new_model.confidence == model.confidence).all()

    # Testing read/write translation sequence model to json:
    model_list = list(TranslationRegistrationModel(shift_vector=[-1 * (i / 100.0), -5 + 0.1 * i, 13 - 0.2 * i], confidence=0.6) for i in range(0, 10))
    model_list += list(WarpRegistrationModel(vector_field=xp.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,)),
                                             confidence=xp.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3)) for _ in range(0, 10))
    model = SequenceRegistrationModel(model_list)
    json_str = model.to_json()
    new_model = from_json(json_str)
    assert model == new_model


def _test_model_io_to_file():
    xp = Backend.get_xp_module()

    magnitude = 15
    warp_grid_size = 4
    vector_field = xp.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    confidence = xp.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3)

    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        aprint(f"Temp file: {tmp.name}")

        model_list_1 = list([TranslationRegistrationModel(shift_vector=[-1, i, 2 * i], confidence=0.6 - 0.01 * i) for i in range(3)])
        model_list_2 = list([WarpRegistrationModel(vector_field=vector_field * (1 + 0.1 * i),
                                                   confidence=confidence / (i + 1e-8)) for i in range(3)])
        model_list = model_list_1 + model_list_2
        model_list_to_file(tmp.name, model_list)

        loaded_model_list = model_list_from_file(tmp.name)

        assert model_list == loaded_model_list
