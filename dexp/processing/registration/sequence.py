from math import log
from typing import List, Optional

from arbol import aprint, asection
from dask.array import Array

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.translation_nd_proj import register_translation_maxproj_nd
from dexp.processing.utils.center_of_mass import center_of_mass
from dexp.processing.utils.linear_solver import linsolve


def sequence_stabilisation(image: 'Array',
                           axis: int = 0,
                           mode: str = 'translation',
                           max_scale: Optional[int] = None,
                           min_confidence: float = 0.3,
                           use_center_of_mass_shifts: bool = False,
                           solver_order: float = 1.05,
                           internal_dtype=None,
                           **kwargs
                           ) -> SequenceRegistrationModel:
    """
    Computes a sequence stabilisation model for an image sequence indexed along a specified axis.


    Parameters
    ----------
    image: image to stabilise
    axis: sequence axis along which to stabilise image
    mode: registartion mode. For now only 'translation' is available.
    max_scale: Integer signifying how many scales to sue for pairwise registration. Each scale corresponds roughly to a power of the golden ratio g**scale,
    but to avoid hitting the same images too often, we snap these integers to the closest prime number (see magic numbers table in code).
    min_confidence: minimal confidence to accept a pairwise registration
    use_center_of_mass_shifts: if True then additional pairwise terms are added onthe basis of center of mass shifts.
    solver_order: order for linear solver. We use the same order for both error and regularisation terms.
    internal_dtype : internal dtype for computation
    **kwargs: argument passthrough to the pairwise registration method.

    Returns
    -------
    sequence registration model

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # These numbers are approximately spaced out as powers of the golden ratio, but are all prime numbers:
    # There is a beautiful reason for this, ask Loic.
    magic_numbers = [1, 2, 3, 7, 11, 19, 29, 47, 79, 127, 199, 317, 521, 839, 1367]

    length = image.shape[axis]
    if max_scale is None:
        max_scale = len(magic_numbers) - 1
    max_scale = min(max_scale, int(log(length, 1.618033988749895)))
    image_sequence = list(xp.take(image, axis=axis, indices=range(0, length)))

    with asection(f"Registering image sequence of length: {length}"):

        ndim = image_sequence[0].ndim

        pairwise_models = []
        with asection(f"Computing pairwise registrations..."):
            uv_set = set()
            for scale in range(0, max_scale):
                for offset in range(0, magic_numbers[scale], max(1, magic_numbers[max(0, scale - 1)])):
                    for u in range(offset, length, magic_numbers[scale]):
                        v = u + magic_numbers[scale]
                        if v >= length:
                            continue
                        if (u, v) not in uv_set:
                            uv_set.add((u, v))
                            image_u = image_sequence[u]
                            image_v = image_sequence[v]

                            if mode == 'translation':
                                model = register_translation_maxproj_nd(image_u, image_v,
                                                                        _display_phase_correlation=False,
                                                                        **kwargs)
                                confidence = model.overall_confidence()

                                if confidence >= min_confidence:
                                    model.u = u
                                    model.v = v
                                    pairwise_models.append(model)
                                    # aprint(f"Success: registered images for indices: ({u}, {v}) with confidence {confidence} (>={min_confidence}) ")
                                else:
                                    aprint(f"Warning: low confidence ({confidence}) for pair: ({u}, {v})")
                            else:
                                raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")
                        else:
                            aprint(f"Pair: ({u}, {v}) already considered, that's all good just info...")

        if use_center_of_mass_shifts:
            with asection(f"Computing pairwise center-of-mass shifts..."):
                step = length // 2
                for offset in range(0, step):
                    for u in range(0, length, step):
                        v = u + step
                        if v >= length:
                            continue

                        image_u = image_sequence[u]
                        image_v = image_sequence[v]

                        if mode == 'translation':
                            com_u = center_of_mass(image_u, projection_type='max', offset_mode='middle')
                            com_v = center_of_mass(image_v, projection_type='max', offset_mode='middle')
                            model = TranslationRegistrationModel(shift_vector=com_u - com_v, confidence=1)
                            model.u = u
                            model.v = v
                            pairwise_models.append(model)
                        else:
                            raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")

        nb_models = len(pairwise_models)
        aprint(f"Number of models obtained: {nb_models} for a sequence of length:{length}")

        with asection(f"Solving for optimal sequence registration"):

            if mode == 'translation':

                sequential_models: List[TranslationRegistrationModel] = list(TranslationRegistrationModel(xp.zeros((ndim,), dtype=internal_dtype)) for _ in range(length))

                for d in range(ndim):
                    a = xp.zeros((nb_models + 1, length), dtype=xp.float32)
                    y = xp.zeros((nb_models + 1,), dtype=xp.float32)

                    for i, model in enumerate(pairwise_models):
                        u = model.u
                        v = model.v

                        y[i] = model.shift_vector[d]
                        a[i, u] = +1
                        a[i, v] = -1

                    a[-1, 0] = 1
                    y[-1] = 0

                    x_opt = linsolve(a, y, order_error=solver_order, order_reg=solver_order, alpha_reg=0.1)

                    for i in range(length):
                        sequential_models[i].shift_vector[d] = -x_opt[i]

                return SequenceRegistrationModel(model_list=sequential_models)


            else:
                raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")

            # if not image_a.dtype == image_b.dtype:
            #     raise ValueError("Arrays must have the same dtype")
            #
            # if internal_dtype is None:
            #     internal_dtype = image_a.dtype
            #
            # if type(Backend.current()) is NumpyBackend:
            #     internal_dtype = xp.float32
            #
            # image_a = Backend.to_backend(image_a, dtype=internal_dtype)
            # image_b = Backend.to_backend(image_b, dtype=internal_dtype)
