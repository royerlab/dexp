import json
from typing import Sequence, Union

from arbol import asection

from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.processing.registration.model.sequence_registration_model import (
    SequenceRegistrationModel,
)
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration.model.warp_registration_model import (
    WarpRegistrationModel,
)
from dexp.utils.backends import Backend


def from_json(json_str: str) -> Union[PairwiseRegistrationModel, SequenceRegistrationModel]:
    xp = Backend.get_xp_module()
    parsed_model = json.loads(json_str)
    if parsed_model["type"] == "translation":
        model = TranslationRegistrationModel(
            shift_vector=xp.asarray(parsed_model["translation"]), confidence=xp.asarray(parsed_model["confidence"])
        )
    elif parsed_model["type"] == "warp":
        model = WarpRegistrationModel(
            vector_field=xp.asarray(parsed_model["vector_field"]), confidence=xp.asarray(parsed_model["confidence"])
        )

    elif parsed_model["type"] == "translation_sequence":
        model_list_json = parsed_model["model_list"]
        model_list = list(from_json(model_json) for model_json in model_list_json)
        model = SequenceRegistrationModel(model_list=model_list)

    else:
        raise NotImplementedError

    return model


def model_list_from_file(file_path: str) -> Sequence[Union[PairwiseRegistrationModel, SequenceRegistrationModel]]:
    with open(file_path) as models_file:
        lines = models_file.readlines()

        model_list = []
        with asection(f"Loading {len(lines)} models from file: {file_path}"):
            for line in lines:
                line = line.strip()
                model = None if line == "None" else from_json(line)
                model_list.append(model)

        return model_list


def model_list_to_file(file_path: str, model_list: Sequence[PairwiseRegistrationModel]):
    with open(file_path, "w") as models_file:
        lines = []

        with asection(f"Writing {len(model_list)} models to file: {file_path}"):
            for model in model_list:
                line = "None" if model is None else model.to_json()
                lines.append(line + "\n")

            models_file.writelines(lines)
