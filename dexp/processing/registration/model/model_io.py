import json
from typing import Sequence

from arbol import asection

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel


def from_json(json_str: str):
    xp = Backend.get_xp_module()
    parsed_model = json.loads(json_str)
    if parsed_model['type'] == 'translation':
        model = TranslationRegistrationModel(shift_vector=xp.asarray(parsed_model['translation']),
                                             confidence=xp.asarray(parsed_model['confidence']),
                                             integral=parsed_model['integral'])
    elif parsed_model['type'] == 'warp':
        model = WarpRegistrationModel(vector_field=xp.asarray(parsed_model['vector_field']),
                                      confidence=xp.asarray(parsed_model['confidence']))

    return model


def model_list_from_file(file_path: str):
    with open(file_path, "r") as models_file:
        lines = models_file.readlines()

        model_list = []
        with asection(f"Loading {len(lines)} models from file: {file_path}"):
            for line in lines:
                line = line.strip()
                model = None if line == 'None' else from_json(line)
                model_list.append(model)

        return model_list


def model_list_to_file(file_path: str, model_list: Sequence[PairwiseRegistrationModel]):
    with open(file_path, "w") as models_file:
        lines = []

        with asection(f"Writing {len(model_list)} models to file: {file_path}"):
            for model in model_list:
                line = 'None' if model is None else model.to_json()
                lines.append(line + '\n')

            models_file.writelines(lines)
