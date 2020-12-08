import json

import numpy

from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel


def from_json(json_str: str):
    parsed_model = json.loads(json_str)
    if parsed_model['type'] == 'translation':
        model = TranslationRegistrationModel(shift_vector=numpy.asarray(parsed_model['translation']),
                                             confidence=numpy.asarray(parsed_model['confidence']),
                                             integral=parsed_model['integral'])
    elif parsed_model['type'] == 'warp':
        model = WarpRegistrationModel(vector_field=numpy.asarray(parsed_model['vector_field']),
                                      confidence=numpy.asarray(parsed_model['confidence']))

    return model
