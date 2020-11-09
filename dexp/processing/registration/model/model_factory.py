import json

from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel


def from_json(json_str: str):
    parsed_model = json.loads(json_str)
    model = TranslationRegistrationModel(shift_vector=parsed_model['translation'],
                                        integral=parsed_model['integral'])
    return model