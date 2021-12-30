import os
from os.path import exists, join

import yaml

__config = None


def get_config():
    if __config is not None:
        return __config

    dexp_home = join(os.getenv("HOME"), ".dexp")
    os.makedirs(dexp_home, exist_ok=True)

    config_file_path = join(dexp_home, "config.yaml")
    if exists(config_file_path):
        with open(config_file_path) as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            return config

    else:
        config = {}
        return config


__config = get_config()
