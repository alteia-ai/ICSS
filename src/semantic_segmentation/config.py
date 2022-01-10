
from omegaconf import OmegaConf
from icecream import ic


def config_factory(config_file: str):
    """
    parse the config file and instanciate the config object
    """
    params = OmegaConf.load(config_file)
    ic(params)

    return params
