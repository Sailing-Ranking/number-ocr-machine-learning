from ruamel.yaml import YAML


def load_yaml(path: str) -> dict:
    yaml = YAML(typ="safe")
    with open(path) as f:
        params = yaml.load(f)
    return params
