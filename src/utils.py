from ruamel.yaml import YAML

def load_params() -> dict:
    yaml = YAML(typ="safe")
    with open("params.yaml") as f:
        params = yaml.load(f)
    return params