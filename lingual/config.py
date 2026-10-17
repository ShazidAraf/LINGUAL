"""Plain-text `key = value` config reader (same format as config_files/*.txt)."""


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return "\n".join(f"{k} = {v}" for k, v in self.__dict__.items())


def _parse(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.startswith("(") and value.endswith(")"):
        return tuple(int(v) for v in value[1:-1].split(","))
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def read_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()  # allow inline comments
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = _parse(value.strip())
    return Config(**cfg)
