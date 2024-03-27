import gzip
import json


def json_dump(filename, obj, gz=False, indent=None):
    if gz:
        open_file = gzip.open
        assert filename.endswith(".gz")
        with open_file(filename, 'wt', encoding="UTF-8") as f:
            json.dump(obj, f, indent=indent)
    else:
        open_file = open
        with open_file(filename, 'w', encoding="UTF-8") as f:
            json.dump(obj, f, indent=indent)


def json_load(filename):
    if filename.endswith(".gz"):
        open_file = gzip.open
        with open_file(filename, 'rt', encoding="UTF-8") as f:
            return json.load(f)
    else:
        open_file = open
        with open_file(filename, 'r', encoding="UTF-8") as f:
            return json.load(f)
