"""Locate versioned measurement resources independent of the process cwd."""

from importlib.resources import files
from pathlib import Path

_ALLOWED = {
    "configs/probes.yaml",
    "configs/ontology.yaml",
    "ontology/shapes.ttl",
    "ontology/ontology.ttl",
}


def resource_path(name):
    if name not in _ALLOWED:
        raise ValueError("Unknown packaged resource " + name)
    source = Path(__file__).resolve().parents[1] / name
    if source.is_file():
        return source
    packaged = Path(str(files("ontoresponse").joinpath("_resources", name)))
    if not packaged.is_file():
        raise FileNotFoundError("Missing packaged measurement resource " + name)
    return packaged
