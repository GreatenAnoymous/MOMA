
from .damc import DAMC

all_versions = {
    "v1": DAMC,
}

get_version = lambda v : all_versions[v]