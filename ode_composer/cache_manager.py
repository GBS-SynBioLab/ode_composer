#  Copyright (c) 2021. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk

from pathlib import Path
import json


class CacheManager(object):
    def __init__(self, cache_id, cache_folder=None):
        # keep this order otherwise the initialization fails
        self.path = None
        if cache_folder:
            self.cache_folder = cache_folder
        else:
            self.cache_folder = "../cache"
        self.cache_id = cache_id

    @property
    def cache_id(self):
        return self._cache_id

    @cache_id.setter
    def cache_id(self, new_cache_id):
        self._cache_id = new_cache_id
        self._set_path()

    def _set_path(self):
        self.path = Path(f"{self.cache_folder}/{self.cache_id}.json")

    def _check_cache(self):
        if self.path.exists():
            return True
        return False

    def cache_hit(self):
        # TODO ZAT other checks, e.g. hash of the data
        return self._check_cache()

    def write(self, dict_to_cache):
        with open(self.path, "w+") as file:
            file.write(json.dumps(dict_to_cache))

    def read(self):
        if self._check_cache():
            with open(self.path, "r") as file:
                return json.load(file)
