# Author: tphiepbk

import json

from src.common.singleton import Singleton

# ==========================================================================================

# Singleton class for handling configuration
class ConfigurationReader(metaclass=Singleton):
    def __init__(self, configPath=None):
        assert configPath, "configPath must not be None"
        self._configPath = configPath
        self._readConfig()

    def _readConfig(self):
        with open(self._configPath, "r", encoding="utf-8") as f:
            try:
                self._data = json.load(f)
            except Exception as e:
                print(f"Failed to read configuration in {self._configPath}, error: {e}") 
                self._data = None

    def __str__(self):
        return json.dumps(self._data, indent=4)

    @property
    def data(self):
        assert self._data, "self._data must not be None"
        return self._data
