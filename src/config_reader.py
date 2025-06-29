# Author: tphiepbk

import json

class ConfigurationReader(object):
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

    def __str__(self):
        return json.dumps(self._data, indent=4)
    
    @property
    def general(self):
        return self._data["general"]

    @property
    def prediction(self):
        return self._data["prediction"]

    @property
    def reduction(self):
        return self._data["reduction"]
