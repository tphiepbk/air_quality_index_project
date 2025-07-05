# Author: tphiepbk

from src.common.singleton import Singleton

# ==========================================================================================

# Singleton class for debugging purpose
class Debug(metaclass=Singleton):
    def __init__(self):
        self._tag = "hiepdebug"

    def info(self, message, *args):
        if args:
            print("[{}] INFO: {}".format(self._tag, message.format(*args)))
        else:
            print("[{}] INFO: {}".format(self._tag, *args))

    def warning(self, message, *args):
        if args:
            print("[{}] WARNING: {}".format(self._tag, message.format(*args)))
        else:
            print("[{}] WARNING: {}".format(self._tag, *args))

    def error(self, message, *args):
        if args:
            print("[{}] ERROR: {}".format(self._tag, message.format(*args)))
        else:
            print("[{}] ERROR: {}".format(self._tag, *args))
