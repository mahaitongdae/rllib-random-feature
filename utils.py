import numpy as np
from datetime import datetime
import os
from ray.tune.logger import UnifiedLogger
import tempfile

def angle_normalize(th):
    return ((th + np.pi) % (2 * np.pi)) - np.pi

def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)
    def logger_creator(config):
        # if not os.path.exists(custom_path):
        os.makedirs(custom_path, exist_ok=True)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator