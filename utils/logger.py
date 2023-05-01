from config import ex

import os
import logging


class Logger:
    @ex.capture
    def __init__(self, output_path: str):
        self.logger = logging.Logger(__name__)
        filehandler = logging.FileHandler(os.path.join(output_path, 'log.txt'), encoding='UTF-8')
        filehandler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        filehandler.setFormatter(formatter)
        self.logger.addHandler(filehandler)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logging.warning(*args, **kwargs)
