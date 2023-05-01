from utils.util import preprocess
from config import ex
from engine import Engine

import os

@ex.automain
def main(_config):
    if not os.path.exists("./dataset/data.mat"):
        preprocess(_config['data_path'], _config['test_size'])
    os.makedirs(_config['output_path'], exist_ok=True)
    engine = Engine(_config)
    if _config.get("xgb", False):
        engine.fit_xgb()
    if _config.get("gbm", False):
        engine.fit_gbm()
    if _config.get("cat", False):
        engine.fit_cat()
    if _config.get("stack", False):
        engine.fit_stack()