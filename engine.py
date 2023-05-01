from metric import Metric
from utils.logger import Logger

from scipy import io
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metric = Metric()
        self.logger = Logger()
        self.init_model()
        self.init_dataset()

    def init_dataset(self):
        data_path = self.cfg['post_data_path']
        data = io.loadmat(data_path)
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train'].squeeze()
        self.y_test = data['y_test'].squeeze()

    def init_model(self):
        self.xgb = XGBClassifier(learning_rate=0.01,
                                 n_estimators=10,
                                 max_depth=4)
        
        self.gbm = LGBMClassifier()
        self.cat = CatBoostClassifier(verbose=False)
        est_model = [('xgb', self.xgb), ('gbm', self.gbm), ('cat', self.cat)]
        self.stack = StackingClassifier(estimators=est_model, n_jobs=-1)
        
        self.make_grid()
    
    def make_grid(self):
        if self.cfg.get("xgb_search", False):
            self.xgb = GridSearchCV(self.xgb, self.cfg['xgb_param'])
        if self.cfg.get("gbm_search", False):
            self.gbm = GridSearchCV(self.gbm, self.cfg['gbm_param'])
        if self.cfg.get("cat_search", False):
            self.cat = GridSearchCV(self.cat, self.cfg['cat_param'])

    
    def fit_xgb(self):
        self.xgb.fit(self.X_train, self.y_train)
        metric_result = self.predict(self.xgb)
        if self.cfg.get("xgb_search", False):
            self.logger.info(f"xgb best param: {self.xgb.best_params_}")
        self.logger.info(f"xgb metric: {metric_result}")


    def fit_gbm(self):
        self.gbm.fit(self.X_train, self.y_train)
        metric_result = self.predict(self.gbm)
        if self.cfg.get("gbm_search", False):
            self.logger.info(f"gbm best param: {self.gbm.best_params_}")
        self.logger.info(f"gbm metric: {metric_result}")

    def fit_cat(self):
        self.cat.fit(self.X_train, self.y_train)
        metric_result = self.predict(self.cat)
        if self.cfg.get("cat_search", False):
            self.logger.info(f"cat best param: {self.cat.best_params_}")
        self.logger.info(f"cat metric: {metric_result}")

    def fit_stack(self):
        self.stack.fit(self.X_train, self.y_train)
        metric_result = self.predict(self.stack)
        if self.cfg.get("stack_search", False):
            self.logger.info(f"stack best param: {self.stack.best_params_}")
        self.logger.info(f"stack metric: {metric_result}")

    def predict(self, model):
        y_pred = model.predict(self.X_test)
        metric_result = self.metric(self.y_test, y_pred)
        return metric_result


        