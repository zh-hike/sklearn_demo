from sacred import Experiment

ex = Experiment("cal_classificaion")

@ex.config
def cfg():
    data_path = "../datasets/other/train.csv"
    post_data_path = "./dataset/data.mat"
    test_size = 0.2
    output_path = "./output/"
    gbm = False
    xgb = False
    cat = False
    stack = False
    search_param = False
    xgb_search = False
    gbm_search = False
    cat_search = False
    stack_search = False
    xgb_param = {'learning_rate': (0.5,),
                 'n_estimators': (80, 90, 100)}
    gbm_param = {'num_leaves': (25, 31, 35),
                 'learning_rate': (0.001, 0.01, 0.1, 0.5),
                 'n_estimators': (80, 90, 100, 110),
                 'min_child_samples': (15, 20, 25)}
    cat_param = {'learning_rate': (0.01, 0.05, 0.1, 0.5),
                  'depth': (5, 6, 7, 8),
                  'l2_leaf_reg': (2, 3, 4),
                  }
    
    