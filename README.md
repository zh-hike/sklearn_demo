# sklearn包的简单使用，三个模型分类，
> 作者： zh-hike \
> 联系方式： 1583124882@qq.com \
> 作者单位： 北京邮电大学在读研究生

## 任务
1. smote过采样解决数据不平衡 
2. 调参(任何方法都行) 
3. 三个模型的预测(ACU)  (xgboost  catboost lightgbm)
4. 三个模型stacking融合

## 数据
数据集文件 `train.csv`，有需要请联系本人。

## 环境准备
本次程序运行在
> `python=3.10.9`

安装python环境后，运行如下命令安装第三方包
```
pip install -r requirements.txt
```

## 运行
> 运行需要的参数请在 `config.py` 中进行修改

1. 程序运行需要按照要求选择运行时的参数，程序可以选择4种不同的模型，1. xgboost, 2. catboost, 3. lightngbm, 4. stacking融合。 假如需要运行`xgboost` 和 `stacking`则可以运行
```
python main.py with xgb=True stack=True
```

2. 程序支持自动调参，请在`config.py` 中设置不同的模型的参数选择范围，假如需要对`xgboost` 和 `catboost` 进行自动调参，请运行
```
python main.py with xgb=True cat=True xgb_search=True cat_search=True
```