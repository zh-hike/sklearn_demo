from utils.logger import Logger

import logging
import pandas as pd
import scipy.io as io
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def preprocess(data_path: str, test_size: 0.2) -> None:
    """
    数据预处理
    """
    logger = Logger()
    logger.info("数据预处理...")
    d = pd.read_csv(data_path)
    d = d.fillna(method="ffill")
    label = d.loc[:, 'isDefault'].astype('int32').values
    d = d.drop(columns=['isDefault'])
    drop_c = ['id', 'title', 'issueDate', 'postCode', 'regionCode', 'earliesCreditLine', 'title']
    convert_onehot = ['term', 'grade', 'subGrade', 'employmentLength', 'homeOwnership', 'verificationStatus', 'purpose', 'delinquency_2years', 'pubRec', 'pubRecBankruptcies', 'initialListStatus', 'applicationType', 'policyCode']
    df = d.drop(columns=drop_c)
    df2 = pd.get_dummies(df.loc[:, convert_onehot].astype('str'))
    df = df.drop(columns=convert_onehot)
    df = pd.concat([df, df2], axis=1)
    X = df.values.astype('float32')
    smo = SMOTE()
    logger.info("SMOTE采样")
    X, label = smo.fit_resample(X, label)
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=test_size)

    data = {'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test}
    logger.info(f"正在保存...")
    io.savemat("./dataset/data.mat", data)
    logger.info(f"保存数据到 ./dataste/data.mat")

    
