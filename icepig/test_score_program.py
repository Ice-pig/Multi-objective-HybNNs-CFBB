from sklearn import metrics
import numpy as np

def score_calculation( y_true, y_prediction):  # 测试程序
    y_true = y_true
    y_predict = y_prediction
    MSE = metrics.mean_squared_error(y_true, y_predict)
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_predict))
    MAE = metrics.mean_absolute_error(y_true, y_predict)
    R2 = metrics.r2_score(y_true, y_predict)
    EVS = metrics.explained_variance_score(y_true, y_predict)
    out = [MSE, RMSE, MAE, R2, EVS]
    return (out)

def mul_score_calculation( y_true, y_prediction):  # 测试程序
    y_true = y_true
    y_predict = y_prediction
    MSE = list( metrics.mean_squared_error(y_true, y_predict,multioutput='raw_values') )
    RMSE = list( np.sqrt(metrics.mean_squared_error(y_true, y_predict,multioutput='raw_values')) )
    MAE = list( metrics.mean_absolute_error(y_true, y_predict,multioutput='raw_values') )
    R2 = list( metrics.r2_score(y_true, y_predict,multioutput='raw_values') )
    EVS = list( metrics.explained_variance_score(y_true, y_predict,multioutput='raw_values'))

    return (MSE, RMSE, MAE, R2, EVS)

