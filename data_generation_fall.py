import numpy as np
import pickle
import pandas as pd
from outlier_dbscan import abnormal_det_out_db
import utils.pre_stdbscan as pst
from sklearn.ensemble import IsolationForest
#不包含重采样的数据生成

df_20_test = pd.read_csv('data/train/data_0.csv') #/kaggle/input/anomaly-detection-falling-people-events
df_21_test = pd.read_csv('data/train/data_1.csv')
df_22_test = pd.read_csv('data/train/data_2.csv')
df_23_test = pd.read_csv('data/train/data_3.csv')
df_24_test = pd.read_csv('data/train/data_4.csv')
# total_test = pd.concat([df_20_test,df_21_test,df_22_test,df_23_test, df_24_test])
data_traj=[df_20_test,df_21_test,df_22_test,df_23_test, df_24_test]
det_module = abnormal_det_out_db()
# def stack_method(data):
num=0
final_results_dataframe = pd.DataFrame(columns=(['x','y','z','1','2','3','4','label','stdbscan', 'isolation','hbos', 'knn','num']))
for data in data_traj:
    data.columns=['x','y','z','1','2','3','4','label']
    results_stdbscan = det_module.stdbscan_fall(data, eps1=0.1, eps2=1, min_samples=10)

    results_iso = det_module.isolate_fall(data)
    results_hbos = det_module.get_TOS_hbos(data, k_list=None)
    results_knn = det_module.get_TOS_knn(data, k=5)

    # results = np.stack((i * np.ones(len(results_JNB)), j * np.ones(len(results_JNB)), results_threshold,
    #                     results_out, results_JNB, results_stdbscan, results_iso, temp_state_label[:, 4]), axis=1)
    results = np.stack((np.array(data['x']), np.array(data['y']),np.array(data['z']),np.array(data['1']),np.array(data['2']),np.array(data['3']),np.array(data['4']),np.array(data['label']),results_stdbscan, results_iso,results_hbos, results_knn, num * np.ones(data.shape[0])), axis=1)
    results_dataframe = pd.DataFrame(results,
                                     columns=(
                                         ['x','y','z','1','2','3','4','label','stdbscan', 'isolation','hbos', 'knn','num']))

    final_results_dataframe = pd.concat([final_results_dataframe, results_dataframe], ignore_index=True)
    num+=1
final_results_dataframe.to_csv("data/final_results_dataframe_fall.csv")
