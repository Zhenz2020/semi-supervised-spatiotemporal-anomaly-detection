import numpy as np
import pickle
import pandas as pd
from outlier_dbscan import abnormal_det_out_db
import utils.pre_stdbscan as pst
from sklearn.ensemble import IsolationForest
#不包含重采样的数据生成

# fileHandle = open('data/Dist_backgroud.txt', 'rb')
fileHandle = open('data/Dist_backgroud_0324.txt', 'rb')
dist_background = pickle.load(fileHandle)
fileHandle.close()

# fileHandle = open('data/Traj_backgroud.txt', 'rb')
fileHandle = open('data/Traj_backgroud_0324.txt', 'rb')
traj_background = pickle.load(fileHandle)
fileHandle.close()

# fileHandle = open('data/true_state_label.txt', 'rb')
fileHandle = open('data/true_state_label_0324.txt', 'rb')
true_state_label = pickle.load(fileHandle)
fileHandle.close()

num_lane = len(dist_background)
num_grid_per_lane = len(dist_background[0])

det_module = abnormal_det_out_db()
final_results_dataframe = pd.DataFrame(columns=(["veh_id", "frame", "world_x", "world_y", "dist", "lane", "grid",
                                                  "threshold",
                                                  "out", "JNB", "stdbscan", "isolation"
                                                  "label"]))

# def stack_method(data):


for i in range(num_lane):
    second_results = []
    for j in range(num_grid_per_lane):

        temp_dist_data = np.array(dist_background[i][j])
        temp_traj_data = traj_background[i][j]
        temp_state_label = np.array(true_state_label[i][j])
        if temp_dist_data.tolist() == []:
            results = []
        else:
            temp_frame = pst.conv_id_to_time(temp_traj_data)
            temp_out_next = temp_frame[["id", "frame", "lati", "longti", "dist"]]
            results_threshold = det_module.threshold(temp_traj_data, i)
            results_out = det_module.outliers(temp_dist_data)
            try:
                results_JNB = det_module.JNB(temp_dist_data, cls_num=5)
            except:
                results_JNB = det_module.JNB(temp_dist_data, cls_num=1)
            results_stdbscan = det_module.stdbscan(temp_traj_data, eps1=0.4, eps2=2, min_samples=10)

            results_iso = det_module.isolate(temp_traj_data)


            # results = np.stack((i * np.ones(len(results_JNB)), j * np.ones(len(results_JNB)), results_threshold,
            #                     results_out, results_JNB, results_stdbscan, results_iso, temp_state_label[:, 4]), axis=1)
            results = np.stack((np.array(temp_out_next['id']), np.array(temp_out_next['frame']),
                                np.array(temp_out_next['lati']), np.array(temp_out_next['longti']),
                                np.array(temp_out_next['dist']), i * np.ones(len(results_JNB)),
                                j * np.ones(len(results_JNB)), results_threshold,
                                results_out, results_JNB, results_stdbscan, results_iso, temp_state_label[:, 4]), axis=1)
            results_dataframe = pd.DataFrame(results,
                                             columns=(
                                                 ["veh_id", "frame", "world_x", "world_y", "dist", "lane", "grid",
                                                  "threshold",
                                                  "out", "JNB", "stdbscan", "isolation",
                                                  "label"]))

            final_results_dataframe = pd.concat([final_results_dataframe, results_dataframe], ignore_index=True)
final_results_dataframe.to_csv("data/final_results_dataframe_0324.csv")
