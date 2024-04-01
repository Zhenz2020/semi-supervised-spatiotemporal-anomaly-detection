import numpy as np
import pickle
import pandas as pd
from outlier_dbscan import abnormal_det_out_db
import utils.pre_stdbscan as pst
import random
from sklearn.ensemble import IsolationForest

# 包含重采样的数据生成

fileHandle = open('data/Dist_backgroud.txt', 'rb')
dist_background = pickle.load(fileHandle)
fileHandle.close()

fileHandle = open('data/Traj_backgroud.txt', 'rb')
traj_background = pickle.load(fileHandle)
fileHandle.close()

fileHandle = open('data/true_state_label.txt', 'rb')
true_state_label = pickle.load(fileHandle)
fileHandle.close()

num_lane = len(dist_background)
num_grid_per_lane = len(dist_background[0])

det_module = abnormal_det_out_db()
final_results_dataframe = pd.DataFrame(columns=(["veh_id", "frame", "world_x", "world_y", "dist", "lane", "grid",
                                                 "threshold",
                                                 "out", "JNB", "stdbscan", "isolation", "hbos","KNN",
                                                                           "label"]))
final_resampled_dataframe = pd.DataFrame(columns=(["veh_id", "frame", "world_x", "world_y", "dist", "lane", "grid",
                                                   "threshold",
                                                   "out", "JNB", "stdbscan", "isolation", "hbos","KNN",
                                                                             "label"]))

# def stack_method(data):
K = 20


def internal_list_resample(data, proportion):
    """
    Args:
      * data -> (examples,attributes) format
      * proportion -> percentage to sample
    """
    resampled_indices = np.random.choice(
        range(len(data)), size=int(len(data) * proportion), replace=True)
    return resampled_indices


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

            results_iso = det_module.isolate(temp_traj_data,max_feature=1.0)
            results_hbos = det_module.get_TOS_hbos(temp_traj_data, k_list=None)
            try:
                results_knn = det_module.get_TOS_knn(temp_traj_data, k=5)
            except:
                results_knn = np.ones(len(temp_traj_data))
            # results = np.stack((i * np.ones(len(results_JNB)), j * np.ones(len(results_JNB)), results_threshold,
            #                     results_out, results_JNB, results_stdbscan, results_iso, temp_state_label[:, 4]), axis=1)
            results = np.stack((np.array(temp_out_next['id']), np.array(temp_out_next['frame']),
                                np.array(temp_out_next['lati']), np.array(temp_out_next['longti']),
                                np.array(temp_out_next['dist']), i * np.ones(len(results_JNB)),
                                j * np.ones(len(results_JNB)), results_threshold,
                                results_out, results_JNB, results_stdbscan, results_iso, results_hbos, results_knn, temp_state_label[:, 4]),
                               axis=1)
            results_dataframe = pd.DataFrame(results,
                                             columns=(
                                                 ["veh_id", "frame", "world_x", "world_y", "dist", "lane", "grid",
                                                  "threshold",
                                                  "out", "JNB", "stdbscan", "isolation","hbos","KNN",
                                                  "label"]))

            final_results_dataframe = pd.concat([final_results_dataframe, results_dataframe], ignore_index=True)
        for k in range(K):
            resampled_indices = internal_list_resample(temp_traj_data, proportion=0.9)
            random_dist_data = [temp_dist_data[i] for i in resampled_indices]
            random_traj_data = [temp_traj_data[i] for i in resampled_indices]
            random_state_data = np.array([temp_state_label[i] for i in resampled_indices])

            if random_dist_data == []:
                results = []
            else:
                temp_frame = pst.conv_id_to_time(random_traj_data)
                temp_out_next = temp_frame[["id", "frame", "lati", "longti", "dist"]]
                results_threshold = det_module.threshold(random_traj_data, i)
                results_out = det_module.outliers(random_dist_data)
                try:
                    results_JNB = det_module.JNB(random_dist_data, cls_num=5)
                except:
                    results_JNB = det_module.JNB(random_dist_data, cls_num=1)
                results_stdbscan = det_module.stdbscan(random_traj_data, eps1=0.4, eps2=2, min_samples=10)

                results_iso = det_module.isolate(random_traj_data)
                results_hbos = det_module.get_TOS_hbos(random_traj_data, k_list=None)
                try:
                    results_knn = det_module.get_TOS_knn(random_traj_data, k=5)
                except:
                    results_knn = np.ones(len(random_traj_data))

                results = np.stack((np.array(temp_out_next['id']), np.array(temp_out_next['frame']),
                                    np.array(temp_out_next['lati']), np.array(temp_out_next['longti']),
                                    np.array(temp_out_next['dist']), i * np.ones(len(results_JNB)),
                                    j * np.ones(len(results_JNB)), results_threshold,
                                    results_out, results_JNB, results_stdbscan, results_iso,results_hbos, results_knn, random_state_data[:, 4]),
                                   axis=1)
                results_dataframe = pd.DataFrame(results,
                                                 columns=(
                                                     ["veh_id", "frame", "world_x", "world_y", "dist", "lane", "grid",
                                                      "threshold",
                                                      "out", "JNB", "stdbscan", "isolation","hbos","KNN",
                                                      "label"]))


                final_resampled_dataframe = pd.concat([final_resampled_dataframe, results_dataframe], ignore_index=True)

labeled_100_index = np.random.choice(range(final_results_dataframe.shape[0]), 100, replace=False)
labeled_200_index = np.random.choice(range(final_results_dataframe.shape[0]), 200, replace=False)
labeled_500_index = np.random.choice(range(final_results_dataframe.shape[0]), 500, replace=False)

final_results_dataframe.iloc[labeled_100_index,:].to_csv("data/final_result_dataframe_100.csv")
final_results_dataframe.iloc[labeled_200_index,:].to_csv("data/final_result_dataframe_200.csv")
final_results_dataframe.iloc[labeled_500_index,:].to_csv("data/final_result_dataframe_500.csv")

merge_100=pd.merge(final_results_dataframe.loc[labeled_100_index,['veh_id','frame']], final_resampled_dataframe, on=['veh_id','frame'])
merge_200=pd.merge(final_results_dataframe.loc[labeled_200_index,['veh_id','frame']], final_resampled_dataframe, on=['veh_id','frame'])
merge_300=pd.merge(final_results_dataframe.loc[labeled_500_index,['veh_id','frame']], final_resampled_dataframe, on=['veh_id','frame'])
final_results_dataframe.to_csv("data/final_results_dataframe_15.csv")
merge_100.to_csv("data/final_validation_dataframe_100.csv")
merge_200.to_csv("data/final_validation_dataframe_200.csv")
merge_300.to_csv("data/final_validation_dataframe_500.csv")
