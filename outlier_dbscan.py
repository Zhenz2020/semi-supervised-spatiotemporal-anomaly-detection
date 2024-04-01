
import numpy as np
import pickle
from jenkspy import jenks_breaks
from st_dbscan import ST_DBSCAN
import utils.pre_stdbscan as pst
from sklearn.ensemble import IsolationForest
from utils.hbos import Hbos
from sklearn.neighbors import NearestNeighbors
class abnormal_det_out_db:

    def __init__(self):

        self.threshold_outliers = 1.5
        self.threshold_press = [[0.8, 2.4], [1.1, 2.1]]
        # self.traj_background = traj_background

    def classify(self, value, breaks):
        for i in range(1, len(breaks)):
            if value < breaks[i]:
                return i
        return len(breaks) - 1

    def threshold(self, temp_dist_data, lane):
        len_grid = len(temp_dist_data)
        results = np.zeros(len_grid)
        temp = pst.conv_id_to_time(temp_dist_data)
        data = temp.loc[:, ['cls', 'dist']]
        data['upper_thres'] = 0
        data['down_thres'] = 0
        data.loc[data['cls'] == 1, 'upper_thres'] = self.threshold_press[1][1]
        data.loc[data['cls'] == 1, 'down_thres'] = self.threshold_press[1][0]
        data.loc[data['cls'] == 2, 'upper_thres'] = self.threshold_press[0][1]
        data.loc[data['cls'] == 2, 'down_thres'] = self.threshold_press[0][0]
        if lane == 0:
            results[data['dist'] > data['upper_thres']] = 1
        elif lane == 4:
            results[data['dist'] < data['down_thres']] = 1
        else:
            results[data['dist'] > data['upper_thres']] = 1
            results[data['dist'] < data['down_thres']] = 1
        return results

    def outliers(self, temp_dist_data, Q=0.75):

        len_grid = len(temp_dist_data)
        results = np.zeros(len_grid)
        Q3 = np.quantile(temp_dist_data, Q)
        Q1 = np.quantile(temp_dist_data, 1-Q)
        up_value = Q3 + self.threshold_outliers * (Q3 - Q1)
        low_value = Q3 - self.threshold_outliers * (Q3 - Q1)
        low_index = temp_dist_data < low_value
        up_index = temp_dist_data > up_value

        results[low_index] = 1
        results[up_index] = 1
        results.tolist()
        return results

    def JNB(self, temp_dist_data, cls_num):
        len_grid = len(temp_dist_data)
        results = np.zeros(len_grid)
        temp_dist_data = np.array(temp_dist_data)
        classes = jenks_breaks(temp_dist_data, cls_num)
        classified = np.array([self.classify(i, classes) for i in temp_dist_data])
        low_index = classified == 1
        up_index = classified == cls_num

        results[up_index] = 1
        results[low_index] = 1
        results.tolist()
        return results

    def stdbscan(self, temp_dist_data, eps1, eps2, min_samples):

        temp = pst.conv_id_to_time(temp_dist_data)
        data = temp.loc[:, ['width', 'height', 'lati', 'longti']].values
        st_dbscan = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=min_samples)
        st_dbscan.fit(data)
        return st_dbscan.labels

    def stdbscan_fall(self, temp, eps1, eps2, min_samples):

        data = temp.loc[:, ['x','y','z']].values
        st_dbscan = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=min_samples)
        st_dbscan.fit(data)
        return st_dbscan.labels
    def isolate(self, temp_dist_data, max_feature=1.0):
        temp = pst.conv_id_to_time(temp_dist_data)
        data = temp.loc[:, ['lati', 'longti','dist','width','height']].values
        model = IsolationForest(n_estimators=200,
                                max_samples='auto',
                                contamination=float(0.1),
                                max_features=max_feature)
        model.fit(data)
        result = model.predict(data)
        return result

    def isolate_fall(self, temp, max_feature=1.0):

        data = temp.loc[:, ['x','y','z']].values
        model = IsolationForest(n_estimators=500,
                                max_samples='auto',
                                contamination=float(0.1),
                                max_features=max_feature)
        model.fit(data)
        result = model.predict(data)
        return result

    def knn(self, X, n_neighbors):
        '''
        Utility function to return k-average, k-median, knn
        Since these three functions are similar, so is inluded in the same func
        :param X: train data
        :param n_neighbors: number of neighbors
        :return:
        '''
        neigh = NearestNeighbors()
        neigh.fit(X)

        res = neigh.kneighbors(n_neighbors=n_neighbors, return_distance=True)
        # k-average, k-median, knn
        return np.mean(res[0], axis=1), np.median(res[0], axis=1), res[0][:, -1]
    def get_TOS_knn(self,temp, k=5):
        knn_clf = ["knn_mean"]
        data = temp.loc[:, ['x','y','z']].values
        k_mean, _, _ = self.knn(data, n_neighbors=k)

        return k_mean


    def get_TOS_hbos(self, temp, k_list):
        # k_list = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50]
        k_list = [20]
        data = temp.loc[:, ['x','y','z']].values
        for i in range(len(k_list)):
            k = k_list[i]
            clf = Hbos(bins=k, alpha=0.3)
            clf.fit(data)
            score_pred = clf.decision_scores
            # result_hbos[:, i] = score_pred
            # print(score_pred)
        return score_pred


if __name__ == '__main__':
    fileHandle = open('Traj_backgroud.txt', 'rb')
    traj_background = pickle.load(fileHandle)
    fileHandle.close()
    lane = 3
    temp_dist_data = traj_background[lane][1]
    ab_det = abnormal_det_out_db()
    results = ab_det.threshold(temp_dist_data, lane)
    results_iso = ab_det.isolate(temp_dist_data)
