import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_20_test = pd.read_csv('D:/semi-supervised-stacking/data/train/data_0.csv')
df_21_test = pd.read_csv('D:/semi-supervised-stacking/data/train/data_1.csv')
df_22_test = pd.read_csv('D:/semi-supervised-stacking/data/train/data_2.csv')
df_23_test = pd.read_csv('D:/semi-supervised-stacking/data/train/data_3.csv')
df_24_test = pd.read_csv('D:/semi-supervised-stacking/data/train/data_4.csv')
total_test = pd.concat([df_20_test, df_21_test, df_22_test, df_23_test, df_24_test])
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)
def graph_draw(file1,file2,file3):
    fig = plt.figure(figsize=(9,9))
    font = {'family': 'Times New Roman',
            'size': 12}
    plt.rc('font', **font)
    #fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(10,8))
    ax = fig.add_subplot(3, 1, 1,projection='3d')

    # где 0 — нормальное явление, а 1 — аномальное событие падения.
    colors = ['#A9A9A9', 'red']
    s=1
    c_list = []
    for val in file1['anomaly']:
        if val == 0:
            c_list.append(colors[0])
        else:
            c_list.append(colors[1])
    #legend_elements =[markers([0], [0], marker='.', color='black', label='normal', markersize=5),markers([0], [0], marker='.', color='r', label='fall', markersize=5)]
    ax.scatter(file1['x'], file1['y'], zs=file1['z'], s=s, c=c_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=20., azim=-130)
    ax.set_box_aspect([12, 12, 4])



    ax = fig.add_subplot(3, 1, 2,projection='3d')
    # где 0 — нормальное явление, а 1 — аномальное событие падения.
    c_list = []
    for val in file2['anomaly']:
        if val == 0:
            c_list.append(colors[0])
        else:
            c_list.append(colors[1])
    #legend_elements =[markers([0], [0], marker='.', color='black', label='normal', markersize=5),markers([0], [0], marker='.', color='r', label='fall', markersize=5)]
    ax.scatter(file2['x'], file2['y'], zs=file2['z'], s=s, c=c_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(elev=20., azim=-130)
    ax.set_zlabel('z')
    ax.set_box_aspect([12, 12, 4])

    ax = fig.add_subplot(3, 1, 3, projection='3d')
    # где 0 — нормальное явление, а 1 — аномальное событие падения.
    c_list = []
    for val in file3['anomaly']:
        if val == 0:
            c_list.append(colors[0])
        else:
            c_list.append(colors[1])
    # legend_elements =[markers([0], [0], marker='.', color='black', label='normal', markersize=5),markers([0], [0], marker='.', color='r', label='fall', markersize=5)]
    ax.scatter(file3['x'], file3['y'], zs=file3['z'], s=s, c=c_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=20., azim=-130)
    ax.set_box_aspect([12, 12, 4])
    #fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.tight_layout()
    plt.savefig('fall_plot.png',dpi=1000)
    plt.show()

graph_draw(df_21_test,df_22_test,df_23_test)