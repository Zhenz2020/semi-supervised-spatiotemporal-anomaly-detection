import pandas as pd
import matplotlib.pyplot as plt
X1 = pd.read_csv('data/final_results_dataframe_0312.csv',usecols=["veh_id", "frame", "world_x", "world_y","label"])
X2 = pd.read_csv('data/final_results_dataframe_0322.csv',usecols=["veh_id", "frame", "world_x", "world_y","label"])
X3 = pd.read_csv('data/final_results_dataframe_0324.csv',usecols=["veh_id", "frame", "world_x", "world_y","label"])


def graph_draw(file1,file2,file3):
    fig = plt.figure(figsize=(9,6))
    font = {'family': 'Times New Roman',
            'size': 12}
    plt.rc('font', **font)
    #fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(10,8))
    ax = fig.add_subplot(1, 3, 1,projection='3d')

    # где 0 — нормальное явление, а 1 — аномальное событие падения.
    colors = ['#A9A9A9', 'red']
    s=1
    c_list = []
    for val in file1['label']:
        if val == 0:
            c_list.append(colors[0])
        else:
            c_list.append(colors[1])
    #legend_elements =[markers([0], [0], marker='.', color='black', label='normal', markersize=5),markers([0], [0], marker='.', color='r', label='fall', markersize=5)]
    ax.scatter(file1['world_x'], file1['world_y'], zs=file1['frame'], s=s, c=c_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('frame',labelpad=10)
    ax.view_init(elev=20., azim=-130)
    ax.set_box_aspect([6, 6, 12])



    ax = fig.add_subplot(1, 3, 2,projection='3d')
    # где 0 — нормальное явление, а 1 — аномальное событие падения.
    c_list = []
    for val in file2['label']:
        if val == 0:
            c_list.append(colors[0])
        else:
            c_list.append(colors[1])
    #legend_elements =[markers([0], [0], marker='.', color='black', label='normal', markersize=5),markers([0], [0], marker='.', color='r', label='fall', markersize=5)]
    ax.scatter(file2['world_x'], file2['world_y'], zs=file2['frame'], s=s, c=c_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(elev=20., azim=-130)
    ax.set_zlabel('frame', labelpad=10)
    ax.set_box_aspect([6, 6, 12])

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    # где 0 — нормальное явление, а 1 — аномальное событие падения.
    c_list = []
    for val in file3['label']:
        if val == 0:
            c_list.append(colors[0])
        else:
            c_list.append(colors[1])
    # legend_elements =[markers([0], [0], marker='.', color='black', label='normal', markersize=5),markers([0], [0], marker='.', color='r', label='fall', markersize=5)]
    ax.scatter(file3['world_x'], file3['world_y'], zs=file3['frame'], s=s, c=c_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('frame', labelpad=10)
    ax.view_init(elev=20., azim=-130)
    ax.set_box_aspect([6, 6, 12])
    #fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.tight_layout()
    plt.savefig('trajectory_plot.png',dpi=1000)
    plt.show()

graph_draw(X1,X2,X3)