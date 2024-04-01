import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)
PAC = pd.read_csv("data/PAC_tuning.csv", usecols=list(range(1, 17)))
F1 = pd.read_csv("data/auc_roc_tuning.csv", usecols=list(range(1, 5)))
model_list = ["Outliers", "JNB", "STDBSCAN", "Isolation Tree"]
out_tuning_list = np.linspace(0.6, 0.8, 10)
JNB_tuning_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
stdbscan_tuning_list = np.linspace(50, 350, 10)
isolation_tuning_list = np.linspace(0.6, 1.0, 10)
x = np.concatenate((out_tuning_list.reshape(1, -1), JNB_tuning_list.reshape(1, -1), stdbscan_tuning_list.reshape(1, -1),
                    isolation_tuning_list.reshape(1, -1)), axis=0)
for i in range(0, len(PAC.columns), 4):
    col1 = PAC.columns[i]
    col2 = PAC.columns[i + 1] if i + 1 < len(PAC.columns) else None
    col3 = PAC.columns[i + 2] if i + 2 < len(PAC.columns) else None
    col4 = PAC.columns[i + 3] if i + 3 < len(PAC.columns) else None
    if col2:
        # 创建新列，存储两列的平均值
        PAC[f'{col1}_{col2}_average'] = (PAC[col1] + PAC[col2] + PAC[col3] + PAC[col4]) / 4
# PAC_plot = PAC.iloc[:, [16, 17, 18, 19]]
PAC_plot = PAC.iloc[:, [5,6,7,8]]
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
# color1 = "#2878B5" # 孔雀绿
# color2 = "#8B4513" # 向日黄
# color3 = "#C82423"
color1 = "#2878B5"  # 孔雀绿
color2 = "#C82423"  # 向日黄
para=['Quantile','Class number','eps1','Max features']
for i in range(4):
    row = i // 2  # 行索引
    col = i % 2  # 列索引

    # 在每个子图上画 PAC_plot 数据集
    ax1 = axs[row, col]
    line1, = ax1.plot(x[i], PAC_plot.iloc[:, i].values, label='PAC', color=color1,marker="o",markersize=8, linewidth=2.0, markeredgecolor="white")
    ax1.set_ylabel('PAC', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0,8)
    ax1.set_xlabel(para[i], color='black')
    # 创建右边的坐标轴，用于显示 F1 数据集
    ax2 = ax1.twinx()
    line2, = ax2.plot(x[i], F1.iloc[:, i].values, label='F1', color=color2,marker="s",markersize=8, linewidth=2.0, markeredgecolor="white")
    ax2.set_ylabel('F1', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    #ax2.set_ylim(1, 7)
    # 设置子图标题
    ax2.set_title(f'{model_list[i]}',fontweight='bold')
    plt.legend(handles=[line1, line2], labels=['PAC', 'F1'], loc='lower right')
# fig.legend(handles=[line1, line2], labels=['PAC', 'F1'],loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           ncol=4, columnspacing=1.5)

# 调整子图之间的间距
plt.tight_layout()
#plt.legend()
plt.savefig('Fig/tuning_plot.png', dpi=800)
plt.show()
