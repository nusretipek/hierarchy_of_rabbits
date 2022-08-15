import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global parameters

cam_no = 9
start_value = 1000
randomization_n = 500
normal_probability = False

# Sample arrays
Tail = [1003.3635, 1013.6198, 1022.1915, 1033.9458, 1041.6777, 1040.5527, 1042.0762, 1049.5306, 1001.3874, 958.6132,
        949.4908, 949.9298, 951.9912, 958.0712, 949.4076, 947.4822, 954.4432, 950.5177, 945.5749, 918.772, 879.3157,
        875.9987, 876.635, 873.7712, 868.3805, 861.4223, 861.2711, 849.3031, 863.6228, 859.0795]
Circle = [1036.0157, 1072.4699, 1128.2113, 1156.7003, 1209.0573, 1188.0424, 1198.5336, 1225.1461, 1212.4352, 1200.2007,
          1200.1878, 1216.8876, 1229.9419, 1242.6629, 1232.9416, 1240.5445, 1242.1466, 1245.6744, 1226.0458, 1248.4819,
          1236.5757, 1239.8151, 1248.7638, 1258.2777, 1237.6393, 1240.0901, 1234.7832, 1232.5745, 1230.7257, 1223.5288]
Neck = [960.6208, 966.3386, 943.8558, 903.4176, 885.6826, 867.1876, 871.5993, 824.616, 848.9553, 893.149, 898.7107,
        883.8787, 883.078, 861.0683, 884.581, 891.9598, 888.5206, 888.3275, 916.9818, 925.0304, 986.6718, 981.6877,
        964.6046, 963.1361, 984.4073, 983.4509, 982.3847, 991.7455, 983.0918, 1002.2799]
Line = [1000, 947.5717, 905.7414, 905.9363, 863.5825, 904.2172, 887.7909, 900.7073, 937.222, 948.0371, 951.6107,
        949.3039, 934.9889, 938.1976, 933.0698, 920.0135, 914.8897, 915.4804, 911.3975, 907.7156, 897.4368, 902.4985,
        909.9966, 904.8151, 909.573, 915.0367, 921.561, 926.3769, 922.5598, 915.1118]

# Load CSV file
df = pd.read_csv('Behaviour_Files/Cam_' + str(cam_no) + '.csv', header=0)

# Time Array
time_df = pd.read_excel('Behaviour_Files/Camera ' + str(cam_no) + '.xlsx', header=0)

time_arr = [time_df.loc[action_idx - 1, 'Video_Name'].rsplit('.', 1)[1][:11] +
            time_df.loc[action_idx - 1, 'Action_Start'] for action_idx in df['action_no'].unique()]
time_arr = [pd.to_datetime(element, format='%Y%m%d_%H%M:%S') for element in time_arr]
time_arr_text = [str(time.month).zfill(2) + '-' + str(time.day).zfill(2) + ' ' +
                 str(time.hour).zfill(2) + ':' + str(time.minute).zfill(2) for time in time_arr]
time_arr_text.insert(0, 'GROUPING')

sample_size = [0]
for idx, action_idx in enumerate(df['action_no'].unique()):
    temp_df = df[df['action_no'].isin(df['action_no'].unique()[:idx + 1])]
    sample_size.append(len(temp_df))

# Export Randomized Elo Plot
color_arr = ['#FF00FF', '#007DFF', '#FF7D00', '#7DFF7D']
shape_arr = ['Circle', 'Tail', 'Line', 'Neck']

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(111)

for idx in df['Perpetrator'].unique():
    temp_arr = None
    if idx == 0:
        temp_arr = Circle
    elif idx == 1:
        temp_arr = Tail
    elif idx == 2:
        temp_arr = Line
    elif idx == 3:
        temp_arr = Neck
    else:
        print('Error')
    temp_arr.insert(0, start_value)
    ax1.plot(time_arr_text, temp_arr, label=shape_arr[idx] + ' (' + str(idx) + ')',
             color=color_arr[idx], linewidth=2, marker="h")

# X-axis
ax1.set_xlabel("Action Date/Time", labelpad=10, color='black', fontweight="bold")
ax1.set_xticks(tuple(time_arr_text), rotation=90)
ax1.set_xticklabels(tuple(time_arr_text), rotation=90)
ax2 = ax1.twiny()
ax2.set_xticks(ax1.get_xticks())
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(tuple(sample_size))
ax2.set_xlabel('# of Actions Detected (Cumulative)', labelpad=10, color='black', fontweight="bold")

# Y-axis
all_values = [e for v in [Circle, Tail, Line, Neck] for e in v]
min_val, max_val = min(all_values), max(all_values)
ax1.set_ylabel("Randomized ELO Rating", labelpad=10, color='black', fontweight="bold")
ax1.set_yticks(np.arange(int(round(min_val / 10) * 10 - 30), int(round(max_val / 10) * 10 + 30), 20))

# Grid
ax1.grid(True, color="grey", linewidth="0.5", linestyle="dashed")

# Legend
handles, labels = ax1.get_legend_handles_labels()
current_pos = [int(label[-2]) for label in labels]
ax1.legend([handles[current_pos.index(i)] for i in np.arange(0, 4)],
           [labels[current_pos.index(i)] for i in np.arange(0, 4)], loc="upper left")

# Layout & Save figure
plt.title('Randomized ELO Ratings (CAGE ' + str(cam_no) + ')', size=16, y=1.12, color='maroon', fontweight="bold")
plt.tight_layout()
plt.savefig('Behaviour_Files/temp_plot' + str(cam_no) + '.png')
