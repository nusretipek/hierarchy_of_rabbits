import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import sys
import scipy.stats as st
from scipy.signal import savgol_filter

with open('randomized_elo_master_5.json', 'r') as f:
    d = json.load(f)

x_arr, y_arr_0, y_arr_1, y_arr_2, y_arr_3 = None, None, None, None, None

# Winner Dict
d_winner = {}
k_i = max(d, key=int)
for k_j in d[k_i]:
    sorted_randomized_elo_rating = [[], [], [], []]
    d_winner[k_j] = list(sorted(d[k_i][k_j], key=d[k_i][k_j].get, reverse=True))

for k_i in d:
    sorted_randomized_elo_rating = [[], [], [], []]

    for k_j in d[k_i]:
        for idx, doe_id in enumerate(d_winner[k_j]):
            if doe_id in d[k_i][k_j]:
                sorted_randomized_elo_rating[idx].append(d[k_i][k_j][doe_id])
            else:
                sorted_randomized_elo_rating[idx].append(1000)

        '''
        doe_list = ['0', '1', '2', '3']
        for idx, doe_id in enumerate(sorted(d[k_i][k_j], key=d[k_i][k_j].get, reverse=True)):
            doe_list.remove(str(doe_id))
        for doe_id in doe_list:
            d[k_i][k_j][doe_id] = 1000
        for idx, doe_id in enumerate(sorted(d[k_i][k_j], key=d[k_i][k_j].get, reverse=True)):
            sorted_randomized_elo_rating[idx].append(d[k_i][k_j][doe_id])
        '''

    x_arr = np.concatenate((x_arr, np.repeat(k_i, len(d[k_i]))),
                           axis=0) if x_arr is not None else np.repeat(k_i, len(d[k_i]))
    y_arr_0 = np.concatenate((y_arr_0, sorted_randomized_elo_rating[0]),
                             axis=0) if y_arr_0 is not None else sorted_randomized_elo_rating[0]
    y_arr_1 = np.concatenate((y_arr_1, sorted_randomized_elo_rating[1]),
                             axis=0) if y_arr_1 is not None else sorted_randomized_elo_rating[1]
    y_arr_2 = np.concatenate((y_arr_2, sorted_randomized_elo_rating[2]),
                             axis=0) if y_arr_2 is not None else sorted_randomized_elo_rating[2]
    y_arr_3 = np.concatenate((y_arr_3, sorted_randomized_elo_rating[3]),
                             axis=0) if y_arr_3 is not None else sorted_randomized_elo_rating[3]

x_arr = x_arr.astype('int64')
y0_mean = [np.mean(y_arr_0[np.where(x_arr == x)]) for x in np.unique(x_arr)]
y1_mean = [np.mean(y_arr_1[np.where(x_arr == x)]) for x in np.unique(x_arr)]
y2_mean = [np.mean(y_arr_2[np.where(x_arr == x)]) for x in np.unique(x_arr)]
y3_mean = [np.mean(y_arr_3[np.where(x_arr == x)]) for x in np.unique(x_arr)]

# Save the ratings in Excel files
'''
df = pd.DataFrame({'Action_No': x_arr,
                   'Doe_Rank_1': y_arr_0,
                   'Doe_Rank_2': y_arr_1,
                   'Doe_Rank_3': y_arr_2,
                   'Doe_Rank_4': y_arr_3})

df = pd.DataFrame({'Action_No': np.unique(x_arr),
                   'Doe_Rank_1': y0_mean,
                   'Doe_Rank_2': y1_mean,
                   'Doe_Rank_3': y2_mean,
                   'Doe_Rank_4': y3_mean})

df.to_csv('group_data_does_2.csv', sep=',')
sys.exit(0)
'''

y0_mean_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_0[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_0[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_0[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y0_mean_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_0[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_0[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_0[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]
y0_pred_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_0[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_0[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_0[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y0_pred_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_0[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_0[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_0[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]

y1_mean_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_1[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_1[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_1[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y1_mean_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_1[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_1[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_1[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]
y1_pred_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_1[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_1[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_1[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y1_pred_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_1[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_1[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_1[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]

y2_mean_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_2[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_2[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_2[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y2_mean_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_2[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_2[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_2[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]
y2_pred_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_2[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_2[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_2[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y2_pred_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_2[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_2[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_2[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]

y3_mean_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_3[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_3[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_3[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y3_mean_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_3[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_3[np.where(x_arr == x)]),
                                    scale=st.sem(y_arr_3[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]
y3_pred_conf_lower = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_3[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_3[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_3[np.where(x_arr == x)]))[0] for x in np.unique(x_arr)]
y3_pred_conf_upper = [st.t.interval(alpha=0.95,
                                    df=len(y_arr_3[np.where(x_arr == x)]) - 1,
                                    loc=np.mean(y_arr_3[np.where(x_arr == x)]),
                                    scale=st.tstd(y_arr_3[np.where(x_arr == x)]))[1] for x in np.unique(x_arr)]

y0_mean_slope = [(y0_mean[i+1]-y0_mean[i])/5 for i in range(len(y0_mean)-1)]
y1_mean_slope = [(y1_mean[i+1]-y1_mean[i])/5 for i in range(len(y1_mean)-1)]
y2_mean_slope = [(y2_mean[i+1]-y2_mean[i])/5 for i in range(len(y2_mean)-1)]
y3_mean_slope = [(y3_mean[i+1]-y3_mean[i])/5 for i in range(len(y3_mean)-1)]

#plt.plot(np.unique(x_arr)[:-1], savgol_filter(y0_mean_slope, 5, 3))
#plt.plot(np.unique(x_arr)[:-1], savgol_filter(y1_mean_slope, 5, 3))
#plt.plot(np.unique(x_arr)[:-1], savgol_filter(y2_mean_slope, 5, 3))
#plt.plot(np.unique(x_arr)[:-1], savgol_filter(y3_mean_slope, 5, 3))
#plt.plot(np.unique(x_arr)[:-1],
#         np.mean([np.absolute(y0_mean_slope),
#                 np.absolute(y1_mean_slope),
#                 np.absolute(y2_mean_slope),
#                 np.absolute(y3_mean_slope)], axis=0),
#         color='black',
#         linestyle='dashed')

#plt.show()
#sys.exit(1)

plt.plot(np.unique(x_arr), y0_mean, color='firebrick', linestyle='dashed')
plt.fill_between(np.unique(x_arr), y0_mean_conf_lower, y0_mean_conf_upper, alpha=.5, label='Confidence interval', color='salmon')
plt.fill_between(np.unique(x_arr), y0_pred_conf_lower, y0_pred_conf_upper, alpha=.5, label='Prediction interval', color='salmon')

plt.plot(np.unique(x_arr), y1_mean, color='royalblue', linestyle='dashed')
plt.fill_between(np.unique(x_arr), y1_mean_conf_lower, y1_mean_conf_upper, alpha=.6, label='Confidence interval', color='dodgerblue')
plt.fill_between(np.unique(x_arr), y1_pred_conf_lower, y1_pred_conf_upper, alpha=.3, label='Prediction interval', color='dodgerblue')

plt.plot(np.unique(x_arr), y2_mean, color='green', linestyle='dashed')
plt.fill_between(np.unique(x_arr), y2_mean_conf_lower, y2_mean_conf_upper, alpha=.6, label='Confidence interval', color='chartreuse')
plt.fill_between(np.unique(x_arr), y2_pred_conf_lower, y2_pred_conf_upper, alpha=.3, label='Prediction interval', color='chartreuse')

plt.plot(np.unique(x_arr), y3_mean, color='darkmagenta', linestyle='dashed')
plt.fill_between(np.unique(x_arr), y3_mean_conf_lower, y3_mean_conf_upper, alpha=.6, label='Confidence interval', color='purple')
plt.fill_between(np.unique(x_arr), y3_pred_conf_lower, y3_pred_conf_upper, alpha=.3, label='Prediction interval', color='purple')

plt.xticks(np.arange(0, max(x_arr), 10))
plt.tight_layout()
plt.show()

sys.exit(1)

# ax = sns.pointplot(x=x_arr, y=y_arr_1)
# ax = sns.pointplot(x=x_arr, y=y_arr_2)
# ax = sns.pointplot(x=x_arr, y=y_arr_3)

# ax = sns.regplot(x=x_arr, y=y_arr_0, scatter=False)
# ax = sns.regplot(x=x_arr, y=y_arr_1, scatter=False)
# ax = sns.regplot(x=x_arr, y=y_arr_2, scatter=False)
# ax = sns.regplot(x=x_arr, y=y_arr_3, scatter=False)
plt.show()
