import scipy.stats as st
import numpy as np
import pandas as pd
import HierarchiaPy
import glob
import json
import ast

sorted_randomized_elo_rating = [[], [], [], []]
improved_landau_h = []

for text_file in glob.glob('Behaviour_Files/Hierarchy_Cam*.txt'):
    with open(text_file, 'r') as f:
        lines = f.readlines()

    randomized_elo_dict = ast.literal_eval(lines[33])
    linearity = ast.literal_eval(lines[53])
    improved_landau_h.append(linearity['Improved_Landau_h'])

    for idx, doe_id in enumerate(sorted(randomized_elo_dict, key=randomized_elo_dict.get, reverse=True)):
        sorted_randomized_elo_rating[idx].append(randomized_elo_dict[doe_id])

for idx, rank_arr in enumerate(sorted_randomized_elo_rating):
    print(idx+1, sum(rank_arr)/len(rank_arr))
    print(st.t.interval(alpha=0.95, df=len(rank_arr) - 1, loc=np.mean(rank_arr), scale=st.tstd(rank_arr)))

print(sum(improved_landau_h)/len(improved_landau_h))
import sys
sys.exit(0)

# Confidence interval of the Randomized Elo ratings
i = 5
j = 5

files = (glob.glob('Behaviour_Files/Cam_[0-9].csv') + glob.glob('Behaviour_Files/Cam_[0-9][0-9].csv'))
processed_cam_list = []

final_randomized_elo_ratings = {}
temp_randomized_elo_ratings = {}
while len(files) > len(processed_cam_list):
    for file in files:
        cam_no = file.rsplit('.', 1)[0].rsplit('_', 1)[1]
        if cam_no not in processed_cam_list:
            temp_df = pd.read_csv('Behaviour_Files/Cam_' + str(cam_no) + '.csv', header=0)
            h_df = HierarchiaPy.Hierarchia(temp_df[:i], 'Perpetrator', 'Target')
            temp_randomized_elo_ratings[cam_no] = h_df.randomized_elo(start_value=1000, n=500, normal_probability=False)

            if i >= len(temp_df):
                processed_cam_list.append(cam_no)

    final_randomized_elo_ratings[i] = temp_randomized_elo_ratings.copy()
    print(final_randomized_elo_ratings)
    i += j
    print(i)

with open('randomized_elo_master_' + str(j) + '.json', 'w') as f:
    json.dump(final_randomized_elo_ratings, f)

