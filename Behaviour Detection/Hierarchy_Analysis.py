import HierarchiaPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import glob
# Global parameters

for file in glob.glob('Behaviour_Files/Cam_*.csv'):
    cam_no = int(file.rsplit('.', 1)[0].rsplit('_', 1)[1])

    start_value = 1000
    randomization_n = 1000
    normal_probability = False
    color_arr = ['#FF00FF', '#007DFF', '#FF7D00', '#7DFF7D']
    shape_arr = ['Circle', 'Tail', 'Line', 'Neck']

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

    # Sample size (upper x-axis)
    sample_size = [0]
    for idx, action_idx in enumerate(df['action_no'].unique()):
        temp_df = df[df['action_no'].isin(df['action_no'].unique()[:idx + 1])]
        sample_size.append(len(temp_df))

    # HierarchiaPy
    hierarchy_df = HierarchiaPy.Hierarchia(df, 'Perpetrator', 'Target')
    mat = hierarchy_df.mat
    processed_mat = mat - np.transpose(mat)
    processed_mat = np.where(processed_mat < 0, 0, processed_mat)

    # Network graph
    plot_processed_mat = (processed_mat / processed_mat.sum()).round(decimals=4)
    network_graph = nx.from_numpy_matrix(plot_processed_mat, create_using=nx.DiGraph(directed=True))
    network_graph = nx.relabel_nodes(network_graph, {idx: name for idx, name in enumerate(shape_arr)})
    network_graph_int = nx.from_numpy_matrix(processed_mat, create_using=nx.DiGraph(directed=True))
    network_graph_int = nx.relabel_nodes(network_graph_int, {idx: name for idx, name in enumerate(shape_arr)})

    pos = nx.planar_layout(network_graph)
    nx.draw_networkx(network_graph,
                     pos,
                     node_color=color_arr,
                     node_size=1250,
                     linewidths=2,
                     node_shape='8',
                     arrows=True)

    weights = nx.get_edge_attributes(network_graph, 'weight')
    weights_int = nx.get_edge_attributes(network_graph_int, 'weight')
    weights_final = {key: (weights_int[key], weights[key]) for key in weights}
    nx.draw_networkx_edge_labels(network_graph, pos, edge_labels=weights_final)
    plt.title('Network Graph (CAGE ' + str(cam_no) + ')', size=16, y=1.12, color='maroon', fontweight="bold")
    plt.tight_layout()
    plt.savefig('Behaviour_Files/Final_Analysis/Network_Graph_Cam_' + str(cam_no) + '.png')
    plt.close()

    # Elo
    elo_dict_list = []
    for idx, action_idx in enumerate(df['action_no'].unique()):
        temp_df = df[df['action_no'].isin(df['action_no'].unique()[:idx + 1])]
        h_df = HierarchiaPy.Hierarchia(temp_df, 'Perpetrator', 'Target')
        temp_elo_ratings = hierarchy_df.elo(start_value=start_value,
                                            normal_probability=normal_probability)
        elo_dict_list.append(temp_elo_ratings)

    # Randomized Elo
    randomized_elo_dict_list = []
    for idx, action_idx in enumerate(df['action_no'].unique()):
        temp_df = df[df['action_no'].isin(df['action_no'].unique()[:idx + 1])]
        h_df = HierarchiaPy.Hierarchia(temp_df, 'Perpetrator', 'Target')
        temp_elo_ratings = h_df.randomized_elo(start_value=start_value,
                                               n=randomization_n,
                                               normal_probability=normal_probability)
        randomized_elo_dict_list.append(temp_elo_ratings)

    # Export Randomized Elo Plot
    fig = plt.figure(figsize=(30, 15))
    ax1 = fig.add_subplot(111)

    for idx in df['Perpetrator'].unique():
        temp_arr = [start_value if idx not in elo_dict else elo_dict[idx] for elo_dict in randomized_elo_dict_list]
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
    all_values = [start_value if idx not in elo_dict else elo_dict[idx]
                  for elo_dict in randomized_elo_dict_list for idx in np.arange(4)]
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
    plt.savefig('Behaviour_Files/Final_Analysis/Randomized_Elo_Plot_Cam_' + str(cam_no) + '.png',
                edgecolor='black',
                dpi=200,
                facecolor='white',
                transparent=True)
    plt.close(fig)

    # DS
    davids_scores_pij = hierarchy_df.davids_score(method='Pij', normalize=False)
    davids_scores_pij_norm = hierarchy_df.davids_score(method='Pij', normalize=True)
    davids_scores_dij = hierarchy_df.davids_score(method='Dij', normalize=False)
    davids_scores_dij_norm = hierarchy_df.davids_score(method='Dij', normalize=True)

    # ADI
    adi = hierarchy_df.average_dominance_index()

    # DCI
    dci = hierarchy_df.dci()

    # Steepness
    stp_test_dij = hierarchy_df.steepness_test(method='Dij', n=10000)

    # Kendall_K
    kendall_k = hierarchy_df.kendall_k()

    # Landau_h
    landau_h = hierarchy_df.landau_h()

    # File Verbose
    with open('Behaviour_Files/Final_Analysis/Hierarchy_Cam_' + str(cam_no) + '.txt', 'w') as f:
        f.write('++++++++++++++\nBEGIN ANALYSIS\n++++++++++++++\n\n')

        f.write('Hierarchy Matrix\n----------------- \n\n')
        f.write('       Doe 0  Doe 1  Doe 2  Doe 3 \n       -----  -----  -----  -----\n')
        for idx, line in enumerate(mat):
            f.write('Doe ' + str(idx) + '|' + '  '.join([str(a).rjust(5) for a in line]) + '\n')

        f.write('\nProcessed Hierarchy Matrix\n--------------------------- \n\n')
        f.write('       Doe 0  Doe 1  Doe 2  Doe 3 \n       -----  -----  -----  -----\n')
        for idx, line in enumerate(processed_mat):
            f.write('Doe ' + str(idx) + '|' + '  '.join([str(a).rjust(5) for a in line]) + '\n')

        f.write('\nElo Ratings\n------------ \n\n')
        f.write(str(elo_dict_list[-1]) + '\n\n')

        f.write('\nRandomized Elo Ratings\n----------------------- \n\n')
        f.write(str(randomized_elo_dict_list[-1]) + '\n\n')

        f.write('\nDavid Score (DS) \n----------------- \n\n')
        f.write('Pij: ' + str(davids_scores_pij) + '\n')
        f.write('Pij + Norm: ' + str(davids_scores_pij_norm) + '\n')
        f.write('Dij: ' + str(davids_scores_dij) + '\n')
        f.write('Dij + Norm: ' + str(davids_scores_dij_norm) + '\n\n')


        f.write('\nAverage Dominance Index (ADI) \n------------------------------ \n\n')
        f.write(str(adi) + '\n\n')

        f.write('\nLinearity Statistics (Kendall K & Landau h) \n--------------------------------- \n\n')
        f.write(str(kendall_k) + '\n\n')
        f.write(str(landau_h) + '\n\n')

        f.write('\nDCI \n--------------------------------- \n\n')
        f.write(str(dci) + '\n\n')

        f.write('\nSteepness \n--------------------------------- \n\n')
        f.write(str(stp_test_dij) + '\n\n')

        f.write('++++++++++++\nEND ANALYSIS\n++++++++++++')
