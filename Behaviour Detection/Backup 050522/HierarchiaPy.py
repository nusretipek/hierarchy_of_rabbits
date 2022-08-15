import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

test_matrix = np.array([[0, 6, 9, 8, 5],
                        [0, 0, 4, 6, 0],
                        [0, 2, 0, 4, 7],
                        [1, 0, 5, 0, 3],
                        [0, 0, 2, 3, 0]], dtype='float32')


class HierarchiaPy:

    def __init__(self, df, winner_col, loser_col):
        self.df = df
        self.winner_col = winner_col
        self.loser_col = loser_col

        # Create matrix
        self.indices = sorted(set(list(df[winner_col]) + list(df[loser_col])))
        self.cross_tab_df = pd.crosstab(index=df[winner_col],
                                        columns=df[loser_col],
                                        dropna=True).reindex(self.indices,
                                                             fill_value=0,
                                                             axis=0).reindex(self.indices,
                                                                             fill_value=0,
                                                                             axis=1)

    def elo(self, k=100):
        elo_dict = {i: 1000 for i in self.indices}

        for idx, row in self.df.iterrows():
            expected_winner = 1 / (1 + 10 ** ((elo_dict[row[self.loser_col]] - elo_dict[row[self.winner_col]]) / 400))
            expected_loser = 1 / (1 + 10 ** ((elo_dict[row[self.winner_col]] - elo_dict[row[self.loser_col]]) / 400))
            elo_dict[row[self.winner_col]] += (k - k * expected_winner)
            elo_dict[row[self.loser_col]] += (-k * expected_loser)

        return elo_dict

    def randomized_elo(self, k=100, n=100):
        elo_dict_master = {i: [] for i in self.indices}

        for _ in range(n):
            random_df = self.df.sample(frac=1)
            elo_dict_temp = {i: 1000 for i in self.indices}

            for idx, row in random_df.iterrows():
                expected_winner = 1 / (1 + 10 ** (
                        (elo_dict_temp[row[self.loser_col]] - elo_dict_temp[row[self.winner_col]]) / 400))
                expected_loser = 1 / (1 + 10 ** (
                        (elo_dict_temp[row[self.winner_col]] - elo_dict_temp[row[self.loser_col]]) / 400))
                elo_dict_temp[row[self.winner_col]] += (k - k * expected_winner)
                elo_dict_temp[row[self.loser_col]] += (-k * expected_loser)

            for key in elo_dict_temp:
                elo_dict_master[key].append(elo_dict_temp[key])

        for key in elo_dict_master:
            elo_dict_master[key] = sum(elo_dict_master[key]) / len(elo_dict_master[key])

        return elo_dict_master

    def davids_score(self):
        mat = self.cross_tab_df.to_numpy().astype('float64')
        np.fill_diagonal(mat, np.nan)
        sum_mat = mat.copy()

        for idx in range(0, mat.shape[0]):
            for idy in range(idx + 1, mat.shape[0]):
                temp_sum = mat[idx, idy] + mat[idy, idx]
                if temp_sum > 0:
                    sum_mat[idx, idy] = temp_sum
                    sum_mat[idy, idx] = temp_sum
                else:
                    sum_mat[idx, idy] = np.nan
                    sum_mat[idy, idx] = np.nan

        prop_mat = mat / sum_mat
        var_l = np.nansum(prop_mat, axis=0)
        var_w = np.nansum(prop_mat, axis=1)
        var_l2 = np.nansum(np.transpose(prop_mat) * var_l, axis=1)
        var_w2 = np.nansum(prop_mat * var_w, axis=1)
        var_ds = var_w + var_w2 - var_l - var_l2

        davids_score_dict = {i: var_ds[idx] for idx, i in enumerate(self.indices)}
        return davids_score_dict

    def average_dominance_index(self):
        mat = self.cross_tab_df.to_numpy().astype('float64')
        np.fill_diagonal(mat, np.nan)
        sum_mat = mat.copy()

        for idx in range(0, mat.shape[0]):
            for idy in range(idx + 1, mat.shape[0]):
                temp_sum = mat[idx, idy] + mat[idy, idx]
                if temp_sum > 0:
                    sum_mat[idx, idy] = temp_sum
                    sum_mat[idy, idx] = temp_sum
                else:
                    sum_mat[idx, idy] = np.nan
                    sum_mat[idy, idx] = np.nan

        prop_mat = mat / sum_mat
        var_w = np.nansum(prop_mat, axis=1)
        var_adi = var_w / np.count_nonzero(~np.isnan(prop_mat), axis=1)

        average_dominance_index_dict = {i: var_adi[idx] for idx, i in enumerate(self.indices)}
        return average_dominance_index_dict

    def adagio(self, preprocessing=False, plot_network=False, rank='topological'):
        mat = self.cross_tab_df.to_numpy().astype('int64')
        mat = np.array([[0, 1, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 2, 1, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0]], dtype='int64')
        # mat = self.cross_tab_df.to_numpy().astype('int64')

        if preprocessing:
            mat = mat - np.transpose(mat)
            mat = np.where(mat < 0, 0, mat)

        network_graph = nx.from_numpy_matrix(mat, create_using=nx.DiGraph(directed=True))

        if plot_network:
            nx.draw_networkx(network_graph, arrows=True)
            plt.show()

        largest = max(nx.strongly_connected_components(network_graph), key=len)
        while len(largest) > 1:
            sliced_mat = mat[list(largest), :][:, list(largest)]
            min_edges = np.where(sliced_mat == np.min(sliced_mat[sliced_mat > 0]))

            for idx in range(len(min_edges[0])):
                network_graph.remove_edge(list(largest)[min_edges[0][idx]], list(largest)[min_edges[1][idx]])

            largest = max(nx.strongly_connected_components(network_graph), key=len)

        if rank == 'topological':
            return {element: idx for idx, element in enumerate(list(nx.topological_sort(network_graph)))}
        elif rank == 'top':
            level_ranked = []
            rank_dict = {}
            i = 0
            topologically_sorted_nodes = list(nx.topological_sort(network_graph))
            while len(topologically_sorted_nodes) > 0:
                if len(list(network_graph.predecessors(topologically_sorted_nodes[0]))) == 0:
                    rank_dict[topologically_sorted_nodes[0]] = i
                    topologically_sorted_nodes.pop(0)
                else:
                    if not all([True if _ in rank_dict else False for _ in
                                list(network_graph.predecessors(topologically_sorted_nodes[0]))]):
                        i += 1
                        for element in level_ranked:
                            rank_dict[element] = i
                        level_ranked = []
                    else:
                        level_ranked.append(topologically_sorted_nodes[0])
                        topologically_sorted_nodes.pop(0)
            for element in level_ranked:
                rank_dict[element] = i + 1
            return rank_dict
        elif rank == 'bottom':
            level_ranked = []
            rank_dict = {}
            i = 0
            topologically_sorted_nodes = list(nx.topological_sort(network_graph))[::-1]
            while len(topologically_sorted_nodes) > 0:
                if len(list(network_graph.successors(topologically_sorted_nodes[0]))) == 0:
                    rank_dict[topologically_sorted_nodes[0]] = i
                    topologically_sorted_nodes.pop(0)
                else:
                    if not all([True if _ in rank_dict else False for _ in
                                list(network_graph.successors(topologically_sorted_nodes[0]))]):
                        i += 1
                        for element in level_ranked:
                            rank_dict[element] = i
                        level_ranked = []
                    else:
                        level_ranked.append(topologically_sorted_nodes[0])
                        topologically_sorted_nodes.pop(0)
            i += 1
            for element in level_ranked:
                rank_dict[element] = i
            return {key: abs(rank_dict[key] - i) for key in rank_dict}
        else:
            raise ValueError('Enter a valid rank: ["topological", "top", "bottom"]')

    @staticmethod
    def ISI98(runs=500):

        def swap_column_2d(arr, index_x, index_y):
            arr[:, [index_x, index_y]] = temp_mat[:, [index_y, index_x]]
            return arr

        def swap_row_2d(arr, index_x, index_y):
            arr[[index_x, index_y], :] = temp_mat[[index_y, index_x], :]
            return arr

        def swap_element_1d(arr, index_x, index_y):
            arr[index_x], arr[index_y] = arr[index_y], arr[index_x]
            return arr

        # define test matrix
        name_seq = np.array(['a', 'v', 'b', 'h', 'g', 'w', 'e', 'k', 'c', 'y'])
        mat = np.array([[0, 5, 4, 6, 3, 0, 2, 2, 3, 1],
                        [0, 0, 0, 0, 2, 1, 2, 0, 7, 7],
                        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
                        [0, 3, 0, 0, 0, 0, 6, 0, 2, 5],
                        [0, 0, 0, 1, 0, 2, 4, 0, 3, 0],
                        [2, 0, 0, 3, 0, 0, 0, 0, 2, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
                        [0, 0, 0, 0, 0, 1, 0, 2, 0, 6],
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]], dtype='int64')

        name_seq = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        mat = np.array([[0, 0, 1, 0.5, 1, 0, 0],
                        [1, 0, 0.5, 0, 1, 0, 0],
                        [0, 0.5, 0, 0, 0, 1, 1],
                        [0.5, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 1, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0]], dtype='int64')

        # calculate number of inconsistencies and number strength of inconsistencies
        diff_mat = mat - np.transpose(mat)
        inconsistencies = np.where(np.triu(diff_mat) < 0)
        str_i = sum([inconsistencies[1][idx] - inconsistencies[0][idx] for idx in range(len(inconsistencies[0]))])
        print('Number of inconsistencies: ', len(inconsistencies[0]))
        print('Number of inconsistencies: ', str_i)

        # define parameters
        min_inconsistencies = len(inconsistencies[0])
        min_str_i = str_i

        temp_mat = mat.copy()
        temp_seq = name_seq.copy()
        best_mat = temp_mat.copy()
        best_seq = temp_seq.copy()

        # iterative process
        for _ in range(runs):
            for idx in range(len(inconsistencies[0])):
                net_incs = 0
                for idy in range(inconsistencies[0][idx], inconsistencies[1][idx]):
                    net_incs += (temp_mat[inconsistencies[1][idx], idy] - temp_mat[idy, inconsistencies[1][idx]])
                if net_incs > 0:
                    temp_mat = swap_column_2d(temp_mat, inconsistencies[0][idx], inconsistencies[1][idx])
                    temp_mat = swap_row_2d(temp_mat, inconsistencies[0][idx], inconsistencies[1][idx])
                    temp_seq = swap_element_1d(temp_seq, inconsistencies[0][idx], inconsistencies[1][idx])

            # compute number of inconsistencies and strength of inconsistencies
            inconsistencies = np.where(np.triu(temp_mat - np.transpose(temp_mat)) < 0)
            str_i = sum([inconsistencies[1][idx] - inconsistencies[0][idx] for idx in range(len(inconsistencies[0]))])

            if (len(inconsistencies[0]) < min_inconsistencies or
                    (len(inconsistencies[0])) == min_inconsistencies and str_i < min_str_i):
                best_seq, best_mat = temp_seq.copy(), temp_mat.copy()
                min_inconsistencies, min_str_i = len(inconsistencies[0]), str_i
            else:
                if min_str_i > 0 and _ < runs - 1:
                    for idx in range(len(inconsistencies[0])):
                        random_swap_idx = 0
                        if inconsistencies[1][idx] - 1 != 0:
                            random_swap_idx = np.random.randint(0, inconsistencies[1][idx] - 1)
                        temp_mat = swap_column_2d(temp_mat, random_swap_idx, inconsistencies[1][idx])
                        temp_mat = swap_row_2d(temp_mat, random_swap_idx, inconsistencies[1][idx])
                        temp_seq = swap_element_1d(temp_seq, random_swap_idx, inconsistencies[1][idx])
                else:
                    print('Optimal or near-optimal linear ranking is found!')
                    print(best_seq)
                    print(best_mat)
                    break

        # final phase
        temp_mat = best_mat.copy()
        temp_seq = best_seq.copy()
        best_diff_mat = (best_mat - np.transpose(best_mat)).astype('float64')
        upper_triangle_indices = np.triu_indices_from(best_diff_mat, k=1)

        for idx in range(len(upper_triangle_indices[0])):
            if best_diff_mat[upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]] == 0 and \
                    upper_triangle_indices[1][idx] - upper_triangle_indices[0][idx] == 1:
                d_i = len(np.where(best_diff_mat[upper_triangle_indices[0][idx], :] > 0)[0])
                s_i = len(np.where(best_diff_mat[upper_triangle_indices[0][idx], :] < 0)[0])
                d_j = len(np.where(best_diff_mat[upper_triangle_indices[1][idx], :] > 0)[0])
                s_j = len(np.where(best_diff_mat[upper_triangle_indices[1][idx], :] < 0)[0])

                if d_i - s_i < d_j - s_j:
                    temp_mat = swap_column_2d(temp_mat, upper_triangle_indices[0][idx], upper_triangle_indices[1][idx])
                    temp_mat = swap_row_2d(temp_mat, upper_triangle_indices[0][idx], upper_triangle_indices[1][idx])
                    temp_seq = swap_element_1d(temp_seq, upper_triangle_indices[0][idx], upper_triangle_indices[1][idx])
                    temp_i = np.where(np.triu(temp_mat - np.transpose(temp_mat)) < 0)
                    temp_str_i = sum([temp_i[1][idz] - temp_i[0][idz] for idz in range(len(temp_i[0]))])
                    if not temp_str_i > min_str_i:
                        best_seq, best_mat = temp_seq.copy(), temp_mat.copy()

        print('After Final Phase')
        print(best_seq)
        print(best_mat)
