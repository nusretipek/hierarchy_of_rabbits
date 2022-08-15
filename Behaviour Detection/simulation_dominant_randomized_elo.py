import HierarchiaPy
import numpy as np
import time


class DominanceSimulator:

    def __init__(self, n, randomized_elo_dict, weight_matrix):
        self.n = n
        self.dict = randomized_elo_dict
        self.mat = np.zeros((len(self.dict), len(self.dict)), dtype='uint32')
        self.weight_matrix = np.array(weight_matrix)
        self.win_prob_matrix = self.get_win_probability_matrix()
        self.indices_matrix = self.get_indices_matrix()
        self.errors = {  # 7 out of 52 (0.153846154) -> empirical values
            'id_swap': 0.019230769,  # 1 out of 52
            'miss_action': 0.057692308,  # 3 out of 52
            'false_detection': 0.076923077}  # 4 out of 52

        # Methods
        self.fill_matrix()
        self.inject_errors()

    def get_win_probability_matrix(self):
        win_prob_matrix = []
        for idx, element_x in enumerate(list(self.dict.values())):
            win_prob_list = []
            for idy, element_y in enumerate(list(self.dict.values())):
                if idx != idy:
                    win_prob_list.append(1 / (1 + (10 ** ((element_y - element_x) / 400))))
                else:
                    win_prob_list.append(0)
            win_prob_matrix.append(win_prob_list)
        return np.array(win_prob_matrix)

    def get_indices_matrix(self):
        indices_matrix = []
        for idx in range(self.mat.shape[0]):
            indices_row = []
            for idy in range(self.mat.shape[0]):
                indices_row.append(str(idx) + str(idy))
            indices_matrix.append(indices_row)
        return indices_matrix

    def fill_matrix(self):
        # Randomly sample fights
        flat_indices = np.array([x for xs in self.indices_matrix for x in xs])
        flat_weights = np.array([x for xs in self.weight_matrix for x in xs])
        choices = np.random.choice(flat_indices, self.n, p=flat_weights)
        for i in choices:
            random_entry = np.random.choice((i, i[::-1]), 1, p=[self.win_prob_matrix[int(i[0]), int(i[1])],
                                                                self.win_prob_matrix[int(i[1]), int(i[0])]])[0]
            self.mat[int(random_entry[0]), int(random_entry[1])] += 1

    def inject_errors(self):
        for error in self.errors.keys():
            error_p = self.n * self.errors[error]
            error_count = int(np.floor(error_p))
            error_count += np.random.choice([0, 1], p=[1-(error_p-error_count), (error_p-error_count)])

            triu_upper_indices = np.triu_indices_from(self.mat, k=1)
            if error == 'id_swap':
                for _ in range(error_count):
                    idx, idy = 0, 0
                    while self.mat[idx, idy] == 0:
                        idx = np.random.choice(triu_upper_indices[0], 1)[0]
                        idy = np.random.choice(triu_upper_indices[1], 1)[0]
                        if self.mat[idx, idy] == 0:
                            idx, idy = idy, idx
                    self.mat[idx, idy] -= 1
                    self.mat[idy, idx] += 1
            elif error == 'miss_action':
                for _ in range(error_count):
                    idx, idy = 0, 0
                    while self.mat[idx, idy] == 0:
                        idx = np.random.choice(list(self.dict.keys()), 1)[0]
                        idy = np.random.choice(list({k: v for k, v in self.dict.items() if k != idx}.keys()), 1)[0]
                    self.mat[idx, idy] -= 1
            elif error == 'false_detection':
                for _ in range(error_count):
                    idx = np.random.choice(list(self.dict.keys()), 1)[0]
                    idy = np.random.choice(list({k: v for k, v in self.dict.items() if k != idx}.keys()), 1)[0]
                    self.mat[idx, idy] += 1


def get_dominance_at_n(action_n):
    re_average = {0: 1179.1388,
                  1: 997.3815,
                  2: 936.3420,
                  3: 887.1378}

    weight_mat = [[0, 0.16515782, 0.16687057, 0.15292391],
                  [0.06019085, 0, 0.07218008, 0.08343528],
                  [0.0415953, 0.06141424, 0, 0.0641057],
                  [0.04037191, 0.04232934, 0.04942501, 0]]

    step_dict = {k: 0 for k, v in re_average.items()}
    for _ in range(100):
        s_time = time.time()
        dom_simulator = DominanceSimulator(n=action_n, randomized_elo_dict=re_average, weight_matrix=weight_mat)
        hierarchia_obj = HierarchiaPy.Hierarchia(mat=dom_simulator.mat, name_seq=np.arange(0, len(dom_simulator.mat)))
        randomized_elo = hierarchia_obj.randomized_elo(n=500, normal_probability=False)
        step_dict[max(randomized_elo, key=randomized_elo.get)] += 1
        print(time.time() - s_time)
    file = open("simulation_results.txt", "a")  # append mode
    file.write(str(action_n) + ' - ' + str(step_dict) + '\n')
    file.close()


get_dominance_at_n(50)

# Development Code

# dom_simulator = DominanceSimulator(n=30, normalized_ds_dict={0: 2.0894, 1: 0.9992, 2: 1.4148, 3: 1.4967})
# print(dom_simulator.mat)
# hierarchia_obj = HierarchiaPy.Hierarchia(mat=dom_simulator.mat)
# ds = hierarchia_obj.davids_score(method='Dij', normalize=True)
# print(ds)
# step_dict = {k: (step_dict[k]+v) for k, v in randomized_elo.items()}
# step_dict = {k: v/10 for k, v in step_dict.items()}
# n_ds = {0: 2.0894, 1: 0.9992, 2: 1.4148, 3: 1.4967}
# n_ds = {0: 1.1675, 1: 1.4162, 2: 2.4873, 3: 0.929}

## Checkpoint Complete! ##
