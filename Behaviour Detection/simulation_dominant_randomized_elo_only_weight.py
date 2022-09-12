import HierarchiaPy
import numpy as np
import time


class DominanceSimulator:

    def __init__(self, n, weight_matrix):
        # Params
        self.n = n
        self.weight_matrix = np.array(weight_matrix)
        self.mat = np.zeros((len(self.weight_matrix), len(self.weight_matrix)), dtype='uint32')
        self.indices_matrix = self.get_indices_matrix()
        self.errors = {  # 7 out of 52 (0.153846154) -> empirical values
            'id_swap': 0.019230769,  # 1 out of 52
            'miss_action': 0.057692308,  # 3 out of 52
            'false_detection': 0.076923077}  # 4 out of 52

        # Methods
        self.fill_matrix()
        #self.inject_errors()

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
            self.mat[int(i[0]), int(i[1])] += 1

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
                        idx = np.random.choice(list(range(len(self.mat))), 1)[0]
                        p_list = list(range(len(self.mat)))
                        p_list.remove(idx)
                        idy = np.random.choice(p_list, 1)[0]
                    self.mat[idx, idy] -= 1
            elif error == 'false_detection':
                for _ in range(error_count):
                    idx = np.random.choice(list(range(len(self.mat))), 1)[0]
                    p_list = list(range(len(self.mat)))
                    p_list.remove(idx)
                    idy = np.random.choice(p_list, 1)[0]
                    self.mat[idx, idy] += 1


def get_dominance_at_n(action_n):
    weight_mat = [[0, 0.14653285, 0.1689781, 0.16532847],
                  [0.05565693, 0, 0.07025547, 0.08083942],
                  [0.05054745, 0.05711679, 0, 0.06076642],
                  [0.04671533, 0.04762774, 0.04963504, 0]]

    step_dict = {k: 0 for k in range(4)}
    step_dict_elo = {k: 0 for k in range(4)}

    for _ in range(100):
        s_time = time.time()
        dom_simulator = DominanceSimulator(n=action_n, weight_matrix=weight_mat)
        hierarchia_obj = HierarchiaPy.Hierarchia(mat=dom_simulator.mat, name_seq=np.arange(0, len(dom_simulator.mat)))
        randomized_elo = hierarchia_obj.randomized_elo(n=100, normal_probability=False)
        step_dict[max(randomized_elo, key=randomized_elo.get)] += 1
        step_dict_elo = {k: (step_dict_elo[k] + v) for k, v in randomized_elo.items()}
    step_dict_elo = {k: v / 100 for k, v in step_dict_elo.items()}
    print(str(step_dict_elo))
    #file = open("simulation_results.txt", "a")  # append mode
    #file.write(str(action_n) + ' - ' + str(step_dict) + '\n')
    #file.close()


get_dominance_at_n(100)

## Checkpoint Complete! ##
