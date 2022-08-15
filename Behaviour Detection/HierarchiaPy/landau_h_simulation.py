from HierarchiaPy import HierarchiaPy
import pandas as pd
import numpy as np

from itertools import permutations
from itertools import product

# Define an environment with 5 animals

n = [0, 1, 2, 3, 4]
name_sequence = ['a', 'b', 'c', 'd', 'e']

# Calculate dyadic permutations
result = permutations(n, r=2)

# Make unique combinations from all possible dyadic relationships
dyadic_arr = []
all_permutations = list(result)
while len(all_permutations) > 0:
    swap_element = (all_permutations[0][1], all_permutations[0][0])
    dyadic_arr.append([all_permutations[0], swap_element])
    all_permutations.pop(0)
    all_permutations.remove(swap_element)

# Create interaction matrix and calculate Landau h'
list_landau_h = []
list_improved_landau_h = []
i = 0
for unique_combination in list(product(*dyadic_arr)):

    for element_x in unique_combination:
        temp_mat = np.zeros((len(n), len(n)), dtype='int64')
        for element_y in unique_combination:
            temp_mat[element_y[0], element_y[1]] = 1
        temp_mat[element_x[0], element_x[1]] = 0
        print(unique_combination)
        h_mat = HierarchiaPy(temp_mat, name_sequence)
        list_improved_landau_h.append(h_mat.landau_h(improved=True, n_random=1000)['Improved_Landau_h'])
        #list_landau_h.append(h_mat.landau_h(improved=False)['Landau_h'])

# Verbose results
print(sorted((list(set(list_landau_h)))))
print(sorted((list(set(list_improved_landau_h)))))
