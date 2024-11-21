import numpy as np

def fix_soc(x,y, SOC, CapMax, etach, etadh,timestep,discretization):
    v = np.array([SOC])#input the initial SOC and in output the process of charging
    for i in range(discretization):
        v = np.append(v,v[-1]+(x[i] * etach - y[i] / etadh)*timestep/60/CapMax)
    v=v[1:]
    return v

def borders(n,l1,u1,l2,u2):
    v = [[l1, u1]]
    for i in range(int(n * 2)):
        if i < n:
            v.extend([[l1, u1]])
        else:
            v.extend([[l2, u2]])
    v = v[1:]
    return v

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

def shifting(index,another_index,a):
    v = a[index]
    a = np.delete(a, index)
    a = np.insert(a, int(another_index), v)
    return a

def replace_negatives_with_zeroes(vector):
    """Replace negative values in a vector with 0."""
    return [max(0, x) for x in vector]

def repeat_vector_elements(vec, num_repeats):
    repeated_vec = []
    for elem in vec:
        repeated_vec += [elem] * num_repeats
    return repeated_vec


