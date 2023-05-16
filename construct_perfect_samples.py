import numpy as np

from utils.eval_utils import construct_perfect_sample, framewise_overlap_areas

if __name__ == '__main__':
    OA_means = np.array([0.796, 0.828])
    OA_vars = np.array([0.01437, 0.01396])

    samples = construct_perfect_sample(OA_means, OA_vars, 1000)
    OAs = np.array([framewise_overlap_areas(s) for s in samples])
    #OAs = np.array([np.random.normal(OA_means[i], np.sqrt(OA_vars[i]), 10) for i in [0, 1]]).T
    consistency = 1 - np.abs(OA_means - OAs.mean(0)) / OA_means
    variance = 1 - np.abs(OA_vars - OAs.var(0)) / OA_vars

    print(consistency, variance)
