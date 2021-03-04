import yaml
import math as ma
from scipy.stats import spearmanr
from collections import defaultdict

MEASURES_LIST = ['fisher', 'grad_norm', 'grasp', 'jacob_cov', 'plain', 'snip', 'synflow', 'synflow_bn']





if __name__ == '__main__':
    
    filename = '.\\precomputed_results\\nasbench201_cifar10_subset_all_results.yaml'
    
    with open(filename, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)

    

    spe_top_percents = defaultdict(list)

    for measure in MEASURES_LIST:
        reg_measure = [(res['testacc'], res['logmeasures'][measure]) for res in data]
        reg_measure.sort(key=lambda x: x[0], reverse=True)

        top_percents = []
        top_percent_range = range(2, 101, 2)

        for top_percent in top_percent_range:
            top_percents.append(top_percent)
            num_to_keep = int(ma.floor(len(reg_measure) * top_percent * 0.01))
            top_percent_evals = reg_measure[:num_to_keep]
            top_percent_reg = [x[0] for x in top_percent_evals]
            top_percent_measure = [x[1] for x in top_percent_evals]

            spe_measure, _ = spearmanr(top_percent_reg, top_percent_measure)
            spe_top_percents[measure].append(spe_measure)


        spe_top_percents['top_percents'] = top_percents

    print('rest')     