import yaml
import math as ma
from scipy.stats import spearmanr
from collections import defaultdict

MEASURES_LIST = ['fisher', 'grad_norm', 'grasp', 'jacob_cov', 'plain', 'snip', 'synflow', 'synflow_bn']


if __name__ == '__main__':
    
    filename = '.\\precomputed_results\\nasbench201_ImageNet16-120_subset_all_results.yaml'
    
    with open(filename, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)


    top_percent_range = range(2, 101, 2)

    # rank correlation vs top percent of architectures
    # -------------------------------------------------
    spe_top_percents = defaultdict(list)

    for measure in MEASURES_LIST:
        reg_measure = [(res['testacc'], res['logmeasures'][measure]) for res in data]
        reg_measure.sort(key=lambda x: x[0], reverse=True)

        top_percents = []
        

        for top_percent in top_percent_range:
            top_percents.append(top_percent)
            num_to_keep = int(ma.floor(len(reg_measure) * top_percent * 0.01))
            top_percent_evals = reg_measure[:num_to_keep]
            top_percent_reg = [x[0] for x in top_percent_evals]
            top_percent_measure = [x[1] for x in top_percent_evals]

            spe_measure, _ = spearmanr(top_percent_reg, top_percent_measure)
            spe_top_percents[measure].append(spe_measure)


        spe_top_percents['top_percents'] = top_percents

    # overlap in top x% of architectures between method and groundtruth
    # ------------------------------------------------------------------
    cr_top_percents = defaultdict(list)

    arch_id_reg_evals = [(res['i'], res['testacc']) for res in data]
    arch_id_reg_evals.sort(key=lambda x: x[1], reverse=True)
    
    for measure in MEASURES_LIST:
        arch_id_measure_evals = [(res['i'], res['logmeasures'][measure]) for res in data]
        arch_id_measure_evals.sort(key=lambda x: x[1], reverse=True)

        assert len(arch_id_reg_evals) == len(arch_id_measure_evals)

        top_percents = []

        measure_ratio_common = []
        
        for top_percent in top_percent_range:
            top_percents.append(top_percent)
            num_to_keep = int(ma.floor(len(arch_id_reg_evals) * top_percent * 0.01))
            top_percent_arch_id_reg_evals = arch_id_reg_evals[:num_to_keep]
            top_percent_arch_id_measure_evals = arch_id_measure_evals[:num_to_keep]

            # take the set of arch_ids in each method and find overlap with top archs
            set_reg = set([x[0] for x in top_percent_arch_id_reg_evals])
            set_measure = set([x[0] for x in top_percent_arch_id_measure_evals])
            measure_num_common = len(set_reg.intersection(set_measure))
            cr_top_percents[measure].append(measure_num_common/num_to_keep)

        cr_top_percents['top_percents'] = top_percents

    
    # save raw data for other aggregate plots over experiments
    raw_data_dict = {}
    raw_data_dict['top_percents'] = spe_top_percents['top_percents']

    for measure in MEASURES_LIST:
        raw_data_dict[measure+'_spe'] = spe_top_percents[measure]
        raw_data_dict[measure+'_ratio_common'] = cr_top_percents[measure]

    savename = '.\\precomputed_results\\raw_data.yaml'
    with open(savename, 'w') as f:
        yaml.dump(raw_data_dict, f)