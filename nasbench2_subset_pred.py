# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pickle
import torch
import argparse
from scipy.stats import spearmanr
import yaml

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201 subset')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true', help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    if args.noacc:
        api = pickle.load(open(args.api_loc,'rb'))
    else:
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)
    
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)

    cached_res = []
    pre='cf' if 'cifar' in args.dataset else 'im'
    pfn=f'nb2_{pre}{get_num_classes(args)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}.p'
    op = os.path.join(args.outdir,pfn)

    
    args.end = len(api) if args.end == 0 else args.end


    # list of archs to compute measures over
    all_archs = [6983, 14708, 1891, 6083, 5727, 10259, 11869, 4336, 5737, 8242, 1759, 12447, 11686, 1604, 8184, 288, 3352, 4026, 6531, 13745, 7319, 3435, 10659, 10850, 9427, 3972, 14392, 8047, 7118, 2491, 4750, 9167, 8039, 13941, 7787, 9438, 9110, 4555, 1835, 9117, 14584, 12112, 9233, 905, 4185, 6744, 9522, 11247, 411, 2778, 6781, 3370, 1032, 11208, 11033, 11652, 2224, 2047, 7572, 5031, 3395, 13911, 9045, 7113, 6050, 2373, 13971, 9650, 13273, 9682, 1944, 1809, 7170, 767, 10236, 3267, 11786, 8702, 3444, 13017, 5557, 14710, 7624, 11751, 6390, 2989, 8246, 8980, 13087, 27, 6607, 9580, 12749, 11341, 13912, 11180, 15543, 7761, 9102, 9509, 9677, 362, 6272, 3079, 13105, 8206, 13708, 191, 10218, 12165, 12613, 920, 6014, 5400, 8279, 13784, 307, 14341, 8797, 11173, 12631, 11363, 7723, 6708, 6918, 9906, 2903, 13869, 15135, 9594, 3088, 9455, 4360, 3201, 7404, 8504, 7933, 7545, 11098, 1554, 7939, 11503, 5637, 3140, 7982, 433, 15460, 12101, 9801, 15617, 6149, 2181, 8306, 15468, 14259, 13431, 2170, 7630, 5541, 5333, 14923, 5828, 5778, 771, 12138, 2924, 59, 14727, 6049, 7341, 13630, 13899, 8200, 13692, 5679, 14861, 14815, 9802, 11801, 2803, 2082, 6621, 1998, 3807, 9691, 7361, 15550, 13775, 851, 6056, 12596, 2884, 4477, 10417, 14455, 10915, 15094, 6180, 5212, 6323, 12684, 15180, 6501, 6890, 5875, 3572, 14526, 12273, 6440, 9078, 1285, 4368, 7383, 9197, 10560, 15439, 3827, 7060, 14000, 12244, 4398, 2685, 7481, 12825, 15299, 6656, 4824, 5903, 11735, 3810, 6137, 11185, 8956, 12813, 1567, 11666, 4478, 12066, 2315, 8517, 2950, 10669, 8096, 10638, 10805, 14524, 14179, 3330, 7045, 2573, 10152, 11851, 4299, 2980, 5099, 6651, 13137, 1902, 1212, 4704, 4522, 3760, 273, 1235, 8407, 8016, 7192, 4429, 3298, 8339, 11255, 7453, 8739, 5756, 3428, 2795, 2729, 13000, 5732, 9962, 658, 11422, 8464, 3455, 15201, 7885, 9053, 5976, 7257, 9598, 3070, 11672, 14113, 21, 12330, 7670, 6608, 3566, 13006, 10356, 7962, 4875, 7656, 7976, 7912, 4099, 6269, 11320, 10035, 3874, 7817, 3829, 11576, 2805, 2377, 8315, 13182, 6214, 7888, 10509, 8877, 1879, 1086, 7977, 3218, 13244, 738, 5122, 8927, 4961, 5367, 2829, 1247, 8994, 8723, 9297, 14924, 3025, 6689, 6605, 14695, 232, 14521, 9323, 12750, 7265, 11262, 3405, 12794, 7949, 5624, 1697, 11802, 3809, 7369, 800, 7650, 2447, 5299, 3803, 190, 9389, 10900, 12754, 13837, 4748, 9186, 7548, 6421, 8657, 11892, 13922, 4159, 11265, 6424, 14754, 1252, 5864, 8146, 9373, 4161, 15541, 8672, 14525, 12957, 14855, 2382, 2309, 7076, 1334, 779, 6986, 3826, 9065, 11874, 10706, 4695, 4686, 13267, 3648, 2844, 12055, 5874, 1528, 2524, 12857, 256, 10265, 8400, 6115, 5348, 4599, 13247, 11399, 15310, 3543, 1430, 4253, 11793, 247, 6413, 12083, 11181, 8864, 2124, 479, 12129, 11743, 15535, 3851, 2640, 10628, 465, 9060, 3415, 7822, 15168, 6490, 14886, 15072, 7440, 15312, 10472, 9397, 10621, 6081, 2818, 5373, 15420, 2348, 9019, 8625, 14961, 15044, 9278, 8011, 4286, 13012, 12213, 14768, 4340, 237, 3684, 12234, 14887, 2559, 5695, 4318, 9903, 14663, 770, 8043, 4699, 10133, 12939, 8614, 860, 967, 14671, 2283, 88, 9704, 14057, 15417, 3953, 14301, 14422, 14472, 5191, 14105, 14730, 4666, 13601, 405, 1510, 11715, 13319, 9932, 6789, 3512, 13861, 7615, 12995, 9855, 3197, 680, 7766, 3514, 4814, 2821, 8057, 3050, 4822, 14086, 12346, 4861, 755, 1089, 1749, 3933, 5957, 1620, 4805, 268, 12627, 6211, 8663, 13140, 12583, 7673, 12497, 1213, 10794, 13780, 4357, 12109, 3091, 11603, 4852, 12291, 10991, 9318, 6091, 14036, 14596, 7179, 7561, 2518, 13257, 1517, 1126, 7094, 901, 487, 8051, 4123, 4295, 13032, 6022, 1307, 6219, 2469, 6080, 6017, 11090, 5678, 8070, 10442, 3602, 3563, 13637, 1778, 10287, 14690, 6955, 6062, 6363, 14609, 977, 12609, 1029, 10297, 9152, 7529, 3258, 10771, 1518, 2417, 3273, 11205, 4973, 4678, 9170, 4499, 10343, 14581, 10368, 9499, 5579, 5609, 11569, 3587, 10789, 13972, 352, 4399, 2662, 14781, 14478, 13292, 4372, 14184, 8249, 15208, 4652, 7541, 13658, 2621, 8758, 3015, 11726, 15513, 14247, 5462, 6204, 13441, 7129, 14465, 9597, 5042, 13483, 10195, 246, 6995, 6034, 10010, 4948, 1640, 4130, 3086, 15503, 8357, 4638, 6839, 2838, 5359, 4575, 11637, 5262, 2023, 11675, 15161, 12147, 9920, 4174, 11190, 4210, 3484, 6597, 7425, 3559, 1052, 6122, 15578, 7225, 13851, 11438, 12412, 4267, 7695, 2175, 5175, 13347, 1355, 938, 8326, 15559, 6538, 2739, 6898, 9963, 10834, 1708, 11298, 10153, 4657, 3931, 11639, 13926, 12495, 6320, 4417, 2789, 3870, 12201, 12608, 9098, 4679, 8817, 9087, 2326, 7007, 3982, 9137, 2957, 11129, 5550, 11279, 12560, 979, 270, 13031, 1067, 867, 5159, 3204, 8729, 6276, 4618, 6578, 2163, 8006, 986, 5656, 1062, 2509, 15461, 3906, 10955, 4184, 4820, 8601, 818, 7974, 5486, 14090, 2458, 3046, 13386, 5885, 1859, 5502, 911, 13126, 12658, 10128, 12931, 3207, 13510, 5494, 15526, 8304, 9461, 3838, 6850, 11322, 8713, 3491, 12691, 12400, 12178, 6999, 1674, 13812, 12907, 3060, 12520, 12015, 13455, 13632, 5176, 747, 2238, 10043, 10007, 13827, 11485, 6823, 14319, 12766, 12985, 9927, 8520, 4163, 5529, 9352, 13619, 15551, 10038, 5760, 3398, 7585, 4701, 9928, 787, 8084, 1381, 1026, 7223, 2945, 7281, 13364, 6785, 2899, 5300, 11700, 6917, 13968, 4439, 2199, 12391, 4019, 12418, 12639, 8604, 12443, 2670, 9079, 11755, 9246, 4157, 8948, 13638, 3077, 7710, 259, 8095, 7706, 11101, 9940, 13287, 3759, 4914, 8003, 13109, 10464, 15238, 15392, 417, 15556, 7135, 2296, 5168, 9242, 2490, 10177, 13045, 13366, 11993, 10654, 3341, 14190, 11272, 9852, 796, 8819, 5813, 5647, 10056, 9215, 2209, 3618, 11474, 7622, 1184, 7894, 603, 2928, 5951, 11804, 4989, 5455, 4568, 3170, 1332, 5250, 10515, 2302, 4690, 571, 864, 4606, 7607, 9595, 1017, 5096, 14930, 12830, 8963, 5154, 484, 13022, 5595, 14404, 14234, 15504, 9391, 4886, 7896, 14993, 2954, 7726, 2955, 2708, 5752, 11432, 13363, 5102, 10618, 11126, 727, 5698, 6534, 12879, 11253, 9343, 15052, 2407, 7255, 11976, 14103, 10945, 5508, 12270, 641, 1362, 12432, 13539, 15364, 9915, 14811, 13359, 3728, 12548, 4441, 14082, 4154, 5499, 13479, 13365, 3638, 5959, 159, 3550, 10182, 13323, 9559, 1087, 14223, 4062, 14717, 10087, 14130, 11430, 630, 1351, 3365, 3303, 2958, 3353, 7441, 14283, 7563, 12974, 2354, 5052, 7753, 10273, 11730, 2235, 12788, 5658, 8231, 6226, 1602, 9631, 3806, 2609, 5479, 4967, 8835, 1574, 6954, 4351, 2041, 9931, 8744, 7104, 14354, 1445, 6565, 4946, 11687, 10718, 6909, 1668, 12027, 4585, 3995, 11048]

    #loop over nasbench2 archs
    for i, arch_str in enumerate(api):

        # only compute over the subset
        if i not in all_archs:
            continue

        if i < args.start:
            continue
        if i >= args.end:
            break 

        res = {'i':i, 'arch':arch_str}

        net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
        net.to(args.device)

        init_net(net, args.init_w_type, args.init_b_type)
        
        arch_str2 = nasbench2.get_arch_str_from_model(net)
        if arch_str != arch_str2:
            print(arch_str)
            print(arch_str2)
            raise ValueError

        measures = predictive.find_measures(net, 
                                            train_loader, 
                                            (args.dataload, args.dataload_info, get_num_classes(args)),
                                            args.device)

        res['logmeasures']= measures

        if not args.noacc:
            info = api.get_more_info(i, 'cifar10-valid' if args.dataset=='cifar10' else args.dataset, iepoch=None, hp='200', is_random=False)

            trainacc = info['train-accuracy']
            valacc   = info['valid-accuracy']
            testacc  = info['test-accuracy']
        
            res['trainacc']=trainacc
            res['valacc']=valacc
            res['testacc']=testacc
        
        #print(res)
        cached_res.append(res)

        # write raw results to file for post processing
        if i % args.write_freq == 0 or i == len(api)-1 or i == 10:
            savename = f'nasbench201_{args.dataset}_subset_all_results.yaml'
            with open(savename, 'w') as f: 
                yaml.dump(cached_res, f)        


    # compute spearman's correlation between measures 
    measures_list = ['grad_norm', 'snip', 'fisher', 'grasp', 'jacob_cov', 'plain', 'synflow_bn', 'synflow']

    evals_dict = {}

    # gather test accuracies
    all_reg_evals = [res['testacc'] for res in cached_res]
    
    # gather measures
    for m in measures_list:
        m_evals = [res['logmeasures'][m] for res in cached_res]
        evals_dict[m] = m_evals

    spes_dict = {}    
    for m in measures_list:
        assert len(all_reg_evals) == len(evals_dict[m])
        spe, _ = spearmanr(all_reg_evals, evals_dict[m])
        spes_dict[m] = spe
        print(f'{m} measure spe: {spe}')
    
    savename = f'nasbench201_{args.dataset}_subset_overall.yaml'
    with open(savename, 'w') as f:
        yaml.dump(spes_dict, f)
