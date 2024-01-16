import torch
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from ruamel.yaml.scalarfloat import ScalarFloat
from config.synthetic_config.config_tauMLP_synthetic import get_config
from torch.utils.data import DataLoader
from lib.datasets import mnist, maze, protein, synthetic
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.dataset_utils as dataset_utils
from lib.datasets.metrics import eval_mmd

def main():
    
    script_dir = os.path.dirname(os.path.realpath(__file__))


    # creating paths
    """
    save_location = os.path.join(script_dir, "SavedModels/SyntheticRMDirect/")
    date = '2023-12-20' # 2023-10-30 'Hollow-2023-10-29'
    config_name = 'config_001_hollowCEDirect500K.yaml' # 'config_001_maze.yaml' 'config_001_rate001.yaml'
    model_name = 'model_199999_hollowCEDirect500K.pt' # 'model_55999_rate001.pt' 'model_5999_maze.pt'
    
    
    save_location = os.path.join(script_dir, "SavedModels/Synthetic/")
    date = '2023-12-20' # 2
    config_name = 'config_001_hollowCEProb500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_hollowCEProb500K.pt'

    """
    save_location = os.path.join(script_dir, "SavedModels/SyntheticBert/")
    date = '2023-12-20' # 2
    config_name = 'config_001_bert500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_bert500K.pt'

    save_location = os.path.join(script_dir, "SavedModels/Synthetic/")
    date = '2023-12-20' # 2
    config_name = 'config_001_hollowCEProb500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_hollowCEProb500K.pt'

    save_location = os.path.join(script_dir, "SavedModels/SyntheticRMDirect/")
    date = '2023-12-20' # 2023-10-30 'Hollow-2023-10-29'
    config_name = 'config_001_hollowCEDirect500K.yaml' # 'config_001_maze.yaml' 'config_001_rate001.yaml'
    model_name = 'model_199999_hollowCEDirect500K.pt' 

    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_masked.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_masked.pt'

  
    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_maskeddirect.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_maskeddirect.pt'
    

    save_location = os.path.join(script_dir, "SavedModels/SyntheticBert/")
    date = '2023-12-28' # 2
    config_name = 'config_001_bert500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_bert500K.pt'

    save_location = os.path.join(script_dir, "SavedModels/SyntheticRMDirect/")
    date = '2023-12-20' # 2023-10-30 'Hollow-2023-10-29'
    config_name = 'config_001_hollowCEDirect500K.yaml' # 'config_001_maze.yaml' 'config_001_rate001.yaml'
    model_name = 'model_199999_hollowCEDirect500K.pt' 


    save_location = os.path.join(script_dir, "SavedModels/Synthetic/")
    date = '2023-12-18' # 2
    config_name = 'config_001_hollowelbo.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_hollowelbo.pt'



    save_location = os.path.join(script_dir, "SavedModels/SyntheticBert/")
    date = '2023-12-28' # 2
    config_name = 'config_001_bert500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_bert500K.pt'



    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_masked.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_masked.pt'


    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_masked.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_masked.pt'

    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_maskeddirect.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_maskeddirect.pt'

    save_location = os.path.join(script_dir, "SavedModels/Synthetic/")
    date = '2023-12-20' # 2
    config_name = 'config_001_hollowCEProb500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_hollowCEProb500K.pt'


    #config_name = 'config_001_r07.yaml' 
    #model_name = 'model_84999_hollowr07.pt' 
    config_path = os.path.join(save_location, date, config_name)
    checkpoint_path = os.path.join(save_location, date, model_name)

    # creating models
    cfg = bookkeeping.load_config(config_path)
    cfg.sampler.name = 'CRMTauL' #'ExactSampling' # ElboLBJF CRMTauL CRMLBJF
    cfg.sampler.num_corrector_steps = 0
    cfg.sampler.corrector_entry_time = ScalarFloat(0.0)
    cfg.sampler.num_steps = 5 #750
    cfg.sampler.is_ordinal = False

    num_steps = [20, 50, 100, 200, 500]
    num_sampler = ['CRMTauL', 'CRMLBJF', 'ExactSampling']
    num_mmd = []

    #for sampler_i in num_sampler:
    #    cfg.sampler.name = sampler_i
    #    for i in num_steps:
        #print(cfg)
    #        cfg.sampler.num_steps = i
    cfg.device = 'cuda'
            
    device = torch.device(cfg.device)

    model = model_utils.create_model(cfg, device)
    print("number of parameters: ", sum([p.numel() for p in model.parameters()]))

    #modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
    #model.load_state_dict(modified_model_state)
    #optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)
    optimizer = torch.optim.Adam(model.parameters(), cfg.optimizer.lr)

    sampler = sampling_utils.get_sampler(cfg)

    state = {"model": model, "optimizer": optimizer, "n_iter": 0}
    state = bookkeeping.load_state(state, checkpoint_path)
    state['model'].eval()

    dataset_location = os.path.join(script_dir, cfg.data.location)

    dataset = dataset_utils.get_dataset(cfg, device, dataset_location)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle)
    print("Model Name:", model_name)
    print("Sampler:", cfg.sampler.name)
    n_samples = 4096#2048 #4096 # 16384  #1024
    n_rounds = 25
    mmd = eval_mmd(cfg, state['model'], sampler, dataloader, n_rounds, n_samples=n_samples)
    #num_mmd.append(mmd.item())
    print("MMD", mmd.item())

if __name__ == "__main__":
    main()

# TauL: 5: 7 7: 6 5.357915870263241e-05 15: 5.31 20: 5.09023702761624e-05
# LBJF: 5: 0.0004332971584517509 10: 0.00011 15: 5.818
# Exact:5:0.00012 7: 8.9 10: 4.81 15: 5.57 20:


# Hollow Prob:
# LBJF: 2.090136331389658e-05 (6) 1.6812768080853857e-05 (7)
# TauL: 1.515944859420415e-05 (6) 1.4381749679159839e-05 (7)
# Exact: 1.67
    
# Hollow Direct
# LBJF: 2.1502375602722168e-05 (8) 1.7217227650689892e-05 (8)
# TauL: 2.4476223188685253e-05 (7) 1.1635678674792871e-05 (7)

# Hollow Prob Elbo:
# TauL: 2.0287930965423584e-05 1.678466833254788e-05 (10)
# LBJF: 2.2509269911097363e-05 2.2564083337783813e-05 (8)
# Exact: 2.319738268852234e-05 2.3145568775362335e-05
    
# Bert:
# TauL: 5.522172068594955e-05 4.608830204233527e-05
# LBJF: 5.2543484343914315e-05 (15)  4.1333834815304726e-05
    
# Hollow Masked Prob:
# LBJF:1.1980533599853516e-05 (1) 2.5756657123565674e-05 (4)
# TauL:  1.5117228031158447e-05
# Exact: 1.0967254638671875e-05 (1)
    
# Hollow Masked Direct:
# LBJF: 2.9295682907104492e-05 (2) 5.730135308112949e-05 (7)
# TauL; 4.2401254177093506e-05 (4)

# Hollow Masked Elbo:
# TauL: 5.631893873214722e-05
# LBJF: 5.0887465476989746e-05 (8)
# Exact: 1.3649463653564453e-05 (4) 6.712675531161949e-05 (5)
    
# Sampler Study:
# steps = np.array([10, 20, 30, 50, 100, 250, 500])
# LBJF: 500: 
#       250: 1.6327416233252734e-05 (14) 2.5535624445183203e-05 (12) 2.1526448108488694e-05 (13) 1.7970800399780273e-05 (7)  2.5190911401296034e-05 (15) 2.63 (8)
#       100:  2.4178623789339326e-05 (10) 2.130269967892673e-05 (10)
#        75: 
#        50: 2.242293703602627e-05 (18) 2.5032295525306836e-05 (18) ##### 2.016127109527588e-05 (12) 2.017158840317279e-05 (13)
#        30: 2.1900981664657593e-05 (16) 2.174718065361958e-05 (14)  2.7424759537097998e-05 (18) 2.03 (13)
#        20: 3.191044379491359e-05(19) 3.637179179349914e-05 
#        10: 0.00010606398427626118 (24) 0.00011 0.00012 (both)
# mmd_taul = np.array([1.1, 0.35, 0.25, 0.21, 0.18, 0.153, 0.122])
# TauL: 500: 
#       250: 3.703344918903895e-05 (11) 2.3318661988014355e-05 (9) 1.7970800399780273e-05(10)  2.1526448108488694e-05(13)
#       100:  2.0078638044651598e-05 (11) 2.138889794878196e-05(13) 2.2402831746148877e-05
#        50: 2.2658705347566865e-05 (10) 1.910825631057378e-05 (12) ###  1.278790568903787e-05 (11) 2.1581139662885107e-05 (14)
#        30: 2.208352270827163e-05 (14) 1.8802973499987274e-05 (13) ### 1.7121434211730957e-05 (10) 2.333273550902959e-05 (12)
#        20: 2.136826515197754e-05 (16) 2.6285648345947266e-05 (17) Nochmal neu 3.9149075746536255e-05 3.052751344512217e-05 3.3215565053978935e-05(17) 4.148833977524191e-05(17)
#        10: 0.00015331625763792545 (25) 0.0001 0.00015230178541969508 (25) 0.00016035675071179867 (25)
# mmd_lbjf = np.array([1.6, 3.48, 0.22, 0.19, 0.17, 0.162, 0.151])
#exact: 500: 
#       250: 1.371900270896731e-05 (12) 1.6324222087860107e-05 (8) 1.8229708075523376e-05 (16) 1.888615770440083e-05 (14) 
#       100: 2.143638630514033e-05 (7) 2.420825148874428e-05 (17) 1.7177600966533646e-05 (13) 1.8470562281436287e-05 (13)
#        50: 2.5057368475245312e-05 (14) 2.4858643882907927e-05 (17)
#        30: 1.879731826193165e-05 (15) 1.6927719116210938e-05 (11) 2.559026142989751e-05 (15) 2.02743449335685e-05 (17
#        20: 3.18058519042097e-05 (22) 2.7716161639546044e-05 (15) 2.2092461222200654e-05 (20) 3.158301115036011e-05 (16)
#        10: 5.816817065351643e-05 (25) 4.2921554268104956e-05 (24) 5.065401637693867e-05 (24)  5.3188799938652664e-05 (25)
# mmd_lbjf = np.array([0.5, 0.318, 0.228, 0.19, 0.1675, 0.152, 0.143])

# python eval_synthetic.py
# best Analytical, then Euler, then TauL mach ich einfach so
    
# EXACT: 0.00011 3.14235694531817e-05
# LBJF: 15: 4.764795085065998e-05
# Taul: 20: 4.758514114655554e-05 50: 4.1479866922600195e-05
    
# TauL: 5: 5.31, 7, 6 5.3 10: 4.43 (12) 4.45 (16) 3.92 (16) 15: 5.95 4 (13) 4.743039608001709e-05 (13) 4.64 (17) 20: 3.88 (14) 2.83 (14) 4.07