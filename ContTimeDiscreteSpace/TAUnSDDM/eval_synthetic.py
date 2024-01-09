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
    
    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-25' # 2
    config_name = 'config_001_maskedelbo.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_maskedelbo.pt'

    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_maskeddirect.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_maskeddirect.pt'

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
    cfg.sampler.num_steps = 200 #750
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
    print("Sampler:", cfg.sampler.name)
    n_samples = 1024 # 16384  #1024
    n_rounds = 25
    mmd = eval_mmd(cfg, state['model'], sampler, dataloader, n_rounds, n_samples=n_samples)
    #num_mmd.append(mmd.item())
    print("MMD", mmd.item())

if __name__ == "__main__":
    main()

# Hollow Prob:
# LBJF: 
# TauL: 9.0085 10^-5 9.153992868959904e-05
# Exact: 10.9 * 10^-5 9.803791181184351e-05
    
# Hollow Direct
# LBJF: 9.13636467885226e-05 0.00011826169065898284
# TauL: 10 9.22

# Hollow Prob Elbo:
# TauL: 0.00010263950389344245 9.1402223915793e-05
# LBJF: 0.0001028686310746707 8.924536086851731e-05
# Exact: 0.00014057054067961872  0.00010845297947525978
    
# Bert:
# TauL: 0.00012461026199162006 0.00012860528659075499
# LBJF: 0.00015199881454464048 0.00012528669321909547
    
# Hollow Masked Prob:
# LBJF:0.00034483475610613823 0.00042633572593331337 8.5
# TauL: 0.0004098012577742338 0.00040654430631548166 9.07
# Exact: 0.00034637871431186795 0.00038752591353841126 8.14
    
# Hollow Masked Direct:
# LBJF: 0.0003944083000533283 0.00037753558717668056 8.56
# TauL; 0.00040866935160011053 MMD 0.00043422437738627195 9.36

# Hollow Masked Elbo:
# TauL: MMD 0.0004588830634020269 0.0005090999184176326
# LBJF: 0.00043064355850219727 0.00043451989768072963
# Exact:  MMD 0.00048021896509453654 MMD 0.00040368011104874313
    
# Sampler Study:
# steps = np.array([10, 20, 30, 50, 100, 250, 500])
# LBJF: 500: 
#       250: 10.3 8.58 8.82 7.67 = 8.84
#       100: 9.68 11 9.07 9.63 = 9.845 => 9.54
#        75: 8.92 8.95 10.32 9.97 = 9.54 => 9.56
#        50: 11 9.17 10 8.1 = 9.56 => 9.845
#        30: 9.95 11.15 12 11.2 = 11.075
#        20: 12.74 12.5 11.7 15.4 = 13.085
#        10: 23.06 19.7 19.2 19.2 = 20.29
# mmd_taul = np.array([19.53, , ])
# LBJF: 500: 
#       250:
#       100: 8.8
#        50: 
#        30: 
#        20: 16 11 8.77 13 
#        10: 10 16 22 16 20:18
# mmd_lbjf = np.array([])
#exact: 500: 
#       250:
#       100:
#        50:
#        30:
#        20: 20: 12-4
#        10: 20 27 20 25 # 20: 16.7
# mmd_lbjf = np.array([])