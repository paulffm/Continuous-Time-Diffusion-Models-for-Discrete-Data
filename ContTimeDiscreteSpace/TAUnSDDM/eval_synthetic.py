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
    model_name = 'model_9999_hollowCEDirect500K.pt' # 'model_55999_rate001.pt' 'model_5999_maze.pt'
    """

    #"""
    save_location = os.path.join(script_dir, "SavedModels/Synthetic/")
    date = '2023-12-20' # 2
    config_name = 'config_001_hollowCEProb500K.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_9999_hollowCEProb500K.pt'
    #"""

    #config_name = 'config_001_r07.yaml' 
    #model_name = 'model_84999_hollowr07.pt' 
    config_path = os.path.join(save_location, date, config_name)
    checkpoint_path = os.path.join(save_location, date, model_name)

    # creating models
    cfg = bookkeeping.load_config(config_path)
    cfg.sampler.name = 'ExactSampling' #'ExactSampling' # ElboLBJF CRMTauL CRMLBJF
    cfg.sampler.num_corrector_steps = 0
    cfg.sampler.corrector_entry_time = ScalarFloat(0.0)
    cfg.sampler.num_steps = 100 #750
    cfg.sampler.is_ordinal = False

    #print(cfg)
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
    n_samples = 1024
    n_rounds = 25
    mmd = eval_mmd(cfg, state['model'], sampler, dataloader, n_rounds, n_samples=n_samples)
    print("MMD", mmd)
if __name__ == "__main__":
    main()

# 500 Steps:
# Bert LBJF: 0.0008 10/10
# Bert TauL: 0.0002 0.0002 7/10
    
# 100 Steps
# Bert LBJF: 0.0008 10/10
# Bert TauL: 0.0002  7/10
    
# 50 Steps
# Bert LBJF: 0.0009 10/10
# Bert TauL: 0.0001  5/10

# 500 
# Hollow Direct LBJF: 5/10 0.0002 0.0001
# Hollow Direct TauL: 5/10 9.0557e-05 7/10 0.0002

# 50 Steps
# Hollow Direct LBJF: 0.0001 5/10 0.0001 4/10
# Hollow Direct TauL: 0.0002 8/10

# 20 Steps
# Hollow Direct LBJF: 0.0002 6/10 0.0002 6/10
# Hollow Direct TauL: 0.0001 4/10 0.0002 4/10

# 500 Steps
# Hollow p0t LBJF: 9.3952e-05 4/10 0.0002 6/10
# Hollow p0t TauL: 0.0002 5/10 0.0002 5/10
# Hollow p0t Exact: 0.0002 5/10 0.0002 4/10

# 20 Steps
# Hollow p0t LBJF: 0.0001 7/10 0.0001 7/10
# Hollow p0t TauL: 0.0002 9/10 0.0002 5/10
# Hollow p0t Exact: 0.0001 5/10 0.0001 6/10

# 100 steps
# Masked p0t LBJF: 
# Masked p0t TauL: 0.0001 6/10
# Masked p0t Exact: 

# 20 steps
# Masked p0t LBJF: 0.0001 4/10 0.0001 2/10
# Masked p0t TauL: 4.5056e-05 6/10 0.0001 8/10
# Masked p0t Exact: 5.5522e-05 2/10 9.5707e-05 5/10
    
# Very similiar: p0t and direct => therefore we will use for the following experiments 
# the p0t as objective, as we can utilize another sampling procedure
    
# TauL and Euler Sampling very similiar. We hypothetise since we are in a binary setting, multiple jumps are not meaningful 
    

# Direct: 100 Steps 9999
# TauL: 4: 0.0001 
# LBJF: 0.0001
    
# RevProb: 100 Steps 9999
# TauL 3: 0.0002; 2 0.00006
    
# 0.0002; 4: 0.000008
# TauL: 18/25: 0.0002
# LBJF: 32/50 0.0002 
# Exact: 