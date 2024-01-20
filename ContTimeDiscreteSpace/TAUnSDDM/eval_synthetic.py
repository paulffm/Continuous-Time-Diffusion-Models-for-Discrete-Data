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