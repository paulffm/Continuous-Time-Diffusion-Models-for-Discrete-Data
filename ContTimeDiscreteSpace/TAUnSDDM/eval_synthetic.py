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
    
    """
    save_location = os.path.join(script_dir, "SavedModels/SyntheticMasked/")
    date = '2023-12-17' # 2
    config_name = 'config_001_masked.yaml' # config_001_hollowMLEProb.yaml
    model_name = 'model_199999_masked.pt'
    """
    #config_name = 'config_001_r07.yaml' 
    #model_name = 'model_84999_hollowr07.pt' 
    config_path = os.path.join(save_location, date, config_name)
    checkpoint_path = os.path.join(save_location, date, model_name)

    # creating models
    cfg = bookkeeping.load_config(config_path)
    cfg.sampler.name = 'ElboLBJF' #'ExactSampling' # ElboLBJF CRMTauL CRMLBJF
    cfg.sampler.num_corrector_steps = 0
    cfg.sampler.corrector_entry_time = ScalarFloat(0.0)
    cfg.sampler.num_steps = 100 #750
    cfg.sampler.is_ordinal = False

    num_steps = [20, 50, 100, 200, 500]
    num_sampler = ['CRMTauL', 'CRMLBJF', 'ExactSampling']
    num_mmd = []

    #for sampler_i in num_sampler:
    #    cfg.sampler.name = sampler_i
    #    for i in num_steps:
        #print(cfg)
    #        cfg.sampler.num_steps = i

            
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
    #num_mmd.append(mmd.item())
    print("MMD", mmd.item())
    #    print("Sampler:", sampler_i)
    #    print("Num steps:", num_steps)
    #    print("Num mmd", num_mmd)

    #print("Sampler ges:", num_sampler)
    #print("Num steps ges :", num_steps)
    #print("Num mmd ges", num_mmd)
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
# TauL: pos MMD: 0.0002686732041183859 neg MMD: -0.00011882558465003967 MMD: 0.0001446735841454938 pos n_rounds: 17/25
# LBJF: pos MMD: 0.00017342269711662084 neg MMD: -0.00010647327144397423 MMD: 6.146430678199977e-05 pos n_rounds: 30/50
# Direct 199 999:
# TauL: pos MMD: 0.00011419731890782714 neg MMD: -0.00013799247972201556 MMD: -2.1985173589200713e-05 pos n_rounds: 23
# pos MMD: 0.00017084764840546995 neg MMD: -0.00013665799633599818 MMD: 4.794597316504223e-06 pos n_rounds: 23
    
# LBJF: pos MMD: 0.00013810396194458008 neg MMD: -0.00011365055979695171 MMD: 1.2226700164319482e-05 pos n_rounds: 25
# pos MMD: 0.00019979715580120683 neg MMD: -0.00013425826909951866 MMD: 3.276943971286528e-05 pos n_rounds: 25
    
# RevProb: 100 Steps 9999 Train
# TauL: pos MMD: 0.0001316326088272035 neg MMD: -6.437843694584444e-05 MMD: 4.5387743739411235e-05 pos n_rounds: 14/25
# pos MMD: 0.0001843631180236116 neg MMD: -0.0001115977720473893 MMD: 3.638267298811115e-05
# pos MMD: 0.0001400125038344413 neg MMD: -0.00011733770224964246 MMD: 1.1337398973410018e-05 pos n_rounds: 25
    
# LBJF:  pos MMD 0.0002092393988277763 neg MMD -0.00011471979087218642 MMD 2.782225601549726e-05 pos n_rounds 11/25
# pos MMD: 0.0001962311362149194 neg MMD: -0.00011041585094062611 MMD: 9.197116014547646e-05 
# pos MMD: 0.0001225145097123459 neg MMD: -0.00013166152348276228 MMD: -2.490758924977854e-05 pos n_rounds: 21
# Exact: pos MMD: 0.00014404380635824054 neg MMD: -0.00011983017611782998 MMD: 1.7384290913469158e-05 pos n_rounds: 26
    
# RevProb 199 999
# TauL: pos MMD: 9.041598968906328e-05 neg MMD: -0.0001231956121046096 MMD: -3.347873644088395e-05
# LBJF:pos MMD: 0.00013347384032793343 neg MMD: -0.00012228958075866103 MMD: -9.5367431640625e-07 pos n_rounds: 22
# Analytical: pos MMD: 0.0001468531947163865 neg MMD: -0.00010747519991127774 MMD: 2.90024272544e-05 24

# Bert
# TauL: pos MMD: 0.00013466611123178154 neg MMD: -8.65033725858666e-05 MMD: 9.433865488972515e-05 pos n_rounds: 18/25
#pos MMD: 0.00019409770902711898 neg MMD: -9.962445619748905e-05 MMD: 8.248328958870843e-05 pos n_rounds: 31
# TauL ohne p0t: pos MMD: 0.00013659503019880503 neg MMD: -0.00013623635459225625 MMD: 7.111549348337576e-05 pos n_rounds: 19
# LBJF: 0.0007480800268240273

# Masked prob 199999
# Taul:pos MMD: 0.00014541375276166946 neg MMD: -0.00014963462308514863 MMD: 2.149343526980374e-05 pos n_rounds: 29
# Taul Last: pos MMD: 0.00018206665117759258 neg MMD: -0.0001321311719948426 MMD: -1.6808509428756224e-07 pos n_rounds: 21
# LBJF: pos MMD: 0.00011410936713218689 neg MMD: -0.000153273344039917 MMD: -2.492964267730713e-05 pos n_rounds: 24
# Euler Last: pos MMD: 0.00012914887338411063 neg MMD: -0.0001249299239134416 MMD: 2.2435784558183514e-05 pos n_rounds: 29
# Exact: pos MMD: 0.00011911988258361816neg MMD: -0.00014259065210353583 MMD: -4.314064790378325e-05 pos n_rounds: 19

# Direct
# TauL: pos MMD: 0.00020308936655055732 neg MMD: -0.00016237089585047215 MMD: 6.42144659650512e-05 pos n_rounds: 31
# TauL: Last: pos MMD: 0.00016415714344475418 neg MMD: -0.00014098762767389417 MMD: 1.158475879492471e-05 pos n_rounds: 25
# Euler last: pos MMD: 0.0001515140465926379 neg MMD: -0.00013864981883671135 MMD: 4.1251776565331966e-05 pos n_rounds: 31
# Elbo: pos MMD: 0.00020035043417010456 neg MMD: -0.00012555404100567102 MMD: 7.65067306929268e-05 pos n_rounds: 31
    
# Hollow implicit Elbo:
# TauL: pos MMD: 0.00012698621139861643 neg MMD: -0.00015479724970646203 MMD: -4.208385871606879e-05 pos n_rounds: 20
# LBJF: pos MMD: 0.00021080255100969225 neg MMD: -0.0001658046239754185 MMD: 2.2498965336126275e-05 pos n_rounds: 25
# Exact: pos MMD: 0.00017154050874523818 neg MMD: -0.0001279190182685852 MMD: -2.0113586288061924e-05 pos n_rounds: 18
    
# LBJF besser bei Hollow prob
# Tau besser bei Hollow dirct
# insgesamt hollow prob besser:
#----------------------------------
# RevProb: 
# LBJF: 0.002525
# TauL: 0.002634
# ExactSampling: 0.00274


# 0.005364969838410616
# 0.0029045911505818367

# Direct:
# LBJF: 0.0026941706892102957
# TauL: pos MMD: 0.002701888093724847
    
# 10, 20
# 0.
    


# Masked:
# LBJF: pos MMD: 0.0027752763126045465
# TauL: pos MMD: 0.0026900090742856264
# 0.002598656341433525