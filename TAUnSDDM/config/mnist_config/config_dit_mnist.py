import ml_collections
import os

# 6 138 946
def get_config():
    save_directory = "SavedModels/MNIST/"

    config = ml_collections.ConfigDict()
    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "NLL"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0  # only for CT-ELBO
    loss.min_time = 0.01
    loss.one_forward_pass = True # only for CT-ELBO

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 600000  # 2000 #2000000
    training.clip_grad = True
    training.grad_norm = 1
    training.warmup = 0  # 5000
    training.max_t = 1
    
    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteMNIST"
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 64  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 28
    data.shape = [1, data.image_size, data.image_size]
    data.use_augm = False
    data.location = 'lib/datasets/'
    

    config.model = model = ml_collections.ConfigDict()
    model.name = "GaussianLogDiTEMA" 
    model.ema_decay = 0.9999  # 0.9999

    model.patch_size = 4  # 128 => 4mal so viele Params
    model.input_channel = 1  
    model.concat_dim = model.input_channel * data.image_size * data.image_size # D
    model.hidden_dim = 512
    model.depth = 28
    model.num_heads = 8
    model.mlp_ratio = 4.0
    model.dropout = 0.1
    model.time_scale_factor = 1000
    model.model_output = 'logistic_pars' #logistic_pars'
    model.fix_logistic = False
    model.data_min_max = (0, data.S - 1)

    # forward model
    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exp = 100.0
    model.time_base = 3.0

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_freq = 1000
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "TauL"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = loss.min_time
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "gaussian"
    sampler.num_corrector_steps = 0
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.is_ordinal = True
    sampler.sample_freq = 1000

    return config
