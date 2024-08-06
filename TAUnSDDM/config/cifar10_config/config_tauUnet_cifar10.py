import ml_collections
import os

# 6 138 946
def get_config():
    save_directory = "SavedModels/CIFAR10/"

    config = ml_collections.ConfigDict()
    config.save_location = save_directory

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CTElboLambda"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 500000  # 2000 #2000000
    training.clip_grad = True
    training.grad_norm = 1
    training.warmup = 0  # 5000
    training.max_t = 1
    
    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteCIFAR10"
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 64  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 32
    data.shape = [3, data.image_size, data.image_size]
    data.random_flips = True
    data.use_augm = False
    

    config.model = model = ml_collections.ConfigDict()
    model.name = "GaussianTargetRateImageX0PredEMAPaul"
    model.padding = False
    model.ema_decay = 0.9999  # 0.9999

    model.ch = 128  # 128 => 4mal so viele Params
    model.num_res_blocks = 2
    model.ch_mult = [1, 2, 2, 2]  # [1, 2, 2, 2]
    model.input_channels = 3  
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 255]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False
    model.model_output = 'logistic_pars' #logistic_pars'
    model.num_heads = 8
    model.attn_resolutions = [int(model.ch / 2)]
    model.concat_dim = data.image_size * data.image_size * model.input_channels
    model.padding = False

    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exp = 100.0
    model.time_base = 3.0

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.checkpoint_freq = 5000
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "TauL"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "gaussian"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.is_ordinal = True
    sampler.sample_freq = 5000

    return config
