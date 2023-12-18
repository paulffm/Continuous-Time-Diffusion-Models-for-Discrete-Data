import ml_collections
import os

# 6 138 946
def get_config():
    save_directory = "SavedModels/BIN-MNIST/"

    config = ml_collections.ConfigDict()
    config.save_location = save_directory

    config.init_model_path = None

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CTElbo"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.007
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 300000  # 2000 #2000000
    training.clip_grad = True
    training.grad_norm = 2
    training.warmup = 0  # 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = "BinMNIST"
    data.train = True
    data.download = True
    data.S = 2
    data.batch_size = 64  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 28
    data.shape = [1, data.image_size, data.image_size]
    data.random_flips = True
    data.use_augm = False
    data.location = 'lib/datasets'
    

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniVarUnetEMA"
    model.padding = False
    model.ema_decay = 0.9999  # 0.9999

    model.ch = 64  # 128 => 4mal so viele Params
    model.num_res_blocks = 2
    model.ch_mult = [1, 2]  # [1, 2, 2, 2]
    model.input_channels = 1  
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 1]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False
    model.model_output = 'logistic_pars'
    model.num_heads = 8
    model.attn_resolutions = [int(model.ch / 2)]
    model.concat_dim = data.image_size * data.image_size * 1
    model.padding = False

    model.rate_const = 2.3
    #model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    #model.time_exp = 5  # b
    #model.time_base = 5 # a
    model.t_func = "sqrt_cos" 

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.checkpoint_freq = 10000
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "ElboTauL"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = loss.min_time
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.is_ordinal = True
    sampler.sample_freq = 220000000

    return config
