import ml_collections
import os


def get_config():
    save_directory = "SavedModels/Synthetic"
    config = ml_collections.ConfigDict()

    config.device = "cpu"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CatRM"
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.logit_type = "direct"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.ce_coeff = 1

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 10  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 5  # 1
    training.warmup = 0  # 50 # 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = "Maze3S"
    data.is_img = False
    data.S = 3
    data.batch_size = 32  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 15
    data.shape = [1, data.image_size, data.image_size]
    data.crop_wall = False
    data.limit = 1

    config.model = model = ml_collections.ConfigDict()
    model.ema_decay = 0.9999 
    model.name = "UniVarMaskUNetEMA"
    model.padding = True
    model.concat_dim = data.image_size * data.image_size * 1
    # Forward model
    model.ch = 32
    model.rate_const = 2.3
    model.t_func = "log_sqr"  # log_sqr

    model.num_res_blocks = 1
    model.num_scales = 4
    model.ch_mult = [1, 2] 
    model.input_channels = 1  # 3
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 2]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False
    model.model_output = 'logistic_pars'
    model.num_heads = 1
    model.attn_resolutions = [int(model.ch / 2)]
    model.Q_sigma = 1

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 1.5e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 500

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "CRMLBJF"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 5
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 200000000
    sampler.is_ordinal = False

    return config
