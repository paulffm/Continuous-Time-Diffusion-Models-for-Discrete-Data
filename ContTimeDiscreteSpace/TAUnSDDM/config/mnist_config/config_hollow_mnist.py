import ml_collections
import os


def get_config():
    save_directory = "SavedModels/MNISTHollow"
    config = ml_collections.ConfigDict()

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CatRM"
    loss.logit_type = "direct"  # direct:  whole train_step with backward < 10 sek, reverse_prob, reverse_logscale
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.ce_coeff = 0  # >0 whole train_step with backward < 10 sek

    loss.eps_ratio = 1e-9
    loss.min_time = 0.007
    loss.one_forward_pass = True
    loss.nll_weight = 0.001

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 600000  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 1  # 1
    training.warmup = 0  # 50 # 5000
    training.max_t = 0.99999

    training.resume = True

    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteMNIST"
    data.is_img = True
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 32  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 28
    data.shape = [1, data.image_size, data.image_size]
    data.use_augm = False
    data.location = 'lib/datasets/'

    config.model = model = ml_collections.ConfigDict()
    model.name = "GaussianHollowEMA"
    model.net_arch = "bidir_transformer"
    model.nets = "bidir_transformer2"
    model.use_cat = False

    # BiDir
    model.embed_dim = 256
    model.bidir_readout = "attention"  # res_concat, attention, concat
    model.use_one_hot_input = False
    model.dropout_rate = 0.1
    model.concat_dim = data.image_size * data.image_size * 1
    model.num_layers = 9
    model.num_heads = 8
    model.attention_dropout_rate = 0.1
    model.transformer_norm_type = "prenorm"  # prenorm
    ## FF
    model.mlp_dim = 512  # d_model in TAU => embed_dim?
    model.out_dim = data.S
    model.readout_dim = data.S
    model.num_output_ffresiduals = 2
    model.qkv_dim = model.embed_dim
    model.ema_decay = 0.9999  # 0.9999
    model.time_scale_factor = 1000

    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exp = 100.0
    model.time_base = 3.0
    model.log_prob = 'cat'

    model.rate_const = 2.1

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 10000

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "CRMLBJF"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = loss.min_time
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "gaussian"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.is_ordinal = True
    sampler.sample_freq = 22000000

    return config
