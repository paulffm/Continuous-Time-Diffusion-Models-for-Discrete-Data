import ml_collections
import os


def get_config():
    save_directory = "SavedModels/Synthetic"
    config = ml_collections.ConfigDict()

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CatRMNLL"
    loss.logit_type = "reverse_prob"  # direct:  whole train_step with backward < 10 sek, reverse_prob, reverse_logscale
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.ce_coeff = 0  # >0 whole train_step with backward < 10 sek
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.007
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 200000  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 1  # 1
    training.warmup = 0  # 50 # 5000
    training.max_t = 0.99999

    config.data = data = ml_collections.ConfigDict()
    data.name = "SyntheticData"
    data.type = "2spirals"
    data.is_img = False
    data.S = 2
    data.binmode = "gray"
    data.int_scale = 6003.0107336488345
    data.plot_size = 4.458594271092115
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [32]
    data.location = f"lib/datasets/Synthetic/data_{data.type}.npy"

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniVarHollowEMA"
    model.log_prob = 'cat'
    # Forward model
    model.rate_const = 2.0
    model.Q_sigma = 512.0
    model.t_func = "sqrt_cos"  # log_sqr
    # hollow:
    model.net_arch = "bidir_transformer"
    model.nets = "bidir_transformer2"
    model.use_cat = False

    # BiDir
    model.embed_dim = 64
    model.bidir_readout = "attention"  # res_concat, attention, concat
    model.use_one_hot_input = False
    model.dropout_rate = 0.1
    model.concat_dim = data.shape[0]
    model.num_layers = 2

    model.num_heads = 8
    model.attention_dropout_rate = 0.1
    model.transformer_norm_type = "prenorm"  # prenorm
    ## FF
    model.mlp_dim = 256  # d_model in TAU => embed_dim?
    model.out_dim = data.S
    model.readout_dim = data.S
    # MLP

    model.num_output_ffresiduals = 2

    # AttentionReadout
    ## CrossAttention
    model.qkv_dim = config.model.embed_dim
    model.ema_decay = 0.9999  # 0.9999
    model.time_scale_factor = 1000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 1.5e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 10000

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "CRMLBJF"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 500
    sampler.min_t = loss.min_time
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 200000000
    sampler.is_ordinal = True

    return config
