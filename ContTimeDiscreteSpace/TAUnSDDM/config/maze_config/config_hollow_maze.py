import ml_collections
import os
import torch


def get_config():
    save_directory = "SavedModels/MAZE/"

    config = ml_collections.ConfigDict()
    config.save_location = save_directory

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CatRM"
    loss.logit_type = "reverse_prob"  # direct:  whole train_step with backward < 10 sek, reverse_prob, reverse_logscale
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.ce_coeff = 1  # >0 whole train_step with backward < 10 sek

    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 6000  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 5  # 1
    training.warmup = 0  # 50 # 5000
    training.resume = True

    config.data = data = ml_collections.ConfigDict()
    data.name = "Maze3S"
    data.S = 3
    data.is_img = True
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 15
    data.shape = [1, data.image_size, data.image_size]
    data.use_augm = False
    data.crop_wall = False
    data.limit = 1

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniformHollowEMA"
    # Forward model
    model.rate_const = 1.54
    model.t_func = "loq_sqr"  # log_sqr
    # hollow:
    model.net_arch = "bidir_transformer"
    model.nets = "bidir_transformer2"
    model.use_cat = False

    # BiDir
    model.embed_dim = 128
    model.bidir_readout = "res_concat"  # res_concat, attention, concat
    model.use_one_hot_input = False
    # UniDirectional
    model.dropout_rate = 0.1
    model.concat_dim = data.image_size * data.image_size * 1
    # config.dtype = torch.float32
    model.num_layers = 4
    # TransformerBlock
    ## SA
    model.num_heads = 8
    model.attention_dropout_rate = 0.1
    model.transformer_norm_type = "prenorm"  # prenorm
    ## FF
    model.mlp_dim = 2048  # d_model in TAU => embed_dim?
    ### TransformerMLPBlock
    model.out_dim = None
    # ConcatReadout
    model.readout_dim = data.S
    # MLP
    # features, activation

    # ResidualReadout
    model.num_output_ffresiduals = 2

    # AttentionReadout
    ## CrossAttention
    model.qkv_dim = config.model.embed_dim
    # config.num_heads = 4
    model.ema_decay = 0.9999  # 0.9999
    model.Q_sigma = 20.0
    model.time_scale_factor = 1000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 1000

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "ElboTauL"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 5000
    sampler.is_ordinal = True

    return config
