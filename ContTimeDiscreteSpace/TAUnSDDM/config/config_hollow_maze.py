import ml_collections
import os
import torch


def get_config():
    save_directory = "SavedModels/MAZE/"

    config = ml_collections.ConfigDict()
    config.save_location = save_directory
    config.experiment_name = "MAZE"

    config.device = "cpu"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "HollowAux"
    loss.logit_type = "reverse_prob"  # direct:  whole train_step with backward < 10 sek, reverse_prob, reverse_logscale
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.ce_coeff = 1  # >0 whole train_step with backward < 10 sek

    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 2  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 5  # 1
    training.warmup = 0  # 50 # 5000
    training.resume = True

    config.data = data = ml_collections.ConfigDict()
    data.S = 2
    data.is_img = True
    data.batch_size = 32  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 30
    data.shape = [1, data.image_size, data.image_size]
    data.use_augm = False

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniformBDTEMA"
    # Forward model
    model.rate_const = 0.72
    model.t_func = "loq_sqr"  # log_sqr
    # hollow:
    model.net_arch = "bidir_transformer"

    # BiDir
    model.embed_dim = 64
    model.bidir_readout = "res_concat"  # res_concat, attention, concat
    model.model.use_one_hot_input = False
    # UniDirectional
    model.dropout_rate = 0.1
    config.concat_dim = data.image_size * data.image_size * 1
    # config.dtype = torch.float32
    model.num_layers = 1
    # TransformerBlock
    ## SA
    model.num_heads = 1
    model.attention_dropout_rate = 0.1
    model.transformer_norm_type = "postnorm"  # prenorm
    ## FF
    model.mlp_dim = 256  # d_model in TAU => embed_dim?
    ### TransformerMLPBlock
    model.out_dim = data.S
    # ConcatReadout
    model.readout_dim = data.S
    # MLP
    # features, activation

    # ResidualReadout
    model.num_output_ffresiduals = 1

    # AttentionReadout
    ## CrossAttention
    model.qkv_dim = config.embed_dim
    # config.num_heads = 4
    model.ema_decay = 0.9999  # 0.9999
    model.Q_sigma = 20.0
    model.time_scale_factor = 1000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 1.5e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 2

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "TauLeaping"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 2
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 2
    sampler.is_ordinal = True

    """
    model.num_layers = 6
    model.d_model = 256
    model.num_heads = 8
    model.dim_feedforward = 1024 # 2048
    model.dropout = 0.1
    model.temb_dim = 256
    model.num_output_FFresiduals = 2
    model.time_scale_factor = 1000
    model.use_one_hot_input = False
    model.ema_decay = 0.9999 #0.9999

    model.rate_const = 0.03
    config.logit_type = "reverse_logscale"

    model.rate_sigma = 3.0
    model.Q_sigma = 20.0
    model.time_exponential = 1000.0
    model.time_base = 0.5

    model.sigma_min = 1.0
    model.sigma_max = 100.0
    """
    # unet
    """
    model.rate_const = 0.03
    model.ch = 32 #128
    model.num_res_blocks = 2
    model.num_scales = 4
    model.ch_mult = [1, 2, 2] # [1, 2, 2, 2]
    model.input_channels = 1 #3
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 255]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False
    model.ema_decay = 0.9999 #0.9999
    model.model_output = "logistic_pars"
    model.Q_sigma = 512.0
    
    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0
    """

    return config
