import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    # data
    config.seed = 1023
    config.batch_size = 64
    config.image_size = 32
    config.data_aug = False

    # forward
    config.uniform_rate_const = 0.01 # 0.007
    config.vocab_size = 256
    config.t_func = "sqrt_cos"  # univariant
    config.diffuse_type = "uniform"

    # gaussian forward rate
    config.rate_sigma = 6.0
    config.Q_sigma = 512.0
    config.time_exponential = 100.0
    config.time_base = 3.0

    # backward
    # config.discrete_dim = config.image_size * config.image_size  * 1 # D = C*H*W
    config.discrete_dim = config.image_size * config.image_size * 1  #
    # only for cond_backward_model
    config.lambda_t = "const"
    config.logit_type = "direct"
    config.loss_type = "elbo"

    # model
    config.model_type = "tauldr"  # hollow, 'cond_hollow, ebm
    """
    # for hollow and 'cond_hollow: bidir_transformer or enum_transformer
    config.net_arch = 'bidir_transformer'
    # bidir_transformer inputs:
    config.bidir_readout = 'res_concat'
    config.conditional_dim = 0
    """
    # enum_transformer

    # unet
    config.unet_dim = 32
    config.unet_data_shape = (config.image_size, config.image_size, 1)
    config.unet_outdim = 1
    config.unet_dim_mults = (1, 2, 2)
    config.unet_resnet_block_groups = 2  # 8
    config.unet_learned_variance = False

    # for ebm: if config.vocab_size > 2: automatic CatScoreMLP takes following inputs:
    """
    config.vocab_size = 256
    config.cat_embed_size = 512
    config.num_layers = 2
    config.embed_dim = 512 # hidden_size
    """
    config.time_scale_factor = 1000

    # optimizer
    config.lr_schedule = "constant"
    config.learning_rate = 0.8e-4 #1e-4
    config.warmup_frac = 0.00
    config.optimizer = 'adam' #"adamw"
    config.weight_decay = 0
    config.grad_norm = 1 #5

    # training
    config.total_train_steps = 1000
    config.phase = "train"
    config.sample_freq = 500
    config.checkpoint_freq = 200

    # saving
    config.save_dir = "SavedModels/MNIST"  #
    config.sample_plot_path = "SavedModels/MNIST/PNGs"  #
    config.ckpt_keep = 5
    # config.plot_num_batches = 10
    # config.log_every_steps = 50

    config.dtype = "float32"

    # loss
    config.time_duration = 1.0

    # ema
    config.ema_decay = 0.9999

    # sampler
    # config.plot_sample = 4096
    config.sampling_steps = 1000  # 1000 #400 # mabye 10000
    config.corrector_scale = 1.0
    config.corrector_steps = 0 #10
    config.sampler_type = "tau_leaping"

    # nets
    """
    config.embed_dim = 512
    config.mlp_dim = 256
    config.num_output_ffresiduals = 2
    config.transformer_norm_type = 'prenorm'
    config.num_heads = 4
    config.qkv_dim = 64
    config.attention_dropout_rate = 0.0
    config.dropout_deterministic = False
    config.dropout_rate = 0.0
    config.num_layers = 2
    config.readout = 'mlp'
    """
    # EBM

    # hollow

    # tau leaping
    config.is_ordinal = True
    config.tauldr_onepass = True

    config.eval_rounds = 10

    # VQ image diffusion
    # config.vq_model_info
    # config.vq_model

    """
    # ebm
    config.model_type='ebm'
    config.net_arch='mlp'
    config.embed_dim=256
    config.num_layers=3
    config.grad_norm=5.0
    config.plot_num_batches=32
    config.weight_decay=1e-6
    config.sampler_type='exact'
    config.logit_type='reverse_logscale'
    config.lambda_t='const'
    config.t_sample='linear'

    # hollow
    config.model_type='hollow'
    config.net_arch='bidir_transformer'
    config.readout='resnet'
    config.bidir_readout='res_concat'
    config.logit_type='direct'
    config.num_output_ffresiduals=2
    config.loss_type='rm'
    config.eval_rounds=1
    config.grad_norm=5.0
    config.weight_decay=1e-6
    config.num_heads=4
    config.embed_dim=64
    config.qkv_dim=64
    config.mlp_dim=256
    config.dropout_rate=0.0
    config.learning_rate=1e-4
    config.attention_dropout_rate=0.0
    config.dropout_deterministic=False
    """

    return config
