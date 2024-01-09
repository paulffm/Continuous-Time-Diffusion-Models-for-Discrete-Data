import ml_collections
import os

# config_bert_001: Param: 7 226 883
def get_config():
    save_directory = "SavedModels/MAZE"
    config = ml_collections.ConfigDict()

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CatRM"
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.logit_type = "reverse_prob"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.007
    loss.ce_coeff = 0

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 400000 #0  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 1  # 1
    training.warmup = 0  # 50 # 5000
    training.resume = True
    training.max_t = 0.9999

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
    data.random_transform = True
    #data.use_augm = False

    config.model = model = ml_collections.ConfigDict()
    model.concat_dim = data.image_size * data.image_size * 1
    model.name = "UniVarMaskedEMA"
    # Forward model
    model.rate_const = 1.7
    model.t_func = "sqrt_cos"  # log_sqr
    # hollow:

    # BiDir
    model.embed_dim = 64
    model.readout = 'resnet' # 'mlp'
    model.use_one_hot_input = False
    model.use_cat = False
    model.is_ebm = False
    model.log_prob = 'cat'

    # UniDirectional
    model.dropout_rate = 0.1
    # config.dtype = torch.float32
    model.num_layers = 4
    # TransformerBlock
    ## SA
    model.num_heads = 8
    model.attention_dropout_rate = 0.1
    model.transformer_norm_type = "prenorm"  # prenorm
    ## FF
    model.mlp_dim = 256 # d_model in TAU => embed_dim?
    ### TransformerMLPBlock
    model.out_dim = data.S
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
    saving.checkpoint_freq = 5000

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "LBJF"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = loss.min_time
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 200000000
    sampler.is_ordinal = True

    return config
