import ml_collections
import os

# config_bert_001: Param: 7 226 883
def get_config():
    save_directory = "SavedModels/Synthetic"
    config = ml_collections.ConfigDict()

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CTElbo"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0 #0.001
    loss.min_time = 0.007
    loss.ce_coeff = 0
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 200000  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 1  # 1
    training.warmup = 0  # 50 # 5000
    training.resume = True
    training.max_t = 0.9999

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
    #data.use_augm = False

    config.model = model = ml_collections.ConfigDict()
    model.concat_dim = data.shape[0]
    model.name = "UniBertD3PM"
    # Forward model
    model.rate_const = 2
    model.t_func = "sqrt_cos"  # log_sqr
    # hollow:

    # BiDir
    model.ema_decay = 0.9999  # 0.9999
    model.embed_dim = 64
    model.readout = 'resnet' # 'mlp'
    model.use_one_hot_input = True
    model.use_cat = True
    model.is_ebm = False
    model.log_prob = 'cat'

    # UniDirectional
    model.dropout_rate = 0.1
    model.concat_dim = data.shape[0]
    # config.dtype = torch.float32
    model.num_layers = 3
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

    # diffusion betas
    model.type='cosine'
                # start, stop only relevant for linear, power, jsdtrunc schedules.
    model.start=0.02 # 1e-4 gauss, 0.02 uniform
    model.stop=1 # 0.02, gauss, 1. uniform
    model.num_timesteps=500
    model.time_scale_factor=1000

            # Settings used in diffusion_categorical.py
    model.model_prediction='x_start' # 'x_start','xprev'
            # 'gaussian','uniform','absorbing'
    model.transition_mat_type='uniform'
    model.transition_bands=None
    model.loss_type='hybrid'# kl,cross_entropy_x_start, hybrid
    model.hybrid_coeff=0.001
    model.model_output='logits'
    model.num_pixel_vals=2
    model.device='cuda'
    model.is_img = False


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 10000

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "ElboTauL"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = loss.min_time
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 200000000
    sampler.is_ordinal = False

    return config
