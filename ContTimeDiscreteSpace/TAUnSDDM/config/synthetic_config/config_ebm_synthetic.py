import ml_collections
import os
# MaskedModel -> TransformerEncoder -> TransformerBlock -> SA + FF

def get_config():
    save_directory = "SavedModels/Synthetic"
    config = ml_collections.ConfigDict()

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "BinEBMAux"
    loss.loss_type = "rm"  # rm, mle, elbo
    loss.logit_type = "direct"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.ce_coeff = 1

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 2  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 5  # 1
    training.warmup = 0  # 50 # 5000
    training.resume = True

    config.data = data = ml_collections.ConfigDict()
    data.name = "SyntheticData"
    data.type = "2spirals"
    data.is_img = False
    data.S = 2
    data.binmode = "gray"
    data.int_scale = 5970.914754907331 #5995.531550196217
    data.plot_size = 4.48793632886299
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [32]
    data.location = f"lib/datasets/Synthetic/data_{data.type}.npy"

    config.model = model = ml_collections.ConfigDict()
    model.concat_dim = data.shape[0]
    model.is_ebm = True
    model.name = "UniVarBinaryEBMEMA"
    # Forward model
    model.rate_const = 0.7
    model.t_func = "log_sqr"  # log_sqr
    # hollow:

    # BiDir
    model.embed_dim = 256
    model.readout = 'mlp' # 'mlp'
    model.use_one_hot_input = False
    model.use_cat = False
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
    model.mlp_dim = 512 # d_model in TAU => embed_dim?
    ### TransformerMLPBlock
    model.out_dim = data.S
    # ConcatReadout
    model.readout_dim = data.S
    model.qkv_dim = config.model.embed_dim
    # config.num_heads = 4
    model.ema_decay = 0.9999  # 0.9999
    model.Q_sigma = 20.0
    model.time_scale_factor = 1000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 1.5e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 500

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "CRMBinaryLBJF"  # TauLeaping or PCTauLeaping
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
