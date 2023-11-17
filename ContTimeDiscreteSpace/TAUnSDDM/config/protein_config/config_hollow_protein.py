import ml_collections
import os
import torch


def get_config():
    save_directory = "SavedModels/Protein"
    config = ml_collections.ConfigDict()

    config.device = "cpu"
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

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    training.n_iters = 10  # 2000 #2000000

    training.clip_grad = True
    training.grad_norm = 5  # 1
    training.warmup = 0  # 50 # 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = "ProteinDataset"
    data.is_img = False
    data.S = 21
    data.batch_size = 32  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [48]
    data.location = "lib/datasets/Protein_sequences/grampa_numarr.npy"

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniformHollowEMA"
    # Forward model
    model.rate_const = 0.33
    model.t_func = "loq_sqr"  # log_sqr
    # hollow:
    model.net_arch = "bidir_transformer"
    model.nets = "bidir_transformer2"

    # BiDir
    model.embed_dim = 64
    model.bidir_readout = "res_concat"  # res_concat, attention, concat
    model.use_one_hot_input = True
    # UniDirectional
    model.dropout_rate = 0.01
    model.concat_dim = data.shape[0]
    # config.concat_dim = data.shape[0]
    # config.dtype = torch.float32
    model.num_layers = 1
    # TransformerBlock
    ## SA
    model.num_heads = 1
    model.attention_dropout_rate = 0.1
    model.transformer_norm_type = "postnorm"  # prenorm
    ## FF
    model.mlp_dim = 128
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
    model.qkv_dim = config.model.embed_dim
    # config.num_heads = 4
    model.ema_decay = 0.9999  # 0.9999
    model.Q_sigma = 20.0
    model.time_scale_factor = 1000

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 1.5e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs/'protein_sequences.txt'")
    saving.checkpoint_freq = 500

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "LBJFSampling"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 5
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.sample_freq = 2000
    sampler.is_ordinal = False

    return config
