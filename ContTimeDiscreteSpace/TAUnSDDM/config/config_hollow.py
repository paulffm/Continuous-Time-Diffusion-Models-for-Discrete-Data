import ml_collections
import torch
def get_config():
    save_directory = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST/' # '../../SavedModels/MNIST/'
    dataset_path = '../lib/datasets'



    config = ml_collections.ConfigDict()
    config.experiment_name = 'mnist'
    config.save_location = save_directory

    config.init_model_path = None

    config.device = 'cpu'
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'HollowAux'
    config.logit_type = "direct"  # direct:  whole train_step with backward < 10 sek, reverse_prob, reverse_logscale
    loss.loss_type = "rm" # rm, mle, elbo
    config.ce_coeff = -0.5 # >0 whole train_step with backward < 10 sek
    

    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = 'Standard'

    training.n_iters = 1 #2000 #2000000

    training.clip_grad = True
    training.warmup = 0 #50 # 5000
    training.resume = True 

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteMNIST'
    data.root = dataset_path
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 32 # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [1,32,32]
    data.random_flips = True
    data.image_size = 32

    config.model = model = ml_collections.ConfigDict()
    model.name = 'UniformBDTEMA'

    model.rate_const = 0.03
    # hollow:
    config.net_arch = "bidir_transformer"
    
    # BiDir
    model.use_one_hot = False
    config.embed_dim = 512
    config.bidir_readout = "concat" # res_concat, attention, concat
    config.use_one_hot_input = False
    # UniDirectional
    config.dropout_rate = 0.1
    config.concat_dim = 32 * 32 *1
    # config.dtype = torch.float32
    config.num_layers = 1
    # TransformerBlock
    ## SA
    config.num_heads = 1
    config.attention_dropout_rate = 0.1
    config.transformer_norm_type = "postnorm" # prenorm
    ## FF
    config.mlp_dim = 512 # d_model in TAU => embed_dim?
    ### TransformerMLPBlock
    config.out_dim = data.S
    # ConcatReadout
    config.readout_dim = data.S
    # MLP
    # features, activation

    # ResidualReadout
    config.num_output_ffresiduals = 1

    # AttentionReadout
    ## CrossAttention
    config.qkv_dim = config.embed_dim
    #config.num_heads = 4
    model.ema_decay = 0.9999 #0.9999
    model.Q_sigma = 20.0
    model.time_scale_factor = 1000
    
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

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 2e-4 #2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/TAUnSDDM/SavedModels/MNIST/PNGs'
    saving.checkpoint_freq = 5


    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'LBJFSampling' # TauLeaping or PCTauLeaping
    sampler.num_steps = 2
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'uniform'
    sampler.num_corrector_steps = 2
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.01

    sampler.sample_freq = 2

    return config
