import ml_collections

def get_config():
    save_directory = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST' 
    datasets_folder = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/lib/datasets/MNIST'



    config = ml_collections.ConfigDict()
    config.experiment_name = 'mnist'
    config.save_location = save_directory

    config.init_model_path = None

    config.device = 'cpu'
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'GenericAux'
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = 'Standard'
    training.n_iters = 2000 #2000 #2000000
    training.clip_grad = True
    training.warmup = 0 # 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteMNIST'
    data.root = datasets_folder
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 64 # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [1,32,32]
    data.random_flips = True

    config.model = model = ml_collections.ConfigDict()
    model.name = 'GaussianTargetRateImageX0PredEMAPaul'

    model.ema_decay = 0.9999 #0.9999

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

    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 2e-4 #2e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None
    saving.checkpoint_freq = 1000
    saving.num_checkpoints_to_keep = 2
    saving.checkpoint_archive_freq = 3000 #200000
    saving.log_low_freq = 10000
    saving.low_freq_loggers = ['denoisingImages']
    saving.prepare_to_resume_after_timeout = False


    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'TauLeaping' # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    sampler.sample_freq = 2000

    return config
