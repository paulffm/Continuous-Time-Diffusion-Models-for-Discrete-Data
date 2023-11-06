import ml_collections
import os


def get_config():
    save_directory = "SavedModels/Synthetic"

    config = ml_collections.ConfigDict()

    config.device = "cpu"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "GenericAux"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True
    loss.logit_type = 'reverse_prob'

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 200000  # 2000 #2000000
    training.clip_grad = True
    training.grad_norm = 5
    training.warmup = 0  # 50  # 5000

    config.data = data = ml_collections.ConfigDict()
    data.name = "SyntheticData"
    data.type = "2spirals"
    data.is_img = True
    data.S = 2
    data.binmode = "gray"
    data.int_scale = 5970.914754907331
    data.plot_size = 4.48793632886299
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [32]

    config.concat_dim = data.shape[0]

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniformRateResMLP"

    #model.ema_decay = 0.9999  # 0.9999
    model.rate_const = 0.7

    model.num_layers = 2  # 6
    model.d_model = 512  # 512
    model.hidden_dim = 2048  # 2048
    model.temb_dim = 512
    model.time_scale_factor = 1000

    model.Q_sigma = None

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_freq = 10000
    saving.num_checkpoints_to_keep = 2
    saving.prepare_to_resume_after_timeout = False
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "LBJFSampling"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.0
    sampler.is_ordinal = False
    sampler.sample_freq = 200000

    return config
