import ml_collections
import os

# 6 138 946
def get_config():
    save_directory = "SavedModels/MNIST/"

    config = ml_collections.ConfigDict()
    config.experiment_name = "mnist"
    config.save_location = save_directory

    config.init_model_path = None

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = "CTElbo"
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.001
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 400000  # 2000 #2000000
    training.clip_grad = True
    training.grad_norm = 2
    training.warmup = 0  # 5000
    training.max_t = 0.99

    config.data = data = ml_collections.ConfigDict()
    data.name = "SudokuDataset"
    data.train = True
    data.download = True
    data.S = 9
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    

    config.model = model = ml_collections.ConfigDict()
    model.name = "UniVarScoreNetEMA"
    model.padding = False
    model.ema_decay = 0.9999  # 0.9999

    model.embed_dim = 256

    model.rate_const = 0.35
    model.t_func = "sqrt_cos"  
    model.Q_sigma = 512.0
    model.concat_dim = 81 * 9 


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 1.5e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.checkpoint_freq = 10000
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = "TauL"  # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = "uniform"
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = float(1.5)
    sampler.corrector_entry_time = float(0.0)
    sampler.is_ordinal = True
    sampler.sample_freq = 220000000


    return config
