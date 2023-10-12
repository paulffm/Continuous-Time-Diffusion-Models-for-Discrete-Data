import ml_collections

def get_config():

    datasets_folder = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/lib/datasets/MNIST'
    model_location = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST/2023-09-03/19-50-13_mnist/checkpoints/ckpt_0000004999.pt'
    model_config_location = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/tauLDR/SavedModels/MNIST/2023-09-03/19-50-13_mnist/config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'MNIST'
    config.train_config_overrides = [
        [['device'], 'cpu'],
        [['data', 'root'], datasets_folder],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cpu'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteCIFAR10'
    data.root = datasets_folder
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 16
    data.shuffle = True
    data.shape = [1,32,32]
    data.random_flips = False

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'TauLeaping' # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config