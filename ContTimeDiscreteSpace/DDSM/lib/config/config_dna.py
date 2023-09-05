import ml_collections

def get_config():
    save_directory = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/DNA' 
    datasets_folder = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/lib/datasets/DNA'


    config = ml_collections.ConfigDict()
    config.experiment_name = 'dna'
    config.save_location = save_directory

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'GenericAux'
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()

    config.data = data = ml_collections.ConfigDict()

    config.noise_sample = noise_sample = ml_collections.ConfigDict()
    noise_sample.n_samples = 100000
    noise_sample.num_cat = 4
    noise_sample.num_time_steps = 400
    noise_sample.speed_balance = True
    noise_sample.max_time = 4.0
    noise_sample.out_path = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/DNA/2023-09-06/'
    noise_sample.order = 1000
    noise_sample.steps_per_tick = 200
    noise_sample.mode = 'path' # 'path', 'independent'
    noise_sample.logspace = True

    return config
