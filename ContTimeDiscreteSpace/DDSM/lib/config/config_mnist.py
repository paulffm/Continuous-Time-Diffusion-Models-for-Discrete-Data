import ml_collections


def get_config():
    save_directory = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/MNIST"
    datasets_folder = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/lib/datasets/MNIST"

    config = ml_collections.ConfigDict()
    config.experiment_name = "mnist"
    config.save_location = save_directory
    config.num_cat = 256

    config.loss = loss = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()

    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 64
    data.image_size = 32
    data.use_augmentation = True
    data.num_workers = 4
    

    config.noise_sample = noise_sample = ml_collections.ConfigDict()
    noise_sample.n_samples = 100000
    noise_sample.num_cat = 4
    noise_sample.num_time_steps = 400
    noise_sample.speed_balance = True  # ohne angabe false
    noise_sample.max_time = 4.0
    noise_sample.out_path = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/DNA/2023-09-06/"
    noise_sample.order = 1000
    noise_sample.steps_per_tick = 200
    noise_sample.mode = "path"  # 'path', 'independent'
    noise_sample.logspace = True  # ohne angabe false

    return config
