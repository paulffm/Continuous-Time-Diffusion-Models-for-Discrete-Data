import ml_collections


def get_config():
    save_directory = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/Bin_MNIST"
    datasets_folder = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/lib/datasets/Bin_MNIST"
    diffusion_weights_path = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/DNA/presampled_noise_mnist.pt'

    config = ml_collections.ConfigDict()
    config.experiment_name = "mnist"
    config.save_location = save_directory
    config.num_cat = 2
    config.diffusion_weights_path = diffusion_weights_path
    config.num_time_steps = 400 #?
    config.random_order = False
    config.device = 'cpu'
    config.speed_balanced = False


    config.loss = loss = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.n_iter = 100

    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 64
    data.image_size = 28
    data.num_cat = 2
    data.shape = (28, 28, 2)
    data.use_augmentation = True
    data.num_workers = 4

    config.sampler = sampler = ml_collections.ConfigDict()
    

    config.noise_sample = noise_sample = ml_collections.ConfigDict()
    noise_sample.num_samples = 100000
    noise_sample.num_cat = 4
    noise_sample.num_time_steps = 400
    noise_sample.speed_balance = False  # ohne angabe false
    noise_sample.max_time = 4.0
    noise_sample.out_path = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/Bin_MNIST/"
    noise_sample.order = 1000
    noise_sample.steps_per_tick = 200
    noise_sample.mode = "path"  # 'path', 'independent'
    noise_sample.logspace = False  # ohne angabe false

    return config
