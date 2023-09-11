import ml_collections


def get_config():
    save_directory = '../../SavedModels/MNIST/'
    dataset_path = "../datasets/"
    sample_plot_path = "../../SavedModels/MNIST/PNGs/"
    diffusion_weights_path = '../../SavedModels/MNIST/diff_weights_steps400.cat256.time4.0.samples10000.pth'
    time_dep_weights_path = '../../SavedModels/MNIST/time_depend_weights_steps400.cat256.pth'


    config = ml_collections.ConfigDict()
    config.num_cat = 256
    config.n_time_steps = 400 #?
    config.random_order = False
    config.device = 'cpu'
    config.speed_balanced = False
    config.use_fast_diff = False
   
    config.loss = loss = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.n_iter = 10 # num_epochs = 50
    training.validation_freq = 5 # 10

    config.data = data = ml_collections.ConfigDict()
    data.batch_size = 64
    data.image_size = 32 # del
    data.num_cat = 256
    data.shape = (32, 32, 2)
    data.use_augmentation = True # del
    data.num_workers = 4
    data.name = "mnist"

    config.model = model = ml_collections.ConfigDict()
    model.ch = 32 #128
    model.num_res_blocks = 2
    model.attn = [1] # [16]
    model.ch_mult = [1, 2, 2] # 
    model.dropout = 0.1

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.lr = 5e-4
    optimizer.weight_decay = 1e-10
    
    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.sampler_freq = 10
    sampler.n_samples = 16

    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_freq = 5
    saving.checkpoint_path = save_directory
    saving.time_dep_weights_path = save_directory
    saving.sample_plot_path = sample_plot_path
    saving.save_location = save_directory 

    config.loading = loading = ml_collections.ConfigDict()
    loading.diffusion_weights_path = diffusion_weights_path
    loading.time_dep_weights_path = time_dep_weights_path
    loading.dataset_path = dataset_path

    config.noise_sample = noise_sample = ml_collections.ConfigDict()
    noise_sample.n_samples = 10000
    noise_sample.num_cat = 256
    noise_sample.n_time_steps = 400
    noise_sample.speed_balance = False  # ohne angabe false
    noise_sample.max_time = 4.0
    noise_sample.out_path = save_directory
    noise_sample.order = 1000
    noise_sample.steps_per_tick = 200
    noise_sample.mode = "path"  # 'path', 'independent'
    noise_sample.logspace = False  # ohne angabe false

    return config