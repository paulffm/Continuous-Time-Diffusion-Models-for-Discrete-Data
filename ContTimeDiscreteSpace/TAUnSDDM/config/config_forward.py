import ml_collections
import os

def get_config():
    save_directory = "SavedModels/MAZE/"

    config = ml_collections.ConfigDict()
    config.save_location = save_directory

    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 0

    config.loss = loss = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"

    config.data = data = ml_collections.ConfigDict()
    # MAZE
    data.name = "Maze3S"
    data.S = 3
    data.is_img = True
    data.batch_size = 128  # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.image_size = 15
    data.shape = [1, data.image_size, data.image_size]
    data.use_augm = False
    data.crop_wall = False
    data.limit = 1

    config.model = model = ml_collections.ConfigDict()
    # Forward model
    model.rate_const = 0.01
    model.t_func = "loq_sqr"  # log_sqr

    model.Q_sigma = 20.0
    model.time_scale_factor = 1000

    # MNIST: Gaussian Target Rate
    model.rate_sigma = 6.0
    model.Q_sigma = 512.0
    model.time_exponential = 100.0
    model.time_base = 3.0

    # Piano: Uniform Constant Rate
    model.rate_const = 0.03

    # Synthetic: Gaussian Target Rate
    model.rate_sigma = 1
    model.Q_sigma = 8
    model.time_exponential = 5
    model.time_base = 5
    return config
