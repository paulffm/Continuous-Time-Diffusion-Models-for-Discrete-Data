import ml_collections


def get_config():
    save_directory = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/DNA"
    datasets_folder = "/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/lib/datasets/DNA"
    diffusion_weights_path = '/Users/paulheller/PythonRepositories/Master-Thesis/ContTimeDiscreteSpace/DDSM/SavedModels/DNA/presampled_noise_dna.pt'

    config = ml_collections.ConfigDict()
    config.experiment_name = "dna"
    config.save_location = save_directory
    config.diffusion_weights_path = diffusion_weights_path
    config.num_time_steps = 400 #?
    config.random_order = True
    config.device = 'cpu'
    config.speed_balanced = False

    config.data = data = ml_collections.ConfigDict()
    data.data.num_cat = 4
    data.batch_size = 256
    data.shape = (1024, 4)
    data.image_size = 32
    data.num_worker = 4
    data.ref_file = '../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    data.ref_file_mmap = '../data/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
    data.tsses_file = '../data/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'
    data.fantom_files = [
                    "../data/agg.plus.bw.bedgraph.bw",
                    "../data/agg.minus.bw.bedgraph.bw"
                    ]
    data.fantom_blacklist_files = [
         "../data/fantom.blacklist8.plus.bed.gz",
         "../data/fantom.blacklist8.minus.bed.gz"
        ]
    
    config.loss = loss = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.n_iter = 5000 #200 epochs

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 5e-4 
    config.sei = sei = ml_collections.ConfigDict()
    sei.seifeatures_file = '../data/target.sei.names'
    sei.seimodel_file = '../data/best.sei.model.pth.tar'

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.sampler_freq = 10000

    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_freq = 1000

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
