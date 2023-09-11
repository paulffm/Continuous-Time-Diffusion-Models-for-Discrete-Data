import ml_collections



def get_config():
    config = ml_collections.ConfigDict()
    # data
    config.data_folder='synthetic/checkerboard',
    config.seed=1023,
    config.batch_size=128,

    # forward
    config.uniform_rate_const=1.0,
    config.vocab_size = 256, #?
    config.t_func = 'sqrt_cos', # UniformVariantForward
    config.diffuse_type == 'uniform',

    # backward
    config.discrete_dim = 32,
    config.lambda_t = 'const',
    config.logit_type = 'direct', #? # reverse_logscale'
    config.loss_type = 'elbo', #mle. elbo, 'x0ce', rm 

    # model
    config.model_type = 'tauldr',  # binary: ebm, hollow, 'tauldr'

    # optimizer
    config.lr_schedule = 'constant', # updown, up_exp_down
    config.learning_rate = 1e-4,
    config.warmup_frac = ,
    config.optimizer='adamw',
    config.weight_decay = 0, # wird nichts gemacht
    config.grad_norm = 0.0, # wird nichtss gemacht

    # training
    config.total_train_steps = 1000, # 300000
    config.phase = 'train',
    config.save_root = '../../SavedModels/MNIST', # ''
    config.fig_folder = '../../SavedModels/MNIST/PNGs',
    config.model_init_folder = ''
    config.ckpt_keep = 1, #config.get("ckpt_keep", 1)
    config.plot_num_batches = 10 ,# config.get("plot_num_batches", 10)
    config.log_every_steps = 50, 
    config.plot_every_steps = 2000,
    config.save_every_steps = 10#10000,
    config.dtype='float32',

    # loss
    config.time_duration = 1.0,

    # ema
    config.ema_decay=0.9999,

    # sampler 
    config.plot_sample = 4096,
    config.sampling_steps=400,
    config.corrector_scale = 1.0, # standard
    config.corrector_steps = 0, # wird nichts gemacht
    config.sampler_type = 'tau_leaping', # exact, tau_leaping, lbjf, binary:

    # nets
    ## ConcatReadout
    config.embed_dim = 512,
    ## ResidualReadout
    config.mlp_dim = 256,
    config.num_output_ffresiduals = 2,
    ## sa-block, ff-block
    config.transformer_norm_type = 'prenorm', #'postnorm'
    config.num_heads = 4,
    config.qkv_dim = 64,
    config.attention_dropout_rate = 0.0,
    config.dropout_deterministic = False,
    config.dropout_rate = 0.0
    ## TransformerEncoder
    config.num_layers = 2,
    ## MaskedTransformer
    config.readout = 'mlp', # mlp, resnet

    # EBM
    ## BinaryTransformerScoreFunc: 
    config.time_scale_factor = 1000,
    ## CatMLPScoreFunc
    ## BinaryScoreModel
    config.net_arch = 'mlp',  # transformerbidir_transformer', 'bidir_combiner_transformer', 'enum_transformer'
    ## CategoricalScoreModel
    config.cat_embed_size

    # hollow
    ## bidir_transformer
    config.bidir_readout = 'res_concat', # concat, res_concat, attention
    ## EnumerativeTransformer
    config.conditional_dim = 0,  #config.get('conditional_dim', 0)


    # tau leaping
    config.is_ordinal = True, #config.get('is_ordinal', True
    config.tauldr_onepass = True,

    # image diffusion
    config.fig_folder 

    #VQ image diffusion
    #config.vq_model_info
    # config.vq_model

    config.eval_rounds=10,

    """
    # ebm
    config.model_type='ebm',
    config.net_arch='mlp',
    config.embed_dim=256,
    config.num_layers=3,
    config.grad_norm=5.0,
    config.plot_num_batches=32,
    config.weight_decay=1e-6,
    config.sampler_type='exact',
    config.logit_type='reverse_logscale',
    config.lambda_t='const',
    config.t_sample='linear',

    # hollow
    config.model_type='hollow',
    config.net_arch='bidir_transformer',
    config.readout='resnet',
    config.bidir_readout='res_concat',
    config.logit_type='direct',
    config.num_output_ffresiduals=2,
    config.loss_type='rm',
    config.eval_rounds=1,
    config.grad_norm=5.0,
    config.weight_decay=1e-6,
    config.num_heads=4,
    config.embed_dim=64,
    config.qkv_dim=64,
    config.mlp_dim=256,
    config.dropout_rate=0.0,
    config.learning_rate=1e-4,
    config.attention_dropout_rate=0.0,
    config.dropout_deterministic=False,
    """

    return config
  


