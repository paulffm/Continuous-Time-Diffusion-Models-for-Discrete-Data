import ml_collections


def config_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    return config_dict(
        seed=0,
        dataset=config_dict(
            name='mnist',
            path="datasets/",
            resolution=32,
            args=config_dict(
                class_conditional=False,
                randflip=False,
            ),
        ),
        model=config_dict(
            # architecture, see main.py and model.py
            name='unet0',
            args=config_dict(
                channel=32,
                in_channel=1,
                out_channel=1,
                channel_multiplier=[1, 2, 2],
                n_res_blocks=2,
                attn_resolutions=[16],
                num_heads=1,
                fold=1,
                dropout=0.1,
                model_output='logistic_pars',  # logits  or logistic_pars
                num_pixel_vals=256
            ),
            # diffusion betas, see diffusion_categorical.get_diffusion_betas
            diffusion_betas=config_dict(
                type='linear',
                # start, stop only relevant for linear, power, jsdtrunc schedules.
                start=1e-4,  # 1e-4 gauss, 0.02 uniform
                stop=0.02,  # 0.02, gauss, 1. uniform
                num_timesteps=1000,
            ),
            # Settings used in diffusion_categorical.py
            model_prediction='x_start',  # 'x_start','xprev'
            # 'gaussian','uniform','absorbing'
            transition_mat_type='gaussian',
            transition_bands=None,
            loss_type='hybrid',  # kl,cross_entropy_x_start, hybrid
            hybrid_coeff=0.001,  # only used for hybrid loss type.
        ),
        train=config_dict(
            # optimizer
            batch_size=32,
            optimizer='adam',
            learning_rate=2e-4,
            learning_rate_warmup_steps=0,
            weight_decay=0.0,
            ema_decay=0.9999,
            grad_clip=1.0,
            substeps=10,
            num_train_steps=1500000,  # multiple of substeps
            # logging
            log_loss_every_steps=1000,
            checkpoint_every_secs=900,  # 15 minutes
            retain_checkpoint_every_steps=100000,
            eval_every_steps=50000,
            log_img_dir="exp/mnist/images",
            eval_every_epoch=50
        ))