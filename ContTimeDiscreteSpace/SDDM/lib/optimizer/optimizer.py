import optax


def build_lr_schedule(config):
    """Build lr schedule."""
    if config.lr_schedule == 'constant':
        lr_schedule = lambda step: step * 0 + config.learning_rate
    elif config.lr_schedule == 'updown':
        warmup_steps = int(config.warmup_frac * config.total_train_steps)
        lr_schedule = optax.join_schedules([
            optax.linear_schedule(0, config.learning_rate, warmup_steps),
            optax.linear_schedule(config.learning_rate, 0,
                                    config.total_train_steps - warmup_steps)
        ], [warmup_steps])
    elif config.lr_schedule == 'up_exp_down':
        warmup_steps = int(config.warmup_frac * config.total_train_steps)
        lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value=0.0, peak_value=config.learning_rate,
            warmup_steps=warmup_steps, transition_steps=20000,
            decay_rate=0.9, end_value=1e-6
        )
    else:
        raise ValueError('Unknown lr schedule %s' % config.lr_schedule)
    return lr_schedule


def build_optimizer(config):
    """Build optimizer."""
    lr_schedule = build_lr_schedule(config)
    optimizer_name = config.get('optimizer', 'adamw')
    optims = []
    grad_norm = config.get('grad_norm', 0.0)
    if grad_norm > 0.0:
        optims.append(optax.clip_by_global_norm(grad_norm))
    opt_args = {}
    if optimizer_name in ['adamw', 'lamb']:
        opt_args['weight_decay'] = config.get('weight_decay', 0.0)
    # chained eigeschaften optimizer und dann optimizer selbst
    optims.append(
        getattr(optax, optimizer_name)(lr_schedule, **opt_args)
    )
    optim = optax.chain(*optims)
    return optim


def get_optimizer(config):
    return optax.adam(config.learning_rate)
