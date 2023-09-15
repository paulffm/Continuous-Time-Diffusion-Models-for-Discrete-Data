import lib.networks.networks as networks


def build_network(config):
    if config.net_arch == 'ebm':
        #net = networks.
        pass
    elif config.model_type == 'hollow':
        #net = networks.
        pass
    elif config.model_type == 'tauldr':
        net = networks.
    else:
        raise ValueError('Unknown network type %s' % config.net_arch)
    return net