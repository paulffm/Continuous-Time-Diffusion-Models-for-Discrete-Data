import lib.networks.networks as networks


def build_network(config):
    if config.net_arch == 'ebm':
        net = networks.CategoricalScoreModel(config, fwd_model, net)
    elif config.model_type == 'hollow':
        net = networks.HollowModel(config, fwd_model, net)
    elif config.model_type == 'tauldr':
        net = networks.TauLDRBackward(config, fwd_model, net)
    else:
        raise ValueError('Unknown network type %s' % config.net_arch)
    return net