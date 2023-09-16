import lib.networks.networks as networks
from lib.models import ebm, hollow_model

def build_network(config):
    if config.model_type == 'ebm':
        if config.net_arch == 'mlp':
            if config.vocab_size == 2:
                net = ebm.BinaryMLPScoreFunc(num_layers=config.num_layers, hidden_size=config.embed_dim, time_scale_factor=config.time_scale_factor)
            else:
                net = ebm.CatMLPScoreFunc(
                    vocab_size=config.vocab_size, cat_embed_size=config.cat_embed_size,
                    num_layers=config.num_layers, hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor)
        else:
            raise ValueError('Unknown net arch: %s' % config.net_arch)
        
    elif config.model_type == 'hollow':
        if "bidir" in config.net_arch and "transformer" in config.net_arch:
            net = hollow_model.BidirectionalTransformer(config)
        elif config.net_arch == "enum_transformer":
            net = hollow_model.EnumerativeTransformer(config)

    elif config.model_type == 'cond_hollow':
        if "bidir" in config.net_arch and "transformer" in config.net_arch:
            net = hollow_model.PrefixConditionalBidirTransformer(config)
        elif config.net_arch == "enum_transformer":
            net = hollow_model.EnumerativeTransformer(config)

    elif config.model_type == 'tauldr':
        net = hollow_model.BidirectionalTransformer(config)
    else:
        raise ValueError('Unknown network type %s' % config.net_arch)
    return net