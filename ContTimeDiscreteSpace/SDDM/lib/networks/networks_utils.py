import lib.networks.networks as networks
from lib.models import ebm, hollow_model
from lib.networks.unet import Unet


def build_network(config):
    if config.model_type == "ebm":
        if config.net_arch == "mlp":
            if config.vocab_size == 2:
                net = ebm.BinaryMLPScoreFunc(
                    num_layers=config.num_layers,
                    hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor,
                )
            else:
                # data must be B, D
                net = ebm.CatMLPScoreFunc(
                    vocab_size=config.vocab_size,
                    cat_embed_size=config.cat_embed_size,
                    num_layers=config.num_layers,
                    hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor,
                )
        else:
            raise ValueError("Unknown net arch: %s" % config.net_arch)

    elif config.model_type == "hollow":
        if "bidir" in config.net_arch and "transformer" in config.net_arch:
            net = hollow_model.BidirectionalTransformer(config)
        elif config.net_arch == "enum_transformer":
            net = hollow_model.EnumerativeTransformer(config)

    elif config.model_type == "cond_hollow":
        if "bidir" in config.net_arch and "transformer" in config.net_arch:
            net = hollow_model.PrefixConditionalBidirTransformer(config)
        elif config.net_arch == "enum_transformer":
            net = hollow_model.EnumerativeTransformer(config)

    elif config.model_type == "tauldr":
        #net = hollow_model.BidirectionalTransformer(config)
        
        net = Unet(
            dim=32,
            shape=config.unet_data_shape,
            out_dim=config.unet_outdim,
            dim_mults=config.unet_dim_mults,
            resnet_block_groups=config.unet_resnet_block_groups,
            learned_variance=config.unet_learned_variance,
            num_classes=config.vocab_size,
        )
        
    else:
        raise ValueError("Unknown network type %s" % config.net_arch)
    return net
