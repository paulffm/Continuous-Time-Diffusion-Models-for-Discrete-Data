from lib.models import ebm
from lib.models import hollow_model
from lib.models import tauldr_model
from lib.models.forward_model import UniformForward, UniformVariantForward
import lib.networks.networks as networks
from lib.models import ebm, hollow_model
from lib.networks.unet import Unet

# to prevent cyclic dependencies

def build_fwd_model(config):
    """Get forward model."""
    if config.diffuse_type == "uniform":
        fwd_model = UniformForward(
            num_states=config.vocab_size, rate_const=config.uniform_rate_const
        )
    elif config.diffuse_type == "uniform_variant":
        fwd_model = UniformVariantForward(config)
    else:
        raise ValueError("Unknown diffusion type %s" % config.diffuse_type)
    return fwd_model


def build_backwd_model(config, fwd_model, net):
    if config.model_type == "ebm":
        backwd_model = ebm.CategoricalScoreModel(config, fwd_model, net)
    elif config.model_type == "hollow":
        backwd_model = hollow_model.HollowModel(config, fwd_model, net)
    elif config.model_type == "tauldr":
        backwd_model = tauldr_model.TauLDRBackward(config, fwd_model, net)
    else:
        raise ValueError("Unknown model type %s" % config.model_type)
    return backwd_model


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