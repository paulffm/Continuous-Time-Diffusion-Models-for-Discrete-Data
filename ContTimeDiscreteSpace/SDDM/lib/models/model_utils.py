
from lib.models import ebm
from lib.models.forward_model import UniformForward, UniformVariantForward
from lib.models import hollow_model
from lib.models import tauldr_model
import jax.numpy as jnp



def build_backwd_model(config, fwd_model, net):
    if config.model_type == 'ebm':
        backwd_model = ebm.CategoricalScoreModel(config, fwd_model, net)
    elif config.model_type == 'hollow':
        backwd_model = hollow_model.HollowModel(config, fwd_model, net)
    elif config.model_type == 'tauldr':
        backwd_model = tauldr_model.TauLDRBackward(config, fwd_model, net)
    else:
        raise ValueError('Unknown model type %s' % config.model_type)
    return backwd_model

def build_fwd_model(config):
    """Get forward model."""
    if config.diffuse_type == 'uniform':
        fwd_model = UniformForward(
            num_states=config.vocab_size, rate_const=config.uniform_rate_const)
    elif config.diffuse_type == 'uniform_variant':
        fwd_model = UniformVariantForward(config)
    else:
        raise ValueError('Unknown diffusion type %s' % config.diffuse_type)
    return fwd_model


def get_lambda_t(config, t):
    """Get lambda schedule."""
    if config.get('lambda_t', 'const') == 'const':
        return jnp.ones(t.shape, dtype=jnp.float32)
    elif config.lambda_t == 'grow_linear':
        return 0.5 + t
    elif config.lambda_t == 'decay_linear':
        return 1.5 - t
    elif config.lambda_t == 'decay_convex':
        return (0.1 + t) ** -0.5
    else:
        raise ValueError('Unknown lambda_t: %s' % config.lambda_t)