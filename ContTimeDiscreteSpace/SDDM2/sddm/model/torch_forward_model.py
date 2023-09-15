import torch
import torch.nn as nn
import numpy as np
from sddm.common import torch_utils
import functorch


class ForwardModel:
    """Generic forward model."""

    def __init__(self, num_states):
        super(ForwardModel, self).__init__()
        self.num_states = num_states

    def rate(self, t):
        raise NotImplementedError

    def rate_mat(self, t):
        raise NotImplementedError

    def transition(self, t):
        raise NotImplementedError

    def transit_between(self, t1, t2):
        raise NotImplementedError

    def sample_xt_with_aux(self, x0, time_duration, rng):
        bsize = x0.size(0)
        t_rng, sample_rng = torch.manual_seed(rng), torch.manual_seed(rng)
        t = torch.rand(bsize)
        t = t * time_duration
        qt = self.transition(t)
        b = torch_utils.expand_dims(torch.arange(bsize), (tuple(range(1, x0.dim()))))
        qt0 = qt[b, x0]
        logits = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))
        xt = torch.distributions.categorical.Categorical(logits=logits).sample()
        return qt0, xt, t

    def sample_xt(self, x0, time_duration, rng):
        _, xt, t = self.sample_xt_with_aux(x0, time_duration, rng)
        return xt, t

    def sample_from_prior(self, rng, shape):
        raise NotImplementedError


def get_rate_matrix(rate):
    rate = rate - np.diag(np.diag(rate))
    rate = rate - np.diag(np.sum(rate, axis=1))
    eigvals, eigvecs = np.linalg.eigh(rate)
    return (
        torch.tensor(rate, dtype=torch.float32),  # (S, S)
        torch.tensor(eigvals, dtype=torch.float32),  # (S, )
        torch.tensor(eigvecs, dtype=torch.float32),  # (S,S)
    )

# multiplication in transitions
def usvt(eigvecs, inv_eigvecs, diag_embed):
    ns = eigvecs.shape[0]
    u = eigvecs.view(1, ns, ns)
    vt = inv_eigvecs.view(1, ns, ns)
    transitions = torch.matmul(torch.matmul(u, diag_embed), vt)
    transitions = transitions / torch.sum(transitions, dim=-1, keepdim=True)
    return transitions  # (1, S, S)


class UniformForward(ForwardModel):
    """Uniform rate."""

    def __init__(self, num_states, rate_const):
        super(UniformForward, self).__init__(num_states=num_states)
        self.rate_const = rate_const
        rate = rate_const * np.ones((num_states, num_states))
        self.rate_matrix, self.eigvals, self.eigvecs = get_rate_matrix(rate)

    def rate_mat(self, t):
        return torch.tile(self.rate_matrix.unsqueeze(0), [t.size(0), 1, 1])

    def rate(self, y, t):
        del t
        return self.rate_matrix[y]

    def transition(self, t):
        bsize = t.size(0)
        diag_embed = functorch.vmap(torch.diag)(
            torch.exp(
                torch.reshape(self.eigvals, (1, self.num_states))
                * torch.reshape(t, (bsize, 1))
            )
        )
        transitions = usvt(self.eigvecs, self.eigvecs.transpose(0, 1), diag_embed)
        return transitions

    def transit_between(self, t1, t2):
        return self.transition(t2 - t1)

    def sample_from_prior(self, rng, shape):
        xt = torch.randint(0, self.num_states, shape, dtype=torch.int32)
        return xt

# theoretisch übertragbar in tauLDR
class UniformVariantForward(UniformForward):
    """Variants of uniform."""

    def __init__(self, config):
        super(UniformVariantForward, self).__init__(
            num_states=config.vocab_size, rate_const=config.uniform_rate_const
        )
        self.t_func = config.t_func

    def _integral(self, t):
        if self.t_func == "log_sqr":
            return torch.log(t**2 + 1)
        elif self.t_func == "sqrt_cos":
            return -torch.sqrt(torch.cos(torch.pi / 2 * t))
        else:
            raise ValueError("Unknown t_func %s" % self.t_func)

    def _rate(self, t):
        if self.t_func == "log_sqr":
            return 2 * t / (t**2 + 1)
        elif self.t_func == "sqrt_cos":
            t = torch.pi / 2 * t
            tmp = torch.sin(t) / torch.sqrt(torch.cos(t))
            return torch.pi / 4.0 * tmp
        else:
            raise ValueError("Unknown t_func %s" % self.t_func)

    def rate_mat(self, t):
        rate_scalars = self._rate(t).view(t.size(0), 1, 1)
        base = self.rate_matrix.t().view(1, self.num_states, self.num_states)
        r = base * rate_scalars
        return r

    def rate(self, y, t):
        r = self.rate_mat(t)
        bidx = torch_utils.expand_dims(
            torch.arange(t.size(0)), axis=tuple(range(1, y.dim()))
        )
        result = r[bidx, y]
        return result

    def transit_between(self, t1, t2):
        bsize = t2.size(0)
        d_integral = self._integral(t2) - self._integral(t1)
        diag_embed = functorch.vmap(torch.diag)(
            torch.exp(
                torch.reshape(self.eigvals, (1, self.num_states))
                * torch.reshape(d_integral, (bsize, 1))
            )
        )
        transitions = usvt(self.eigvecs, self.eigvecs.t(), diag_embed)
        return transitions

    def transition(self, t):
        # difference to jnp => they give only 0
        return self.transit_between(torch.zeros_like(t), t)


def get_fwd_model(config):
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
