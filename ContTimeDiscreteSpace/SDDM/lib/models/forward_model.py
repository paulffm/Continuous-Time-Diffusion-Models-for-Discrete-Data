"""Forward diffusion models."""

import math
import jax
import jax.numpy as jnp
import numpy as np


class ForwardModel(object):
    """Generic forward model."""

    def __init__(self, num_states):
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
        """Sample x_t and t with aux info returned."""
        bsize = x0.shape[0]
        t_rng, sample_rng = jax.random.split(rng)
        t = jax.random.uniform(t_rng, (bsize,))
        t = t * time_duration
        qt = self.transition(t)
        b = jnp.expand_dims(jnp.arange(bsize), tuple(range(1, x0.ndim)))
        qt0 = qt[b, x0]
        logits = jnp.where(qt0 <= 0.0, -1e9, jnp.log(qt0))
        xt = jax.random.categorical(sample_rng, logits)
        return qt0, xt, t

    def sample_xt(self, x0, time_duration, rng):
        """Sample x_t and t."""
        _, xt, t = self.sample_xt_with_aux(x0, time_duration, rng)
        return xt, t

    def sample_from_prior(self, rng, shape):
        raise NotImplementedError


def get_rate_matrix(rate):
    rate = rate - np.diag(np.diag(rate))  # diag = 0
    rate = rate - np.diag(np.sum(rate, axis=1))  # diag = - sum of rows
    eigvals, eigvecs = np.linalg.eigh(rate)
    return (
        jnp.array(rate, dtype=jnp.float32),
        jnp.array(eigvals, dtype=jnp.float32),
        jnp.array(eigvecs, dtype=jnp.float32),
    )


def usvt(eigvecs, inv_eigvecs, diag_embed):
    ns = eigvecs.shape[0]
    u = jnp.reshape(eigvecs, (1, ns, ns))  # 1, S, S
    vt = jnp.reshape(inv_eigvecs, (1, ns, ns))
    transitions = (
        u @ diag_embed @ vt
    )  # # 3d or 2d tensor such that dimension fit (lambda)
    transitions = transitions / jnp.sum(transitions, axis=-1, keepdims=True)
    return transitions


class UniformForward(ForwardModel):
    """Uniform rate."""

    def __init__(self, num_states, rate_const):
        super(UniformForward, self).__init__(num_states=num_states)
        self.rate_const = rate_const
        rate = rate_const * np.ones((num_states, num_states))
        self.rate_matrix, self.eigvals, self.eigvecs = get_rate_matrix(rate)

    def rate_mat(self, t):
        return jnp.tile(
            jnp.expand_dims(self.rate_matrix, axis=0), [t.shape[0], 1, 1]
        )  # dimension from 1, S, S to B, S, S

    def rate(self, y, t):
        del t
        return self.rate_matrix[y]

    def transition(self, t):
        bsize = t.shape[0]
        diag_embed = jax.vmap(jnp.diag)(
            jnp.exp(
                jnp.reshape(self.eigvals, (1, self.num_states))
                * jnp.reshape(t, (bsize, 1))
            )
        )
        transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
        return transitions  # shape (B, S, S)

    def transit_between(self, t1, t2):
        return self.transition(t2 - t1)

    def sample_from_prior(self, rng, shape):
        xt = jax.random.randint(
            rng, shape, minval=0, maxval=self.num_states, dtype=jnp.int32
        )
        return xt  # (Shape) = (Num_samples, config.discrete_dim)


class UniformVariantForward(UniformForward):
    """Variants of uniform."""

    def __init__(self, config):
        super(UniformVariantForward, self).__init__(
            num_states=config.vocab_size, rate_const=config.uniform_rate_const
        )
        self.t_func = config.t_func

    def _integral(self, t):
        if self.t_func == "log_sqr":
            return jnp.log(t**2 + 1)
        elif self.t_func == "sqrt_cos":
            return -jnp.sqrt(jnp.cos(jnp.pi / 2 * t))
        else:
            raise ValueError("Unknown t_func %s" % self.t_func)

    def _rate(self, t):
        if self.t_func == "log_sqr":
            return 2 * t / (t**2 + 1)
        elif self.t_func == "sqrt_cos":
            t = jnp.pi / 2 * t
            tmp = jnp.sin(t) / jnp.sqrt(jnp.cos(t))
            return jnp.pi / 4.0 * tmp
        else:
            raise ValueError("Unknown t_func %s" % self.t_func)

    def rate_mat(self, t):
        rate_scalars = jnp.reshape(self._rate(t), (t.shape[0], 1, 1))
        base = jnp.reshape(self.rate_matrix.T, (1, self.num_states, self.num_states))
        r = base * rate_scalars
        return r

    def rate(self, y, t):
        r = self.rate_mat(t)
        bidx = jnp.expand_dims(jnp.arange(t.shape[0]), tuple(range(1, y.ndim)))
        result = r[bidx, y]
        return result

    def transit_between(self, t1, t2):
        bsize = t2.shape[0]
        d_integral = self._integral(t2) - self._integral(t1)
        diag_embed = jax.vmap(jnp.diag)(
            jnp.exp(
                jnp.reshape(self.eigvals, (1, self.num_states))
                * jnp.reshape(d_integral, (bsize, 1))
            )
        )
        transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
        return transitions

    def transition(self, t):
        return self.transit_between(0, t)


class GaussianTargetRate:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exponential = cfg.model.time_exponential
        self.time_base = cfg.model.time_base
        self.device = device

        rate = np.zeros((S, S))

        vals = np.exp(-np.arange(0, S) ** 2 / (self.rate_sigma**2))
        for i in range(S):
            for j in range(S):
                if i < S // 2:
                    if j > i and j < S - i:
                        rate[i, j] = vals[j - i - 1]
                elif i > S // 2:
                    if j < i and j > -i + S - 1:
                        rate[i, j] = vals[i - j - 1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(
                        -((j + 1) ** 2 - (i + 1) ** 2 + S * (i + 1) - S * (j + 1))
                        / (2 * self.Q_sigma**2)
                    )
        self.rate_matrix, self.eigvals, self.eigvecs = get_rate_matrix(rate)

    def _integral_rate_scalar(self, t):
        # so ähnlich für between s und t
        return self.time_base * (self.time_exponential**t) - self.time_base

    def _rate_scalar(self, t):
        return (
            self.time_base
            * math.log(self.time_exponential)
            * (self.time_exponential**t)
        )

    def rate(self, t):
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return jnp.reshape(self.rate_matrix, (1, S, S)) * jnp.reshape(
            rate_scalars, (1, S, S)
        )

    def transition(self, t):
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = jnp.reshape(integral_rate_scalars, (B, 1)) * jnp.reshape(
            self.eigvals, (1, S)
        )
        diag_embed = jax.vmap(jnp.diag)(jnp.exp(adj_eigvals))

        transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
        # Some entries that are supposed to be very close to zero might be negative
        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions


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
