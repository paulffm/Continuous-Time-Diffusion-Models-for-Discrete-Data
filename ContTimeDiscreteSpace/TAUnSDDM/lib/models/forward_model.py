import torch
import numpy as np
import lib.models.model_utils as model_utils
from torchtyping import TensorType
import math
from lib.utils import utils


class BirthDeathForwardBase:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.sigma_min, self.sigma_max = cfg.model.sigma_min, cfg.model.sigma_max
        self.device = device
        # base rate is a (S-1, S-1) Matrix with ones on first off diagonals and - the sum of these entries on diagonal
        base_rate = np.diag(np.ones((S - 1,)), 1)
        base_rate += np.diag(np.ones((S - 1,)), -1)
        base_rate -= np.diag(np.sum(base_rate, axis=1))
        eigvals, eigvecs = np.linalg.eigh(base_rate)

        self.base_rate = torch.from_numpy(base_rate).float().to(self.device)
        self.base_eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.base_eigvecs = torch.from_numpy(eigvecs).float().to(self.device)

    # R_t = beta_t * R_b
    # R_b = self.base_rate
    def _rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        # time dependent scalar beta_(t)
        # choice: beta_t = a * (b ** t) * logb
        return (
            self.sigma_min**2
            * (self.sigma_max / self.sigma_min) ** (2 * t)
            * math.log(self.sigma_max / self.sigma_min)
        )

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        # integral of beta_t over 0 to t: a * b ** t - a
        #
        return (
            0.5 * self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t)
            - 0.5 * self.sigma_min**2
        )

    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        # R_t = beta_t * R_b
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.base_eigvals.view(1, S)
        # P_t = Q * exp(lambda * integrate_scalar) * Q^-1
        transitions = (
            self.base_eigvecs.view(1, S, S)
            @ torch.diag_embed(torch.exp(adj_eigvals))
            @ self.base_eigvecs.T.view(1, S, S)
        )
        transitions = transitions / torch.sum(transitions, axis=-1, keepdims=True)
        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] BirthDeathForwardBase, large negative transition values {torch.min(transitions)}"
            )

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions  # [B, S, S]


class UniformRate:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_const = cfg.model.rate_const
        self.device = device

        rate = self.rate_const * np.ones((S, S))
        rate = rate - np.diag(np.diag(rate))  # diag = 0
        rate = rate - np.diag(np.sum(rate, axis=1))  # diag = - sum of rows
        eigvals, eigvecs = np.linalg.eigh(rate)

        self.rate_matrix = torch.from_numpy(rate).float().to(device)
        self.eigvals = torch.from_numpy(eigvals).float().to(device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(device)
        # above same as get_rate_matrix from ForwardModel

    # rate_mat
    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        return torch.tile(
            self.rate_matrix.view(1, S, S), (B, 1, 1)
        )  # dimension from 1, S, S to B, S, S

    def rate_mat(self, y, t):
        del t
        return self.rate_matrix[y]

    # func usvt from forward model
    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S
        transitions = (
            self.eigvecs.view(1, S, S)  # Q
            @ torch.diag_embed(
                torch.exp(self.eigvals.view(1, S) * t.view(B, 1))
            )  # 3d or 2d tensor such that dimension fit (lambda)
            @ self.eigvecs.T.view(1, S, S)  # Q^-1
        )

        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}"
            )

        transitions[transitions < 1e-8] = 0.0

        return transitions  # q_{t | 0}

    def transit_between(self, t1, t2):
        return self.transition(t2 - t1)


class UniformVariantRate(UniformRate):
    """Variants of uniform."""

    def __init__(self, config, device):
        super(UniformVariantRate, self).__init__(config, device)
        self.config = config
        self.t_func = config.model.t_func
        if self.t_func == "log":
            self.time_base = config.model.time_base
            self.time_exp = config.model.time_exp

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        if self.t_func == "log_sqr":
            return torch.log(t**2 + 1)
        elif self.t_func == "sqrt_cos":
            return -torch.sqrt(torch.cos(torch.pi / 2 * t))  # + 1
        elif self.t_func == "log":
            return self.time_base * (self.time_exp**t) - self.time_base
        else:
            raise ValueError("Unknown t_func %s" % self.t_func)

    def _rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        if self.t_func == "log_sqr":
            return 2 * t / (t**2 + 1)
        elif self.t_func == "sqrt_cos":
            t = torch.pi / 2 * t
            tmp = torch.sin(t) / torch.sqrt(torch.cos(t))
            return torch.pi / 4.0 * tmp
        elif self.t_func == "log":
            return self.time_base * math.log(self.time_exp) * self.time_exp**t
        else:
            raise ValueError("Unknown t_func %s" % self.t_func)

    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        rate_scalars = self._rate_scalar(t).view(t.size(0), 1, 1)
        base = self.rate_matrix.view(
            1, self.S, self.S
        )  # why t()? self.rate_matrix.t().view(1, self.S, self.S)
        r = base * rate_scalars
        return r

    def rate_mat(self, y, t):
        r = self.rate(t)
        bidx = utils.expand_dims(torch.arange(t.size(0)), axis=tuple(range(1, y.dim())))
        result = r[bidx, y]
        return result

    def transit_between(self, t1, t2):
        B = t2.size(0)
        d_integral = self._integral_rate_scalar(t2) - self._integral_rate_scalar(t1)

        transitions = (
            self.eigvecs.view(1, self.S, self.S)  # Q
            @ torch.diag_embed(
                torch.exp(d_integral.view(B, 1) * self.eigvals.view(1, self.S))
            )
            @ self.eigvecs.T.view(1, self.S, self.S)  # Q^-1
        )
        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] UniformVariantRate, large negative transition values {torch.min(transitions)}"
            )

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions = transitions / torch.sum(transitions, axis=-1, keepdims=True)
        transitions[transitions < 1e-8] = 0.0
        return transitions

    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        # difference to jnp => they give only 0
        return self.transit_between(torch.zeros_like(t), t)


class GaussianTargetRate:
    def __init__(self, cfg, device):
        self.S = S = cfg.data.S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exp = cfg.model.time_exp
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

        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))

        eigvals, eigvecs = np.linalg.eig(rate)
        inv_eigvecs = np.linalg.inv(eigvecs)

        self.base_rate = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        return self.time_base * (self.time_exp**t) - self.time_base

    def _rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        return self.time_base * math.log(self.time_exp) * (self.time_exp**t)

    def rate(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def rate_mat(self, y, t):
        r = self.rate(t)
        bidx = utils.expand_dims(torch.arange(t.size(0)), axis=tuple(range(1, y.dim())))
        result = r[bidx, y]
        return result

    def transition(self, t: TensorType["B"]) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)

        transitions = (
            self.eigvecs.view(1, S, S)
            @ torch.diag_embed(torch.exp(adj_eigvals))
            @ self.inv_eigvecs.view(1, S, S)
        )
        transitions = transitions / torch.sum(transitions, axis=-1, keepdims=True)
        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}"
            )

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions

    def transit_between(self, t1, t2):
        B = t2.size(0)
        d_integral = self._integral_rate_scalar(t2) - self._integral_rate_scalar(t1)

        transitions = (
            self.eigvecs.view(1, self.S, self.S)  # Q
            @ torch.diag_embed(
                torch.exp(d_integral.view(B, 1) * self.eigvals.view(1, self.S))
            )
            @ self.eigvecs.T.view(1, self.S, self.S)  # Q^-1
        )
        transitions = transitions / torch.sum(transitions, axis=-1, keepdims=True)
        if torch.min(transitions) < -1e-6:
            print(
                f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}"
            )

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0
        return transitions
