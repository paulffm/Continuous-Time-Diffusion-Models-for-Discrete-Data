import torch
import torch.nn.functional as F
from sddm.model import torch_backward_model
from sddm.model import torch_forward_model
from sddm.model import utils


class TauLDRBackward(torch_backward_model.BackwardModel):
    """Tau LDR backward model, from https://github.com/andrew-cr/tauLDR"""

    def __init__(self, config):
        super(TauLDRBackward, self).__init__(config)
        self.fwd_model = torch_forward_model.get_fwd_model(self.config)
        self.net = torch_backward_model.FreeformTransformer(config)

    def _sample_categorical(self, rng, prob):
        rng, local_rng = torch.manual_seed(rng), torch.manual_seed(rng)
        log_prob = torch.where(prob <= 0, -1e9, torch.log(prob))
        val = torch.distributions.Categorical(logits=log_prob).sample()
        return rng, val

    def get_ratio(self, params, xt, t, xt_target=None):
        raise NotImplementedError

    def get_logits(self, params, xt, t):
        pass

    def get_logprob(self, params, xt, t, xt_target=None):
        xt_target = xt if xt_target is None else xt_target
        xt_onehot = F.one_hot(xt_target, num_classes=self.config.vocab_size)
        x0_logits = self.net(x=xt, t=t)
        p0t = F.softmax(x0_logits, dim=-1)
        qt0 = torch.clamp(self.fwd_model.transition(t), min=1e-8)
        qt0_denorm = qt0.transpose(1, 2)[torch.arange(xt.shape[0]), xt]

        qt0_numer = qt0.unsqueeze(list(range(1, xt.dim() - 1)))
        inner_sum = torch.matmul(p0t / qt0_denorm.unsqueeze(-1), qt0_numer)
        ll_all = inner_sum * (1 - xt_onehot) + xt_onehot
        ll_all = torch.where(ll_all < 1e-35, -1e9, torch.log(ll_all))
        ll_xt = torch.zeros(ll_all.shape[:-1], dtype=torch.float32)
        return ll_all, ll_xt

    def loss(self, params, rng, x0, xt, t):
        eps = 1e-9
        config = self.config
        qt0 = self.fwd_model.transition(t)
        qt0 = torch.clamp(qt0, min=1e-8)
        rate_mat = self.fwd_model.rate_mat(t)
        bsize = xt.shape[0]
        xt = xt.view(bsize, -1)
        d = xt.shape[1]
        s = config.vocab_size
        xt_onehot = F.one_hot(xt, num_classes=s).to(torch.float32)
        cat_dims = xt.shape[1]
        b = torch.arange(bsize).view(-1, 1)

        rate_given_xt = rate_mat[b, xt]
        rate_given_xt = rate_given_xt * (1 - xt_onehot)
        rate_xt_offdiag = torch.sum(rate_given_xt, dim=-1)
        rng, dimcat = self._sample_categorical(rng, rate_xt_offdiag)

        rate_newval = rate_given_xt[torch.arange(bsize), dimcat]
        rng, valcat = self._sample_categorical(rng, rate_newval)
        dimcat_onehot = F.one_hot(dimcat, num_classes=cat_dims).to(torch.int32)
        valcat = valcat.view(-1, 1)
        xtilde = xt * (1 - dimcat_onehot) + dimcat_onehot * valcat

        if config.tauldr_onepass:
            x_logits = self.net(
                xtilde, t
            )  # Anstatt self.net.apply({'params': params}, x=xtilde, t=t)
            reg_x = xtilde
        else:
            x_logits = self.net(
                xt, t
            )  # Anstatt self.net.apply({'params': params}, x=xt, t=t)
            reg_x = xt
        p0t_reg = F.softmax(x_logits, dim=2)

        reg_x_onehot = F.one_hot(reg_x, num_classes=s).to(torch.float32)
        rate2xt = rate_mat[b, reg_x]
        rate2xt = rate2xt * (1 - reg_x_onehot)
        reg_tmp = torch.matmul(rate2xt, qt0.transpose(1, 2))
        qt0_denom_reg = qt0.transpose(1, 2)[b, reg_x]
        reg_term = torch.sum((p0t_reg / (qt0_denom_reg + eps)) * reg_tmp, dim=(1, 2))

        # second term
        if config.tauldr_onepass:
            p0t_sig = p0t_reg
        else:
            x_logits = self.net(x=xtilde, t=t)
            p0t_sig = F.softmax(x_logits, dim=2)

        outer_qt0_numer_sig = qt0[
            torch.repeat(torch.arange(bsize), d * s),
            torch.repeat(torch.repeat(x0, s), bsize * d),
            torch.tile(torch.arange(s), bsize * d),
        ].view(bsize, d, s)

        outer_qt0_denom_sig = (
            qt0[
                torch.repeat(torch.arange(bsize), d),
                torch.repeat(x0, d),
                torch.repeat(xtilde, s),
            ]
            + eps
        )

        qt0_denom_sig = qt0.transpose(1, 2)[b, xtilde] + eps

        inner_log_sig = torch.log(
            torch.matmul(p0t_sig / qt0_denom_sig.unsqueeze(-1), qt0) + eps
        )
        xtilde_onehot = F.one_hot(xtilde, num_classes=s).to(torch.float32)
        outer_rate_sig = rate_mat[
            torch.repeat(torch.arange(bsize), d * s),
            torch.tile(torch.arange(s), bsize * d),
            torch.repeat(xtilde, s),
        ].view(bsize, d, s)

        oss_tmp = outer_qt0_numer_sig / outer_qt0_denom_sig.view(bsize, d, 1)
        outer_sum_sig = torch.sum(
            (1 - xtilde_onehot) * outer_rate_sig * oss_tmp * inner_log_sig, dim=(1, 2)
        )

        rate_row_sums = -rate_mat[
            torch.repeat(torch.arange(bsize), s),
            torch.tile(torch.arange(s), bsize),
            torch.tile(torch.arange(s), bsize),
        ].view(bsize, s)

        base_z_tmp = rate_row_sums[
            torch.repeat(torch.arange(bsize), d), torch.repeat(xtilde, d)
        ].view(bsize, d)
        base_z = torch.sum(base_z_tmp, dim=1)

        z_subtraction = base_z_tmp
        z_addition = rate_row_sums
        z_sig_norm = (
            base_z.view(bsize, 1, 1)
            - z_subtraction.view(bsize, d, 1)
            + z_addition.view(bsize, 1, s)
        )

        rate_sig_norm = rate_mat[
            torch.repeat(torch.arange(bsize), d * s),
            torch.tile(torch.arange(s), bsize * d),
            torch.repeat(xtilde, s),
        ].view(bsize, d, s)
        qt0_sig_norm_numer = qt0[
            torch.repeat(torch.arange(bsize), d * s),
            torch.repeat(x0, s),
            torch.tile(torch.arange(s), bsize * d),
        ].view(bsize, d, s)
        qt0_sig_norm_denom = (
            qt0[
                torch.repeat(torch.arange(bsize), d),
                torch.repeat(x0, d),
                torch.repeat(xtilde, s),
            ]
            + eps
        )
        qt0_sig_norm_denom = qt0_sig_norm_denom.view(bsize, d)

        sig_norm_numer = rate_sig_norm * qt0_sig_norm_numer * (1 - xtilde_onehot)
        sig_norm_denom = z_sig_norm * qt0_sig_norm_denom.view(bsize, d, 1) + eps
        sig_norm = torch.sum(sig_norm_numer / sig_norm_denom, dim=(1, 2))

        sig_mean = torch.mean(-outer_sum_sig / sig_norm)
        reg_mean = torch.mean(reg_term)
        neg_elbo = sig_mean + reg_mean
        aux = {"loss": neg_elbo}
        return neg_elbo, aux
