import torch
import torch.nn as nn
import lib.losses.losses_utils as losses_utils
import math
import numpy as np
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lib.utils.utils as utils
import time

@losses_utils.register_loss
class GenericAux:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.cross_ent = nn.CrossEntropyLoss()

    def calc_loss(self, minibatch, state, writer=None):
        model = state["model"]
        S = self.cfg.data.S
        # if 4 Dim => like images: True
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)

        B, D = minibatch.shape
        device = model.device

        # get random timestep between 1.0 and self.min_time
        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = model.transition(
            ts
        )  # (B, S, S) # transition q_{t | s=0} eq.15 => here randomness because of ts => for every ts another q_{t|0}

        # R_t = beta_t * R_b
        rate = model.rate(
            ts
        )  # (B, S, S) # no proability in here (diagonal = - sum of rows)

        # --------------- Sampling x_t, x_tilde --------------------
        # qt0_rows_reg = (B * D, S) probability distribution
        # diagonal elements of qt0 (higher probability) will be put at column of value of x_t
        # we do this because then we sample from qt0_rows_reg and then it is most likely more similar to x0=batch
        # example: q_t0 =   [0.4079, 0.2961, 0.2961],
        #                   [0.2961, 0.4079, 0.2961],
        #                   [0.2961, 0.2961, 0.4079]],
        # batch = (2, 0, 1)
        # qt0_rows_reg = [0.2961, 0.2961, 0.4079],
        #                [0.4079, 0.2961, 0.4079],
        #                [0.2961, 0.4079, 0.2961]

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(
                D
            ),  # repeats every element 0 to B-1 D-times
            minibatch.flatten().long(),  # minibatch.flatten() => (B, D) => (B*D) (1D-Tensor)
            :,
        ]  # (B*D, S)

        # set of (B*D) categorical distributions with probabilities from qt0_rows_reg
        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(  # sampling B * D times => from every row of qt0_rows_reg once => then transform it to shape B, D
            B, D
        )  # (B*D,) mit view => (B, D) Bsp: x_t = (0, 1, 2, 4, 3) (for B =1 )

        # --------------- x_t = noisy data => x_tilde one transition in every batch of x_t --------------------
        # puts diagonals (- values) (in a B*D, S) in this column where x_t has its entry => x_t[0,0] = 1
        # => in rate_vals_square[0, 1] = - values
        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(D), x_t.long().flatten(), :
        ]  # (B*D, S)

        rate_vals_square[
            torch.arange(B * D, device=device), x_t.long().flatten()
        ] = 0.0  # - values = 0 => in rate_vals_square[0, 1] = 0

        rate_vals_square = rate_vals_square.view(B, D, S)  # (B*D, S) => (B, D, S)

        #  Summe der Werte entlang der Dimension S
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(
            B, D
        )  # B, D with every entry = S-1? => for entries of x_t same prob to transition?
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )

        # Samples where transitions takes place in every row of B
        # if x_t = (0, 1, 2, 4, 3) (B = 1) and square_dims = 3
        # x_t = (0, 1, 2, X, 3) => will change
        square_dims = square_dimcat.sample()  # (B,) taking values in [0, D)

        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device), square_dims, :
        ]  # (B, S) => every row has only one entry = 0, everywhere else 1; chooses the row square_dim of rate_vals_square
        # => now rate_new_val_probs: (B, S) with every row (1, 1, 0)

        # samples from rate_new_val_probs and chooses state to transition to => more likely where entry is 1 instead of 0?
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )

        # Samples state, where we going
        # if x_t = (0, 1, 2, X, 3) and square_newval_samples = 1
        # x_tilde = (0, 1, 2, 1, 3)
        square_newval_samples = (
            square_newvalcat.sample()
        )  # (B, ) taking values in [0, S)

        # x_noisy => in every Batch exactly one transition
        # so in every row: one difference between x_t and x_tilde
        # x_t =     (0, 1, 2, 4, 3)
        # x_tilde = (0, 1, 2, 1, 3)
        x_tilde = x_t.clone()
        x_tilde[torch.arange(B, device=device), square_dims] = square_newval_samples

        # Now, when we minimize LCT, we are sampling (x, x ̃) from the forward process and then maximizing
        # the assigned model probability for the pairing in the reverse direction, just as in LDT

        # ---------- First term of ELBO (regularization) ---------------
        # use forward from UNet, MLP, Sequencetransformer

        # softmax(logits) => probabilities
        # p0t_reg = ptheta_{0|t}(x_0|x) = q_{0|t}(x0|x)
        if self.one_forward_pass:
            x_logits = model(x_tilde, ts)  # (B, D, S)
            # ensures that positive
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
            reg_x = x_tilde
        else:
            # x_t = x from Paper
            x_logits = model(x_t, ts)  # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
            reg_x = x_t

        # For (B, D, S, S) first S is x_0 second S is x'
        # => first ones and then place 0 where x_tilde/x_t has entry: Why? => x != x'
        # x_tilde = (1, 2, 0)
        # => mask_reg = (1, 0, 1)
        #               (1, 1, 0)
        #               (0, 1, 1)

        mask_reg = torch.ones((B, D, S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten(),
        ] = 0.0  # (B, D, S)

        # q_{t|0} (x ̃|x_0)
        qt0_numer_reg = qt0.view(B, S, S)

        # q_{t|0} (x|x_0)
        # puts diagonal (highstest probabilty) of qt0 where x_tilde has its entry
        qt0_denom_reg = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                :,
                reg_x.long().flatten(),
            ].view(B, D, S)
            + self.ratio_eps
        )

        # puts diagonal (-values) of rate where x_tilde has its entry
        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten(),
        ].view(B, D, S)
        # mask_reg * rate_vals_reg => - values => 0
        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(
            1, 2
        )  # (B, D, S)

        reg_term = torch.sum((p0t_reg / qt0_denom_reg) * reg_tmp, dim=(1, 2))

        # R^theta_t(x,x ̃) = R_t(x ̃,x) * sum_x_0 (q_{t|0} (x ̃|x_0) / q_{t | 0} (x|x_0)) * ptheta_{0|t}(x_0|x)
        # (mask_reg * rate_vals_reg) = sum_x' R^theta_t(x',x) for x != x'
        # qt0_numer_reg  = q_{t|0} (x ̃|x_0)
        # qt0_denom_reg = q_{t|0} (x|x_0)
        # p0t_reg = ptheta_{0|t}(x_0|x)

        # ----- second term of continuous ELBO (signal term) ------------

        # To evaluate the LCT objective, we naively need to perform two forward passes of the denoising
        # network: p^{θ}_{0|t}(x0|x) to calculate Rˆtθ(x, x′) and p^{θ}_{0|t}(x0|x ̃) to calculate Rˆtθ(x ̃, x). This is wasteful

        # because x ̃ is created from x by applying a single forward transition which on multi-dimensional problems
        # means x ̃ differs from x in only a single dimension. To exploit the fact that x ̃ and x are very similar,
        # we approximate the sample x ∼ qt(x) with the sample x ̃ ∼ Px qt(x)rt(x ̃|x).

        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            p0t_sig = F.softmax(model(x_tilde, ts), dim=2)  # (B, D, S)

        # q_{t|0} (x_0|x ̃)
        qt0_numer_sig = qt0.view(B, S, S)  # first S is x_0, second S is x

        # q_{t | 0} (x_0|x ̃)
        qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                :,
                x_tilde.long().flatten(),
            ].view(B, D, S)
            + self.ratio_eps
        )

        # log(R^theta_t(x ̃,x)) = R_t(x,x ̃) * sum_x_0 (q_{t|0} (x_0|x ̃) / q_{t | 0} (x_0|x ̃)) * ptheta_{0|t}(x_0|x)
        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        )  # (B, D, S)

        x_tilde_mask = torch.ones((B, D, S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten(),
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(B * D),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, D, S)

        # When we have B,D,S,S first S is x_0, second is x
        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D * S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * D),
        ].view(B, D, S)

        outer_qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                minibatch.long().flatten(),
                x_tilde.long().flatten(),
            ]
            + self.ratio_eps
        )  # (B, D)

        outer_sum_sig = torch.sum(
            x_tilde_mask
            * outer_rate_sig
            * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B, D, 1))
            * inner_log_sig,
            dim=(1, 2),
        )

        # now getting the 2nd term normalization

        rate_row_sums = -rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B),
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten(),
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp  # (B,D)
        Z_addition = rate_row_sums

        Z_sig_norm = (
            base_Z.view(B, 1, 1)
            - Z_subtraction.view(B, D, 1)
            + Z_addition.view(B, 1, S)
        )

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(B * D),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, D, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D * S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * D),
        ].view(B, D, S)

        qt0_sig_norm_denom = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(D),
                minibatch.long().flatten(),
                x_tilde.long().flatten(),
            ].view(B, D)
            + self.ratio_eps
        )

        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask)
            / (Z_sig_norm * qt0_sig_norm_denom.view(B, D, 1)),
            dim=(1, 2),
        )

        sig_mean = torch.mean(-outer_sum_sig / sig_norm)

        reg_mean = torch.mean(reg_term)

        if writer is not None:
            writer.add_scalar("sig", sig_mean.detach(), state["n_iter"])
            writer.add_scalar("reg", reg_mean.detach(), state["n_iter"])

        neg_elbo = sig_mean + reg_mean
        print("neg_elbo", type(neg_elbo), neg_elbo)
        perm_x_logits = torch.permute(x_logits, (0, 2, 1))

        nll = self.cross_ent(perm_x_logits, minibatch.long())
        print("nll", type(nll), nll)
        return neg_elbo + self.nll_weight * nll


@losses_utils.register_loss
class ConditionalAux:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.condition_dim = cfg.loss.condition_dim
        self.cross_ent = nn.CrossEntropyLoss()

    def calc_loss(self, minibatch, state, writer):
        model = state["model"]
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)
        B, D = minibatch.shape
        device = model.device

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = model.transition(ts)  # (B, S, S)

        rate = model.rate(ts)  # (B, S, S)

        conditioner = minibatch[:, 0 : self.condition_dim]
        data = minibatch[:, self.condition_dim :]
        d = data.shape[1]

        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.flatten().long(),
            :,
        ]  # (B*d, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, d)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(d), x_t.long().flatten(), :
        ]  # (B*d, S)
        rate_vals_square[
            torch.arange(B * d, device=device), x_t.long().flatten()
        ] = 0.0  # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, d, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, d)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample()  # (B,) taking values in [0, d)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device), square_dims, :
        ]  # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = (
            square_newvalcat.sample()
        )  # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[torch.arange(B, device=device), square_dims] = square_newval_samples
        # x_tilde (B, d)

        # ---------- First term of ELBO (regularization) ---------------

        if self.one_forward_pass:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits_full = model(model_input, ts)  # (B, D, S)
            x_logits = x_logits_full[:, self.condition_dim :, :]  # (B, d, S)
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, d, S)
            reg_x = x_tilde
        else:
            model_input = torch.concat((conditioner, x_t), dim=1)
            x_logits_full = model(model_input, ts)  # (B, D, S)
            x_logits = x_logits_full[:, self.condition_dim :, :]  # (B, d, S)
            p0t_reg = F.softmax(x_logits, dim=2)  # (B, d, S)
            reg_x = x_t

        # For (B, d, S, S) first S is x_0 second S is x'
        # ToDO: Why masking?
        mask_reg = torch.ones((B, d, S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            reg_x.long().flatten(),
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)

        qt0_denom_reg = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                :,
                reg_x.long().flatten(),
            ].view(B, d, S)
            + self.ratio_eps
        )

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            reg_x.long().flatten(),
        ].view(B, d, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(
            1, 2
        )  # (B, d, S)

        reg_term = torch.sum((p0t_reg / qt0_denom_reg) * reg_tmp, dim=(1, 2))

        # ----- second term of continuous ELBO (signal term) ------------

        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits_full = model(model_input, ts)  # (B, d, S)
            x_logits = x_logits_full[:, self.condition_dim :, :]
            p0t_sig = F.softmax(x_logits, dim=2)  # (B, d, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d * S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * d),
        ].view(B, d, S)

        outer_qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                data.long().flatten(),
                x_tilde.long().flatten(),
            ]
            + self.ratio_eps
        )  # (B, d)

        qt0_numer_sig = qt0.view(B, S, S)  # first S is x_0, second S is x

        qt0_denom_sig = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                :,
                x_tilde.long().flatten(),
            ].view(B, d, S)
            + self.ratio_eps
        )

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        )  # (B, d, S)

        x_tilde_mask = torch.ones((B, d, S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            x_tilde.long().flatten(),
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(d * S),
            torch.arange(S, device=device).repeat(B * d),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, d, S)

        outer_sum_sig = torch.sum(
            x_tilde_mask
            * outer_rate_sig
            * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B, d, 1))
            * inner_log_sig,
            dim=(1, 2),
        )

        # now getting the 2nd term normalization

        rate_row_sums = -rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B),
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(d),
            x_tilde.long().flatten(),
        ].view(B, d)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp  # (B,d)
        Z_addition = rate_row_sums

        Z_sig_norm = (
            base_Z.view(B, 1, 1)
            - Z_subtraction.view(B, d, 1)
            + Z_addition.view(B, 1, S)
        )

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(d * S),
            torch.arange(S, device=device).repeat(B * d),
            x_tilde.long().flatten().repeat_interleave(S),
        ].view(B, d, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(d * S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * d),
        ].view(B, d, S)

        qt0_sig_norm_denom = (
            qt0[
                torch.arange(B, device=device).repeat_interleave(d),
                data.long().flatten(),
                x_tilde.long().flatten(),
            ].view(B, d)
            + self.ratio_eps
        )

        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask)
            / (Z_sig_norm * qt0_sig_norm_denom.view(B, d, 1)),
            dim=(1, 2),
        )

        sig_mean = torch.mean(-outer_sum_sig / sig_norm)
        reg_mean = torch.mean(reg_term)

        writer.add_scalar("sig", sig_mean.detach(), state["n_iter"])
        writer.add_scalar("reg", reg_mean.detach(), state["n_iter"])

        neg_elbo = sig_mean + reg_mean

        perm_x_logits = torch.permute(x_logits, (0, 2, 1))

        nll = self.cross_ent(perm_x_logits, data.long())

        return neg_elbo + self.nll_weight * nll


def get_logprob_with_logits(cfg, model, xt, t, logits, xt_target=None):
    """Get logprob with logits."""
    start = time.time()
    #checked
    if xt_target is None:
        xt_target = xt
    xt_onehot = F.one_hot(xt_target.long(), cfg.data.S)
    if cfg.logit_type == "direct":
        log_prob = F.log_softmax(logits, dim=-1)
    else:
        qt0 = model.transition(t) 
        if cfg.logit_type == "reverse_prob":
            p0t = F.softmax(logits, dim=-1)
            qt0 = utils.expand_dims(qt0, axis=list(range(1, xt.dim() - 1)))
            prob_all = p0t @ qt0
            log_prob = torch.log(prob_all + 1e-35)
            # check
        elif cfg.logit_type == "reverse_logscale":
            log_p0t = F.log_softmax(logits, dim=-1)
            log_qt0 = torch.where(qt0 <= 1e-35, -1e9, torch.log(qt0))
            log_qt0 = utils.expand_dims(log_qt0, axis=list(range(1, xt.dim())))
            log_p0t = log_p0t.unsqueeze(-1)
            log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2)
            # check
        else:
            raise ValueError("Unknown logit_type: %s" % cfg.logit_type)
    log_xt = torch.sum(log_prob * xt_onehot, dim=-1)
    end = time.time()
    print("get_logprob_logits time", end - start)
    return log_prob, log_xt


# checked
@losses_utils.register_loss
class HollowAux:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time

    def _comp_loss(self, model, xt, t, ll_all, ll_xt): # <1sec
        start = time.time()
        B = xt.shape[0]
        if self.cfg.loss.loss_type == "rm":
            loss = -ll_xt # direct + rm => - loss
        elif self.cfg.loss.loss_type == "mle":
            # check
            loss = -(
                (self.cfg.data.S - 1) * ll_xt
                + torch.sum(utils.log1mexp(ll_all), dim=-1)
                - utils.log1mexp(ll_xt)
            )
        elif self.cfg.loss.loss_type == "elbo": #direct + elbo => - loss
            xt_onehot = F.one_hot(xt.long(), num_classes=self.cfg.data.S)
            b = utils.expand_dims(torch.arange(xt.shape[0]), tuple(range(1, xt.dim())))
            qt0_x2y = model.transition(t)
            qt0_y2x = qt0_x2y.permute(0, 2, 1)
            qt0_y2x = qt0_y2x[b, xt.long()]
            ll_xt = ll_xt.unsqueeze(-1)

            backwd = torch.exp(ll_all - ll_xt) * qt0_y2x
            first_term = torch.sum(backwd * (1 - xt_onehot), dim=-1)

            qt0_x2y = qt0_x2y[b, xt.long()]
            fwd = (ll_xt - ll_all) * qt0_x2y
            second_term = torch.sum(fwd * (1 - xt_onehot), dim=-1)
            loss = first_term - second_term

        else:
            raise ValueError("Unknown loss_type: %s" % self.cfg.loss_type)
        weight = torch.ones((B,), dtype=torch.float32)
        weight = utils.expand_dims(weight, axis=list(range(1, loss.dim())))
        loss = loss * weight
        end = time.time()
        print("_comp_loss ", end - start)
        return loss

    def calc_loss(self, minibatch, state, writer=None):
        start_calc = time.time()
        model = state["model"]
        S = self.cfg.data.S
        # if 4 Dim => like images: True
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W) 
        # hollow xt, t, l_all, l_xt geht rein
        device = self.cfg.device
        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = model.transition(ts)  # (B, S, S)

        # rate = model.rate(ts)  # (B, S, S)

        b = utils.expand_dims(torch.arange(B), (tuple(range(1, minibatch.dim()))))
        qt0 = qt0[b, minibatch.long()]
        # log loss
        logits = torch.where(qt0 <= 0.0, -1e9, torch.log(qt0))
        xt = torch.distributions.categorical.Categorical(logits=logits).sample() # bis hierhin <1 sek
        # get logits from CondFactorizedBackwardModel
        logits = model(xt, ts)  # B, D, S <10 sek
        # check
        # ce_coeff < 0
        if self.cfg.ce_coeff > 0: # whole train step <10 sek
            x0_onehot = F.one_hot(minibatch.long(), self.cfg.data.S)
            ll = F.log_softmax(logits, dim=-1)
            loss = -torch.sum(ll * x0_onehot, dim=-1) * self.cfg.ce_coeff
        else:
            ll_all, ll_xt = get_logprob_with_logits(self.cfg, model, xt, ts, logits) # 
            ll_xt = ll_xt * (1 - self.cfg.ce_coeff)
            ll_all = ll_all * (1 - self.cfg.ce_coeff)
            loss = self._comp_loss(model, xt, ts, ll_all, ll_xt) * (
                1 - self.cfg.ce_coeff
            )
        print("type")
        print(type(loss), loss)
        print(type(B), B)
            # loss type new param
        end_calc = time.time()
        print("calc loss time", end_calc - start_calc)
        return torch.sum(loss) / B
        # calc loss from CondFactorizedBackwardModel
