import torch
import torch.nn.functional as F
import lib.utils.utils as utils

_MODELS = {}


def register_model(cls):
    name = cls.__name__
    if name in _MODELS:
        raise ValueError(f"{name} is already registered!")
    _MODELS[name] = cls
    return cls


def get_model(name):
    return _MODELS[name]


def create_model(cfg, device, encoding=None, rank=None):
    if encoding == None:
        model = get_model(cfg.model.name)(cfg, device, rank)
    else:
        model = get_model(cfg.model.name)(cfg, device, encoding, rank)
    model = model.to(device)

    return model


def get_logprob_with_logits(cfg, model, xt, t, logits, xt_target=None):
    """Get logprob with logits."""
    # model = state["model"] # copy expensive?
    # mabye less expensive to insert qt0 = model.transition(t) ?
    # checked
    if xt_target is None:
        xt_target = xt
    xt_onehot = F.one_hot(xt_target.long(), cfg.data.S)
    if cfg.loss.logit_type == "direct":
        log_prob = F.log_softmax(logits, dim=-1)
    else:
        qt0 = model.transition(t)
        if cfg.loss.logit_type == "reverse_prob":
            p0t = F.softmax(logits, dim=-1)
            qt0 = utils.expand_dims(qt0, axis=list(range(1, xt.dim() - 1)))
            prob_all = p0t @ qt0
            log_prob = torch.log(prob_all + 1e-35)

            # check
        elif cfg.loss.logit_type == "reverse_logscale":
            log_p0t = F.log_softmax(logits, dim=-1)
            log_qt0 = torch.where(qt0 <= 1e-35, -1e9, torch.log(qt0))
            log_qt0 = utils.expand_dims(log_qt0, axis=list(range(1, xt.dim())))
            log_p0t = log_p0t.unsqueeze(-1)
            log_prob = torch.logsumexp(log_p0t + log_qt0, dim=-2)
            # check
        else:
            raise ValueError("Unknown logit_type: %s" % cfg.loss.logit_type)
    log_xt = torch.sum(log_prob * xt_onehot, dim=-1)

    return log_prob, log_xt
