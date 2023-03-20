import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Extension(nn.Module):

    def __init__(self, temperature=0., scale_by_temperature=True):
        super(Extension, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None: 
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:  
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # create mask
        logits_mask = torch.ones_like(mask) - (torch.eye(batch_size)).cuda()
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
       
        num_positives_per_row = torch.sum(positives_mask, axis=1)  
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

class SoftCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, label, weight=None):
        assert logit.size() == label.size(), "logit.size() != label.size()" 
        dim = logit.dim() 
        max_logit = logit.max(dim - 1, keepdim=True)[0] 
        logit = logit - max_logit 
                                  
        exp_logit = logit.exp() 
        exp_sum = exp_logit.sum(dim - 1, keepdim=True)
        prob = exp_logit / exp_sum  
        log_exp_sum = exp_sum.log() 
        neg_log_prob = log_exp_sum - logit 

        if weight is None:
            weighted_label = label
        else:
            if weight.size() != (logit.size(-1),):
                raise ValueError(
                    "since logit.size() = {}, weight.size() should be ({},), but got {}".format(
                        logit.size(),
                        logit.size(-1),
                        weight.size(),
                    )
                )
            size = [1] * label.dim()
            size[-1] = label.size(-1)
            weighted_label = label * weight.view(size)
        ctx.save_for_backward(weighted_label, prob)
        out = (neg_log_prob * weighted_label).sum(dim - 1)  
                                                            
        return out

    @staticmethod
    def backward(ctx, grad_output): 
        weighted_label, prob = ctx.saved_tensors
        old_size = weighted_label.size() 
        # num_classes
        K = old_size[-1]  
        # batch_size
        B = weighted_label.numel() // K 
                                         

        grad_output = grad_output.view(B, 1) 
        weighted_label = weighted_label.view(B, K)
        prob = prob.view(B, K)
        grad_input = grad_output * (prob * weighted_label.sum(1, True) - weighted_label)
        grad_input = grad_input.view(old_size) 
        return grad_input, None, None


def soft_cross_entropy(logit, label, weight=None, reduce=None, reduction="mean"):
    if weight is not None and weight.requires_grad:
        raise RuntimeError("gradient for weight is not supported")
    losses = SoftCrossEntropyFunction.apply(logit, label, weight) 
    reduction = {
        True: "mean",
        False: "none",
        None: reduction,
    }[reduce]  
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses 
    else:
        raise ValueError("invalid value for reduction: {}".format(reduction))


class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, reduce=None, reduction="mean"):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, logit, label, weight=None):
        if weight is None:
            weight = self.weight
        return soft_cross_entropy(logit, label, weight, self.reduce, self.reduction)


def rew_ce(logits, labels, sample_weights):
    losses = F.cross_entropy(logits, labels, reduction="none") 
    return (losses * sample_weights.type_as(losses)).mean() 


def rew_sce(logits, soft_labels):
    losses = soft_cross_entropy(logits, soft_labels, reduce=False)
    #return torch.mean(losses * sample_weights.type_as(losses)) 
    return torch.mean(losses.type_as(losses))

def prob_2_entropy(softmax_prob):
    n, c = softmax_prob.size()
    entropy = -torch.mul(softmax_prob, torch.log2(softmax_prob + 1e-30))
    return torch.mean(entropy, dim=1) / np.log2(c)


def log_sum_exp(value, dim=None, keepdim=False):   #value is logits
    weight_energy = torch.nn.Linear(10, 1).cuda()
    torch.nn.init.uniform_(weight_energy.weight)
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)  #MAX_logits
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)   #压缩维度
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
