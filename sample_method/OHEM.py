"""
this is a pytorch implement of OHEM
what online  means in this regard.

OHEM solves these two problems by performing hard example selection batch-wise
given a batch sized k, it performs regular forward propagation abd computes per instance
losses. Then , it finds M<k hard examples in the batch with high loss values.

it only back-propagates the loss computed over the selelcted instances.

"""
import torch as th
import torch.nn as nn
class NLLL_OHEM(nn.NLLLoss):
    """
    inputs nn.LogSoftmax
    """
    def __init__(self, ratio):
        super(NLLL_OHEM, self).__init__(None, True)
        self.ratio = ratio
    def forward(self, x, y, ratio =None):
        if ratio is not None:
            self.ratio = ratio
        num_inst =x.size(0)
        num_hns = int(self.ratio*num_inst)
        x_ = x.clone()
        inst_losses = th.autograd.Variable(th.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return th.nn.functional.nll_loss(x_hn, y_hn)










