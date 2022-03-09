import tensorly
from tools.decomposition import VBMF


def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """

    weights = layer.weight.data.numpy()
    # weights = layer
    unfold_0 = tensorly.base.unfold(weights, 0)
    unfold_1 = tensorly.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks