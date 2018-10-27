

from sandbox.rocky.tf.algos.maesn_npo import MAESN_NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class MAESN_TRPO(MAESN_NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(MAESN_TRPO, self).__init__(optimizer=optimizer, **kwargs)
