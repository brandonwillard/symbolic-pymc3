import pytest

import theano
import theano.tensor as tt
import numpy as np

from theano.gof import FunctionGraph
from theano.gof.opt import EquilibriumOptimizer
from theano.gof.graph import inputs as tt_inputs

from kanren import run, var

from spymc3 import (NormalRV, HalfCauchyRV, ExponentialRV, GammaRV,
                    TruncExponentialRV, observed)
from spymc3.opt import KanrenRelationSub
from spymc3.utils import optimize_graph
from spymc3.relations.scale_mixtures import slice_samplers


theano.config.mode = 'FAST_COMPILE'
theano.config.cxx = ''


@pytest.mark.skip(reason="Not done, yet.")
def test_horseshoe():
    """Confirm that we can produce a slice sampler for the standard HS model by
    using scale mixture representations.
    """
    theano.config.compute_test_value = 'ignore'

    N_tt = tt.iscalar('N')
    N_tt.tag.test_value = 10

    lambda_tt = HalfCauchyRV(size=N_tt, name='\\lambda')
    tau_tt = HalfCauchyRV(name='\\tau')

    theta_rv = NormalRV(tt.zeros((N_tt,)),
                        tt.sqrt(lambda_tt * tau_tt),
                        name='\\theta')

    Y_rv = NormalRV(theta_rv,
                    tt.ones((N_tt,)),
                    name='Y')

    y_tt = tt.as_tensor_variable(Y_rv.tag.test_value)
    y_tt.name = 'y'
    Y_obs = observed(y_tt, Y_rv)

    samplers = run(0, var('q'), slice_samplers(Y_obs, var('q')))

    sampler = samplers[0]

    sampler.eval()

    # fgraph = FunctionGraph(
    #     tt_inputs([lambda_tt, tau_tt, theta_rv, Y_obs]),
    #     [lambda_tt, tau_tt, theta_rv, Y_obs],
    #     clone=True)
    #
    # slice_opt = EquilibriumOptimizer(
    #     [KanrenRelationSub(slice_samplers)],
    #     max_use_ratio=10)
    # fgraph_opt = optimize_graph(fgraph, slice_opt)
