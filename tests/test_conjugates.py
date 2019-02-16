# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::conjugate-rules-tests][conjugate-rules-tests]]
import pytest

import theano
import theano.tensor as tt
import numpy as np

from theano.gof.opt import EquilibriumOptimizer
from theano.gof import FunctionGraph
from theano.gof.graph import inputs as tt_inputs, ancestors

from spymc3 import MvNormalRV
from spymc3.opt import KanrenRelationSub
from spymc3.utils import optimize_graph
from spymc3.relations.conjugates import observed, conjugate_posteriors

theano.config.mode = 'FAST_COMPILE'
theano.config.cxx = ''
# conjugate-rules-tests ends here


# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::mvnormal-posterior-rule-test][mvnormal-posterior-rule-test]]
def test_mvnormal_mvnormal():
    theano.config.compute_test_value = 'ignore'
    a_tt = tt.vector('a')
    R_tt = tt.matrix('R')
    F_t_tt = tt.matrix('F')
    V_tt = tt.matrix('V')

    a_tt.tag.test_value = np.r_[1., 0.]
    R_tt.tag.test_value = np.diag([10., 10.])
    F_t_tt.tag.test_value = np.c_[-2., 1.]
    V_tt.tag.test_value = np.diag([0.5])

    beta_rv = MvNormalRV(a_tt, R_tt, name='\\beta')

    E_y_rv = F_t_tt.dot(beta_rv)
    Y_rv = MvNormalRV(E_y_rv, V_tt, name='Y')

    y_tt = tt.as_tensor_variable(np.r_[-3.])
    y_tt.name = 'y'
    Y_obs = observed(y_tt, Y_rv)

    fgraph = FunctionGraph(tt_inputs([beta_rv, Y_obs]),
                           [beta_rv, Y_obs],
                           clone=True)

    posterior_opt = EquilibriumOptimizer(
        [KanrenRelationSub(conjugate_posteriors)],
        max_use_ratio=10)

    fgraph_opt = optimize_graph(fgraph, posterior_opt)

    # Make sure that it removed the old, integrated observation distribution.
    assert fgraph_opt[1].owner.inputs[1].equals(tt.NoneConst)

    # Check that the SSE has decreased from prior to posterior.
    # TODO: Use a better test.
    beta_prior_mean_val = a_tt.tag.test_value
    F_val = F_t_tt.tag.test_value
    beta_post_mean_val = fgraph_opt[0].owner.inputs[0].tag.test_value
    priorp_err = np.square(
        y_tt.data - F_val.dot(beta_prior_mean_val)).sum()
    postp_err = np.square(
        y_tt.data - F_val.dot(beta_post_mean_val)).sum()

    # First, make sure the prior and posterior means are simply not equal.
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        priorp_err, postp_err)
    # Now, make sure there's a decrease (relative to the observed point).
    np.testing.assert_array_less(postp_err, priorp_err)

# mvnormal-posterior-rule-test ends here
