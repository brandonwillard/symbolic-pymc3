# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::posterior-rules-imports][posterior-rules-imports]]
import theano.tensor as tt

from theano.tensor.nlinalg import matrix_inverse
from theano.sandbox.linalg.ops import Hint, psd

from unification import var
from kanren import conde, eq
from kanren.facts import Relation, fact

from .. import MvNormalRV, observed
from ..rv import RandomVariable
from ..meta import mt
from ..utils import mt_type_params


# posterior-rules-imports ends here


# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::mvnormal-posterior-rule][mvnormal-posterior-rule]]
# Create the pattern/form of the prior normal distribution
beta_name_lv = var('beta_name')
beta_size_lv = var('beta_size')
beta_rng_lv = var('beta_rng')
a_lv = var('a')
R_lv = var('R')
beta_prior_mt = mt.MvNormalRV(a_lv, R_lv,
                              beta_size_lv, beta_rng_lv,
                              name=beta_name_lv)
beta_type_lvars = mt_type_params(beta_prior_mt)

y_name_lv = var('y_name')
y_size_lv = var('y_size')
y_rng_lv = var('y_rng')
F_t_lv = var('f')
V_lv = var('V')
E_y_mt = mt.dot(F_t_lv, beta_prior_mt)
Y_mt = mt.MvNormalRV(E_y_mt, V_lv,
                     y_size_lv, y_rng_lv,
                     name=y_name_lv)

# The variable specifying the fixed sample value of the random variable
# given by `Y_mt`
y_mt = var('y')

# TODO: It's a little restrictive that we limit this conjugate
# update to only those random variables attached to observations.
# Right now it serves as a way to pull in a distinct tensor carrying
# observation values, since the tensor corresponding to a sample
# from the observation distribution (i.e. `Y_mt`) cannot be a shared
# or constant tensor with a user-set value.
# We can simply add the non-observation pattern, let it do the Bayes
# update, then follow that up with an relation/optimization that exchanges
# the `RandomVariable` with its associated observed tensor (e.g. swap
# `Y_mt` with `y_mt` once all the conjugate updates are done).  This
# would have to be restricted to updated `RandomVariables` that are
# also `FunctionGraph` outputs (i.e. the ones we might want to sample).

# Make the observation relationship explicit in the graph.
Y_obs_mt = mt.observed(y_mt, Y_mt)

# Create tuple-form expressions for the posterior
e_expr = mt.sub(Y_obs_mt, mt.dot(F_t_lv, a_lv))
F_expr = (mt.transpose, F_t_lv)
R_F_expr = (mt.dot, R_lv, F_expr)
Q_expr = (mt.add,
          V_lv,
          (mt.dot,
           F_t_lv,
           R_F_expr))
A_expr = (mt.dot, R_F_expr, (mt.matrix_inverse, Q_expr))
# m = C \left(F V^{-1} y + R^{-1} a\right)
m_expr = (mt.add, a_lv, (mt.dot, A_expr, e_expr))
# C = \left(R^{-1} + F V^{-1} F^{\top}\right)^{-1}
# TODO: We could use the naive posterior forms and apply identities, like
# Woodbury's, in another set of "simplification" relations.
# In some cases, this might make the patterns simpler and more broadly
# applicable.
C_expr = (mt.sub,
          R_lv,
          (mt.dot,
           (mt.dot, A_expr, Q_expr),
           (mt.transpose, A_expr)))

norm_posterior_exprs = (mt.MvNormalRV,
                        m_expr, C_expr,
                        y_size_lv, y_rng_lv)

conjugate = Relation('conjugate')

fact(conjugate,
     # MvNormal likelihood, MvNormal prior mean
     Y_obs_mt, norm_posterior_exprs)


def conjugate_posteriors(x, y):
    """A goal relating conjugate priors and their posterior forms.

    This goal unifies `y` with a tuple-form expression for a dictionary
    specifying the graph's node replacements.
    Those replacements map prior random variable terms to their posteriors
    forms.  All other terms depending on those terms essentially become
    posterior predictives.
    """
    z = var()
    return (conde, ((conjugate, x, z),
                    (eq, y,
                     (dict,
                      [
                          # Replace observation with one that doesn't link to
                          # the integrated one
                          (Y_obs_mt, (mt.observed, y_mt, None)),
                          # (Y_mt, None),
                          # Replace the prior with the posterior
                          (beta_prior_mt, z),
                      ]))))
# mvnormal-posterior-rule ends here
