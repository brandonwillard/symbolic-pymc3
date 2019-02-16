import theano.tensor as tt
import numpy as np

from theano.tensor.nlinalg import matrix_inverse

from unification import var
from kanren import conde, eq
from kanren.facts import Relation, fact

from spymc3 import (NormalRV, HalfCauchyRV, ExponentialRV, GammaRV,
                    UniformRV, TruncExponentialRV, observed)
from spymc3.meta import mt, MetaSymbol
from spymc3.utils import mt_type_params

from spymc3.relations import hierarchical_model


conde_clauses = tuple()


def create_normal_hs_slice():
    """A simple slice sampler for the normal observation Horseshoe prior.

    Bhadra, Anindya, Jyotishka Datta, Nicholas G. Polson, and Brandon
    Willard. 2016. “Default Bayesian Analysis with Global-Local Shrinkage
    Priors.” Biometrika 103 (4): 955–69. https://doi.org/10.1093/biomet/asw041.
    """
    term_goals = tuple()

    # XXX: Debug.
    # from theano.printing import debugprint as tt_dprint

    # TODO: Should work for all types of ints.
    # N_dtype_mt = var('N_dtype')
    # N_mt = mt.scalar(dtype=N_dtype_mt)
    # term_goals += tuple((membero, N_dtype_mt, tt.integer_dtypes))
    N_mt = mt.scalar(dtype='int32')
    N_mt.name = var()

    # TODO: Remember to make a normalization/canonicalization relation/rule
    # that combines vectors/arrays of independent scalar RVs into single one
    # with a non-empty `size` parameter.
    # TODO: Likewise, it's probably best to make this pattern match the
    # canonicalized forms of these kinds of graphs.
    lamda_prior_mt = mt.HalfCauchyRV(size=N_mt, name=var())
    tau_prior_mt = mt.HalfCauchyRV(name=var())

    a_mt = mt.zeros((N_mt,))
    R_mt = mt.mul(lamda_prior_mt, tau_prior_mt)
    theta_prior_mt = mt.NormalRV(a_mt, mt.sqrt(R_mt), name=var())

    Y_mt = mt.NormalRV(theta_prior_mt,
                       mt.ones((N_mt,)),
                       name=var())

    y_mt = var('obs_sample')
    Y_obs_mt = mt.observed(y_mt, Y_mt)

    kappa_initial_mt = mt.inv(mt.add(1., mt.square(R_mt)))
    tau_2_initial_mt = mt.square(tau_prior_mt)
    tau_2_min_1_mt = mt.sub(tau_2_initial_mt, 1.)

    omega_mt = mt.ExponentialRV(
        mt.add(1., mt.mul(kappa_initial_mt, tau_2_min_1_mt)))
    gamma_mt = mt.ExponentialRV(mt.square(mt.add(1., tau_2_initial_mt)))

    one_min_kap_inv_mt = mt.inv(mt.sqrt(mt.sub(1., kappa_initial_mt)))

    u_mt = mt.UniformRV(0., one_min_kap_inv_mt)
    tau_2_dist_mt = (mt.GammaRV,
                     mt.true_div(mt.add(1., N_mt), 2.),
                     (mt.add,
                      gamma_mt,
                      (mt.sum, mt.mul(kappa_initial_mt, omega_mt))))
    kappa_dist_mt = (
        mt.TruncExponentialRV,
        mt.maximum(mt.sub(1., mt.inv(mt.square(u_mt))), 0.),
        1.,
        (mt.add,
         mt.true_div(mt.square(Y_obs_mt), 2.),
         (mt.mul,
          omega_mt,
          (mt.sub,
           tau_2_dist_mt,
           1.))))

    theta_post_mt = (mt.NormalRV,
                     (mt.mul, (mt.min, 1., kappa_dist_mt), Y_obs_mt),
                     (mt.sqrt, (mt.min, 1., kappa_dist_mt)))

    lambda_post_mt = (
        mt.sqrt,
        (mt.true_div,
         (mt.sub, (mt.inv, kappa_dist_mt), 1),
         tau_2_dist_mt))

    tau_post_mt = (mt.sqrt, tau_2_dist_mt)

    replacements = {Y_mt: mt(tt.NoneConst.clone()),
                    theta_prior_mt: theta_post_mt,
                    lamda_prior_mt: lambda_post_mt,
                    tau_prior_mt: tau_post_mt}

    fact(hierarchical_model,
         Y_obs_mt,
         tuple(replacements.items()))

    return term_goals


def slice_samplers(x, y):
    """A goal relating [sub]models to slice sampler reformulations.
    """
    z = var()

    # First, find a basic hierarchical model structure match.
    goals = tuple((hierarchical_model, x, z))

    # Second, each corresponding sampler case (and their special
    # conditions).
    goals += tuple((conde,) + conde_clauses)

    # Third, connect the discovered pieces and produce the necessary output.
    # TODO: We could have a "reifiable" goal that makes sure the output is
    # a valid base/non-meta object.
    goals += tuple((eq, y, z))

    # This conde is just a lame way to form the conjunction
    # TODO: Use one of the *all* functions.
    res = (conde, goals)
    return res
