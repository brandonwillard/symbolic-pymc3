import numpy as np
import theano
import theano.tensor as tt

# Don't let pymc3 play with this setting!
_ctv = tt.config.compute_test_value
import pymc3 as pm
tt.config.compute_test_value = _ctv

from warnings import warn

from multipledispatch import dispatch

from theano.gof import FunctionGraph
# from theano.gof.toolbox import Feature
from theano.gof.graph import inputs as tt_inputs, clone_get_equiv

from . import (observed,
               NormalRV, NormalRVType,
               MvNormalRV, MvNormalRVType)
from .rv import RandomVariable
from .utils import replace_nodes


@dispatch(RandomVariable)
def convert_rv_to_pymc(rv):
    # TODO: How best to do this?
    if hasattr(rv, 'fgraph') and hasattr(rv.fgraph, 'shape_feature'):
        shape = rv.fgraph.shape_feature.shape_tuple(rv)
    else:
        shape = rv.shape

    return _convert_rv_to_pymc(rv, shape)


@dispatch(pm.Normal, object)
def convert_pymc_to_rv(dist, rng):
    # size = dist.shape.astype(int)
    size = None
    res = NormalRV(dist.mu, dist.sd, size=size, rng=rng)
    return res


@dispatch(NormalRVType, object)
def _convert_rv_to_pymc(rv, shape):
    res = pm.NormalRV(rv.inputs[0], rv.inputs[1], shape=shape)
    return res


@dispatch(pm.MvNormal, object)
def convert_pymc_to_rv(dist, rng):
    # size = dist.shape[1:].astype(int)
    size = None
    res = MvNormalRV(dist.mu, dist.cov, size=size, rng=rng)
    return res


@dispatch(MvNormalRVType, object)
def _convert_rv_to_pymc(rv, shape):
    res = pm.MvNormal(rv.inputs[0], rv.inputs[1], shape=shape)
    return res


# TODO: More RV conversions!


def model_graph(pymc_model, output_vars=None, convert_rvs=True,
                rand_state=None):
    """Convert a PyMC3 model into a Theano `FunctionGraph`.

    Parameters
    ==========
    pymc_model: `Model`
        A PyMC3 model object.
    output_vars: list (optional)
        Variables to use as `FunctionGraph` outputs.  If not specified,
        the model's observed random variables are used.
    convert_rvs: bool (optional)
        Convert the PyMC3 random variables to `RandomVariable`s.
    rand_state: Numpy rng (optional)
        When converting to `RandomVariable`s, use this random state object.

    Results
    =======
    out: `FunctionGraph`
    """

    model = pm.modelcontext(pymc_model)

    if output_vars is None:
        output_vars = [o for o in model.observed_RVs]

    if not output_vars:
        raise ValueError('No derived or observable variables specified')

    model_inputs = [
        inp for inp in tt_inputs(model.unobserved_RVs + output_vars)
    ]

    model_memo = clone_get_equiv(model_inputs, output_vars,
                                 copy_orphans=False)

    # Make sure the distribution and observation info sticks around through all
    # the object cloning.
    for k, v in model_memo.items():
        if hasattr(k, 'distribution'):
            v.tag.distribution = k.distribution
        if hasattr(k, 'observations'):
            v.tag.observations = k.observations

    fg_features = [tt.opt.ShapeFeature()]
    model_fg = FunctionGraph([model_memo[i] for i in model_inputs],
                             [model_memo[i] for i in output_vars],
                             clone=False, features=fg_features)
    model_fg.memo = model_memo
    model_fg.rev_memo = {v: k for k, v in model_memo.items()}

    if convert_rvs:
        convert_pymc3_rvs(model_fg, clone=False, rand_state=rand_state)

    return model_fg


def convert_pymc3_rvs(fgraph, clone=True, rand_state=None):
    """Replace PyMC3 random variables with `RandomFunction` Ops.

    TODO: Could use a Theano graph `Feature` to trace--or even
    replace--random variables.

    Parameters
    ----------
    fgraph: FunctionGraph
        A graph containing PyMC3 random variables.

    clone: bool, optional
        Clone the original graph.

    rand_state: RandomStateType, optional
        The Theano random state.

    Returns
    -------
    out: A cloned graph with random variables replaced and a `memo` attribute.
    """
    if clone:
        fgraph_, fgraph_memo_ = fgraph.clone_get_equiv(attach_feature=False)
        fgraph_.memo = fgraph_memo_
        fgraph_.rev_memo = {v: k for k, v in fgraph_memo_.items()}
    else:
        fgraph_ = fgraph
        assert hasattr(fgraph_, 'memo')
        assert hasattr(fgraph_, 'rev_memo')

    if rand_state is None:
        rand_state = theano.shared(np.random.RandomState())

    # fgraph_replacements = {}
    nodes = set(o for o in fgraph_.outputs
                if hasattr(o.tag, 'distribution'))
    while nodes:
        pm_var = nodes.pop()
        dist = pm_var.tag.distribution
        new_rv = convert_pymc_to_rv(dist, rand_state)

        new_rv.name = pm_var.name

        if isinstance(pm_var, pm.model.ObservedRV):
            new_rv = observed(pm_var.tag.observations, new_rv)

        # Let's attempt to fix the PyMC3 broadcastable dims "oracle" issue,
        # if present.  We'll basically find the dimensions PyMC3 says
        # are broadcastable--but don't need to be--and restrict our
        # `RandomVariable`s to be broadcastable there, too.
        diff_bcasts = tuple(
            i for i, (a, b) in enumerate(
                zip(pm_var.type.broadcastable,
                    new_rv.type.broadcastable))
            if a > b)

        if len(diff_bcasts) > 0:
            warn(f'The tensor type for {pm_var} has an overly restrictive'
                    ' broadcast dimension.  Try re-creating the model without'
                    ' specifying a shape with a dimension value of 1'
                    ' (e.g. `(1,)`).')
            new_rv = tt.addbroadcast(new_rv, *diff_bcasts)

        # The variables in these distribution objects--and the new
        # `RandomVariable`--are *not* the same as the ones in the fgraph
        # object (those ones are clones)!
        # We need to use the memo mappings to replace those old variables.
        nodes_updates = replace_nodes(
            tt_inputs([new_rv]), [new_rv], fgraph_.memo)

        new_rv = nodes_updates.get(new_rv, new_rv)

        # Update the memo so that next time around we don't
        # needlessly clone objects already in the fgraph.
        fgraph_.memo.update(nodes_updates)

        for i in tt_inputs([new_rv]):
            # We've probably introduced new inputs via the distribution
            # parameters.
            if i not in fgraph_.inputs:
                fgraph_.add_input(i)

            # Make sure we don't lose distribution information on any new
            # PyMC3 RVs that weren't already in the fgraph.  Also, add any
            # new PyMC3 RVs that need to be converted.
            if hasattr(i, 'distribution'):
                i.tag.distribution = i.distribution
                nodes.add(i)
            elif hasattr(i.tag, 'distribution'):
                nodes.add(i)

        # Finally, replace the old PyMC3 RV with the new one.
        fgraph_.replace(pm_var, new_rv)

        # Finally, remove the unused inputs.  For instance, if the original
        # inputs were PyMC3 RVs, then they've been replaced; however,
        # `FunctionGraph.replace` won't remove them for some reason.
        # TODO: Put in an issue/PR?
        if pm_var in fgraph_.inputs:
            fgraph_.inputs.remove(pm_var)

    return fgraph_


# TODO: Create this?  Could be a nicer way to accomplish the same thing
# during the initial `FunctionGraph` creation.
# class PymcToRandomVariable(Feature):
#     """Converts PyMC3 random variables to `RandomVariable`s.
#     """
#     def on_attach(self, function_graph):
#         # TODO: Loop through variables and convert `pm.graph.Factor`s to
#         # `RandomVariable`s.
#         # Also, keep a map of replacements
#
#     def on_detach(self, function_graph):
#         pass
#
#     def on_import(self, function_graph, node, reason):
#         pass
#
#     def on_prune(self, function_graph, node, reason):
#         pass
#
#     def on_change_input(self, function_graph, node, i, r, new_r, reason=None):
#         pass
#
#     def orderings(self, function_graph):
#         pass