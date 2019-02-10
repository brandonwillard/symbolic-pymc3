# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::theano-object-tools][theano-object-tools]]
import theano
import theano.tensor as tt

from collections import OrderedDict

from theano.gof import FunctionGraph, Query
from theano.gof.graph import (inputs as tt_inputs, clone_get_equiv,
                              io_toposort)
from theano.compile import optdb


from .meta import MetaSymbol, _check_eq


canonicalize_opt = optdb.query(Query(include=['canonicalize']))


def replace_nodes(inputs, outputs, replacements,
                  memo=None, clone_inputs=True):
    """Recreate a graph, replacing some variables according to a given map.

    This is helpful if you want to replace the variable dependencies of
    an existing variable according to a `clone_get_equiv` map and/or
    replacement variables that already exist within a `FunctionGraph`.

    The latter is especially annoying, because you can't simply make a
    `FunctionGraph` for the variable to be adjusted and then use that to
    perform the replacement; if the variables to be replaced are already in a
    `FunctionGraph` any such replacement will err-out saying "...these
    variables are already owned by another graph..."

    Parameters
    ==========
    inputs: list
        List of input nodes.
    outputs: list
        List of output nodes.  Everything between `inputs` and these `outputs`
        is the graph under consideration.
    replacements: dict
        A dictionary mapping existing nodes to their new ones.
    memo: dict (optional)
        A dictionary to update with the initial `replacements` and maps from
        any old-to-new nodes arising from an actual replacement.
    clone_inputs: bool (optional)
        If enabled, clone all the input nodes that aren't mapped in
        `replacements`.  These cloned nodes are mapped in `memo`, as well.

    Results
    =======
    out: memo
    """
    if memo is None:
        memo = {}
    memo.update(replacements)
    for apply in io_toposort(inputs, outputs):
        if clone_inputs:
            apply_inputs = [memo.setdefault(i, i.clone())
                            for i in apply.inputs]
        else:
            apply_inputs = apply.inputs

        if any(i in memo for i in apply.inputs):
            new_apply = apply.clone_with_new_inputs(
                [memo.get(i, apply_inputs[n])
                 for n, i in enumerate(apply.inputs)])

            memo.setdefault(apply, new_apply)

            for output, new_output in zip(apply.outputs, new_apply.outputs):
                memo.setdefault(output, new_output)
    return memo


def parts_unequal(x, y):
    """Traverse meta objects and return the first pair of elements
    that are not equal.
    """
    if type(x) != type(y):
        print('unequal types')
        return x, y
    elif isinstance(x, MetaSymbol):
        if x.base != y.base:
            print('unequal bases')
            return x.base, y.base
        for a, b in zip(x.rands(), y.rands()):
            z = parts_unequal(a, b)
            if z is not None:
                return z
    elif isinstance(x, (tuple, list)):
        for a, b in zip(x, y):
            z = parts_unequal(a, b)
            if z is not None:
                return z
    elif not _check_eq(x, y):
        return x, y


def expand_meta(x, tt_print=tt.pprint):
    """Produce a dictionary representation of a meta object."""
    if isinstance(x, MetaSymbol):
        return OrderedDict([('rator', x.base),
                            ('rands', tuple(expand_meta(p)
                                            for p in x.rands())),
                            ('obj', expand_meta(getattr(x, 'obj', None)))])
    elif tt_print and isinstance(x, theano.gof.op.Op):
        return x.name
    elif tt_print and isinstance(x, theano.gof.graph.Variable):
        return tt_print(x)
    else:
        return x


def graph_equal(x, y):
    """Compare elements in a Theano graph using their object properties and not
    just identity.
    """
    try:
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return (len(x) == len(y) and
                    all(MetaSymbol.from_obj(xx) == MetaSymbol.from_obj(yy)
                        for xx, yy in zip(x, y)))
        return MetaSymbol.from_obj(x) == MetaSymbol.from_obj(y)
    except ValueError:
        return False
# theano-object-tools ends here


# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::mt_type_params][mt_type_params]]
def mt_type_params(x):
    return {'ttype': x.type, 'index': x.index, 'name': x.name}
# mt_type_params ends here


# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::theano-optimize-helper][theano-optimize-helper]]
def optimize_graph(x, optimization):
    """Apply an optimization to either the graph formed by a Theano variable or
    an existing graph and return the resulting optimized graph.

    When given an existing `FunctionGraph`, the optimization is performed
    without side-effects (i.e. won't change the given graph).
    """
    if not isinstance(x, FunctionGraph):
        inputs = tt_inputs([x])
        outputs = [x]
        model_memo = clone_get_equiv(inputs, outputs,
                                     copy_orphans=False)
        cloned_inputs = [model_memo[i] for i in inputs]
        cloned_outputs = [model_memo[i] for i in outputs]

        x_graph = FunctionGraph(cloned_inputs, cloned_outputs, clone=False)
        x_graph.memo = model_memo
    else:
        x_graph = x

    x_graph_opt = x_graph.clone()
    optimization.optimize(x_graph_opt)
    return x_graph_opt.outputs[0]


def canonicalize(x):
    """Canonicalize a Theano variable and/or graph.
    """
    return optimize_graph(x, canonicalize_opt)


# theano-optimize-helper ends here
