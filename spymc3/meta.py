# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::theano-meta-imports][theano-meta-imports]]
import abc
import types
import inspect

import numpy as np

import theano
import theano.tensor as tt

from itertools import chain
from functools import partial, wraps
from collections.abc import Iterator

from unification import var, isvar, Var

from .rv import RandomVariable

# TODO: Replace `from_obj` with a dispatched function?
# from multipledispatch import dispatch
# theano-meta-imports ends here


# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::theano-meta-objects][theano-meta-objects]]
def _meta_reify_iter(rands):
    # We want as many of the rands reified as possible,
    any_unreified = False
    reified_rands = []
    for s in rands:
        if isinstance(s, MetaSymbol):
            rrand = s.reify()
            reified_rands += [rrand]
            any_unreified |= isinstance(rrand, MetaSymbol)
            any_unreified |= isvar(rrand)
        elif MetaSymbol.is_meta(s):
            reified_rands += [s]
            any_unreified |= True
        elif isinstance(s, (list, tuple)):
            _reified_rands, _any_unreified = _meta_reify_iter(s)
            reified_rands += [type(s)(_reified_rands)]
            any_unreified |= _any_unreified
        else:
            reified_rands += [s]

    return reified_rands, any_unreified


def _check_eq(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b


class MetaSymbolType(abc.ABCMeta):
    def __new__(cls, name, bases, clsdict):

        # We need to track the cumulative slots, because subclasses can define
        # their own--yet we'll need to track changes across all of them.
        all_slots = set(chain.from_iterable(
            s.__all_slots__ for s in bases
            if hasattr(s, '__all_slots__')))
        all_slots |= set(clsdict.get('__slots__', []))
        clsdict['__all_slots__'] = all_slots

        def __setattr__(self, attr, obj):
            """If a slot value is changed, discard any associated non-meta/base
            objects.
            """
            if (getattr(self, 'obj', None) is not None and
                    not isinstance(self.obj, Var) and
                    attr in getattr(self, '__all_slots__', {}) and
                    hasattr(self, attr) and getattr(self, attr) != obj):
                self.obj = None
            elif attr == 'obj':
                if isinstance(obj, MetaSymbol):
                    raise ValueError('base object cannot be a meta object!')

            object.__setattr__(self, attr, obj)

        clsdict['__setattr__'] = __setattr__

        res = super().__new__(cls, name, bases, clsdict)

        # TODO: Could register base classes.
        # E.g. cls.register(bases)
        return res


class MetaSymbol(metaclass=MetaSymbolType):
    """Meta objects for unification and such.
    """
    @property
    @abc.abstractmethod
    def base(self):
        """The base type/rator for this meta object.
        """
        pass

    @classmethod
    def base_classes(cls, mro_order=True):
        res = tuple(c.base for c in cls.__subclasses__())
        if cls is not MetaSymbol:
            res = (cls.base,) + res
        sorted(res, key=lambda cls: len(cls.mro()), reverse=mro_order)
        return res

    @classmethod
    def is_meta(cls, obj):
        return isinstance(obj, MetaSymbol) or isvar(obj)

    @classmethod
    def from_obj(cls, obj):
        """Create a meta object for a given base object.

        XXX: Be careful when overriding this: `isvar` checks are necessary!
        """
        if (cls.is_meta(obj) or obj is None or
                isinstance(obj, (types.FunctionType, partial,
                                 str, dict))):
            return obj

        if isinstance(obj, (set, list, tuple, Iterator)):
            # Convert elements of the iterable
            return type(obj)([cls.from_obj(o) for o in obj])

        if inspect.isclass(obj) and issubclass(obj, cls.base_classes()):
            # This is a class/type covered by a meta class/type.
            try:
                obj_cls = next(filter(lambda t: issubclass(obj, t.base),
                                      cls.__subclasses__()))
            except StopIteration:
                # The current class is the best fit.
                if cls.base == obj:
                    return cls.base
                # This object is a subclass of the base type.
                new_type = type(f'Meta{obj.__name__}', (cls,), {})
                return new_type(obj)
            else:
                return obj_cls.from_obj(obj)

        if not isinstance(obj, cls.base_classes()):
            # We might've been given something convertible to a type with a
            # meta type, so let's try that
            try:
                obj = tt.as_tensor_variable(obj)
            except (ValueError, tt.AsTensorError):
                pass

            # Check for a meta type again
            if not isinstance(obj, cls.base_classes()):
                raise ValueError(
                    'Could not find a MetaSymbol class for {}'.format(obj))

        try:
            obj_cls = next(filter(lambda t: isinstance(obj, t.base),
                                  cls.__subclasses__()))
        except StopIteration:
            res = cls(*[getattr(obj, s)
                        for s in getattr(cls, '__slots__', [])],
                      obj=obj)
        else:
            # Descend into this class to find a more suitable one, if any.
            res = obj_cls.from_obj(obj)

        return res

    def __init__(self, obj=None):
        self.obj = obj

    def rands(self):
        """Create a tuple of the meta object's operator parameters (i.e. "rands").
        """
        return tuple(getattr(self, s)
                     for s in getattr(self, '__slots__', []))

    def reify(self):
        """Create a concrete base object from this meta object (and its
        rands).
        """
        if self.obj and not isinstance(self.obj, Var):
            return self.obj
        else:
            reified_rands, any_unreified = _meta_reify_iter(self.rands())

            # If not all the rands reified, then create another meta
            # object--albeit one with potentially more non-`None` `obj` fields.
            rator = self.base if not any_unreified else type(self)
            res = rator(*reified_rands)

            if not any_unreified:
                self.obj = res

            return res

    def __eq__(self, other):
        """Syntactic equality between meta objects and their bases.
        """
        # TODO: Allow a sort of cross-inheritance equivalence (e.g. a
        # `tt.Variable` or `tt.TensorVariable`)?
        # a_sub_b = isinstance(self, type(other))
        # b_sub_a = isinstance(other, type(self))
        # if not (a_sub_b or b_sub_a):
        #     return False
        if not (type(self) == type(other)):
            return False

        # TODO: ?
        # Same for base objects
        # a_sub_b = isinstance(self.base, type(other.base))
        # b_sub_a = isinstance(other.base, type(self.base))
        # if not (a_sub_b or b_sub_a):
        #     return False
        if not (self.base == other.base):
            return False

        # TODO: ?
        # # `self` is the super class, that might be generalizing
        # # `other`
        a_slots = getattr(self, '__slots__', [])
        # b_slots = getattr(other, '__slots__', [])
        # if (b_sub_a and not a_sub_b and
        #     not all(getattr(self, attr) == getattr(other, attr)
        #             for attr in a_slots)):
        #     return False
        # # `other` is the super class, that might be generalizing
        # # `self`
        # elif (a_sub_b and not b_sub_a and
        #       not all(getattr(self, attr) == getattr(other, attr)
        #               for attr in b_slots)):
        #     return False
        if not all(_check_eq(getattr(self, attr), getattr(other, attr))
                   for attr in a_slots):
            return False

        # if (self.obj and not isvar(self.obj) and
        #         other.obj and not isvar(other.objj)):
        #     assert self.obj == other.obj

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        def _make_hashable(x):
            if isinstance(x, list):
                return tuple(x)
            elif isinstance(x, np.ndarray):
                return x.data.tobytes()
            else:
                return x
        rands = tuple(_make_hashable(p) for p in self.rands())
        return hash(rands + (self.base,))

    def __str__(self):
        obj = getattr(self, 'obj', None)
        if obj is None:
            params = self.rands()
            args = ', '.join([str(p) for p in params])
            res = '{}({})'.format(self.__class__.__name__, args)
        else:
            res = str(obj)
        return res

    def __repr__(self):
        obj = getattr(self, 'obj', None)
        args = ', '.join([repr(p) for p in self.rands()] +
                         ['obj={}'.format(repr(obj))])
        return '{}({})'.format(
            self.__class__.__name__, args)


class MetaType(MetaSymbol):
    base = theano.Type

    def __call__(self, name=None):
        if self.obj:
            return MetaSymbol.from_obj(self.obj(name=name))
        return MetaSymbol.from_obj(self.base.Variable)(self, name)


class MetaRandomStateType(MetaType):
    base = tt.raw_random.RandomStateType


class MetaTensorType(MetaType):
    base = tt.TensorType
    __slots__ = ['dtype', 'broadcastable', 'name']

    def __init__(self, dtype, broadcastable, name, obj=None):
        super().__init__(obj=obj)
        self.dtype = dtype
        self.broadcastable = broadcastable
        self.name = name


class MetaOp(MetaSymbol):
    """A meta object that represents Theano `Op`s.

    NOTE: By default it will use `Op.make_node`'s signature to produce meta
    `Apply` node inputs, so be sure to override that signature when
    `Op.make_node`'s arguments aren't one-to-one with the expected `Apply` node
    inputs.  See `MetaOp.__call__` for more details.

    Also, make sure to override `Op.out_meta_type` and make it return the
    expected meta variable type, if it isn't the default: `MetaTensorVariable`.
    """
    base = tt.Op

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_sig = inspect.signature(self.obj.make_node)

    def out_meta_type(self, inputs=None):
        """Return the type of meta variable this `Op` is expected to produce
        given the inputs.

        The default is `MetaTensorVariable` (corresponding to
        `TheanoTensorVariable` outputs from the base `Op`).
        """
        return MetaTensorVariable

    def __call__(self, *args, ttype=None, index=None, **kwargs):
        """Emulate `make_node` for this `Op` and return .

        NOTE: Meta objects will use positional arguments and non-"name" keyword
        args as `Apply` node inputs.  Also, if some of the `Op` constructor
        arguments that end up as `Apply` node input arguments are keywords,
        *use the keywords* and not their positions!

        Otherwise, if a base object can't be referenced, unknown Theano types
        and index values will be fill-in with logic variables (that can also
        be specified manually though the keyword arguments `ttype` and `index`).

        Parameters
        ==========
        ttype: object (optional)
            Value to use for an unknown Theano type.  Defaults to a logic
            variable.
        index: object (optional)
            Value to use for an unknown output index value.  Defaults to a
            logic variable.
        """
        name = kwargs.pop('name', None)

        # Use the `Op`'s default `make_node` arguments, if any.
        op_arg_bind = self.op_sig.bind(*args, **kwargs)
        op_arg_bind.apply_defaults()
        op_args, op_args_unreified = _meta_reify_iter(op_arg_bind.args)

        if not op_args_unreified:
            tt_out = self.obj(*op_args)
            res_var = MetaVariable.from_obj(tt_out)

            # If the name is indeterminate, we still want all the reified info,
            # but we need to make sure that certain parts aren't known.
            # TODO: In this case, the reified Theano object is a sort of
            # "proxy" object; we should use this approach for dtype, as well.
            # TODO: We should also put this kind of logic in the appropriate places
            # (e.g. `MetaVariable.reify`), when possible.
            if MetaSymbol.is_meta(name):
                # This should also invalidate `res_var.obj`.
                res_var.name = name
                # Allow the base object to be unified, so that reification
                # can recover the underlying object--instead of recreating
                # it and sacrificing equality.
                res_var.obj = var()

            elif tt_out.name != name:
                tt_out.name = name
                res_var.name = name

        else:
            # XXX: It's not always clear how `Op.make_node` arguments map to
            # `Apply` node inputs, which is one of the big problem with
            # Theano's design.  (More generally, it's that `Op`s don't provide
            # a spec for `Apply` node inputs and outputs at all.)

            # Also, `Apply` inputs can't be `None` (they could be
            # `tt.none_type_t()`, though).
            res_apply = MetaApply(
                self, tuple(filter(lambda x: x, op_arg_bind.args)))

            # TODO: Elemwise has an `output_types` method that can be
            # used to infer the output type of this variable.
            ttype = ttype or var()
            index = index if index is not None else var()

            # XXX: We don't have a higher-order meta object model, so being
            # wrong about the exact type of output variable will cause
            # problems.
            out_meta_type = self.out_meta_type(op_args)
            res_var = out_meta_type(ttype, res_apply, index, name)
            res_var.obj = var()

        return res_var

    def __eq__(self, other):
        # Since these have no rands/slots, we can only really compare against
        # the underlying base objects (which should be there!).
        if not super().__eq__(other):
            return False

        assert self.obj

        if self.obj != other.obj:
            return False

        return True

    def __hash__(self):
        return hash((self.base, self.obj))


class MetaElemwise(MetaOp):
    base = tt.Elemwise

    def __call__(self, *args, ttype=None, index=None, **kwargs):
        obj_nout = getattr(self.obj, 'nfunc_spec', None)
        obj_nout = obj_nout[-1] if obj_nout is not None else None
        if obj_nout == 1 and index is None:
            index = 0
        return super().__call__(*args, ttype=ttype, index=index, **kwargs)


class MetaDimShuffle(MetaOp):
    base = tt.DimShuffle
    __slots__ = ['input_broadcastable', 'new_order', 'inplace']

    def __init__(self, input_broadcastable, new_order, inplace=True, obj=None):
        super().__init__(obj=obj)
        self.input_broadcastable = input_broadcastable
        self.new_order = new_order
        self.inplace = inplace


class MetaRandomVariable(MetaOp):
    base = RandomVariable

    def __init__(self, obj=None):
        super().__init__(obj=obj)
        # The `name` keyword parameter isn't an `Apply` node input, so we need
        # to remove it from the automatically generated signature.
        self.op_sig = self.op_sig.replace(
            parameters=list(self.op_sig.parameters.values())[0:4])


class MetaApply(MetaSymbol):
    base = tt.Apply
    __slots__ = ['op', 'inputs']

    def __init__(self, op, inputs, outputs=None, obj=None):
        super().__init__(obj=obj)
        self.op = MetaOp.from_obj(op)
        self.inputs = tuple(MetaSymbol.from_obj(i) for i in inputs)
        self.outputs = outputs

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj
        else:
            tt_op = self.op.reify()
            if not self.is_meta(tt_op):
                reified_rands, any_unreified = _meta_reify_iter(self.inputs)
                if not any_unreified:
                    tt_var = tt_op(*reified_rands)
                    self.obj = tt_var.owner
                    return tt_var.owner
            return self

    @property
    def nin(self):
        return len(self.inputs)

    @property
    def nout(self):
        if self.outputs is not None:
            return len(self.outputs)
        elif self.obj:
            return len(self.obj.outputs)
        # TODO: Would be cool if we could return
        # a logic variable representing this.


class MetaVariable(MetaSymbol):
    base = theano.Variable
    __slots__ = ['type', 'owner', 'index', 'name']

    def __init__(self, type, owner, index, name, obj=None):
        super().__init__(obj=obj)
        self.type = MetaType.from_obj(type)
        self.owner = MetaApply.from_obj(owner)
        self.index = index
        self.name = name

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj

        if not self.owner:
            return super().reify()

        # Having an `owner` causes issues (e.g. being consistent about
        # other, unrelated outputs of an `Apply` node), and, in this case,
        # the `Apply` node that owns this variable needs to construct it.
        reified_rands, any_unreified = _meta_reify_iter(self.rands())
        tt_apply = self.owner.obj

        if tt_apply and not isvar(tt_apply):
            # If the owning `Apply` reified, then one of its `outputs`
            # corresponds to this variable.  Our `self.index` value should
            # tell us which, but, when that's not available, we can
            # sometimes infer it.
            if tt_apply.nout == 1:
                tt_index = 0
                # Make sure we didn't have a mismatched non-meta index value.
                assert (isvar(self.index) or
                        self.index is None or
                        self.index == 0)
                # Set/replace `None` or meta value
                self.index = 0
                tt_var = tt_apply.outputs[tt_index]
            elif not self.is_meta(self.index):
                tt_var = tt_apply.outputs[self.index]
            elif self.index is None:
                tt_var = tt_apply.default_output()
                self.index = tt_apply.outputs.index(tt_var)
            else:
                return self
            # If our name value is not set/concrete, then use the reified
            # value's.  Otherwise, use ours.
            if isvar(self.name) or self.name is None:
                self.name = tt_var.name
            else:
                tt_var.name = self.name
            assert tt_var is not None
            self.obj = tt_var
            return tt_var

        return super().reify()


class MetaTensorVariable(MetaVariable):
    # TODO: Could extend `theano.tensor.var._tensor_py_operators`, too.
    base = tt.TensorVariable

    @property
    def ndim(self):
        if (isinstance(self.type, MetaTensorType) and
                isinstance(self.type.broadastable, (list, tuple))):
            return len(self.type.broadcastable)
        # TODO: Would be cool if we could return
        # a logic variable representing this.


class MetaConstant(MetaVariable):
    base = theano.Constant
    __slots__ = ['type', 'data']

    def __init__(self, type, data, name=None, obj=None):
        super().__init__(type, None, None, name, obj=obj)
        self.data = data


class MetaTensorConstant(MetaConstant):
    # TODO: Could extend `theano.tensor.var._tensor_py_operators`, too.
    base = tt.TensorConstant
    __slots__ = ['type', 'data', 'name']

    def __init__(self, type, data, name=None, obj=None):
        super().__init__(type, data, name, obj=obj)


class MetaSharedVariable(MetaVariable):
    base = tt.sharedvar.SharedVariable
    __slots__ = ['name', 'type', 'data', 'strict']

    @classmethod
    def from_obj(cls, obj):
        if isvar(obj):
            return obj
        res = cls(obj.name, obj.type, obj.container.data, obj.container.strict,
                  obj=obj)
        return res

    def __init__(self, name, type, data, strict, obj=None):
        super().__init__(type, None, None, name, obj=obj)
        self.data = data
        self.strict = strict


class MetaTensorSharedVariable(MetaSharedVariable):
    # TODO: Could extend `theano.tensor.var._tensor_py_operators`, too.
    base = tt.sharedvar.TensorSharedVariable


class MetaScalarSharedVariable(MetaSharedVariable):
    base = tt.sharedvar.ScalarSharedVariable
# theano-meta-objects ends here


# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-mcmc.org::theano-meta-accessor][theano-meta-accessor]]
class TheanoMetaAccessor(object):
    """Creates an object that can be used to implicitly
    convert Theano functions and object into meta objects.

    Use it like a namespace/module/package object, e.g.

    >>> mt = TheanoMetaAccessor()
    >>> mt.vector('a')
    MetaTensorVariable(MetaTensorType('float64', (False,), None,
    obj=TensorType(float64, vector)), None, None, 'a', obj=a)

    Call it as a function to perform direct conversion to a meta
    object, e.g.

    >>> mt(tt.vector('a'))
    MetaTensorVariable(MetaTensorType('float64', (False,), None,
    obj=TensorType(float64, vector)), None, None, 'a', obj=a)

    """
    namespaces = [tt]

    def __init__(self):
        pass

    def __call__(self, x):
        return MetaSymbol.from_obj(x)

    def __getattr__(self, obj):

        ns_obj = next((getattr(ns, obj)
                       for ns in self.namespaces
                       if hasattr(ns, obj)), None)

        if ns_obj is None:
            # Try caller's namespace
            frame = inspect.currentframe()
            f_back = frame.f_back
            if f_back:
                ns_obj = f_back.f_locals.get(obj, None)
                if ns_obj is None:
                    ns_obj = f_back.f_globals.get(obj)

        if isinstance(ns_obj, (types.FunctionType, partial)):
            # It's a function, so let's provide a wrapper
            # that converts to-and-from theano and meta objects.
            @staticmethod
            @wraps(ns_obj)
            def meta_obj(*args, **kwargs):
                args = [o.reify() if hasattr(o, 'reify') else o
                        for o in args]
                res = ns_obj(*args, **kwargs)
                return MetaSymbol.from_obj(res)
        else:
            meta_obj = MetaSymbol.from_obj(ns_obj)

        setattr(TheanoMetaAccessor, obj, meta_obj)

        return getattr(TheanoMetaAccessor, obj)


mt = TheanoMetaAccessor()

mt.dot = MetaSymbol.from_obj(tt.basic._dot)


#
# The wrapped Theano functions will only work when the meta objects
# are fully reifiable (i.e. can be turned to Theano objects), but it's
# fairly straight-forward to adjust many of those functions so that they
# work with meta objects.
# TODO: Would be nice if we could trick Theano into using meta objects, or
# a robust use of "proxy" Theano objects
#

def mt_zeros(shape, dtype=None):
    if not isinstance(shape,
                      (list, tuple,
                       MetaTensorVariable,
                       tt.TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = tt.config.floatX
    return mt.alloc(np.array(0, dtype=dtype), *shape)


mt.zeros = mt_zeros


def mt_diag(v, k=0):
    if v.ndim == 1:
        return mt.AllocDiag(k)(v)
    elif v.ndim is not None and v.ndim >= 2:
        return mt.diagonal(v, offset=k)
    else:
        raise ValueError("Input must has v.ndim >= 1.")
# theano-meta-accessor ends here
