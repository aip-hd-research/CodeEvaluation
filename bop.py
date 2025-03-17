from typing import Dict, Tuple, Type, Callable, get_type_hints


class BoPColumn():
    """Interface for columns used in BoP. Implementations must define name and datatype."""

    @property
    def name(self) -> str:
        """Returns the column name."""
        return self._name

    @property
    def datatype(self) -> Type:
        """Returns the column datatype."""
        return self._datatype

class StringColumn(BoPColumn):
    """A string-based column."""
    _datatype = str


class IntegerColumn(BoPColumn):
    """An integer-based column."""
    _datatype = int


class BoPMeta(type):
    """Metaclass for tracking type parameters in generic classes with validation."""

    _registry: Dict[Tuple[Type, frozenset], Type] = {}

    def __getitem__(cls, item):
        """Handles BoP[ID, CodeJava] and ensures only BoPColumn-based types are allowed."""

        typesTuple = item if isinstance(item, tuple) else (item,)

        for typ in typesTuple:
            if not issubclass(typ, BoPColumn):
                raise TypeError(
                    f"Invalid type {typ.__name__}. BoP only supports subclasses of BoPColumn."
                )
            if not hasattr(typ, "_name") or not hasattr(typ, "_datatype"):
                raise TypeError(
                    f"Invalid BoPColumn implementation: {typ.__name__} must define '_name' and '_datatype'."
                )

        key = frozenset(typesTuple)  # Make it order-invariant

        if (cls, key) in cls._registry:
            return cls._registry[(cls, key)]

        new_class = type(
            f"{cls.__name__}[{', '.join(t.__name__ for t in sorted(typesTuple, key=lambda x: x.__name__))}]",
            (cls,),
            {"_type_params": typesTuple},
        )
        cls._registry[(cls, key)] = new_class
        return new_class

    def __instancecheck__(cls, instance):
        """Allow isinstance(boP, BoP) and BoP[ID, CodeJava] checking."""
        instance_cls = type(instance)

        if not isinstance(instance_cls, BoPMeta):
            return False

        if cls is BoP:
            return hasattr(instance_cls, "_type_params")

        if not hasattr(instance_cls, "_type_params") or not hasattr(
            cls, "_type_params"
        ):
            return False

        instance_types = set(instance_cls._type_params)
        class_types = set(cls._type_params)

        return instance_types == class_types

    def __subclasscheck__(cls, subclass):
        """Allow subset checking in issubclass."""
        if not isinstance(subclass, BoPMeta):
            return False

        if cls is BoP:
            return hasattr(subclass, "_type_params")

        if not hasattr(subclass, "_type_params") or not hasattr(cls, "_type_params"):
            return False

        subclass_types = set(subclass._type_params)
        class_types = set(cls._type_params)

        return subclass_types.issubset(class_types)


class BoP(metaclass=BoPMeta):
    def get_types(self) -> set:
        """Retrieve the stored generic types of this instance."""
        return set(getattr(type(self), "_type_params", ()))


def castBoPtype(func: Callable) -> Callable:
    """
    A decorator to cast BoP[X, Y, Z] to BoP[Y] inside the function,
    if the function expects BoP[Y] as its parameter.
    """

    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)

        for param, expected_type in hints.items():
            if isinstance(expected_type, type) and issubclass(expected_type, BoP):
                required_types = set(expected_type._type_params)

                for i, arg in enumerate(args):
                    if isinstance(arg, BoP):
                        actual_types = set(type(arg)._type_params)

                        if required_types.issubset(actual_types):
                            casted_class = BoP[tuple(required_types)]
                            args = list(args)
                            args[i] = casted_class()
                            break

        return func(*args, **kwargs)

    return wrapper
