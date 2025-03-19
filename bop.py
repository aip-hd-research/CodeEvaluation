from typing import Dict, Tuple, Type, Callable, get_type_hints
from abc import ABC, abstractmethod


class BoPColumn(ABC):
    """Interface for columns used in BoP. Implementations must define name and datatype."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the column name."""
        ...

    @property
    @abstractmethod
    def datatype(self) -> Type:
        """Returns the column datatype."""
        ...

class StringColumn(BoPColumn):
    """A string-based column."""
    datatype = str


class IntegerColumn(BoPColumn):
    """An integer-based column."""
    datatype = int


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
            
            #isinstance(getattr(typ, "name", None), str) and isinstance(getattr(cls, "datatype", None), str)
            if not isinstance(getattr(typ, "name", None), str) or not isinstance(getattr(typ, "datatype", None), Type):
                raise TypeError(
                    f"Invalid BoPColumn implementation: {typ.__name__} must define 'name' and 'datatype'."
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

        instance_types = get_typ_params_or_empty_set(instance_cls)
        class_types = get_typ_params_or_empty_set(cls)

        return class_types.issubset(instance_types)

    def __subclasscheck__(cls, subclass):
        """Allow subset checking in issubclass."""

        if not isinstance(subclass, BoPMeta):
            return False

        subclass_types = get_typ_params_or_empty_set(subclass)
        class_types = get_typ_params_or_empty_set(cls)

        return class_types.issubset(subclass_types)

def get_typ_params_or_empty_set(bopClass):
    types = (
        set(bopClass._type_params) 
        if hasattr(bopClass, "_type_params") 
        else set()
    )
    return types

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
