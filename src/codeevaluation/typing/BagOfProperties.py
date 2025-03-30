from typing import Dict, Tuple, Type, Callable, List, Union, get_type_hints
from abc import ABC, abstractmethod

import json

import polars as pl
from datasets import Dataset, DatasetDict, load_dataset


class BoPColumn(ABC):
    """
    Interface for columns used in BoP.
    Implementations must define name and datatype.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the column name."""
        ...

    @property
    @abstractmethod
    def datatype(self) -> type:
        """Returns the column datatype."""
        ...

    @classmethod
    def create(cls, name: str, datatype: type) -> Type["BoPColumn"]:
        """Creates a new column subclass dynamically."""
        return type(name, (cls,), {"name": name, "datatype": datatype})


class BoPMeta(type):
    """Metaclass for tracking type parameters"""

    _registry: Dict[Tuple[Type, frozenset], Type] = {}

    def __getitem__(cls, item) -> Type["BoP"]:
        """
        Handles BoP[*] objects.
        Ensures only BoPColumn-based types are allowed.
        """

        typesTuple = item if isinstance(item, tuple) else (item,)

        for typ in typesTuple:
            if not issubclass(typ, BoPColumn):
                raise TypeError(
                    f"Invalid type {typ.__name__}. BoP only supports subclasses of BoPColumn."
                )
            if not (
                isinstance(getattr(typ, "name", None), str)
                and isinstance(getattr(typ, "datatype", None), type)
            ):
                raise TypeError(
                    f"Invalid BoPColumn implementation: {typ.__name__} must define 'name' and 'datatype'."
                )
        typesSet = frozenset(typesTuple)  # Make it order-invariant

        if (cls, typesSet) in cls._registry:
            return cls._registry[(cls, typesSet)]

        new_class = type(
            f"{cls.__name__}[{', '.join(t.__name__ for t in typesSet)}]",
            (cls,),
            {"_type_params": typesSet},
        )
        cls._registry[(cls, typesSet)] = new_class
        return new_class

    def __instancecheck__(cls, instance) -> bool:
        """Allow isinstance(boP, BoP) and BoP[ID, CodeJava] checking."""
        instance_cls = type(instance)

        if not isinstance(instance_cls, BoPMeta):
            return False

        instance_types = get_typ_params_or_empty_set(instance_cls)
        class_types = get_typ_params_or_empty_set(cls)

        return class_types.issubset(instance_types)

    def __subclasscheck__(cls, subclass) -> bool:
        """Allow subset checking in issubclass."""

        if not isinstance(subclass, BoPMeta):
            return False

        subclass_types = get_typ_params_or_empty_set(subclass)
        class_types = get_typ_params_or_empty_set(cls)

        return class_types.issubset(subclass_types)


def get_typ_params_or_empty_set(bopClass) -> set:
    types = set(bopClass._type_params) if hasattr(bopClass, "_type_params") else set()
    return types


class BoP(metaclass=BoPMeta):
    """
    Base class for Bag of Properties.
    Handling structured data storage in a Polars DataFrame.
    """

    _type_params: frozenset = frozenset()

    def __init__(self, data: Union[List[Dict], None] = None) -> None:
        """Initialize the BoP instance with a Polars DataFrame."""
        self.df = self.create_dataframe(data)

    @classmethod
    def from_json(cls, filepath: str) -> "BoP":
        """Loads data from a JSON file and initializes a BoP instance."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain a list of dictionaries.")
                return cls(data)
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")

    @classmethod
    def from_csv(cls, filepath: str) -> "BoP":
        """Loads data from a CSV file and initializes a BoP instance."""
        try:
            df = pl.read_csv(filepath)
            return cls(df.to_dicts())
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    @classmethod
    def from_dicts(cls, data: List[Dict]) -> "BoP":
        """Initializes a BoP instance from a list of dictionaries."""
        return cls(data)

    @classmethod
    def from_huggingface_dataset(cls, dataset: Dataset) -> "BoP":
        """Loads data from a Hugging Face dataset."""

        if not isinstance(dataset, Dataset):
            raise ValueError("Provided dataset must be a Hugging Face Dataset.")

        return cls(dataset.to_list())

    @classmethod
    def load_from_huggingface(cls, path: str) -> "BoP":
        """Loads a dataset from Hugging Face using the provided path."""

        dataset = load_dataset(path)

        # If it's a DatasetDict, default to the 'train' split
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

        return cls.from_huggingface_dataset(dataset)

    def create_dataframe(self, data: Union[List[Dict], None] = None) -> pl.DataFrame:
        """
        Creates a Polars DataFrame based on the BoPColumn definitions.
        Ensures:
        - Only schema-defined fields are included.
        - Missing fields are set to None.
        - Extra fields are dropped.
        """
        schema = {col.name: col.datatype for col in self._type_params}

        # Create an empty DataFrame if no data is provided
        if not data:
            return pl.DataFrame(schema=schema)

        processed_data = [
            {col_name: row.get(col_name, None) for col_name in schema} for row in data
        ]

        return pl.DataFrame(processed_data, schema=schema)

    def get_types(self) -> set:
        """Retrieve the stored generic types of this instance."""
        return set(getattr(type(self), "_type_params", ()))

    def show(self) -> None:
        """Print the DataFrame."""
        print(self.df)

    def join(self, other: "BoP", on: str = "id", how: str = "inner") -> "BoP":
        """
        Joins the current BoP instance with another BoP instance on the specified column (default: 'id').
        The join operation is done using Polars' join functionality.

        Args:
            other (BoP): The other BoP instance to join with.
            on (str): The column name on which to join. Defaults to 'id'.
            how (str): The type of join. Defaults to "inner". Other options: "left", "right", "outer".

        Returns:
            BoP: A new BoP instance containing the joined DataFrame.
        """
        if on not in self.df.columns or on not in other.df.columns:
            raise ValueError(
                f"The column '{on}' must exist in both DataFrames for joining."
            )

        # Perform the join using Polars' join method
        joined_df = self.df.join(other.df, on=on, how=how)

        # Create a new BoP instance with the joined DataFrame and the same type parameters
        joined_bop = BoP[*self._type_params, *other._type_params]()
        joined_bop.df = joined_df
        return joined_bop


def castBoPtype(func: Callable) -> Callable:
    """
    A decorator to cast BoP[X, Y, Z] to BoP[Y] inside the function,
    if the function expects BoP[Y] as its parameter.
    This also updates the `df` (Polars DataFrame) accordingly.
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
                            # Create a new BoP subclass with only the required columns
                            casted_class = BoP[tuple(required_types)]
                            casted_instance = casted_class()

                            # Filter the DataFrame to match only the required columns
                            required_columns = {col.name for col in required_types}
                            casted_instance.df = arg.df.select(
                                [
                                    col
                                    for col in arg.df.columns
                                    if col in required_columns
                                ]
                            )

                            # Replace the original argument with the casted instance
                            args = list(args)
                            args[i] = casted_instance
                            break

        return func(*args, **kwargs)

    return wrapper
