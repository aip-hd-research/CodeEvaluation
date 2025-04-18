from typing import (
    TypeVar,
    Tuple,
    Union,
    List,
    Dict,
    Type,
    cast,
    Callable,
    get_type_hints,
    Set,
    get_origin,
    get_args,
    Literal,
)

import json
import polars as pl
from datasets import Dataset, DatasetDict, load_dataset

T = TypeVar("T", contravariant=True)

# Type alias for convenience
type BagOfProperties[*T] = _BagOfPropertiesBase[Union[*T]]


class _BagOfPropertiesBase[T]:
    def __class_getitem__(cls, item):
        # Create a new class that holds the type argument(s)
        if item and hasattr(item, "__args__"):
            type_list = cast(List[type], item.__args__)
            for _type in type_list:
                if not isinstance(getattr(_type, "datatype", None), Type):
                    raise TypeError(
                        "BagOfPropertiesBase type args need the datatype entry"
                    )

            class Typed(cls):
                __type_args__: List[type] = type_list

            return Typed
        raise TypeError("_BagOfPropertiesBase has to be used with Union class")

    # func is needed for static typechecking currently, why???
    def _func(self, x: T) -> None:
        pass

    def __init__(self, data: Union[List[Dict], None] = None):
        self.schema = self._generate_schema()
        self.df = self._create_dataframe(data)

    def _generate_schema(self) -> Dict[str, type]:
        type_args = cast(List[type], getattr(self.__class__, "__type_args__", None))
        return {typ.__name__: typ.datatype for typ in type_args}

    def _create_dataframe(self, data: Union[List[Dict], None]) -> pl.DataFrame:
        if not data:
            return pl.DataFrame(schema=self.schema)

        processed_data = [
            {col_name: row.get(col_name, None) for col_name in self.schema}
            for row in data
        ]
        return pl.DataFrame(processed_data)

    def join(
        self,
        other: BagOfProperties,
        on: str = "id",
        how: Literal[
            "inner", "left", "right", "full", "semi", "anti", "cross", "outer"
        ] = "inner",
    ) -> BagOfProperties:
        """
        Joins the current BoP instance with another BoP instance on the specified column (default: 'id').
        The join operation is done using Polars' join functionality.
        """
        if on not in self.df.columns or on not in other.df.columns:
            raise ValueError(
                f"The column '{on}' must exist in both DataFrames for joining."
            )

        # Perform the join using Polars' join method
        joined_df = self.df.join(other.df, on=on, how=how)

        # Create a new BoP instance with the joined DataFrame and the same type parameters
        own_args = cast(List[type], getattr(self.__class__, "__type_args__", None))
        other_types = getTypes(other)

        new_types = set(own_args) | set(other_types)

        joined_bop = BagOfPropertiesFactory[*new_types].new()
        joined_bop.df = joined_df
        return joined_bop


class BagOfPropertiesFactory[*T]:
    def __class_getitem__(cls, item):
        class Typed(cls):
            __type_args__: List[type] = item

        return Typed

    @classmethod
    def new(cls) -> BagOfProperties[*T]:
        myargs = getattr(cls, "__type_args__", None)
        return _BagOfPropertiesBase[Union[*myargs]]()

    @classmethod
    def from_json(cls, filepath: str) -> BagOfProperties[*T]:
        """Loads data from a JSON file and initializes a BoP instance."""
        myargs = getattr(cls, "__type_args__", None)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain a list of dictionaries.")
                return _BagOfPropertiesBase[Union[*myargs]](data)
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")

    @classmethod
    def from_csv(cls, filepath: str) -> BagOfProperties[*T]:
        """Loads data from a CSV file and initializes a BoP instance."""
        myargs = getattr(cls, "__type_args__", None)
        try:
            df = pl.read_csv(filepath)
            return _BagOfPropertiesBase[Union[*myargs]](df.to_dicts())
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    @classmethod
    def from_dicts(cls, data: List[Dict]) -> BagOfProperties[*T]:
        myargs = getattr(cls, "__type_args__", None)
        return _BagOfPropertiesBase[Union[*myargs]](data)

    @classmethod
    def from_huggingface_dataset(cls, dataset: Dataset) -> BagOfProperties[*T]:
        """Loads data from a Hugging Face dataset."""
        myargs = getattr(cls, "__type_args__", None)
        return _BagOfPropertiesBase[Union[*myargs]](dataset.to_list())

    @classmethod
    def load_from_huggingface(cls, path: str) -> BagOfProperties[*T]:
        """Loads a dataset from Hugging Face using the provided path."""

        dataset = load_dataset(path)

        # If it's a DatasetDict, default to the 'train' split
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

        dataset = cast(Dataset, dataset)

        return cls.from_huggingface_dataset(dataset)


def getTypes(boP: BagOfProperties) -> Set[Type]:
    return set(cast(Tuple[Type], getattr(type(boP), "__type_args__", None)))


def SliceBoPType(func: Callable) -> Callable:
    """
    A decorator to slice BagOfProperties[X, Y, Z] to BagOfProperties[Y] inside the function,
    if the function expects BoP[Y] as its parameter.
    This also updates the `df` (Polars DataFrame) accordingly.
    """

    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)

        for param, expected_type in hints.items():
            origin = get_origin(expected_type)
            if origin is BagOfProperties:
                required_types = set(get_args(expected_type))

                for i, arg in enumerate(args):
                    if isinstance(arg, _BagOfPropertiesBase):
                        actual_types = cast(
                            Set[type], getattr(type(arg), "__type_args__", None)
                        )

                        if required_types.issubset(actual_types):
                            # Create a new BoP subclass with only the required columns
                            Sliced_class = _BagOfPropertiesBase[Union[*required_types]]
                            sliced_instance = Sliced_class()

                            # Filter the DataFrame to match only the required columns
                            required_columns = {col.__name__ for col in required_types}
                            sliced_instance.df = arg.df.select(
                                [
                                    col
                                    for col in arg.df.columns
                                    if col in required_columns
                                ]
                            )

                            # Replace the original argument with the sliced instance
                            args = list(args)
                            args[i] = sliced_instance
                            break
                        else:
                            raise TypeError(
                                "Fields missing for BagOfProperties slicing"
                            )

        return func(*args, **kwargs)

    return wrapper
