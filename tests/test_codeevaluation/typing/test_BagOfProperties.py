import pytest

from codeevaluation.typing.BagOfProperties import BoP, castBoPtype, BoPColumn
from codeevaluation.typing.column_types import ID, DExecutable, Error, Success


# === Test: isinstance() Behavior === #
def test_isinstance_exact_match():
    """Ensure isinstance correctly identifies an exact BoP match."""
    boP = BoP[ID, DExecutable]()
    assert isinstance(boP, BoP[ID, DExecutable])  # ✅ True (exact match)
    assert isinstance(boP, BoP[DExecutable, ID])  # ✅ True (order should not matter)


def test_isinstance_permutations():
    """Ensure isinstance ignores parameter order."""
    boP = BoP[ID, DExecutable]()
    assert isinstance(boP, BoP[DExecutable, ID])  # ✅ True (order should not matter)


def test_isinstance_partial_overlap():
    """Ensure BoP instances with overlapping but different generics are not interchangeable."""
    assert not isinstance(BoP[ID, DExecutable](), BoP[DExecutable, Error])  # ❌ False


def test_isinstance_missing_types():
    """Ensure isinstance works for bigger boPs implementing smaller ones."""
    boP = BoP[ID, DExecutable]()
    assert isinstance(boP, BoP[ID])  # ✅ True (missing one type)
    assert isinstance(boP, BoP[DExecutable])  # ✅ True (missing one type)


def test_bop_invalid_type():
    """❌ Test that BoP raises a TypeError for invalid types."""
    with pytest.raises(TypeError):
        BoP[str]()  # Should raise an error

    with pytest.raises(TypeError):
        BoP[DExecutable, int]()  # Mixed case, still should fail


def test_isinstance_non_bop_type():
    """Ensure isinstance fails for completely unrelated types."""
    assert not isinstance(True, BoP)  # ❌ False (bool should not be considered BoP)
    assert not isinstance("test", BoP)  # ❌ False
    assert not isinstance(42, BoP)  # ❌ False


def test_ininstance_with_unrelated_type():
    """Ensure BoP does not match completely unrelated types."""
    boP = BoP[ID, DExecutable]()
    assert not isinstance(boP, Success)  # ❌ False (completely unrelated)
    assert not isinstance(boP, BoP[Error, Success])  # ❌ False


def test_isinstance_single_generic_type():
    """Test BoP with only one generic type."""
    boP = BoP[ID]()
    assert isinstance(boP, BoP[ID])  # ✅ True
    assert not isinstance(boP, BoP[DExecutable])  # ❌ False
    assert not isinstance(boP, BoP[ID, DExecutable])  # ❌ False


def test_isinstance_no_generic_params():
    """Ensure BoP without parameters behaves correctly."""
    generic_boP = BoP()
    assert isinstance(generic_boP, BoP)  # ✅ True
    assert not isinstance(generic_boP, BoP[ID])  # ❌ False


# === Test: issubclass() Behavior === #
def test_issubclass_subset():
    """Ensure issubclass works when one type set is a subset of another."""
    assert issubclass(
        BoP[ID, DExecutable, Error], BoP[ID, DExecutable]
    )  # ✅ True (special implementation with one more column)


def test_issubclass_superset():
    """Ensure issubclass fails when one type set is a superset."""
    assert not issubclass(
        BoP[ID, DExecutable], BoP[ID, DExecutable, Error]
    )  # ❌ False (Error misses, no subclass)


def test_issubclass_base_class():
    """Ensure all BoP[*V] types are subclasses of BoP."""
    assert issubclass(BoP[ID, DExecutable], BoP)  # ✅ True


def test_issubclass_non_bop_type():
    """Ensure issubclass fails for completely unrelated types."""
    assert not issubclass(bool, BoP)  # ❌ False (bool should not be considered BoP)
    assert not issubclass(str, BoP)  # ❌ False
    assert not issubclass(int, BoP)  # ❌ False


def test_issubclass_symmetric():
    """Ensure issubclass works symmetrically for permutations of types."""
    assert issubclass(BoP[ID, DExecutable], BoP[DExecutable, ID])  # ✅ True


def test_issubclass_large_subset():
    """Ensure issubclass correctly identifies large subsets."""
    assert not issubclass(BoP[ID, DExecutable], BoP[ID, DExecutable, Error])  # ❌ False
    assert issubclass(BoP[ID, DExecutable, Error], BoP[ID, DExecutable])  # ✅ True


def test_issubclass_partial_overlap():
    """Ensure BoP instances with overlapping but different generics are not interchangeable."""
    assert not issubclass(
        BoP[ID, DExecutable], BoP[DExecutable, Error]
    )  # ❌ False (overlap but not a subset)


# === Test: Instance Methods === #
def test_get_types():
    """Ensure get_types() correctly returns stored generic types."""
    boP = BoP[ID, DExecutable]()
    assert boP.get_types() == {ID, DExecutable}  # ✅ Expected stored types


# === Test: Casting with @castBoPtype === #
def test_casting_valid():
    """Ensure @castBoPtype correctly casts BoP[ID, DExecutable] to BoP[DExecutable]."""

    @castBoPtype
    def checkDExecutable(boP: BoP[DExecutable]) -> bool:
        return isinstance(boP, BoP[DExecutable])

    boP = BoP[ID, DExecutable]()
    assert checkDExecutable(boP)  # ✅ True (should be casted)


def test_casting_invalid():
    """Ensure @castBoPtype does not allow incorrect casting."""

    @castBoPtype
    def checkDExecutable(boP: BoP[DExecutable]) -> bool:
        return isinstance(boP, BoP[DExecutable])

    assert not checkDExecutable(BoP[ID]())  # ❌ False (wrong type)


def test_casting_does_not_mutate_original():
    """Ensure @castBoPtype does not modify original arguments."""

    @castBoPtype
    def checkDExecutable(boP: BoP[DExecutable]) -> bool:
        return isinstance(boP, BoP[DExecutable])

    boP = BoP[ID, DExecutable]()
    original_types = boP.get_types()
    checkDExecutable(boP)
    assert (
        boP.get_types() == original_types
    )  # ✅ The original object should remain unchanged


def test_bop_order_invariance():
    BoP1 = BoP[ID, DExecutable]
    BoP2 = BoP[DExecutable, ID]
    assert BoP1 is BoP2  # ✅ Should be the same class regardless of order


def test_bop_unique_registration():
    BoP1 = BoP[ID, DExecutable]
    BoP2 = BoP[ID, DExecutable]
    assert BoP1 is BoP2  # ✅ Should be the same reference (singleton pattern)


def test_bop_missing_attributes():
    """❌ Ensure that BoP raises a TypeError if a BoPColumn is missing _name or _datatype."""

    class IntegerColumn(BoPColumn):
        datatype = int

    with pytest.raises(
        TypeError,
        match="Invalid BoPColumn implementation: IntegerColumn must define 'name' and 'datatype'.",
    ):
        BoP[IntegerColumn]()
