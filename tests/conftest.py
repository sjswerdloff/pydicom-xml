"""Shared test fixtures for pydicom-xml tests."""

from __future__ import annotations

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.valuerep import PersonName


@pytest.fixture
def simple_dataset() -> Dataset:
    """Dataset with basic patient demographics."""
    ds = Dataset()
    ds.PatientName = "Smith^John"
    ds.PatientID = "P001"
    ds.StudyDate = "20240101"
    return ds


@pytest.fixture
def binary_dataset() -> Dataset:
    """Dataset with OW pixel data."""
    ds = Dataset()
    ds.PatientID = "BIN001"
    ds.add_new(Tag(0x7FE0, 0x0010), "OW", bytes(range(16)))
    return ds


@pytest.fixture
def sequence_dataset() -> Dataset:
    """Dataset with a SQ element containing one item."""
    ds = Dataset()
    item = Dataset()
    item.CodeValue = "T-D0050"
    item.CodingSchemeDesignator = "SNM3"
    item.CodeMeaning = "Body"
    ds.add_new(Tag(0x0008, 0x2218), "SQ", Sequence([item]))
    return ds


@pytest.fixture
def pn_full_dataset() -> Dataset:
    """Dataset with a PN element using all five Alphabetic components."""
    ds = Dataset()
    ds.PatientName = PersonName("Smith^John^Robert^Dr.^Jr.")
    return ds


@pytest.fixture
def pn_multigroup_dataset() -> Dataset:
    """Dataset with PN element using Alphabetic, Ideographic, and Phonetic groups."""
    ds = Dataset()
    ds.PatientName = PersonName("Yamada^Tarou=山田^太郎=やまだ^たろう")
    return ds
