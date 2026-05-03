"""Tests for multipart/related XML serialization utilities (PS3.18)."""

import pytest
from pydicom.dataset import Dataset

from pydicom_xml import (
    datasets_from_multipart_xml,
    datasets_to_multipart_xml,
    extract_boundary,
)


@pytest.fixture
def sample_datasets():
    ds1 = Dataset()
    ds1.PatientName = "Smith^John"
    ds1.PatientID = "PAT-001"

    ds2 = Dataset()
    ds2.PatientName = "Doe^Jane"
    ds2.PatientID = "PAT-002"

    ds3 = Dataset()
    ds3.PatientName = "Brown^Bob"
    ds3.PatientID = "PAT-003"

    return [ds1, ds2, ds3]


class TestDatasetsToMultipartXml:
    def test_single_dataset_produces_one_part(self):
        ds = Dataset()
        ds.PatientID = "TEST-001"
        body, ct = datasets_to_multipart_xml([ds])
        assert b"TEST-001" in body
        assert "multipart/related" in ct
        assert "boundary=" in ct

    def test_multiple_datasets_produces_multiple_parts(self, sample_datasets):
        body, ct = datasets_to_multipart_xml(sample_datasets)
        assert b"PAT-001" in body
        assert b"PAT-002" in body
        assert b"PAT-003" in body
        boundary = extract_boundary(ct)
        parts = body.split(f"--{boundary}".encode())
        # First part is empty (before first boundary), last is closing "--"
        xml_parts = [p for p in parts if b"Content-Type" in p]
        assert len(xml_parts) == 3

    def test_empty_list_produces_closing_boundary(self):
        body, ct = datasets_to_multipart_xml([])
        assert "multipart/related" in ct
        boundary = extract_boundary(ct)
        assert body == f"--{boundary}--\r\n".encode()

    def test_custom_boundary(self):
        ds = Dataset()
        ds.PatientID = "TEST"
        body, ct = datasets_to_multipart_xml([ds], boundary="my-custom-boundary")
        assert "boundary=my-custom-boundary" in ct
        assert b"--my-custom-boundary" in body

    def test_content_type_includes_type_parameter(self):
        ds = Dataset()
        ds.PatientID = "TEST"
        _, ct = datasets_to_multipart_xml([ds])
        assert 'type="application/dicom+xml"' in ct

    def test_each_part_has_xml_content_type_header(self, sample_datasets):
        body, _ = datasets_to_multipart_xml(sample_datasets)
        assert body.count(b"Content-Type: application/dicom+xml") == 3

    def test_each_part_is_valid_dicom_xml(self, sample_datasets):
        body, ct = datasets_to_multipart_xml(sample_datasets)
        boundary = extract_boundary(ct)
        recovered = datasets_from_multipart_xml(body, boundary)
        assert len(recovered) == 3
        for ds in recovered:
            assert isinstance(ds, Dataset)


class TestDatasetsFromMultipartXml:
    def test_round_trip_single(self):
        ds = Dataset()
        ds.PatientID = "RT-001"
        ds.PatientName = "Roundtrip^Test"
        body, ct = datasets_to_multipart_xml([ds])
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert len(result) == 1
        assert result[0].PatientID == "RT-001"
        assert str(result[0].PatientName) == "Roundtrip^Test"

    def test_round_trip_multiple(self, sample_datasets):
        body, ct = datasets_to_multipart_xml(sample_datasets)
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert len(result) == 3
        ids = {ds.PatientID for ds in result}
        assert ids == {"PAT-001", "PAT-002", "PAT-003"}

    def test_round_trip_empty(self):
        body, ct = datasets_to_multipart_xml([])
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert result == []

    def test_preserves_nested_sequences(self):
        ds = Dataset()
        ds.PatientID = "SQ-001"
        item = Dataset()
        item.CodeValue = "T-D1100"
        item.CodingSchemeDesignator = "SRT"
        ds.AnatomicRegionSequence = [item]
        body, ct = datasets_to_multipart_xml([ds])
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert len(result) == 1
        assert result[0].AnatomicRegionSequence[0].CodeValue == "T-D1100"


class TestExtractBoundary:
    def test_simple_boundary(self):
        ct = 'multipart/related; type="application/dicom+xml"; boundary=my-boundary'
        assert extract_boundary(ct) == "my-boundary"

    def test_quoted_boundary(self):
        ct = 'multipart/related; type="application/dicom+xml"; boundary="quoted-boundary"'
        assert extract_boundary(ct) == "quoted-boundary"

    def test_boundary_first(self):
        ct = 'multipart/related; boundary=first-param; type="application/dicom+xml"'
        assert extract_boundary(ct) == "first-param"

    def test_case_insensitive(self):
        ct = 'multipart/related; type="application/dicom+xml"; Boundary=my-boundary'
        assert extract_boundary(ct) == "my-boundary"

    def test_uppercase(self):
        ct = "multipart/related; BOUNDARY=my-boundary"
        assert extract_boundary(ct) == "my-boundary"

    def test_whitespace_around_equals(self):
        ct = "multipart/related; boundary = my-boundary"
        assert extract_boundary(ct) == "my-boundary"

    def test_no_boundary_raises(self):
        ct = 'multipart/related; type="application/dicom+xml"'
        with pytest.raises(ValueError, match="No boundary parameter"):
            extract_boundary(ct)
