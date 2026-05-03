"""Tests for multipart/related XML serialization utilities (PS3.18)."""

import pytest
from pydicom.dataset import Dataset

from pydicom_xml import (
    datasets_from_multipart_xml,
    datasets_to_multipart_xml,
    extract_boundary,
)
from pydicom_xml.xmlrep import _check_part_content_type, _quote_boundary, _validate_boundary


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


class TestValidateBoundary:
    """Tests for RFC 2046 boundary validation."""

    def test_valid_alphanumeric(self):
        _validate_boundary("simple-boundary-123")

    def test_valid_with_allowed_special_chars(self):
        _validate_boundary("boundary'()+_,-./:=?")

    def test_valid_with_space_not_trailing(self):
        _validate_boundary("boundary with space")

    def test_leading_space_accepted(self):
        _validate_boundary(" boundary")

    def test_single_space_rejected_trailing(self):
        # A single space is a trailing space
        with pytest.raises(ValueError, match="must not end with a space"):
            _validate_boundary(" ")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_boundary("")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="exceeds maximum length"):
            _validate_boundary("x" * 71)

    def test_max_length_accepted(self):
        _validate_boundary("x" * 70)

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match=r"invalid characters.*'@'"):
            _validate_boundary("boundary@invalid")

    def test_trailing_space_raises(self):
        with pytest.raises(ValueError, match="must not end with a space"):
            _validate_boundary("boundary ")

    def test_tab_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_boundary("boundary\there")

    def test_newline_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_boundary("boundary\nhere")


class TestQuoteBoundary:
    """Tests for boundary quoting in Content-Type header."""

    def test_simple_boundary_not_quoted(self):
        assert _quote_boundary("simple-boundary") == "simple-boundary"

    def test_boundary_with_space_quoted(self):
        assert _quote_boundary("boundary value") == '"boundary value"'

    def test_boundary_with_parentheses_quoted(self):
        assert _quote_boundary("bound(ary)") == '"bound(ary)"'

    def test_boundary_with_slash_quoted(self):
        # '/' is a tspecial per RFC 2045 Section 5.1, requiring quoting
        assert _quote_boundary("bound/ary") == '"bound/ary"'

    def test_boundary_with_equals_quoted(self):
        # '=' is a tspecial requiring quoting
        assert _quote_boundary("bound=ary") == '"bound=ary"'


class TestBoundaryQuotingIntegration:
    """Integration tests: boundary quoting in datasets_to_multipart_xml."""

    def test_boundary_with_space_quoted_in_header(self):
        ds = Dataset()
        ds.PatientID = "TEST"
        body, ct = datasets_to_multipart_xml([ds], boundary="my boundary")
        # Content-Type header must have quoted boundary
        assert 'boundary="my boundary"' in ct
        # Body uses unquoted boundary for delimiter
        assert b"--my boundary" in body

    def test_boundary_with_space_round_trips(self):
        ds = Dataset()
        ds.PatientID = "SPACE-TEST"
        body, ct = datasets_to_multipart_xml([ds], boundary="my boundary")
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert len(result) == 1
        assert result[0].PatientID == "SPACE-TEST"

    def test_invalid_boundary_rejected(self):
        ds = Dataset()
        ds.PatientID = "TEST"
        with pytest.raises(ValueError, match="invalid characters"):
            datasets_to_multipart_xml([ds], boundary="bad@boundary")

    def test_auto_generated_boundary_not_quoted(self):
        ds = Dataset()
        ds.PatientID = "TEST"
        _, ct = datasets_to_multipart_xml([ds])
        # Auto-generated boundaries use only safe chars, no quoting needed
        assert 'boundary="' not in ct
        assert "boundary=pydicom-xml-" in ct

    def test_tspecial_boundary_without_space_round_trips(self):
        """Boundary with tspecial (no space) is quoted and round-trips."""
        ds = Dataset()
        ds.PatientID = "TSPECIAL-TEST"
        body, ct = datasets_to_multipart_xml([ds], boundary="bound=ary")
        # Header must be quoted
        assert 'boundary="bound=ary"' in ct
        # Body uses unquoted delimiter
        assert b"--bound=ary" in body
        # Round-trip succeeds
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert len(result) == 1
        assert result[0].PatientID == "TSPECIAL-TEST"


class TestPartContentTypeValidation:
    """Tests for per-part Content-Type checking in datasets_from_multipart_xml."""

    def test_valid_dicom_xml_content_type_accepted(self):
        header_block = b"Content-Type: application/dicom+xml"
        _check_part_content_type(header_block)

    def test_text_xml_accepted(self):
        header_block = b"Content-Type: text/xml"
        _check_part_content_type(header_block)

    def test_application_xml_accepted(self):
        header_block = b"Content-Type: application/xml"
        _check_part_content_type(header_block)

    def test_content_type_with_charset_accepted(self):
        header_block = b"Content-Type: application/dicom+xml; charset=utf-8"
        _check_part_content_type(header_block)

    def test_case_insensitive_header_name(self):
        header_block = b"CONTENT-TYPE: application/dicom+xml"
        _check_part_content_type(header_block)

    def test_no_content_type_header_accepted(self):
        header_block = b"Content-Length: 1234"
        _check_part_content_type(header_block)

    def test_empty_header_block_accepted(self):
        _check_part_content_type(b"")

    def test_folded_content_type_header_accepted(self):
        """RFC 2822 folded headers (continuation lines) are unfolded."""
        header_block = b"Content-Type:\r\n application/dicom+xml"
        _check_part_content_type(header_block)

    def test_folded_content_type_with_tab_accepted(self):
        header_block = b"Content-Type:\r\n\tapplication/dicom+xml; charset=utf-8"
        _check_part_content_type(header_block)

    def test_json_content_type_rejected(self):
        header_block = b"Content-Type: application/dicom+json"
        with pytest.raises(ValueError, match="Unexpected Content-Type"):
            _check_part_content_type(header_block)

    def test_octet_stream_rejected(self):
        header_block = b"Content-Type: application/octet-stream"
        with pytest.raises(ValueError, match="Unexpected Content-Type"):
            _check_part_content_type(header_block)

    def test_html_rejected(self):
        header_block = b"Content-Type: text/html"
        with pytest.raises(ValueError, match="Unexpected Content-Type"):
            _check_part_content_type(header_block)

    def test_rejection_in_full_parse(self):
        """Non-XML parts in multipart body are rejected."""
        boundary = "test-boundary"
        # Build a multipart body with a JSON part
        part = (
            f"--{boundary}\r\n"
            "Content-Type: application/dicom+json\r\n"
            "\r\n"
            '{"00100020": {"vr": "LO", "Value": ["PAT-001"]}}\r\n'
            f"--{boundary}--\r\n"
        ).encode()
        with pytest.raises(ValueError, match="Unexpected Content-Type"):
            datasets_from_multipart_xml(part, boundary)

    def test_valid_xml_parts_parsed_normally(self):
        """Standard round-trip still works with content-type checking."""
        ds = Dataset()
        ds.PatientID = "CT-CHECK"
        body, ct = datasets_to_multipart_xml([ds])
        boundary = extract_boundary(ct)
        result = datasets_from_multipart_xml(body, boundary)
        assert len(result) == 1
        assert result[0].PatientID == "CT-CHECK"
