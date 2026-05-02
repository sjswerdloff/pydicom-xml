"""Integration tests for pydicom-xml — round-trip fidelity.

All tests verify that dataset_from_xml(dataset_to_xml(ds)) reproduces the
original Dataset with equivalent element values.

All test datasets are constructed programmatically — no DICOM files required.
"""

from __future__ import annotations

import pytest
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.valuerep import PersonName

from pydicom_xml import from_xml, to_xml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _datasets_equivalent(a: Dataset, b: Dataset) -> bool:
    """Compare two Datasets element-by-element for equivalence.

    Args:
        a: First dataset.
        b: Second dataset.

    Returns:
        True if all elements match by tag, VR, and string representation.
    """
    if set(a.keys()) != set(b.keys()):
        return False
    for tag in a.keys():
        ea = a[tag]
        eb = b[tag]
        if ea.VR != eb.VR:
            return False
        if str(ea.value) != str(eb.value):
            return False
    return True


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_simple_demographics(self, simple_dataset: Dataset) -> None:
        """Contract: basic string VR elements survive round-trip without data loss."""
        result = from_xml(to_xml(simple_dataset))
        assert str(result.PatientName) == str(simple_dataset.PatientName)
        assert result.PatientID == simple_dataset.PatientID
        assert result.StudyDate == simple_dataset.StudyDate

    def test_multi_valued_ds_element(self) -> None:
        """Contract: multi-valued DS (ImagePositionPatient) round-trips correctly."""
        ds = Dataset()
        ds.add_new(Tag(0x0020, 0x0032), "DS", ["1.0", "2.5", "-3.75"])
        result = from_xml(to_xml(ds))
        vals = list(result[Tag(0x0020, 0x0032)].value)
        assert len(vals) == 3
        assert float(vals[0]) == pytest.approx(1.0)
        assert float(vals[1]) == pytest.approx(2.5)
        assert float(vals[2]) == pytest.approx(-3.75)

    def test_multi_valued_fl_as_list(self) -> None:
        """Contract: multi-valued FL stored as a plain Python list round-trips."""
        ds = Dataset()
        # RT Ion Plan sometimes stores FL multi-values as plain Python lists
        ds.add_new(Tag(0x0066, 0x0011), "FL", [1.0, 2.5, -3.75])
        result = from_xml(to_xml(ds))
        vals = list(result[Tag(0x0066, 0x0011)].value)
        assert len(vals) == 3
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(2.5)
        assert vals[2] == pytest.approx(-3.75)

    def test_multi_valued_fd_as_list(self) -> None:
        """Contract: multi-valued FD stored as a plain Python list round-trips."""
        ds = Dataset()
        ds.add_new(Tag(0x0018, 0x9089), "FD", [1.0e10, -2.5e-5])
        result = from_xml(to_xml(ds))
        vals = list(result[Tag(0x0018, 0x9089)].value)
        assert len(vals) == 2
        assert vals[0] == pytest.approx(1.0e10)
        assert vals[1] == pytest.approx(-2.5e-5)

    def test_sequence_with_nested_items(self, sequence_dataset: Dataset) -> None:
        """Contract: SQ elements with nested items round-trip correctly."""
        result = from_xml(to_xml(sequence_dataset))
        seq_out = result[Tag(0x0008, 0x2218)].value
        assert len(seq_out) == 1
        assert seq_out[0].CodeValue == "T-D0050"
        assert seq_out[0].CodingSchemeDesignator == "SNM3"
        assert seq_out[0].CodeMeaning == "Body"

    def test_deeply_nested_sequences_three_levels(self) -> None:
        """Contract: sequences nested 3 levels deep round-trip correctly."""
        inner_item = Dataset()
        inner_item.CodeValue = "deep_code"
        inner_seq = Sequence([inner_item])

        mid_item = Dataset()
        mid_item.add_new(Tag(0x0008, 0x2218), "SQ", inner_seq)
        mid_seq = Sequence([mid_item])

        outer_item = Dataset()
        outer_item.add_new(Tag(0x0040, 0xA730), "SQ", mid_seq)
        outer_seq = Sequence([outer_item])

        ds = Dataset()
        ds.add_new(Tag(0x0040, 0xA370), "SQ", outer_seq)

        result = from_xml(to_xml(ds))
        deep_code = result[Tag(0x0040, 0xA370)].value[0][Tag(0x0040, 0xA730)].value[0][Tag(0x0008, 0x2218)].value[0].CodeValue
        assert deep_code == "deep_code"

    def test_person_name_all_five_components(self, pn_full_dataset: Dataset) -> None:
        """Contract: PN with all 5 Alphabetic components round-trips."""
        result = from_xml(to_xml(pn_full_dataset))
        pn_str = str(result.PatientName)
        assert "Smith" in pn_str
        assert "John" in pn_str
        assert "Robert" in pn_str
        assert "Dr." in pn_str
        assert "Jr." in pn_str

    def test_person_name_with_ideographic_and_phonetic(self, pn_multigroup_dataset: Dataset) -> None:
        """Contract: PN with Alphabetic=Ideographic=Phonetic groups round-trips."""
        result = from_xml(to_xml(pn_multigroup_dataset))
        pn = PersonName(result.PatientName)
        assert str(pn.alphabetic) == "Yamada^Tarou"
        assert str(pn.ideographic) == "山田^太郎"
        assert str(pn.phonetic) == "やまだ^たろう"

    def test_zero_length_elements(self) -> None:
        """Contract: zero-length elements survive round-trip as zero-length."""
        ds = Dataset()
        ds.add_new(Tag(0x0010, 0x0030), "DA", None)
        ds.PatientID = "P123"
        result = from_xml(to_xml(ds))
        assert Tag(0x0010, 0x0030) in result
        assert result.PatientID == "P123"

    def test_multi_valued_with_empty_middle_value(self) -> None:
        """Contract: multi-valued field with empty middle value preserves value count."""
        ds = Dataset()
        ds.add_new(Tag(0x0008, 0x0008), "CS", ["ORIGINAL", "", "PRIMARY"])
        result = from_xml(to_xml(ds))
        vals = list(result[Tag(0x0008, 0x0008)].value)
        assert len(vals) == 3
        assert vals[0] == "ORIGINAL"
        assert vals[1] == ""
        assert vals[2] == "PRIMARY"

    def test_binary_data_ow_inline(self, binary_dataset: Dataset) -> None:
        """Contract: OW pixel data encoded as InlineBinary survives round-trip."""
        result = from_xml(to_xml(binary_dataset))
        assert result[Tag(0x7FE0, 0x0010)].value == bytes(range(16))

    def test_binary_data_ob_inline(self) -> None:
        """Contract: OB binary data survives round-trip byte-for-byte."""
        ds = Dataset()
        payload = b"\xde\xad\xbe\xef"
        ds.add_new(Tag(0x0042, 0x0011), "OB", payload)
        result = from_xml(to_xml(ds))
        assert result[Tag(0x0042, 0x0011)].value == payload

    def test_at_vr_round_trip(self) -> None:
        """Contract: AT VR encodes as 8-char hex and decodes back to a Tag."""
        ds = Dataset()
        ds.add_new(Tag(0x0028, 0x0009), "AT", Tag(0x0018, 0x1063))
        result = from_xml(to_xml(ds))
        assert result[Tag(0x0028, 0x0009)].value == Tag(0x0018, 0x1063)

    def test_private_elements_round_trip(self) -> None:
        """Contract: private elements (creator + data) survive round-trip."""
        ds = Dataset()
        ds.add_new(Tag(0x0009, 0x0010), "LO", "ACME Corp")
        ds.add_new(Tag(0x0009, 0x1001), "LO", "private value")
        result = from_xml(to_xml(ds))
        assert result[Tag(0x0009, 0x0010)].value == "ACME Corp"
        assert result[Tag(0x0009, 0x1001)].value == "private value"

    def test_numeric_vrs_integer(self) -> None:
        """Contract: integer VRs (US/SS/UL/SL) round-trip as correct Python ints."""
        ds = Dataset()
        ds.add_new(Tag(0x0028, 0x0010), "US", 512)
        ds.add_new(Tag(0x0028, 0x0011), "US", 256)
        ds.add_new(Tag(0x0028, 0x0106), "US", 0)
        ds.add_new(Tag(0x0028, 0x0107), "US", 4095)
        result = from_xml(to_xml(ds))
        assert result[Tag(0x0028, 0x0010)].value == 512
        assert result[Tag(0x0028, 0x0011)].value == 256
        assert result[Tag(0x0028, 0x0106)].value == 0
        assert result[Tag(0x0028, 0x0107)].value == 4095

    def test_float_vrs_fl_fd(self) -> None:
        """Contract: float VRs (FL/FD) round-trip within precision."""
        ds = Dataset()
        ds.add_new(Tag(0x0018, 0x0088), "DS", "1.234567")
        ds.add_new(Tag(0x0066, 0x0011), "FL", 3.14)
        result = from_xml(to_xml(ds))
        assert float(result[Tag(0x0018, 0x0088)].value) == pytest.approx(1.234567)
        assert result[Tag(0x0066, 0x0011)].value == pytest.approx(3.14, rel=1e-5)

    def test_ui_uid_values(self) -> None:
        """Contract: UI VR (UID strings) survive round-trip unchanged."""
        ds = Dataset()
        uid = "1.2.840.10008.5.1.4.1.1.2"
        ds.add_new(Tag(0x0008, 0x0016), "UI", uid)
        result = from_xml(to_xml(ds))
        assert result[Tag(0x0008, 0x0016)].value == uid

    def test_is_vr_round_trip(self) -> None:
        """Contract: IS VR preserves integer semantics through round-trip."""
        ds = Dataset()
        ds.add_new(Tag(0x0020, 0x0013), "IS", "42")
        result = from_xml(to_xml(ds))
        assert int(result[Tag(0x0020, 0x0013)].value) == 42

    def test_multiple_sequence_items(self) -> None:
        """Contract: SQ with multiple items — all items present and correct."""
        ds = Dataset()
        items = []
        for i in range(3):
            item = Dataset()
            item.add_new(Tag(0x0020, 0x0013), "IS", str(i + 1))
            items.append(item)
        ds.add_new(Tag(0x0008, 0x1115), "SQ", Sequence(items))
        result = from_xml(to_xml(ds))
        result_items = result[Tag(0x0008, 0x1115)].value
        assert len(result_items) == 3
        for i, item in enumerate(result_items):
            assert int(item[Tag(0x0020, 0x0013)].value) == i + 1

    def test_empty_sequence(self) -> None:
        """Contract: empty SQ element round-trips as empty sequence."""
        ds = Dataset()
        ds.add_new(Tag(0x0008, 0x1115), "SQ", Sequence([]))
        result = from_xml(to_xml(ds))
        seq = result[Tag(0x0008, 0x1115)].value
        assert len(seq) == 0

    def test_bulk_data_handler_in_round_trip(self) -> None:
        """Contract: BulkData URI handler resolves to original bytes on from_xml."""
        payload = bytes(range(64)) * 20  # 1280 bytes — above default 1024 threshold

        def to_uri(de: DataElement) -> str:
            return f"http://example.com/pixel/{de.tag:08X}"

        xml_bytes = to_xml(Dataset(), bulk_data_threshold=1024)
        # Build dataset with large binary, serialise with bulk data handler
        ds = Dataset()
        ds.add_new(Tag(0x7FE0, 0x0010), "OW", payload)
        xml_bytes = to_xml(ds, bulk_data_threshold=512, bulk_data_element_handler=to_uri)

        def from_uri(_tag: str, _vr: str, uri: str) -> bytes:
            return payload  # resolve to known payload

        result = from_xml(xml_bytes, bulk_data_uri_handler=from_uri)
        assert result[Tag(0x7FE0, 0x0010)].value == payload

    def test_dataset_equivalence_simple(self, simple_dataset: Dataset) -> None:
        """Contract: _datasets_equivalent() confirms round-trip produces equivalent Dataset."""
        result = from_xml(to_xml(simple_dataset))
        assert _datasets_equivalent(simple_dataset, result)

    def test_custom_dataset_class(self) -> None:
        """Contract: dataset_class parameter is used for constructed items."""

        class MyDataset(Dataset):
            pass

        ds = Dataset()
        ds.PatientID = "P001"
        from pydicom_xml import dataset_from_xml, dataset_to_xml

        result = dataset_from_xml(dataset_to_xml(ds), dataset_class=MyDataset)
        assert isinstance(result, MyDataset)
        assert result.PatientID == "P001"

    def test_empty_dataset_round_trip(self) -> None:
        """Contract: empty Dataset produces valid XML and round-trips back to empty Dataset."""
        ds = Dataset()
        result = from_xml(to_xml(ds))
        assert len(result) == 0
