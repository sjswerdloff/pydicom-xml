"""Unit tests for pydicom_xml.xmlrep module.

Tests are organized by concern:
- XmlDataElementConverter — deserialization of individual XML elements
- data_element_to_xml_element — serialization of individual DataElements
- Bulk data threshold and handler behaviour
- PersonName encoding and decoding
- AT VR encoding and decoding
- Error handling paths

All test datasets are constructed programmatically — no DICOM files required.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET

import pytest
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.valuerep import PersonName

from pydicom_xml.xmlrep import (
    NAMESPACE,
    DicomXmlAtValueHexError,
    DicomXmlAtValueLengthError,
    DicomXmlRootError,
    DicomXmlTagHexError,
    DicomXmlTagLengthError,
    XmlDataElementConverter,
    data_element_to_xml_element,
    dataset_from_xml,
    dataset_to_xml,
)

_NS = f"{{{NAMESPACE}}}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dicom_attribute(tag: str, vr: str, children: list[ET.Element] | None = None) -> ET.Element:
    """Build a minimal <DicomAttribute> element for converter tests."""
    attr = ET.Element(f"{_NS}DicomAttribute")
    attr.set("tag", tag)
    attr.set("vr", vr)
    for child in children or []:
        attr.append(child)
    return attr


def _make_value_child(number: int, text: str) -> ET.Element:
    """Build a <Value number="N"> child element."""
    v = ET.Element(f"{_NS}Value")
    v.set("number", str(number))
    v.text = text
    return v


def _make_inline_binary_child(data: bytes) -> ET.Element:
    """Build an <InlineBinary> child element with base64 content."""
    ib = ET.Element(f"{_NS}InlineBinary")
    ib.text = base64.b64encode(data).decode("ascii")
    return ib


def _make_bulk_data_child(uri: str) -> ET.Element:
    """Build a <BulkData uri="..."> child element."""
    bd = ET.Element(f"{_NS}BulkData")
    bd.set("uri", uri)
    return bd


# ---------------------------------------------------------------------------
# XmlDataElementConverter — zero-length / empty
# ---------------------------------------------------------------------------


class TestXmlDataElementConverterEmpty:
    def test_no_children_returns_empty_value_for_vr(self) -> None:
        """Contract: DicomAttribute with no children returns empty VR value."""
        attr = _make_dicom_attribute("00100020", "LO")
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        # empty_value_for_VR("LO") is ""
        assert result == "" or result is None

    def test_empty_value_element_returns_empty_string(self) -> None:
        """Contract: a single empty <Value> element returns empty string for LO."""
        attr = _make_dicom_attribute("00100020", "LO", [_make_value_child(1, "")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert result == ""


# ---------------------------------------------------------------------------
# XmlDataElementConverter — Value elements (various VRs)
# ---------------------------------------------------------------------------


class TestXmlDataElementConverterValues:
    def test_lo_single_value(self) -> None:
        """Contract: single LO Value element returns a plain string."""
        attr = _make_dicom_attribute("00100020", "LO", [_make_value_child(1, "P001")])
        converter = XmlDataElementConverter(Dataset, attr)
        assert converter.get_element_values() == "P001"

    def test_us_single_value_returns_int(self) -> None:
        """Contract: single US Value element returns int."""
        attr = _make_dicom_attribute("00280010", "US", [_make_value_child(1, "512")])
        converter = XmlDataElementConverter(Dataset, attr)
        assert converter.get_element_values() == 512
        assert isinstance(converter.get_element_values(), int)

    def test_ss_single_value_returns_int(self) -> None:
        """Contract: SS Value element returns int."""
        attr = _make_dicom_attribute("00281052", "SS", [_make_value_child(1, "-1024")])
        converter = XmlDataElementConverter(Dataset, attr)
        assert converter.get_element_values() == -1024

    def test_ul_single_value_returns_int(self) -> None:
        """Contract: UL Value element returns int."""
        attr = _make_dicom_attribute("00280120", "UL", [_make_value_child(1, "65536")])
        converter = XmlDataElementConverter(Dataset, attr)
        assert converter.get_element_values() == 65536

    def test_fl_single_value_returns_float(self) -> None:
        """Contract: FL Value element returns float."""
        attr = _make_dicom_attribute("00660011", "FL", [_make_value_child(1, "3.14")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_fd_single_value_returns_float(self) -> None:
        """Contract: FD Value element returns float."""
        attr = _make_dicom_attribute("00186024", "FD", [_make_value_child(1, "2.718281828")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert isinstance(result, float)
        assert result == pytest.approx(2.718281828)

    def test_ds_returns_float_value(self) -> None:
        """Contract: DS Value element returns a float (DS is in FLOAT_VR in pydicom)."""
        attr = _make_dicom_attribute("00180088", "DS", [_make_value_child(1, "1.5")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        # pydicom's FLOAT_VR includes DS, so _coerce_value returns float
        assert float(result) == pytest.approx(1.5)

    def test_is_returns_int_value(self) -> None:
        """Contract: IS Value element returns an int (IS is in INT_VR in pydicom)."""
        attr = _make_dicom_attribute("00200013", "IS", [_make_value_child(1, "42")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        # pydicom's INT_VR includes IS, so _coerce_value returns int
        assert int(result) == 42

    def test_multivalued_lo_returns_multivalue(self) -> None:
        """Contract: multiple Value elements return MultiValue list."""
        from pydicom.multival import MultiValue

        attr = _make_dicom_attribute(
            "00080008",
            "CS",
            [_make_value_child(1, "ORIGINAL"), _make_value_child(2, "PRIMARY")],
        )
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert isinstance(result, MultiValue)
        assert list(result) == ["ORIGINAL", "PRIMARY"]

    def test_multivalued_us_returns_multivalue_of_ints(self) -> None:
        """Contract: multiple US Value elements return MultiValue of ints."""
        from pydicom.multival import MultiValue

        attr = _make_dicom_attribute(
            "00280010",
            "US",
            [_make_value_child(1, "512"), _make_value_child(2, "256")],
        )
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert isinstance(result, MultiValue)
        assert list(result) == [512, 256]


# ---------------------------------------------------------------------------
# XmlDataElementConverter — AT VR
# ---------------------------------------------------------------------------


class TestXmlDataElementConverterAT:
    def test_at_valid_returns_tag(self) -> None:
        """Contract: AT Value with valid 8-char hex returns a Tag."""
        attr = _make_dicom_attribute("00280009", "AT", [_make_value_child(1, "00181063")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert result == Tag(0x0018, 0x1063)

    def test_at_too_short_raises_length_error(self) -> None:
        """Contract: AT Value with fewer than 8 hex chars raises DicomXmlAtValueLengthError."""
        attr = _make_dicom_attribute("00280009", "AT", [_make_value_child(1, "1234")])
        converter = XmlDataElementConverter(Dataset, attr)
        with pytest.raises(DicomXmlAtValueLengthError):
            converter.get_element_values()

    def test_at_non_hex_raises_hex_error(self) -> None:
        """Contract: AT Value with non-hex characters raises DicomXmlAtValueHexError."""
        attr = _make_dicom_attribute("00280009", "AT", [_make_value_child(1, "ZZZZYYYY")])
        converter = XmlDataElementConverter(Dataset, attr)
        with pytest.raises(DicomXmlAtValueHexError):
            converter.get_element_values()


# ---------------------------------------------------------------------------
# XmlDataElementConverter — InlineBinary
# ---------------------------------------------------------------------------


class TestXmlDataElementConverterInlineBinary:
    def test_inline_binary_decodes_to_bytes(self) -> None:
        """Contract: <InlineBinary> element decodes back to original bytes."""
        payload = bytes(range(16))
        attr = _make_dicom_attribute("7FE00010", "OW", [_make_inline_binary_child(payload)])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        assert result == payload

    def test_empty_inline_binary_returns_empty_bytes(self) -> None:
        """Contract: empty <InlineBinary> element returns b''."""
        ib = ET.Element(f"{_NS}InlineBinary")
        ib.text = ""
        attr = _make_dicom_attribute("7FE00010", "OW", [ib])
        converter = XmlDataElementConverter(Dataset, attr)
        assert converter.get_element_values() == b""


# ---------------------------------------------------------------------------
# XmlDataElementConverter — BulkData
# ---------------------------------------------------------------------------


class TestXmlDataElementConverterBulkData:
    def test_bulk_data_no_handler_returns_empty_value(self) -> None:
        """Contract: BulkData element without handler returns empty_value_for_VR result."""
        from pydicom.dataelem import empty_value_for_VR

        attr = _make_dicom_attribute("7FE00010", "OW", [_make_bulk_data_child("http://example.com/pixel")])
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        # empty_value_for_VR("OW") in pydicom 3.x returns None
        assert result == empty_value_for_VR("OW")

    def test_bulk_data_3arg_handler_called_with_tag_vr_uri(self) -> None:
        """Contract: 3-argument bulk_data_uri_handler receives tag, vr, and uri."""
        calls: list[tuple[str, str, str]] = []

        def handler(tag: str, vr: str, uri: str) -> bytes:
            calls.append((tag, vr, uri))
            return b"resolved"

        attr = _make_dicom_attribute("7FE00010", "OW", [_make_bulk_data_child("http://example.com/pixel")])
        converter = XmlDataElementConverter(Dataset, attr, handler)
        result = converter.get_element_values()
        assert result == b"resolved"
        assert calls == [("7FE00010", "OW", "http://example.com/pixel")]

    def test_bulk_data_1arg_handler_wraps_correctly(self) -> None:
        """Contract: 1-argument bulk_data_uri_handler is wrapped and receives only URI."""
        calls: list[str] = []

        def handler(uri: str) -> bytes:
            calls.append(uri)
            return b"single-arg"

        attr = _make_dicom_attribute("7FE00010", "OW", [_make_bulk_data_child("http://example.com/px")])
        converter = XmlDataElementConverter(Dataset, attr, handler)
        result = converter.get_element_values()
        assert result == b"single-arg"
        assert calls == ["http://example.com/px"]


# ---------------------------------------------------------------------------
# XmlDataElementConverter — PersonName
# ---------------------------------------------------------------------------


class TestXmlDataElementConverterPersonName:
    def _make_pn_attr(self, pn_str: str) -> ET.Element:
        """Build a DicomAttribute element containing one PersonName from a string."""
        # Build the XML by serializing from a Dataset (reuses our to_xml path)
        ds = Dataset()
        ds.PatientName = PersonName(pn_str)
        xml_bytes = dataset_to_xml(ds)
        root = ET.fromstring(xml_bytes.decode("utf-8"))
        attr = root.find(f"{_NS}DicomAttribute[@tag='00100010']")
        assert attr is not None
        return attr

    def test_alphabetic_only(self) -> None:
        """Contract: alphabetic-only PN decodes correctly."""
        attr = self._make_pn_attr("Smith^John")
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        pn = PersonName(result)
        assert str(pn.alphabetic) == "Smith^John"

    def test_all_three_groups(self) -> None:
        """Contract: PN with all three groups (Alphabetic=Ideographic=Phonetic) decodes correctly."""
        attr = self._make_pn_attr("Yamada^Tarou=山田^太郎=やまだ^たろう")
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        pn = PersonName(result)
        assert str(pn.alphabetic) == "Yamada^Tarou"
        assert str(pn.ideographic) == "山田^太郎"
        assert str(pn.phonetic) == "やまだ^たろう"

    def test_all_five_alphabetic_components(self) -> None:
        """Contract: all five PN sub-components (Family^Given^Middle^Prefix^Suffix) round-trip."""
        attr = self._make_pn_attr("Smith^John^Robert^Dr.^Jr.")
        converter = XmlDataElementConverter(Dataset, attr)
        result = converter.get_element_values()
        pn_str = str(PersonName(result))
        assert "Smith" in pn_str
        assert "John" in pn_str
        assert "Robert" in pn_str
        assert "Dr." in pn_str
        assert "Jr." in pn_str


# ---------------------------------------------------------------------------
# data_element_to_xml_element — serialization
# ---------------------------------------------------------------------------


class TestDataElementToXmlElement:
    def _root(self) -> ET.Element:
        return ET.Element(f"{_NS}NativeDicomModel")

    def test_lo_produces_value_element(self) -> None:
        """Contract: LO DataElement produces a <Value number="1"> child."""
        elem = DataElement(Tag(0x0010, 0x0020), "LO", "P001")
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        value = attr.find(f"{_NS}Value")
        assert value is not None
        assert value.text == "P001"

    def test_us_encodes_as_string_in_value(self) -> None:
        """Contract: US value is written as a decimal string in <Value>."""
        elem = DataElement(Tag(0x0028, 0x0010), "US", 512)
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        value = attr.find(f"{_NS}Value")
        assert value is not None
        assert value.text == "512"

    def test_at_encodes_as_8char_hex(self) -> None:
        """Contract: AT value is written as 8-char uppercase hex in <Value>."""
        elem = DataElement(Tag(0x0028, 0x0009), "AT", Tag(0x0018, 0x1063))
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        value = attr.find(f"{_NS}Value")
        assert value is not None
        assert value.text == "00181063"

    def test_ow_binary_produces_inline_binary(self) -> None:
        """Contract: OW DataElement produces an <InlineBinary> child."""
        payload = bytes(range(8))
        elem = DataElement(Tag(0x7FE0, 0x0010), "OW", payload)
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        ib = attr.find(f"{_NS}InlineBinary")
        assert ib is not None
        assert base64.b64decode(ib.text or "") == payload

    def test_ow_large_binary_uses_bulk_data_uri(self) -> None:
        """Contract: large OW DataElement uses BulkData URI when handler provided."""

        def handler(de: DataElement) -> str:
            return "http://example.com/bulk"

        payload = bytes(1024)  # 1 KB — above default threshold
        elem = DataElement(Tag(0x7FE0, 0x0010), "OW", payload)
        root = self._root()
        data_element_to_xml_element(elem, root, bulk_data_threshold=512, bulk_data_element_handler=handler)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        bd = attr.find(f"{_NS}BulkData")
        assert bd is not None
        assert bd.get("uri") == "http://example.com/bulk"

    def test_small_binary_below_threshold_uses_inline(self) -> None:
        """Contract: binary data below threshold uses InlineBinary even with handler."""

        def handler(de: DataElement) -> str:
            return "http://example.com/bulk"

        payload = b"\x00\x01\x02"  # well below any threshold
        elem = DataElement(Tag(0x7FE0, 0x0010), "OW", payload)
        root = self._root()
        data_element_to_xml_element(elem, root, bulk_data_threshold=1024, bulk_data_element_handler=handler)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        # Should use InlineBinary, not BulkData
        assert attr.find(f"{_NS}InlineBinary") is not None
        assert attr.find(f"{_NS}BulkData") is None

    def test_empty_element_produces_no_children(self) -> None:
        """Contract: zero-length DataElement produces <DicomAttribute> with no children."""
        elem = DataElement(Tag(0x0010, 0x0030), "DA", None)
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        assert len(list(attr)) == 0

    def test_tag_attribute_is_8_char_uppercase_hex(self) -> None:
        """Contract: tag attribute on <DicomAttribute> is 8-char uppercase hex."""
        elem = DataElement(Tag(0x0010, 0x0010), "PN", "Smith^John")
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        tag_str = attr.get("tag", "")
        assert len(tag_str) == 8
        assert tag_str == tag_str.upper()
        assert tag_str == "00100010"

    def test_keyword_attribute_populated_for_known_tags(self) -> None:
        """Contract: keyword attribute is set from pydicom data dictionary for known tags."""
        elem = DataElement(Tag(0x0010, 0x0010), "PN", "Smith^John")
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        assert attr.get("keyword") == "PatientName"

    def test_sq_produces_item_elements(self) -> None:
        """Contract: SQ DataElement produces numbered <Item> children."""
        item1 = Dataset()
        item1.PatientID = "P001"
        seq_elem = DataElement(Tag(0x0008, 0x1115), "SQ", Sequence([item1]))
        root = self._root()
        data_element_to_xml_element(seq_elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        items = attr.findall(f"{_NS}Item")
        assert len(items) == 1
        assert items[0].get("number") == "1"

    def test_pn_produces_person_name_element(self) -> None:
        """Contract: PN DataElement produces a <PersonName> child."""
        elem = DataElement(Tag(0x0010, 0x0010), "PN", "Smith^John")
        root = self._root()
        data_element_to_xml_element(elem, root)
        attr = root.find(f"{_NS}DicomAttribute")
        assert attr is not None
        pn = attr.find(f"{_NS}PersonName")
        assert pn is not None


# ---------------------------------------------------------------------------
# dataset_to_xml — structural requirements
# ---------------------------------------------------------------------------


class TestDatasetToXml:
    def test_output_is_valid_xml(self, simple_dataset: Dataset) -> None:
        """Contract: output is parseable as valid XML."""
        xml_bytes = dataset_to_xml(simple_dataset)
        ET.fromstring(xml_bytes.decode("utf-8"))

    def test_root_element_is_native_dicom_model(self, simple_dataset: Dataset) -> None:
        """Contract: root element is NativeDicomModel in the correct namespace."""
        xml_bytes = dataset_to_xml(simple_dataset)
        root = ET.fromstring(xml_bytes.decode("utf-8"))
        assert root.tag == f"{_NS}NativeDicomModel"

    def test_xml_space_preserve_on_root(self, simple_dataset: Dataset) -> None:
        """Contract: xml:space="preserve" is on the root element."""
        xml_bytes = dataset_to_xml(simple_dataset)
        root = ET.fromstring(xml_bytes.decode("utf-8"))
        space = root.get("{http://www.w3.org/XML/1998/namespace}space")
        assert space == "preserve"

    def test_xml_declaration_in_output(self, simple_dataset: Dataset) -> None:
        """Contract: output starts with XML declaration."""
        assert dataset_to_xml(simple_dataset).startswith(b"<?xml")

    def test_group_length_elements_excluded(self) -> None:
        """Contract: group length elements (gggg,0000) are not included."""
        ds = Dataset()
        ds.PatientID = "P001"
        ds.add_new(Tag(0x0010, 0x0000), "UL", 100)
        root = ET.fromstring(dataset_to_xml(ds).decode("utf-8"))
        for attr in root.iter(f"{_NS}DicomAttribute"):
            assert not attr.get("tag", "").endswith("0000")

    def test_file_meta_information_excluded(self) -> None:
        """Contract: File Meta Information (group 0002) elements are not included."""
        ds = Dataset()
        ds.PatientID = "P001"
        ds.add_new(Tag(0x0002, 0x0010), "UI", "1.2.840.10008.1.2.1")
        root = ET.fromstring(dataset_to_xml(ds).decode("utf-8"))
        for attr in root.iter(f"{_NS}DicomAttribute"):
            assert not attr.get("tag", "").startswith("0002")

    def test_elements_sorted_by_tag_ascending(self) -> None:
        """Contract: DicomAttribute elements appear in ascending tag order."""
        ds = Dataset()
        ds.add_new(Tag(0x0028, 0x0011), "US", 256)
        ds.add_new(Tag(0x0010, 0x0010), "PN", "Smith^John")
        ds.add_new(Tag(0x0008, 0x0020), "DA", "20240101")
        root = ET.fromstring(dataset_to_xml(ds).decode("utf-8"))
        tags = [a.get("tag", "") for a in root.iter(f"{_NS}DicomAttribute")]
        assert tags == sorted(tags)

    def test_private_creator_attribute_set(self) -> None:
        """Contract: private elements have privateCreator attribute."""
        ds = Dataset()
        ds.add_new(Tag(0x0009, 0x0010), "LO", "ACME Corp")
        ds.add_new(Tag(0x0009, 0x1001), "LO", "private value")
        root = ET.fromstring(dataset_to_xml(ds).decode("utf-8"))
        priv = root.find(f".//{_NS}DicomAttribute[@tag='00091001']")
        assert priv is not None
        assert priv.get("privateCreator") == "ACME Corp"

    def test_sequence_items_numbered_from_one(self, sequence_dataset: Dataset) -> None:
        """Contract: Item elements in SQ are numbered starting from 1."""
        root = ET.fromstring(dataset_to_xml(sequence_dataset).decode("utf-8"))
        items = root.findall(f".//{_NS}Item")
        assert items[0].get("number") == "1"

    def test_person_name_value_elements_numbered_from_one(self, simple_dataset: Dataset) -> None:
        """Contract: PersonName elements are numbered from 1."""
        root = ET.fromstring(dataset_to_xml(simple_dataset).decode("utf-8"))
        pn_elems = root.findall(f".//{_NS}PersonName")
        for pn in pn_elems:
            assert pn.get("number") == "1"


# ---------------------------------------------------------------------------
# dataset_from_xml — parsing
# ---------------------------------------------------------------------------


class TestDatasetFromXml:
    def test_wrong_root_element_raises(self) -> None:
        """Contract: wrong root element raises DicomXmlRootError."""
        xml = b'<?xml version="1.0" encoding="utf-8"?><Root/>'
        with pytest.raises(DicomXmlRootError):
            dataset_from_xml(xml)

    def test_tag_too_short_raises_tag_length_error(self) -> None:
        """Contract: DicomAttribute with tag shorter than 8 chars raises DicomXmlTagLengthError."""
        ns = NAMESPACE
        xml = (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<NativeDicomModel xmlns="{ns}">'
            f'<DicomAttribute tag="1234" vr="LO"/>'
            f"</NativeDicomModel>"
        ).encode()
        with pytest.raises(DicomXmlTagLengthError):
            dataset_from_xml(xml)

    def test_tag_non_hex_raises_tag_hex_error(self) -> None:
        """Contract: DicomAttribute with non-hex tag raises DicomXmlTagHexError."""
        ns = NAMESPACE
        xml = (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<NativeDicomModel xmlns="{ns}">'
            f'<DicomAttribute tag="ZZZZYYYY" vr="LO"/>'
            f"</NativeDicomModel>"
        ).encode()
        with pytest.raises(DicomXmlTagHexError):
            dataset_from_xml(xml)

    def test_accepts_string_input(self) -> None:
        """Contract: dataset_from_xml accepts str as well as bytes."""
        ds = Dataset()
        ds.PatientID = "P001"
        xml_str = dataset_to_xml(ds).decode("utf-8")
        result = dataset_from_xml(xml_str)
        assert result.PatientID == "P001"

    def test_bulk_data_1arg_handler_is_called(self) -> None:
        """Contract: 1-argument bulk_data_uri_handler is normalised and called."""
        # Build XML with a BulkData element
        payload = bytes(2048)
        ds = Dataset()
        ds.add_new(Tag(0x7FE0, 0x0010), "OW", payload)

        def bulk_handler(de: DataElement) -> str:
            return "http://example.com/pixel"

        xml_bytes = dataset_to_xml(ds, bulk_data_threshold=512, bulk_data_element_handler=bulk_handler)

        calls: list[str] = []

        def resolve(uri: str) -> bytes:
            calls.append(uri)
            return payload

        result = dataset_from_xml(xml_bytes, bulk_data_uri_handler=resolve)
        assert result[Tag(0x7FE0, 0x0010)].value == payload
        assert calls == ["http://example.com/pixel"]

    def test_bulk_data_3arg_handler_is_called(self) -> None:
        """Contract: 3-argument bulk_data_uri_handler receives tag, vr, and uri."""
        payload = bytes(2048)
        ds = Dataset()
        ds.add_new(Tag(0x7FE0, 0x0010), "OW", payload)

        def bulk_handler(de: DataElement) -> str:
            return "http://example.com/pixel"

        xml_bytes = dataset_to_xml(ds, bulk_data_threshold=512, bulk_data_element_handler=bulk_handler)

        calls: list[tuple[str, str, str]] = []

        def resolve(tag: str, vr: str, uri: str) -> bytes:
            calls.append((tag, vr, uri))
            return payload

        result = dataset_from_xml(xml_bytes, bulk_data_uri_handler=resolve)
        assert result[Tag(0x7FE0, 0x0010)].value == payload
        assert calls == [("7FE00010", "OW", "http://example.com/pixel")]
