"""Methods for converting Datasets and DataElements to/from the DICOM XML Native Model (PS3.19 Annex A).

References:
    * PS3.19 Annex A.1 — Native DICOM Model XML Format
    * PS3.19 Annex A.1.5 — DicomAttribute encoding rules (Table A.1.5-2)
    * PS3.19 Annex A.1.6 — RELAX NG Compact normative specification

This module mirrors the architecture of pydicom's jsonrep.py and is designed
to be suitable for upstream contribution to pydicom.
"""

from __future__ import annotations

import base64
import io
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from inspect import signature
from typing import TYPE_CHECKING, Any

from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.tag import BaseTag, Tag
from pydicom.valuerep import AMBIGUOUS_VR, BYTES_VR, FLOAT_VR, INT_VR, VR, PersonName

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

NAMESPACE = "http://dicom.nema.org/PS3.19/models/NativeDICOM"

# XML element names matching PS3.19 schema (Table A.1.5-2)
XML_VALUE_ELEMENTS = ("Value", "BulkData", "InlineBinary", "PersonName", "Item")

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_NS_PREFIX = f"{{{NAMESPACE}}}"

# VRs that store raw binary payloads (Table A.1.5-2)
_BINARY_VRS: set[VR] = BYTES_VR

# PersonName XML component group names in order (= delimiter)
_PN_GROUPS = ("Alphabetic", "Ideographic", "Phonetic")

# PersonName sub-component element names in order (^ delimiter)
_PN_COMPONENTS = ("FamilyName", "GivenName", "MiddleName", "NamePrefix", "NameSuffix")

# ---------------------------------------------------------------------------
# Type aliases — mirrors jsonrep.py
# ---------------------------------------------------------------------------

BulkDataType = None | str | int | float | bytes
BulkDataHandlerType = Callable[[str, str, str], BulkDataType] | None

# ---------------------------------------------------------------------------
# Custom exceptions (TRY003 compliance)
# ---------------------------------------------------------------------------


class DicomXmlError(ValueError):
    """Raised when DICOM XML is structurally invalid or unsupported."""

    def __init__(self, detail: str) -> None:
        """Initialize with a descriptive detail string.

        Args:
            detail: Human-readable explanation of the problem.
        """
        super().__init__(f"DICOM XML error: {detail}")
        self.detail = detail


class DicomXmlAtValueLengthError(DicomXmlError):
    """Raised when an AT Value element does not contain exactly 8 hex characters."""

    def __init__(self, text: str) -> None:
        """Initialize with the offending text.

        Args:
            text: The Value element text that failed length validation.
        """
        super().__init__(f"AT value must be 8 hex chars, got {text!r}")
        self.text = text


class DicomXmlAtValueHexError(DicomXmlError):
    """Raised when an AT Value element contains non-hex characters."""

    def __init__(self, text: str) -> None:
        """Initialize with the offending text.

        Args:
            text: The Value element text that failed hex parsing.
        """
        super().__init__(f"Invalid AT hex value {text!r}")
        self.text = text


class DicomXmlTagLengthError(DicomXmlError):
    """Raised when a DicomAttribute tag attribute is not exactly 8 characters."""

    def __init__(self, tag_str: str) -> None:
        """Initialize with the offending tag string.

        Args:
            tag_str: The tag attribute value that failed length validation.
        """
        super().__init__(f"DicomAttribute 'tag' must be 8 chars, got {tag_str!r}")
        self.tag_str = tag_str


class DicomXmlTagHexError(DicomXmlError):
    """Raised when a DicomAttribute tag attribute contains non-hex characters."""

    def __init__(self, tag_str: str) -> None:
        """Initialize with the offending tag string.

        Args:
            tag_str: The tag attribute value that failed hex parsing.
        """
        super().__init__(f"Invalid tag hex value {tag_str!r}")
        self.tag_str = tag_str


class DicomXmlRootError(DicomXmlError):
    """Raised when the XML root element is not NativeDicomModel in the correct namespace."""

    def __init__(self, actual_tag: str) -> None:
        """Initialize with the actual root tag found.

        Args:
            actual_tag: The root element tag that was found instead of NativeDicomModel.
        """
        super().__init__(f"Root element must be NativeDicomModel in namespace {NAMESPACE!r}, got {actual_tag!r}")
        self.actual_tag = actual_tag


class DicomXmlParseError(DicomXmlError):
    """Raised when the XML input cannot be parsed (malformed XML)."""

    def __init__(self, detail: str) -> None:
        """Initialize with the XML parse error detail.

        Args:
            detail: Human-readable explanation from the XML parser.
        """
        super().__init__(f"XML parse error: {detail}")


# ---------------------------------------------------------------------------
# XmlDataElementConverter — mirrors JsonDataElementConverter
# ---------------------------------------------------------------------------


class XmlDataElementConverter:
    """Convert from an XML DicomAttribute element to a DataElement.

    Mirrors ``JsonDataElementConverter`` from ``pydicom.jsonrep``.

    References:
        PS3.19 Annex A.1 — Native DICOM Model
    """

    def __init__(
        self,
        dataset_class: type[Dataset],
        dicom_attribute: ET.Element,
        bulk_data_uri_handler: BulkDataHandlerType | Callable[[str], BulkDataType] | None = None,
    ) -> None:
        """Create a new converter instance.

        Args:
            dataset_class: The class object to use for SQ element items.
            dicom_attribute: A ``<DicomAttribute>`` XML element from PS3.19 Native DICOM Model.
            bulk_data_uri_handler: Callable that accepts ``(tag, vr, uri)`` or just ``(uri)``
                and returns the resolved bulk data value.  When ``None`` (default) a BulkData
                element produces an empty value for its VR.
        """
        self.dataset_class = dataset_class
        self.dicom_attribute = dicom_attribute
        self.tag_str: str = dicom_attribute.get("tag", "")
        self.vr: str = dicom_attribute.get("vr", "")

        # Normalise handler to 3-argument form, same as JsonDataElementConverter
        handler = bulk_data_uri_handler
        if handler is not None and len(signature(handler).parameters) == 1:
            # Callable[[str], BulkDataType] → wrap to 3-arg form
            _h: Callable[[str], BulkDataType] = handler  # type: ignore[assignment]

            def _wrapper(_tag: str, _vr: str, value: str) -> BulkDataType:
                return _h(value)

            self.bulk_data_element_handler: BulkDataHandlerType = _wrapper
        else:
            self.bulk_data_element_handler = handler  # type: ignore[assignment]

    def get_element_values(self) -> Any:
        """Return the element value(s) parsed from the DicomAttribute XML element.

        Returns:
            The value or value list of the newly created data element.  Type
            depends on VR: ``None``, ``str``, ``int``, ``float``, ``bytes``,
            ``Dataset`` subclass or a list of these.

        Raises:
            DicomXmlTagLengthError: If the ``tag`` attribute is not exactly 8 chars.
            DicomXmlTagHexError: If the ``tag`` attribute contains non-hex characters.
        """
        from pydicom.dataelem import empty_value_for_VR

        attr = self.dicom_attribute

        # Classify child content
        items = attr.findall(f"{_NS_PREFIX}Item")
        pn_elems = attr.findall(f"{_NS_PREFIX}PersonName")
        inline_binary = attr.find(f"{_NS_PREFIX}InlineBinary")
        bulk_data = attr.find(f"{_NS_PREFIX}BulkData")
        value_elems = attr.findall(f"{_NS_PREFIX}Value")

        if items:
            # SQ — recurse, forwarding the bulk_data_element_handler so BulkData
            # elements inside sequence items are resolved correctly.
            return Sequence([_element_to_dataset(self.dataset_class, item, self.bulk_data_element_handler) for item in items])

        if pn_elems:
            return _decode_person_name(attr)

        if inline_binary is not None:
            raw_b64 = (inline_binary.text or "").strip()
            return base64.b64decode(raw_b64) if raw_b64 else b""

        if bulk_data is not None:
            uri = bulk_data.get("uri", "")
            if self.bulk_data_element_handler is None:
                return empty_value_for_VR(self.vr)
            return self.bulk_data_element_handler(self.tag_str, self.vr, uri)

        if value_elems:
            return _decode_values(attr, self.vr)

        # Zero-length element — no children
        return empty_value_for_VR(self.vr)


# ---------------------------------------------------------------------------
# to_xml helpers
# ---------------------------------------------------------------------------


def _tag_str(tag: BaseTag) -> str:
    """Format a BaseTag as 8-char uppercase hex, no delimiters (per PS3.19 A.1).

    Args:
        tag: pydicom BaseTag.

    Returns:
        8-character uppercase hex string such as ``"00100010"``.
    """
    return f"{tag.group:04X}{tag.element:04X}"


def _encode_values(elem: DataElement, parent: ET.Element) -> None:
    """Encode scalar / string / numeric VR values as ``<Value number="N">`` elements.

    Handles single values, MultiValue sequences, and AT (Attribute Tag) VRs.
    Per PS3.19 A.1.5 Table A.1.5-2, each value in a multi-valued field gets
    its own ``<Value>`` child element numbered from 1.

    Args:
        elem: pydicom DataElement with a non-SQ, non-PN, non-binary VR.
        parent: XML element to append ``<Value>`` children to.
    """
    value = elem.value
    vr = elem.VR

    # Normalise to list for uniform iteration.
    # pydicom stores multi-valued FL/FD as plain Python lists (RT Ion Plan data),
    # while string-based multi-valued VRs (DS, IS, etc.) use MultiValue.
    if isinstance(value, list | MultiValue):
        values: list[Any] = list(value)
    else:
        values = [value]

    for idx, val in enumerate(values, start=1):
        child = ET.SubElement(parent, f"{_NS_PREFIX}Value")
        child.set("number", str(idx))
        if vr == "AT":
            # AT stores a BaseTag; encode as 8-char uppercase hex (matches JSON format)
            bt: BaseTag = val
            child.text = f"{bt.group:04X}{bt.element:04X}"
        else:
            child.text = str(val) if val is not None else ""


def _encode_person_name(elem: DataElement, parent: ET.Element) -> None:
    """Encode a PN value as ``<PersonName>`` with Alphabetic/Ideographic/Phonetic groups.

    Supports multi-valued PN elements (multiple patients — rare but valid).
    Per PS3.19 A.1.5 Table A.1.5-2, each group contains zero or more of
    FamilyName, GivenName, MiddleName, NamePrefix, NameSuffix sub-elements.

    Uses ``PersonName.components`` (same approach as ``DataElement.to_json_dict()``).

    Args:
        elem: pydicom DataElement with ``VR == "PN"``.
        parent: XML element to append ``<PersonName>`` children to.
    """
    value = elem.value
    if isinstance(value, MultiValue):
        pn_list: list[PersonName] = [PersonName(v) for v in value]
    else:
        pn_list = [PersonName(value)]

    for idx, pn in enumerate(pn_list, start=1):
        pn_elem = ET.SubElement(parent, f"{_NS_PREFIX}PersonName")
        pn_elem.set("number", str(idx))

        # components: tuple of component group strings (Alphabetic, Ideographic, Phonetic)
        # Each component is a '^'-delimited string of up to 5 name parts.
        # We use str() on each component, same as DataElement.to_json_dict() line 383.
        raw_components = (
            str(pn.alphabetic) if pn.alphabetic is not None else "",
            str(pn.ideographic) if pn.ideographic is not None else "",
            str(pn.phonetic) if pn.phonetic is not None else "",
        )

        for group_name, raw in zip(_PN_GROUPS, raw_components, strict=True):
            if not raw:
                continue
            parts = raw.split("^")
            # Trim trailing empty parts; non-empty middle parts are preserved
            while parts and parts[-1] == "":
                parts.pop()
            if not parts:
                continue

            group_elem = ET.SubElement(pn_elem, f"{_NS_PREFIX}{group_name}")
            for comp_name, comp_val in zip(_PN_COMPONENTS, parts, strict=False):
                if comp_val:
                    ET.SubElement(group_elem, f"{_NS_PREFIX}{comp_name}").text = comp_val


def _encode_sequence(
    elem: DataElement,
    parent: ET.Element,
    bulk_data_threshold: int,
    bulk_data_element_handler: Callable[[DataElement], str] | None,
) -> None:
    """Encode a SQ element as numbered ``<Item>`` children, each containing DicomAttributes.

    Per PS3.19 A.1.5, sequence items are numbered from 1.

    Args:
        elem: pydicom DataElement with ``VR == "SQ"``.
        parent: XML element to append ``<Item>`` children to.
        bulk_data_threshold: Forwarded to recursive encoding for binary elements.
        bulk_data_element_handler: Forwarded to recursive encoding for bulk data URIs.
    """
    for idx, item in enumerate(elem.value, start=1):
        item_elem = ET.SubElement(parent, f"{_NS_PREFIX}Item")
        item_elem.set("number", str(idx))
        _dataset_to_xml_element(item, item_elem, bulk_data_threshold, bulk_data_element_handler)


def _encode_binary(
    elem: DataElement,
    parent: ET.Element,
    bulk_data_threshold: int,
    bulk_data_element_handler: Callable[[DataElement], str] | None,
) -> None:
    """Encode a binary VR (OB/OD/OF/OL/OV/OW/UN) as InlineBinary or BulkData.

    Mirrors the threshold logic in ``DataElement.to_json_dict()``:
    - If a ``bulk_data_element_handler`` is provided and ``len(value) > (threshold // 4) * 3``
      (accounting for base64 expansion), emit a ``<BulkData>`` element with a ``uri`` attribute.
    - Otherwise emit an ``<InlineBinary>`` element with base64-encoded content.

    Per PS3.19, values are NOT padded to even length (unlike binary DICOM).

    Args:
        elem: pydicom DataElement with a binary VR.
        parent: XML element to append the encoding to.
        bulk_data_threshold: Threshold in bytes for switching to BulkData URI.
            The comparison is ``len(raw_bytes) > (threshold // 4) * 3``, which
            converts the base64-encoded length threshold to the equivalent raw byte
            count (base64 expands by 4/3).  This matches the semantics of
            ``DataElement.to_json_dict()``.
        bulk_data_element_handler: Callable that returns the BulkData URI string.
    """
    raw: bytes = elem.value
    if bulk_data_element_handler is not None and len(raw) > (bulk_data_threshold // 4) * 3:
        uri = bulk_data_element_handler(elem)
        bulk_elem = ET.SubElement(parent, f"{_NS_PREFIX}BulkData")
        bulk_elem.set("uri", uri)
    else:
        child = ET.SubElement(parent, f"{_NS_PREFIX}InlineBinary")
        child.text = base64.b64encode(raw).decode("ascii")


def _dataset_to_xml_element(
    dataset: Dataset,
    parent: ET.Element,
    bulk_data_threshold: int,
    bulk_data_element_handler: Callable[[DataElement], str] | None,
) -> None:
    """Recursively convert a Dataset to DicomAttribute XML children.

    Per PS3.19 A.1:
    - Elements are sorted by tag (ascending).
    - Group Length elements (gggg,0000) are excluded.
    - File Meta Information (group 0002) is excluded.
    - Zero-length elements produce an empty ``<DicomAttribute>`` with no children.

    VR routing mirrors ``DataElement.to_json_dict()``:
    - ``BYTES_VR | AMBIGUOUS_VR`` → InlineBinary or BulkData
    - ``SQ`` → recursive Item elements
    - ``PN`` → PersonName with component group decomposition
    - ``AT`` → Value with 8-char uppercase hex
    - Everything else → Value elements

    Args:
        dataset: pydicom Dataset to serialize.
        parent: XML element to append ``<DicomAttribute>`` children to.
        bulk_data_threshold: Byte threshold for BulkData vs InlineBinary.
        bulk_data_element_handler: Callable that returns a BulkData URI string.
    """
    import pydicom.datadict as dd

    for elem in sorted(dataset, key=lambda e: e.tag):
        # Skip Group Length (gggg,0000) and File Meta Information (group 0002)
        if elem.tag.element == 0x0000 or elem.tag.group == 0x0002:
            continue

        attr_elem = ET.SubElement(parent, f"{_NS_PREFIX}DicomAttribute")
        attr_elem.set("tag", _tag_str(elem.tag))
        attr_elem.set("vr", elem.VR)

        # Populate keyword attribute from pydicom data dictionary when available
        kw = dd.keyword_for_tag(elem.tag)
        if kw:
            attr_elem.set("keyword", kw)

        # privateCreator attribute for private (non-creator) elements
        if elem.tag.is_private and not elem.tag.is_private_creator:
            block = elem.tag.element >> 8
            creator_tag = Tag(elem.tag.group, block)
            if creator_tag in dataset:
                creator_val = dataset[creator_tag].value
                if creator_val:
                    attr_elem.set("privateCreator", str(creator_val))

        # Zero-length → empty element, no children
        if elem.is_empty:
            continue

        vr = elem.VR
        # Route by VR, mirroring DataElement.to_json_dict()
        if vr in (_BINARY_VRS | AMBIGUOUS_VR) - {VR.US_SS}:
            _encode_binary(elem, attr_elem, bulk_data_threshold, bulk_data_element_handler)
        elif vr == VR.SQ:
            _encode_sequence(elem, attr_elem, bulk_data_threshold, bulk_data_element_handler)
        elif vr == VR.PN:
            _encode_person_name(elem, attr_elem)
        else:
            _encode_values(elem, attr_elem)


# ---------------------------------------------------------------------------
# from_xml helpers
# ---------------------------------------------------------------------------


def _parse_pn_group(group_elem: ET.Element) -> str:
    """Parse one PersonName component group into a ``^``-delimited string.

    Per PS3.19 A.1.5, the group element may contain any subset of:
    FamilyName, GivenName, MiddleName, NamePrefix, NameSuffix.

    Args:
        group_elem: An ``<Alphabetic>``, ``<Ideographic>``, or ``<Phonetic>`` XML element.

    Returns:
        Caret-delimited component string, e.g. ``"Smith^John^^Dr.^Jr."``.
    """
    parts = ["", "", "", "", ""]
    for i, comp_name in enumerate(_PN_COMPONENTS):
        child = group_elem.find(f"{_NS_PREFIX}{comp_name}")
        if child is not None and child.text:
            parts[i] = child.text
    # Trim trailing empty parts
    while parts and parts[-1] == "":
        parts.pop()
    return "^".join(parts)


def _decode_person_name(attr_elem: ET.Element) -> PersonName | MultiValue[PersonName]:
    """Decode all ``<PersonName>`` children of a DicomAttribute into a PersonName value.

    Args:
        attr_elem: A ``<DicomAttribute vr="PN">`` element.

    Returns:
        A single PersonName if only one is present, or a MultiValue for multi-valued PN.
    """
    pn_elems = attr_elem.findall(f"{_NS_PREFIX}PersonName")
    results: list[PersonName] = []
    for pn_elem in pn_elems:
        group_strings: list[str] = []
        for group_name in _PN_GROUPS:
            group_child = pn_elem.find(f"{_NS_PREFIX}{group_name}")
            if group_child is not None:
                group_strings.append(_parse_pn_group(group_child))
            else:
                group_strings.append("")
        # Trim trailing empty groups
        while group_strings and group_strings[-1] == "":
            group_strings.pop()
        results.append(PersonName("=".join(group_strings)))

    if len(results) == 1:
        return results[0]
    return MultiValue(PersonName, results)


def _coerce_value(text: str, vr: str) -> Any:
    """Coerce an XML Value text string to the appropriate Python type for a VR.

    Uses ``convert_to_python_number`` from ``pydicom.jsonrep`` for numeric coercion,
    matching the JSON implementation's behaviour exactly.

    Per PS3.19 A.1.5 (Table A.1.5-2):

    - US, SS, UL, SL, SV, UV → int
    - FL, FD → float
    - DS → pydicom DS (preserves string representation)
    - IS → pydicom IS
    - AT → Tag (parsed from 8-char hex)
    - All other string VRs → str

    Args:
        text: String content of a ``<Value>`` element.
        vr: Two-letter VR code.

    Returns:
        Python value appropriate for the VR.

    Raises:
        DicomXmlAtValueLengthError: If AT value text is not exactly 8 characters.
        DicomXmlAtValueHexError: If AT value text contains non-hex characters.
    """
    from pydicom.dataelem import empty_value_for_VR
    from pydicom.valuerep import DS, IS

    if not text or not text.strip():
        return empty_value_for_VR(vr)
    # DS and IS must be checked before generic FLOAT_VR/INT_VR because pydicom's
    # FLOAT_VR includes DS and INT_VR includes IS.  Checking first preserves the
    # pydicom wrapper type (DS/IS) which carries the original string representation.
    if vr == "DS":
        return DS(text)
    if vr == "IS":
        return IS(text)
    if vr in (INT_VR - {VR.AT}) | {VR.US_SS}:
        return int(text)
    if vr in FLOAT_VR:
        return float(text)
    if vr == "AT":
        if len(text) != 8:
            raise DicomXmlAtValueLengthError(text)
        try:
            return Tag(int(text[:4], 16), int(text[4:], 16))
        except ValueError as exc:
            raise DicomXmlAtValueHexError(text) from exc
    return text


def _decode_values(attr_elem: ET.Element, vr: str) -> Any:
    """Decode ``<Value>`` children into a single value or MultiValue list.

    Args:
        attr_elem: A ``<DicomAttribute>`` XML element.
        vr: Two-letter VR code.

    Returns:
        Single Python value if VM==1, or MultiValue list if VM>1.
    """
    value_elems = attr_elem.findall(f"{_NS_PREFIX}Value")
    if not value_elems:
        return None

    coerced: list[Any] = [_coerce_value(ve.text or "", vr) for ve in value_elems]

    if len(coerced) == 1:
        return coerced[0]
    return MultiValue(type(coerced[0]), coerced)


def _parse_tag_and_vr(attr_elem: ET.Element) -> tuple[BaseTag, str]:
    """Parse the ``tag`` and ``vr`` attributes from a ``<DicomAttribute>`` element.

    Extracted from ``dataset_from_xml`` and ``_element_to_dataset`` to avoid
    duplicating the same tag-parsing logic in both functions.

    Args:
        attr_elem: A ``<DicomAttribute>`` XML element.

    Returns:
        Tuple of ``(tag, vr)`` where ``vr`` is resolved from the data dictionary
        when the attribute is absent, and falls back to ``"UN"`` for unknown tags.

    Raises:
        DicomXmlTagLengthError: If the ``tag`` attribute is not exactly 8 characters.
        DicomXmlTagHexError: If the ``tag`` attribute contains non-hex characters.
    """
    import pydicom.datadict as dd

    tag_str = attr_elem.get("tag", "")
    if len(tag_str) != 8:
        raise DicomXmlTagLengthError(tag_str)
    try:
        tag = Tag(int(tag_str[:4], 16), int(tag_str[4:], 16))
    except ValueError as exc:
        raise DicomXmlTagHexError(tag_str) from exc

    vr: str = attr_elem.get("vr", "")
    if not vr:
        try:
            entry = dd.get_entry(tag)
            vr = entry[0]
        except KeyError:
            vr = "UN"

    return tag, vr


def _element_to_dataset(
    dataset_class: type[Dataset],
    parent: ET.Element,
    bulk_data_element_handler: BulkDataHandlerType = None,
) -> Dataset:
    """Recursively convert ``<DicomAttribute>`` XML children to a pydicom Dataset.

    Per PS3.19 A.1, each ``<DicomAttribute>`` has a ``tag`` and ``vr`` attribute.
    If ``vr`` is absent the data dictionary is consulted.

    Args:
        dataset_class: The Dataset subclass to instantiate for each item.
        parent: A ``<NativeDicomModel>`` or ``<Item>`` XML element.
        bulk_data_element_handler: Optional handler forwarded to converters so
            BulkData elements inside sequence items can be resolved.

    Returns:
        pydicom Dataset populated from the XML children.

    Raises:
        DicomXmlTagLengthError: If a tag attribute string is not exactly 8 characters.
        DicomXmlTagHexError: If a tag attribute string contains non-hex characters.
    """
    ds = dataset_class()

    for attr_elem in parent.findall(f"{_NS_PREFIX}DicomAttribute"):
        tag, vr = _parse_tag_and_vr(attr_elem)
        converter = XmlDataElementConverter(dataset_class, attr_elem, bulk_data_element_handler)
        value = converter.get_element_values()
        ds.add(DataElement(tag, vr, value))

    return ds


# ---------------------------------------------------------------------------
# Public API — dataset_to_xml / dataset_from_xml
# ---------------------------------------------------------------------------


def data_element_to_xml_element(
    data_element: DataElement,
    parent: ET.Element,
    bulk_data_threshold: int = 1024,
    bulk_data_element_handler: Callable[[DataElement], str] | None = None,
    private_creator: str | None = None,
) -> ET.Element:
    """Convert a DataElement to an XML ``<DicomAttribute>`` element appended to ``parent``.

    Mirrors ``DataElement.to_json_dict()`` behaviour.  VR routing:

    - ``BYTES_VR | AMBIGUOUS_VR`` → ``<InlineBinary>`` or ``<BulkData uri="...">``
    - ``SQ`` → recursive ``<Item>`` elements
    - ``PN`` → ``<PersonName>`` with Alphabetic/Ideographic/Phonetic groups
    - ``AT`` → ``<Value>`` with 8-char uppercase hex
    - Everything else → ``<Value number="N">`` elements

    Args:
        data_element: pydicom DataElement to convert.
        parent: XML element to append the ``<DicomAttribute>`` to.
        bulk_data_threshold: Byte threshold above which binary values are provided as
            BulkData URI rather than InlineBinary.  Ignored when no handler is given.
        bulk_data_element_handler: Callable accepting a DataElement and returning the
            ``BulkDataURI`` string.  Mirrors ``DataElement.to_json_dict()`` parameter.
        private_creator: Optional private creator string.  When provided and the element
            is a private non-creator element, the ``privateCreator`` attribute is set.
            Use this when calling the function outside a Dataset context where the creator
            cannot be looked up automatically.

    Returns:
        The newly created ``<DicomAttribute>`` XML element.
    """
    import pydicom.datadict as dd

    attr_elem = ET.SubElement(parent, f"{_NS_PREFIX}DicomAttribute")
    attr_elem.set("tag", _tag_str(data_element.tag))
    attr_elem.set("vr", data_element.VR)

    kw = dd.keyword_for_tag(data_element.tag)
    if kw:
        attr_elem.set("keyword", kw)

    if private_creator and data_element.tag.is_private and not data_element.tag.is_private_creator:
        attr_elem.set("privateCreator", private_creator)

    if not data_element.is_empty:
        vr = data_element.VR
        if vr in (_BINARY_VRS | AMBIGUOUS_VR) - {VR.US_SS}:
            _encode_binary(data_element, attr_elem, bulk_data_threshold, bulk_data_element_handler)
        elif vr == VR.SQ:
            _encode_sequence(data_element, attr_elem, bulk_data_threshold, bulk_data_element_handler)
        elif vr == VR.PN:
            _encode_person_name(data_element, attr_elem)
        else:
            _encode_values(data_element, attr_elem)

    return attr_elem


def dataset_to_xml(
    dataset: Dataset,
    bulk_data_threshold: int = 1024,
    bulk_data_element_handler: Callable[[DataElement], str] | None = None,
) -> bytes:
    """Convert a Dataset to DICOM XML bytes (Native DICOM Model, PS3.19 Annex A).

    Parameters mirror ``Dataset.to_json()`` for consistency.

    The output XML:

    - Has an XML declaration (UTF-8 encoding).
    - Uses the namespace ``http://dicom.nema.org/PS3.19/models/NativeDICOM``.
    - Carries ``xml:space="preserve"`` on the root element.
    - Orders elements by ascending tag.
    - Omits Group Length (gggg,0000) and File Meta Information (group 0002).
    - Encodes binary VRs as base64 ``<InlineBinary>`` or ``<BulkData uri="...">``
      depending on ``bulk_data_threshold`` and ``bulk_data_element_handler``.

    Args:
        dataset: pydicom Dataset to convert.
        bulk_data_threshold: Size of base64-encoded data element (in bytes) above which
            a value will be provided as a BulkData URI rather than InlineBinary.
            Ignored when no ``bulk_data_element_handler`` is given.
        bulk_data_element_handler: Callable that accepts a bulk DataElement and returns
            the ``BulkDataURI`` as a string.  Mirrors ``Dataset.to_json()`` parameter.

    Returns:
        UTF-8 encoded XML bytes with XML declaration.
    """
    # Register namespace here rather than at module import time to limit the
    # process-global side effect to callers that actually serialise XML.
    ET.register_namespace("", NAMESPACE)
    root = ET.Element(f"{_NS_PREFIX}NativeDicomModel")
    root.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    _dataset_to_xml_element(dataset, root, bulk_data_threshold, bulk_data_element_handler)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def dataset_from_xml(
    xml_data: bytes | str,
    dataset_class: type[Dataset] = Dataset,
    bulk_data_uri_handler: BulkDataHandlerType | Callable[[str], BulkDataType] | None = None,
) -> Dataset:
    """Convert DICOM XML (Native DICOM Model, PS3.19 Annex A) to a Dataset.

    Parameters mirror ``Dataset.from_json()`` for consistency.

    Supported elements:

    - ``DicomAttribute`` with all VRs from Table A.1.5-2.
    - ``PersonName`` with Alphabetic, Ideographic, and Phonetic component groups.
    - Sequences (SQ) with arbitrary nesting depth.
    - ``InlineBinary`` (base64-encoded binary data).
    - ``BulkData`` (URI resolved via ``bulk_data_uri_handler`` or stored as empty value).
    - Zero-length (empty) DicomAttribute elements.

    Args:
        xml_data: XML bytes or string containing a NativeDicomModel document.
        dataset_class: The Dataset subclass to use when constructing items.
            Mirrors ``Dataset.from_json()`` parameter.
        bulk_data_uri_handler: Callable that accepts ``(tag, vr, uri)`` or just ``(uri)``
            and returns the resolved bulk data value.  When ``None`` (default) a BulkData
            element produces an empty value for its VR.

    Returns:
        pydicom Dataset populated from the XML.

    Raises:
        DicomXmlParseError: If the input cannot be parsed as XML.
        DicomXmlRootError: If the XML root element is not NativeDicomModel in the
            correct namespace.
        DicomXmlTagLengthError: If any DicomAttribute has a malformed tag attribute.
        DicomXmlTagHexError: If any DicomAttribute has a non-hex tag attribute.
        DicomXmlAtValueLengthError: If an AT Value element is not exactly 8 hex chars.
        DicomXmlAtValueHexError: If an AT Value element contains non-hex characters.
    """
    if isinstance(xml_data, str):
        xml_bytes = xml_data.encode("utf-8")
    else:
        xml_bytes = xml_data

    try:
        root = ET.fromstring(xml_bytes.decode("utf-8"))
    except ET.ParseError as exc:
        raise DicomXmlParseError(str(exc)) from exc

    if root.tag != f"{_NS_PREFIX}NativeDicomModel":
        raise DicomXmlRootError(root.tag)

    # Normalise bulk_data_uri_handler to 3-arg form for XmlDataElementConverter
    handler: BulkDataHandlerType
    if bulk_data_uri_handler is not None and len(signature(bulk_data_uri_handler).parameters) == 1:
        _h: Callable[[str], BulkDataType] = bulk_data_uri_handler  # type: ignore[assignment]

        def _wrapper(_tag: str, _vr: str, value: str) -> BulkDataType:
            return _h(value)

        handler = _wrapper
    else:
        handler = bulk_data_uri_handler  # type: ignore[assignment]

    # Use a dataset_class-aware converter for the top-level dataset
    ds = dataset_class()
    for attr_elem in root.findall(f"{_NS_PREFIX}DicomAttribute"):
        tag, vr = _parse_tag_and_vr(attr_elem)
        converter = XmlDataElementConverter(dataset_class, attr_elem, handler)
        value = converter.get_element_values()
        ds.add(DataElement(tag, vr, value))

    return ds


# ---------------------------------------------------------------------------
# Multipart XML utilities for DICOMWeb (PS3.18)
# ---------------------------------------------------------------------------

# RFC 2046 Section 5.1.1: boundary characters that do NOT require quoting.
# bcharsnospace := DIGIT / ALPHA / "'" / "(" / ")" / "+" / "_" / "," /
#                  "-" / "." / "/" / ":" / "=" / "?"
# bchars := bcharsnospace / " "
# A boundary value that contains ONLY these characters (excluding trailing
# spaces) does not need to be quoted in the Content-Type header.
_BCHARS_NO_SPACE = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'()+_,-./:=?")
_BCHARS = _BCHARS_NO_SPACE | {" "}

# Maximum boundary length per RFC 2046 is 70 characters.
_MAX_BOUNDARY_LENGTH = 70

# RFC 2045 Section 5.1: tspecials that require a parameter value to be quoted.
# tspecials := "(" / ")" / "<" / ">" / "@" / "," / ";" / ":" / "\" /
#              <"> / "/" / "[" / "]" / "?" / "=" / SPACE
_TSPECIALS = frozenset('()<>@,;:\\"/[]?= ')

# Acceptable MIME media types for parts in a DICOMWeb multipart/related response.
# application/dicom+xml is the canonical type (PS3.18).  text/xml and
# application/xml are accepted for interoperability with conformant but
# loosely-typed servers.
_ACCEPTABLE_XML_MEDIA_TYPES = frozenset(
    {
        "application/dicom+xml",
        "text/xml",
        "application/xml",
    }
)


def _validate_boundary(boundary: str) -> None:
    """Validate that a boundary string conforms to RFC 2046 Section 5.1.1.

    Args:
        boundary: The MIME boundary string to validate.

    Raises:
        ValueError: If the boundary is empty, too long, contains invalid
            characters, or ends with a space.

    """
    if not boundary:
        msg = "Boundary must not be empty"
        raise ValueError(msg)
    if len(boundary) > _MAX_BOUNDARY_LENGTH:
        msg = f"Boundary exceeds maximum length of {_MAX_BOUNDARY_LENGTH}: {len(boundary)}"
        raise ValueError(msg)
    invalid_chars = set(boundary) - _BCHARS
    if invalid_chars:
        char_reprs = ", ".join(repr(c) for c in sorted(invalid_chars))
        msg = f"Boundary contains invalid characters: {char_reprs}"
        raise ValueError(msg)
    if boundary.endswith(" "):
        msg = "Boundary must not end with a space (RFC 2046 Section 5.1.1)"
        raise ValueError(msg)


def _quote_boundary(boundary: str) -> str:
    """Quote a boundary for use in a Content-Type header if needed.

    Per RFC 2045 Section 5.1, parameter values containing characters outside
    the token character set must be quoted.  We always quote if the boundary
    contains spaces (which are legal bchars but not legal token characters).

    """
    if _TSPECIALS & set(boundary):
        return f'"{boundary}"'
    return boundary


def _generate_boundary() -> str:
    """Generate a unique MIME boundary string."""
    return f"pydicom-xml-{uuid.uuid4().hex}"


def datasets_to_multipart_xml(
    datasets: list[Dataset],
    boundary: str | None = None,
    bulk_data_threshold: int = 1024,
    bulk_data_element_handler: Callable[[DataElement], str] | None = None,
) -> tuple[bytes, str]:
    """Serialize multiple Datasets as ``multipart/related`` with XML parts.

    Per PS3.18, DICOMWeb search results (e.g., QIDO-RS) with
    ``Accept: application/dicom+xml`` return a ``multipart/related``
    response where each part is a standalone NativeDicomModel XML document.

    Args:
        datasets: List of pydicom Datasets to serialize.
        boundary: MIME boundary string.  Auto-generated if not provided.
        bulk_data_threshold: Passed through to ``dataset_to_xml``.
        bulk_data_element_handler: Passed through to ``dataset_to_xml``.

    Returns:
        A tuple of ``(body_bytes, content_type_header)``.
        The content_type_header includes the boundary and type parameters.

    """
    if boundary is None:
        boundary = _generate_boundary()
    else:
        _validate_boundary(boundary)

    quoted = _quote_boundary(boundary)

    if not datasets:
        content_type = f'multipart/related; type="application/dicom+xml"; boundary={quoted}'
        return f"--{boundary}--\r\n".encode(), content_type

    parts: list[bytes] = []
    for ds in datasets:
        xml_bytes = dataset_to_xml(
            ds,
            bulk_data_threshold=bulk_data_threshold,
            bulk_data_element_handler=bulk_data_element_handler,
        )
        part_header = (f"--{boundary}\r\nContent-Type: application/dicom+xml\r\n\r\n").encode()
        parts.append(part_header + xml_bytes + b"\r\n")

    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    content_type = f'multipart/related; type="application/dicom+xml"; boundary={quoted}'
    return body, content_type


def _check_part_content_type(header_block: bytes) -> None:
    """Validate that a multipart part has an acceptable Content-Type.

    Acceptable types are ``application/dicom+xml`` or any ``text/xml`` /
    ``application/xml`` variant.  If the Content-Type header is absent,
    the part is accepted (permissive fallback for interoperability).

    Args:
        header_block: The raw header bytes from the multipart part.

    Raises:
        ValueError: If the Content-Type is present but not XML-compatible.

    """
    if not header_block:
        return

    # Normalize line endings and unfold continuation lines (RFC 2822 §2.2.3:
    # a line starting with whitespace is a continuation of the previous line).
    normalized = header_block.replace(b"\r\n", b"\n")
    # Join continuation lines: replace "\n<SP/TAB>" with a single space
    unfolded = b""
    for i, line in enumerate(normalized.split(b"\n")):
        if i > 0 and line and line[0:1] in (b" ", b"\t"):
            # Continuation line — append to previous
            unfolded += b" " + line.strip()
        else:
            if i > 0:
                unfolded += b"\n"
            unfolded += line

    content_type_value: str | None = None
    for line in unfolded.split(b"\n"):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        # Match exactly "content-type:" (case-insensitive) to avoid
        # false matches on headers that merely start with the same prefix.
        if stripped_line.lower().startswith(b"content-type:"):
            _, _, ct_value = stripped_line.partition(b":")
            content_type_value = ct_value.strip().decode("ascii", errors="replace")
            break

    if content_type_value is None:
        # No Content-Type header — permissive: accept and attempt parse
        return

    # Normalize: take only the media type (ignore parameters like charset)
    media_type = content_type_value.split(";")[0].strip().lower()

    if media_type not in _ACCEPTABLE_XML_MEDIA_TYPES:
        msg = (
            f"Unexpected Content-Type in multipart part: '{content_type_value}'. "
            f"Expected one of: {sorted(_ACCEPTABLE_XML_MEDIA_TYPES)}"
        )
        raise ValueError(msg)


def datasets_from_multipart_xml(
    body: bytes,
    boundary: str,
    dataset_class: type[Dataset] = Dataset,
    bulk_data_uri_handler: BulkDataHandlerType | Callable[[str], BulkDataType] | None = None,
) -> list[Dataset]:
    """Parse a ``multipart/related`` XML response into a list of Datasets.

    Per PS3.18, DICOMWeb search results with ``application/dicom+xml``
    use ``multipart/related`` framing with one NativeDicomModel per part.

    Args:
        body: The raw multipart response body bytes.
        boundary: The MIME boundary string (from the Content-Type header).
        dataset_class: Dataset class to use for deserialization.
        bulk_data_uri_handler: Handler for BulkData URI references.

    Returns:
        List of pydicom Datasets, one per multipart part.

    """
    boundary_bytes = f"--{boundary}".encode()

    # Split on boundary markers
    raw_parts = body.split(boundary_bytes)

    datasets_out: list[Dataset] = []
    for raw_part in raw_parts:
        stripped = raw_part.strip()
        # Skip empty parts (before first boundary) and closing marker
        if not stripped or stripped == b"--" or stripped.startswith(b"--"):
            continue

        # Split headers from body on double CRLF
        if b"\r\n\r\n" in stripped:
            header_block, xml_body = stripped.split(b"\r\n\r\n", 1)
            xml_body = xml_body.strip()
        elif b"\n\n" in stripped:
            header_block, xml_body = stripped.split(b"\n\n", 1)
            xml_body = xml_body.strip()
        else:
            # No headers present — treat entire content as XML body
            header_block = b""
            xml_body = stripped

        if not xml_body:
            continue

        # Validate per-part Content-Type
        _check_part_content_type(header_block)

        result = dataset_from_xml(
            xml_body,
            dataset_class=dataset_class,
            bulk_data_uri_handler=bulk_data_uri_handler,
        )
        datasets_out.append(result)

    return datasets_out


def extract_boundary(content_type: str) -> str:
    """Extract the boundary parameter from a multipart Content-Type header.

    Args:
        content_type: The Content-Type header value,
            e.g. ``'multipart/related; type="application/dicom+xml"; boundary=my-boundary'``

    Returns:
        The boundary string.

    Raises:
        ValueError: If no boundary parameter is found.

    """
    for param in content_type.split(";"):
        param = param.strip()
        if "=" in param:
            key, _, value = param.partition("=")
            if key.strip().lower() == "boundary":
                return value.strip().strip('"')
    msg = f"No boundary parameter in Content-Type: {content_type}"
    raise ValueError(msg)
