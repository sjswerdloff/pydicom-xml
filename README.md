# pydicom-xml

DICOM XML (Native DICOM Model) serialization for [pydicom](https://github.com/pydicom/pydicom).

Converts pydicom Datasets to and from the XML format defined in [DICOM PS3.19 Annex A](https://dicom.nema.org/medical/dicom/current/output/chtml/part19/chapter_A.html), using the `application/dicom+xml` media type required by [DICOMweb](https://www.dicomstandard.org/using/dicomweb).

## Why

DICOMweb servers are **required** to support both `application/dicom+json` and `application/dicom+xml` ([PS3.18 Table 8.7.3-3](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/sect_8.7.3.html)). pydicom provides `Dataset.to_json()` / `Dataset.from_json()` but has no XML equivalent. This package fills that gap.

## Install

```bash
pip install pydicom-xml
```

Requires Python 3.10+ and pydicom 3.0+.

## Usage

```python
import pydicom
from pydicom_xml import to_xml, from_xml

# Read a DICOM file
ds = pydicom.dcmread("example.dcm")

# Convert to XML bytes
xml_bytes = to_xml(ds)

# Convert back to Dataset
ds_roundtrip = from_xml(xml_bytes)
```

### Bulk data handling

For large binary elements (pixel data, LUTs), use bulk data references instead of inline base64:

```python
def my_bulk_handler(data_element):
    # Store the data and return a URI
    uri = store_bulk_data(data_element.value)
    return uri

xml_bytes = to_xml(ds, bulk_data_element_handler=my_bulk_handler)
```

When reading XML with BulkData URIs:

```python
def my_uri_handler(tag, vr, uri):
    # Fetch data from the URI
    return fetch_data(uri)

ds = from_xml(xml_bytes, bulk_data_uri_handler=my_uri_handler)
```

These callback signatures match pydicom's existing `Dataset.to_json()` / `Dataset.from_json()` parameters.

## API

### `to_xml(dataset, bulk_data_threshold=1024, bulk_data_element_handler=None) -> bytes`

Convert a pydicom Dataset to DICOM XML (Native DICOM Model, PS3.19 Annex A).

- **dataset** — pydicom Dataset
- **bulk_data_threshold** — byte size above which binary elements use BulkData references (default 1024). Ignored if no handler is given.
- **bulk_data_element_handler** — callable accepting a DataElement, returns a BulkData URI string

Returns UTF-8 encoded XML with XML declaration.

### `from_xml(xml_data, dataset_class=Dataset, bulk_data_uri_handler=None) -> Dataset`

Convert DICOM XML to a pydicom Dataset.

- **xml_data** — XML as `bytes` or `str`
- **dataset_class** — Dataset class to use (default `pydicom.dataset.Dataset`)
- **bulk_data_uri_handler** — callable accepting `(tag, vr, uri)` or `(uri,)`, returns the element value

## What it handles

All DICOM Value Representations are supported per PS3.19 Table A.1.5-2:

| VR Category | XML Encoding |
|---|---|
| String/numeric (AE, CS, DA, DS, FL, FD, IS, LO, SH, etc.) | `<Value number="N">text</Value>` |
| Person Name (PN) | `<PersonName>` with Alphabetic/Ideographic/Phonetic groups |
| Sequence (SQ) | `<Item number="N">` with recursive DicomAttribute elements |
| Binary (OB, OD, OF, OL, OV, OW, UN) | `<InlineBinary>` (base64) or `<BulkData uri="..."/>` |
| Attribute Tag (AT) | `<Value>` with 8-char uppercase hex |

Additional behaviors:
- File Meta Information (group 0002) and Group Length elements are excluded
- Elements are ordered by tag (ascending)
- `keyword` attribute is populated from the DICOM data dictionary
- `privateCreator` attribute is set for private elements
- Multi-valued FL/FD stored as Python lists (common in RT Ion Plan data) are handled correctly
- Empty/zero-length elements produce empty `<DicomAttribute>` elements

## Architecture

Designed to mirror pydicom's existing JSON architecture for potential upstream contribution:

| JSON (pydicom) | XML (this package) |
|---|---|
| `jsonrep.py` | `xmlrep.py` |
| `JsonDataElementConverter` | `XmlDataElementConverter` |
| `Dataset.to_json()` | `to_xml()` / `dataset_to_xml()` |
| `Dataset.from_json()` | `from_xml()` / `dataset_from_xml()` |
| `DataElement.to_json_dict()` | `data_element_to_xml_element()` |
| `bulk_data_uri_handler` | Same callback pattern |
| `bulk_data_element_handler` | Same callback pattern |

## Standards references

- [PS3.19 Annex A.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part19/chapter_A.html) — Native DICOM Model definition
- [PS3.19 Annex A.1.6](https://dicom.nema.org/medical/dicom/current/output/chtml/part19/sect_A.1.6.html) — RELAX NG Compact normative schema
- [PS3.18 Annex F](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/chapter_F.html) — DICOM JSON Model (counterpart)
- [PS3.18 Section F.3.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part18/sect_F.3.html) — XML-to-JSON transformation rules

## License

MIT — see [LICENSE](LICENSE).
