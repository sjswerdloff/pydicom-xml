"""pydicom-xml — DICOM XML (Native DICOM Model, PS3.19 Annex A) serialization.

Public API::

    from pydicom_xml import to_xml, from_xml, XmlDataElementConverter, NAMESPACE

Convenience aliases ``to_xml`` and ``from_xml`` match pydicom's Dataset method naming
conventions (``Dataset.to_json`` / ``Dataset.from_json``).
"""

from pydicom_xml.xmlrep import (
    NAMESPACE,
    XmlDataElementConverter,
    data_element_to_xml_element,
    dataset_from_xml,
    dataset_to_xml,
)

# Convenience aliases matching pydicom's Dataset method naming
to_xml = dataset_to_xml
from_xml = dataset_from_xml

__all__ = [
    "NAMESPACE",
    "XmlDataElementConverter",
    "data_element_to_xml_element",
    "dataset_from_xml",
    "dataset_to_xml",
    "from_xml",
    "to_xml",
]
