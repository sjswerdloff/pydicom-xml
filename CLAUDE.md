# Insights — pydicom-xml

## pydicom VR classification surprises

`INT_VR` in pydicom includes `IS` (Integer String) and `FLOAT_VR` includes `DS` (Decimal String).
This means `_coerce_value("42", "IS")` returns `int`, not `pydicom.valuerep.IS`, and similarly
DS returns `float` not `pydicom.valuerep.DS`. When writing tests, test for the actual numeric
type rather than the string-wrapper type unless you are explicitly testing string preservation
semantics.

## empty_value_for_VR("OW") returns None in pydicom 3.x

`pydicom.dataelem.empty_value_for_VR("OW")` (and most binary VRs) returns `None` in pydicom 3.x,
not `b""`. Test assertions that expect `b""` for empty binary VRs will fail.

## `_BINARY_VRS` type annotation

`BYTES_VR` from `pydicom.valuerep` is typed as `set[VR]`, not `frozenset[str]`. Using
`frozenset[str]` as the annotation causes a mypy incompatible-types error. The correct annotation
is `set[VR]`.

## ruff PLC0415 (import-outside-toplevel)

When using local imports inside functions to avoid circular dependencies (same pattern as pydicom's
own code), the cleanest approach is to add `PLC` to the ruff `select` ruleset and then globally
`ignore` `PLC0415`. This avoids RUF100 "unused noqa directive" errors that arise from adding
`# noqa: PLC0415` when the rule isn't enabled.

## ruff S314 globally ignored vs per-file noqa

If `S314` (suspicious-xml-element-tree-usage) is already in the global `ignore` list, adding
`# noqa: S314` comments produces RUF100 violations. Either remove the global ignore and use
per-call noqa, or keep the global ignore and remove all inline noqa directives for that rule.

## zip() strict= parameter (B905)

`zip(_PN_GROUPS, raw_components)` where both iterables are the same fixed length (3 elements)
is safe with `strict=True`. For `zip(_PN_COMPONENTS, parts)` where `parts` can be shorter
(up to 5 elements), `strict=False` is the correct explicit choice.

## uv + hatchling package discovery

For a `src/` layout project, `uv init` generates a pyproject.toml without build system config.
Running `uv pip install -e .` requires hatchling (or another build backend) to be specified.
Adding `[build-system]` and `[tool.hatch.build.targets.wheel]` sections is needed for editable
installs to work. Alternatively, `uv sync` with the package as a workspace member handles this
automatically.

## pydicom DS is a factory function, not a class

`pydicom.valuerep.DS` is a factory function that returns either `DSfloat` or `DSdecimal`
(depending on `pydicom.config.use_DS_decimal`). `isinstance(x, DS)` always returns False.
Use `isinstance(x, DSfloat)` or `isinstance(x, (DSfloat, DSdecimal))` for type checks.
`DSfloat` is a subclass of `float`, so `isinstance(DSfloat("1.0"), float)` is True.

## _decode_person_name return type requires MultiValue not list

The `_encode_person_name` function checks `isinstance(value, MultiValue)` to detect multi-valued
PN fields. Returning a plain `list` from `_decode_person_name` breaks this check on the next
serialization. Always return `MultiValue(PersonName, results)` for multiple PN values.

## BulkData handler must be forwarded through SQ recursion

`XmlDataElementConverter.get_element_values()` recurses into SQ items via `_element_to_dataset`.
The `bulk_data_element_handler` must be explicitly passed to `_element_to_dataset` so that BulkData
elements inside sequence items can be resolved. Without this, BulkData inside SQ silently returns
`None` (the result of `empty_value_for_VR`).

## ET.register_namespace is process-global

`ET.register_namespace("", NAMESPACE)` affects every XML serialization in the process for that
namespace. Moving it from module import time into `dataset_to_xml()` limits the side effect to
callers that actually serialize — tests that only call `dataset_from_xml` are not affected.
