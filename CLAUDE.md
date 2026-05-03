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
