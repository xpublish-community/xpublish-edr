# Compliance testing

The OGC integration in xpublish-edr is developed against the official OGC
artifacts through three layers of testing:

1. **Schema validation** — `tests/test_ogc_core_integration.py` validates the
   composed xpublish-ogc-core + xpublish-edr responses against the official
   EDR 1.1 bundled OpenAPI schemas vendored by xpublish-ogc-core
   (`xpublish_ogc_core.testing.validate_response`).
2. **Schemathesis fuzzing** — `tests/test_ogc_core_schemathesis.py` generates
   requests from the composed app's own OpenAPI description and validates
   the responses against it.
3. **OGC CITE executable test suite** — `tests/test_teamengine.py` serves the
   composed app over HTTP and runs the official `ets-ogcapi-edr10` suite in
   the `ogccite` Docker image (marked `cite`; requires Docker; deselect with
   `-m "not cite"`).

## What the tests found

Each of these was caught by a test layer and fixed:

| Found by | Error | Change |
| -------- | ----- | ------ |
| `validate_response("collection", ...)` against the official `extent` schema | `temporal.interval` and `vertical.interval` were flat lists, but the spec requires lists of begin/end pairs; vertical values must be strings | `TemporalExtent` / `VerticalExtent` in `xpublish_edr/metadata.py` |
| `validate_response("collection", ...)` against `cubeDataQuery` | Cube data queries were missing the required `height_units` member | `height_units` added to `VariablesMetadata` and `cube_query_description()`, derived from the dataset's vertical axis units |
| Schemathesis `status_code_conformance` | The OGC routes returned undocumented 404s | `responses={404: ...}` declared on the OGC position/area/cube routes |
| Schemathesis `positive_data_acceptance` | WKT `coords` and comma-separated `bbox` parameters are looser than their OpenAPI parameter schemas can express, so schema-valid fuzzed requests are rejected with 422 | Check excluded in the fuzz test with a comment (not expressible without misdocumenting the API) |
| CITE `Conformance.validateConformanceOperationAndResponse` | Only EDR **1.1** conformance URIs were declared, but the suite (and EDR 1.0 clients) check the 1.0 URIs | Both 1.0 and 1.1 class URIs declared (EDR 1.1 is backwards compatible) |
| CITE `GeoJSONEncoding.validateResponseForGeoJSON` (EDR 1.0 Abstract Test 21) | Declaring the `geojson` conformance class implies Locations resources, which are not implemented (even though `f=geojson` is a supported output format) | `geojson` removed from the declared conformance classes, with a comment |
| CITE `CollectionsResponse.verifyCollectionsMetadata` (EDR 1.0 Abstract Test 15) | Collections had no `data`/`collection` rel links | Fixed in xpublish-ogc-core's `build_collection()` |

## Known failures

Documented in `tests/test_teamengine.py::KNOWN_FAILURES` and asserted not to
drift in either direction:

- `ApiDefinition.apiDefinitionValidation` — FastAPI generates an OpenAPI 3.1
  document; the suite validates it as OpenAPI 3.0 and rejects the
  `"type": "null"` members pydantic emits for optional fields.
- `*.{area,position,cube}DateTimeParameterDefinition` — the suite requires an
  explicit `style: form` on the `datetime` query parameter definition, which
  FastAPI does not emit (it is the OpenAPI default for query parameters).

Suite status as of the last run: **28 passed, 4 known failures, 8 skipped**
(the skips cover unimplemented query types: radius, trajectory, corridor,
locations, and the GeoJSON/EDR-GeoJSON encodings).
