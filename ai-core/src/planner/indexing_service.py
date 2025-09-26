"""
Unit tests for the indexing service merge error fixes.

Covers type validation + normalization for array sections to prevent:
- `'dict' object has no attribute 'append'`
- `'str' object has no attribute 'get'`

Pytest-style function tests with a __main__ that runs pytest.
"""

import logging
import pytest
import unittest

from docai.services.indexing_service import IndexingService


@pytest.fixture
def indexing_service():
    return IndexingService()

def test_parties_dict_then_list_merges_cleanly(indexing_service):
    existing = {"parties": {"name": "JOHN DOE", "role": "GRANTOR"}}
    new = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}

    merged = indexing_service._merge_results(existing, new)

    assert "parties" in merged
    assert isinstance(merged["parties"], list)
    assert {p["name"] for p in merged["parties"]} == {"JOHN DOE", "JANE DOE"}
    john = next(p for p in merged["parties"] if p["name"] == "JOHN DOE")
    jane = next(p for p in merged["parties"] if p["name"] == "JANE DOE")
    assert john["role"] == "GRANTOR"
    assert jane["role"] == "GRANTEE"


def test_weird_types_are_skipped_with_metric(indexing_service, caplog):
    existing = {"parties": "oops"}
    new = {"parties": [{"name": "A", "role": "B"}]}

    with caplog.at_level(logging.WARNING, logger="docai.services.indexing_service"):
        merged = indexing_service._merge_results(existing, new)

    assert "parties" in merged
    assert isinstance(merged["parties"], list)
    assert len(merged["parties"]) >= 1
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_string_to_array_normalization(indexing_service):
    existing = {"parties": "JOHN DOE"}
    new = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}

    merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    assert len(merged["parties"]) == 2
    john_entry = next(p for p in merged["parties"] if "JOHN DOE" in str(p))
    assert john_entry["value"] == "JOHN DOE"
    assert john_entry["metadata"]["confidence"] == 0.5


def test_none_to_array_handling(indexing_service):
    existing = {"parties": None}
    new = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}

    merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    # None → [] so only the new party remains
    assert len(merged["parties"]) == 1
    assert merged["parties"][0]["name"] == "JANE DOE"


def test_mixed_types_in_existing_array(indexing_service, caplog):
    existing = {"parties": [{"name": "JOHN DOE", "role": "GRANTOR"}, "INVALID_STRING"]}
    new = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}

    with caplog.at_level(logging.WARNING, logger="docai.services.indexing_service"):
        merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    # invalid string is skipped → only JOHN + JANE remain
    assert len([p for p in merged["parties"] if isinstance(p, dict)]) == 2
    names = {p["name"] for p in merged["parties"] if isinstance(p, dict)}
    assert {"JOHN DOE", "JANE DOE"} == names
    assert any("Expected dict item" in rec.message or "Invalid" in rec.message for rec in caplog.records)


def test_empty_array_to_dict_merge(indexing_service):
    existing = {"parties": []}
    new = {"parties": {"name": "JANE DOE", "role": "GRANTEE"}}

    merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    assert len(merged["parties"]) == 1
    assert merged["parties"][0]["name"] == "JANE DOE"
    assert merged["parties"][0]["role"] == "GRANTEE"


def test_complex_nested_structure_merge(indexing_service):
    existing = {"parties": {"name": "JOHN DOE", "role": "GRANTOR", "address": {"street": "123 Main St"}}}
    new = {"parties": [{"name": "JANE DOE", "role": "GRANTEE", "address": "456 Oak Ave"}]}

    merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    assert len(merged["parties"]) == 2
    john = next(p for p in merged["parties"] if p.get("name") == "JOHN DOE")
    jane = next(p for p in merged["parties"] if p.get("name") == "JANE DOE")
    assert john["address"]["street"] == "123 Main St"
    assert jane["address"] == "456 Oak Ave"


def test_multiple_array_sections(indexing_service):
    existing = {
        "parties": {"name": "JOHN DOE", "role": "GRANTOR"},
        "signatures": "INVALID_STRING",
        "notaryInformation": None,
    }
    new = {
        "parties": [{"name": "JANE DOE", "role": "GRANTEE"}],
        "signatures": [{"name": "JANE DOE", "role": "SIGNATORY"}],
        "notaryInformation": [{"name": "NOTARY PUBLIC", "commission": "12345"}],
    }

    merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    assert isinstance(merged["signatures"], list)
    assert isinstance(merged["notaryInformation"], list)
    assert len(merged["parties"]) == 2
    assert len([s for s in merged["signatures"] if isinstance(s, dict)]) >= 1
    assert len([n for n in merged["notaryInformation"] if isinstance(n, dict)]) >= 1


def test_unknown_array_section_handling(indexing_service):
    existing = {"customArray": {"item": "value"}}
    new = {"customArray": [{"item": "new_value"}]}

    merged = indexing_service._merge_results(existing, new)

    # Not a configured array section → recursive merge path → stays dict
    assert "customArray" in merged
    assert isinstance(merged["customArray"], dict)


def test_error_recovery_and_logging(indexing_service, caplog):
    existing = {"parties": 12345}
    new = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}

    with caplog.at_level(logging.WARNING, logger="docai.services.indexing_service"):
        merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    assert len([p for p in merged["parties"] if isinstance(p, dict)]) >= 1
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_production_scenario_replication(indexing_service):
    existing_data = {
        "parties": {"name": "JOHN DOE", "role": "GRANTOR", "address": "123 Main St"}
    }
    new_data = {
        "parties": [
            {"name": "JANE DOE", "role": "GRANTEE", "address": "456 Oak Ave"},
            {"name": "BOB SMITH", "role": "WITNESS", "address": "789 Pine St"},
        ]
    }

    result = indexing_service._merge_results(existing_data, new_data)

    assert "parties" in result and isinstance(result["parties"], list)
    assert {p["name"] for p in result["parties"]} == {"JOHN DOE", "JANE DOE", "BOB SMITH"}


# Extra coverage: dedupe, idempotency, ingress guards, warnings

def test_dedupe_and_case_normalization(indexing_service):
    existing = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}
    new = {"parties": [{"name": " jane doe ", "role": "grantee"}]}

    merged = indexing_service._merge_results(existing, new)

    norm = {(p["name"].strip().upper(), p["role"].strip().upper()) for p in merged["parties"]}
    assert norm == {("JANE DOE", "GRANTEE")}


def test_idempotent_merge(indexing_service):
    base = {"parties": [{"name": "JOHN DOE", "role": "GRANTOR"}]}
    add = {"parties": [{"name": "JANE DOE", "role": "GRANTEE"}]}

    once = indexing_service._merge_results(base, add)
    twice = indexing_service._merge_results(once, add)

    assert len([p for p in twice["parties"] if isinstance(p, dict)]) == 2
    assert {p["name"] for p in twice["parties"] if isinstance(p, dict)} == {"JOHN DOE", "JANE DOE"}


def test_ingress_normalizes_ai_dict(indexing_service):
    existing = {"parties": {"name": "JOHN", "role": "GRANTOR"}}
    new = {"parties": {"name": "JANE", "role": "GRANTEE"}}

    merged = indexing_service._merge_results(existing, new)

    assert isinstance(merged["parties"], list)
    assert {p["name"] for p in merged["parties"] if isinstance(p, dict)} == {"JOHN", "JANE"}


def test_bad_item_logs_warning(indexing_service, caplog):
    with caplog.at_level(logging.WARNING, logger="docai.services.indexing_service"):
        existing = {"parties": [{"name": "OK", "role": "R"}, 42, "junk"]}
        new = {"parties": [{"name": "OK2", "role": "R2"}]}
        merged = indexing_service._merge_results(existing, new)

    assert {p["name"] for p in merged["parties"] if isinstance(p, dict)} == {"OK", "OK2"}
    assert any("Expected dict item" in rec.message or "Invalid" in rec.message for rec in caplog.records)


def test_normalize_array_section_method(indexing_service):
    result = indexing_service._normalize_array_section({"name": "TEST"}, "parties")
    assert isinstance(result, list) and result[0]["name"] == "TEST"

    result = indexing_service._normalize_array_section("TEST_STRING", "parties")
    assert isinstance(result, list) and result[0]["value"] == "TEST_STRING"

    result = indexing_service._normalize_array_section([{"name": "TEST"}], "parties")
    assert isinstance(result, list) and result[0]["name"] == "TEST"


def test_validate_array_section_method(indexing_service):
    assert indexing_service._validate_array_section([], "parties") is True
    assert indexing_service._validate_array_section({}, "parties") is True
    assert indexing_service._validate_array_section("test", "parties") is True
    assert indexing_service._validate_array_section(None, "parties") is True
    assert indexing_service._validate_array_section(12345, "parties") is False


if __name__ == "__main__":
    unittest.main()
