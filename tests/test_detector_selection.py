"""Unified profiles never conflate SAM weights with crYOLO weights."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from detector_selection import (
    DETECTION_PROFILES, canonical_detection_profile, load_detector_settings,
    profile_from_backend, resolve_detection_profile, save_detector_settings,
    validate_detector_settings,
)


@pytest.mark.parametrize("selection,backend,contour", [
    ("capsid", "sam", "capsid"),
    ("capsid-learned", "sam", "capsid-learned"),
    ("generic", "sam", "generic"),
    ("crYOLO", "cryolo", "capsid-learned"),
])
def test_four_profiles_round_trip_without_architecture_choices(selection, backend, contour):
    assert resolve_detection_profile(selection) == (backend, contour)
    assert profile_from_backend(backend, contour) == selection
    assert DETECTION_PROFILES == ("capsid", "capsid-learned", "generic", "crYOLO")


def test_lowercase_cryolo_alias_is_canonicalized():
    assert canonical_detection_profile("cryolo") == "crYOLO"
    assert resolve_detection_profile("cryolo") == ("cryolo", "capsid-learned")


@pytest.mark.parametrize("value", [None, [], {}, "vit_b", "vit_l", "vit_h", "SAM", "crYOLO "])
def test_invalid_profile_is_rejected(value):
    with pytest.raises(ValueError):
        canonical_detection_profile(value)


@pytest.mark.parametrize("backend,profile", [
    ("cryolo", "generic"), ("cryolo", "capsid"), ("sam", "crYOLO"),
    ("sam", None), ("other", "capsid"),
])
def test_unrepresentable_backend_profile_pairs_are_rejected(backend, profile):
    with pytest.raises(ValueError):
        profile_from_backend(backend, profile)


def test_settings_round_trip_keeps_both_models_and_no_interpreter(tmp_path):
    path = tmp_path / "new" / "detector-settings.json"
    payload = {
        "detection_profile": "cryolo", "sam_checkpoint": "models/sam_vit_b.pth",
        "sam_model_type": "vit_b", "cryolo_checkpoint": "models/capsids.h5",
        "cryolo_config": "models/config.json", "cryolo_threshold": .3,
        "cryolo_timeout": 180, "device": "auto", "points_per_side": 8,
        "min_capsid_area": 3500, "cryolo_python": "private/python.exe",
        "backend": "cryolo", "checkpoint": "wrong-shared.h5",
    }
    save_detector_settings(path, payload)
    actual = load_detector_settings(path)
    assert actual["detection_profile"] == "crYOLO"
    assert actual["sam_checkpoint"] == "models/sam_vit_b.pth"
    assert actual["cryolo_checkpoint"] == "models/capsids.h5"
    assert not {"cryolo_python", "backend", "checkpoint"} & actual.keys()
    assert actual == json.loads(path.read_text())


@pytest.mark.parametrize("key,value", [
    ("sam_checkpoint", "capsids.h5"), ("cryolo_checkpoint", "sam.pth"),
    ("sam_checkpoint", 42), ("cryolo_checkpoint", ["model.h5"]),
    ("cryolo_config", "bad\x00path"), ("cryolo_threshold", float("nan")),
    ("cryolo_threshold", float("inf")), ("cryolo_threshold", 10**1000), ("cryolo_threshold", True),
    ("cryolo_threshold", 1.1), ("cryolo_threshold", -.1),
    ("cryolo_timeout", 0), ("cryolo_timeout", "180"),
    ("min_capsid_area", -1), ("min_capsid_area", 3.5),
    ("points_per_side", True), ("points_per_side", 0), ("points_per_side", 65),
    ("device", "metal"), ("sam_model_type", "cryolo"),
])
def test_invalid_settings_cannot_be_saved(key, value, tmp_path):
    path = tmp_path / "detector-settings.json"
    with pytest.raises(ValueError):
        save_detector_settings(path, {key: value})
    assert not path.exists()


def test_optional_paths_and_case_insensitive_suffixes():
    assert validate_detector_settings({"sam_checkpoint": None, "cryolo_checkpoint": "   ",
                                       "cryolo_config": ""}) == {
        "sam_checkpoint": None, "cryolo_checkpoint": None, "cryolo_config": None}
    assert validate_detector_settings({"sam_checkpoint": "MODEL.PTH", "cryolo_checkpoint": "MODEL.H5"}) == {
        "sam_checkpoint": "MODEL.PTH", "cryolo_checkpoint": "MODEL.H5"}


def test_missing_file_does_not_import_or_modify_legacy_settings(tmp_path):
    legacy = tmp_path / "cryolo-settings.json"
    legacy.write_text('{"checkpoint": "old.h5"}')
    assert load_detector_settings(tmp_path / "detector-settings.json") == {}
    assert legacy.read_text() == '{"checkpoint": "old.h5"}'


@pytest.mark.parametrize("contents", ["{broken", "[]", "null", '{"device": "invalid"}'])
def test_malformed_or_invalid_saved_data_raises_value_error(tmp_path, contents):
    path = tmp_path / "detector-settings.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_detector_settings(path)


def test_failed_atomic_replace_preserves_previous_settings_and_cleans_temporary(tmp_path):
    path = tmp_path / "detector-settings.json"
    save_detector_settings(path, {"detection_profile": "capsid"})
    previous = path.read_bytes()
    with patch("detector_selection.os.replace", side_effect=OSError("disk failure")), pytest.raises(OSError):
        save_detector_settings(path, {"detection_profile": "crYOLO"})
    assert path.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
