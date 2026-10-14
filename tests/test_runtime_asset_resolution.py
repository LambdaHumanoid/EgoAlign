"""Check the bundled asset resolver without importing simulation services."""
import ast
import os
from pathlib import Path

import pytest


def bundled_resolver(runtime_root):
    patch = Path(__file__).resolve().parents[1] / "runtime/patches/0002-replay.patch"
    lines = patch.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("+def _resolve_sonic_brainco_assets("))
    source = [lines[start][1:]]
    for line in lines[start + 1:]:
        if not line.startswith("+"):
            break
        value = line[1:]
        if value and not value[0].isspace():
            break
        source.append(value)
    module = ast.parse("\n".join(source))
    assert len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef)
    namespace = {"os": os, "Path": Path, "GEAR_SONIC_ROOT": runtime_root}
    exec(compile(module, str(patch), "exec"), namespace)
    return namespace["_resolve_sonic_brainco_assets"]


def model_pair(root):
    root.mkdir(parents=True)
    pair = (root / "scene_29dof_brainco.xml", root / "g1_29dof_brainco.xml")
    for path in pair:
        path.write_text("<mujoco/>")
    return pair


def test_explicit_asset_directory_has_priority(tmp_path, monkeypatch):
    model_pair(tmp_path / "gear_sonic/data/robot_model/model_data/g1/brainco")
    expected = model_pair(tmp_path / "configured model")
    monkeypatch.setenv("SONIC_BRAINCO_MODEL_DIR", str(expected[0].parent))
    assert bundled_resolver(tmp_path)() == expected


def test_runtime_asset_directory_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("SONIC_BRAINCO_MODEL_DIR", raising=False)
    expected = model_pair(tmp_path / "gear_sonic/data/robot_model/model_data/g1/brainco")
    assert bundled_resolver(tmp_path)() == expected


@pytest.mark.parametrize("missing", ["scene_29dof_brainco.xml", "g1_29dof_brainco.xml"])
def test_incomplete_explicit_directory_never_substitutes_model(tmp_path, monkeypatch, missing):
    model_pair(tmp_path / "gear_sonic/data/robot_model/model_data/g1/brainco")
    configured = tmp_path / "configured"
    model_pair(configured)
    (configured / missing).unlink()
    monkeypatch.setenv("SONIC_BRAINCO_MODEL_DIR", str(configured))
    with pytest.raises(FileNotFoundError, match="SONIC_BRAINCO_MODEL_DIR"):
        bundled_resolver(tmp_path)()


def test_missing_assets_report_configuration_key(tmp_path, monkeypatch):
    monkeypatch.delenv("SONIC_BRAINCO_MODEL_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="SONIC_BRAINCO_MODEL_DIR"):
        bundled_resolver(tmp_path)()
