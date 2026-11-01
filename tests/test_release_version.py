"""Publication rejects branch refs, stale tags and inconsistent metadata."""

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / 'scripts' / 'release_version.py'
_spec = importlib.util.spec_from_file_location('release_version', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_current_tag_is_required_and_other_refs_fail(tmp_path):
    (tmp_path/'VERSION').write_text('0.02.000\n')
    (tmp_path/'pyproject.toml').write_text('[project]\nversion="0.2.0"\n')
    metadata = _module.release_metadata(tmp_path, 'refs/tags/v0.02.000')
    assert metadata['sdist'] == 'softqp-0.2.0.tar.gz'
    for ref in ('refs/heads/main', 'refs/heads/task/directional-peak',
                'refs/tags/v0.01.000', 'refs/tags/v0.2.0', ''):
        with pytest.raises(ValueError):
            _module.release_metadata(tmp_path, ref)
    (tmp_path/'pyproject.toml').write_text('[project]\nversion="0.1.0"\n')
    with pytest.raises(ValueError, match='disagree'):
        _module.release_metadata(tmp_path)
