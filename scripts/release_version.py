"""Validate display/distribution coherence and an optional immutable release ref."""

import argparse
import json
from pathlib import Path
import re
import tomllib

from packaging.version import Version


def release_metadata(root, ref=None):
    display = (root / 'VERSION').read_text(encoding='utf-8').strip()
    distribution = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if not re.fullmatch(r'\d+\.\d{2}\.\d{3}', display):
        raise ValueError('invalid display version')
    if Version(display) != Version(distribution):
        raise ValueError('display and distribution versions disagree')
    tag = 'v' + display
    if ref is not None and ref != 'refs/tags/' + tag:
        raise ValueError('publication requires the exact display-version tag')
    return dict(display=display, distribution=distribution, tag=tag,
                sdist='softqp-' + distribution + '.tar.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref')
    args = parser.parse_args()
    print(json.dumps(release_metadata(Path(__file__).resolve().parents[1], args.ref)))
