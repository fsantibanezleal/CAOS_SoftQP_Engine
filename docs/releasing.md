# Distribution release procedure

Version 0.1.0 supports CPython 3.12 on Windows x64 and Linux x86_64. Other Python versions are not declared supported from an untested metadata range. The numerical source is separately pinned in each scientific experiment; packaging changes do not rewrite a completed measurement.

The release workflow builds a source distribution, then builds both wheels from that source archive with cibuildwheel 4.3.5. Linux uses manylinux_2_28. Every wheel is installed into an isolated environment; the smoke checks reject a source-checkout import, verify license metadata/notices and run the complete native test suite. Twine validates all distribution metadata. Artifacts are retained by exact source SHA.

The CMake build uses a short independent temporary directory instead of nesting MSBuild compiler checks inside a long pip source checkout. A clean Windows Git-source install exposed the old 260-character path failure. Numerical C source and solver settings are unchanged by this build-path repair.

For publication, configure a PyPI trusted publisher for project `softqp`, owner `fsantibanezleal`, repository `CAOS_SoftQP_Engine`, workflow `release.yml`, environment `pypi`. No token is stored in this repository. After validated task/develop/main promotion, create the immutable `v0.01.000` display-version tag and dispatch the release workflow on that tag with `publish=true`. PyPI normalizes that version to `0.1.0`. The source job checks that the display and distribution versions agree; the publishing job refuses other refs and runs only after both clean wheel gates pass.

A source merge, a draft release or a successful build is not PyPI publication. Verify the project/version metadata and both wheel hashes from the public index, then install the public distribution into a fresh environment and run the smoke suite. Until that completes, report version 0.1.0 as prepared and unpublished.

Primary build documentation: [cibuildwheel build/test options](https://cibuildwheel.pypa.io/en/stable/options/). A successful implementation benchmark does not establish global algorithmic novelty or operational safety.
