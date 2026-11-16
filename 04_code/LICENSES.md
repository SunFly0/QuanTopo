# Dependency and asset notices

This source tree does not vendor third-party package source, datasets, pretrained
weights, or baseline implementations. Runtime dependencies are installed separately
through `pyproject.toml` and remain subject to their upstream licenses.

- PyTorch
- NumPy
- PyYAML
- OpenCLIP / `open_clip_torch`

HumanML3D, KIT-ML, CLIP weights, and any evaluation assets are not redistributed here.
Users are responsible for obtaining them from their official sources and following the
corresponding licenses and terms. Files placed under `data/` or `assets/` are ignored by
the implementation's provenance records except through content digests and anonymous,
project-relative identifiers.

