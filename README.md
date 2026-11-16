# QuanTopo materials

[Download the complete numbered archive](https://github.com/SunFly0/QuanTopo/releases)

Inside the complete archive:

| Directory | Contents |
|---|---|
| `01_paper` | Current manuscript PDF, TeX and referenced figures |
| `02_figure_videos` | 32 figure-example videos and captions |
| `03_composite180` | 180 captions, five methods, 900 videos, annotations and scoring |
| `04_baseline_evaluation` | S3 baseline evaluation records, embeddings and statistics |
| `05_code` | Model code and PSD/PCA numerical and checkpoint checks |
| `06_rebuttal` | English reviewer responses |

The archive contains the complete files, including videos, evaluation runs and
large arrays. The repository tree provides the implementation and provenance for browsing. See `metadata.json` for data provenance.

After extracting the archive, from `03_composite180`, run:

```bash
python score.py
python paired_score.py
```

For the model code and its PSD/PCA checks:

```bash
cd 05_code
PYTHONPATH=src python -m unittest discover -s tests -v
```
