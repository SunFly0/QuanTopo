# QuanTopo materials

Download the material ZIPs below, or browse the folders farther down this page. Links stay within this repository.

Each ZIP is independently extractable. For a group with multiple ZIPs, download all its parts and extract them into the same destination folder; no file joining is required. The ZIPs preserve the original directory names and contents.

| Download group | ZIP files |
|---|---|
| Figure videos | [ZIP 01](downloads/Figure_videos_01.zip), [ZIP 02](downloads/Figure_videos_02.zip), [ZIP 03](downloads/Figure_videos_03.zip) |
| Composite comparison | [ZIP 01](downloads/Composite180_01.zip), [ZIP 02](downloads/Composite180_02.zip), [ZIP 03](downloads/Composite180_03.zip), [ZIP 04](downloads/Composite180_04.zip), [ZIP 05](downloads/Composite180_05.zip), [ZIP 06](downloads/Composite180_06.zip), [ZIP 07](downloads/Composite180_07.zip), [ZIP 08](downloads/Composite180_08.zip), [ZIP 09](downloads/Composite180_09.zip), [ZIP 10](downloads/Composite180_10.zip), [ZIP 11](downloads/Composite180_11.zip), [ZIP 12](downloads/Composite180_12.zip), [ZIP 13](downloads/Composite180_13.zip), [ZIP 14](downloads/Composite180_14.zip) |
| Baseline reevaluation | [ZIP 01](downloads/S3_baselines_01.zip), [ZIP 02](downloads/S3_baselines_02.zip) |
| Implementation | [ZIP](downloads/QuanTopo_code.zip) |

For batch download, save [download_materials.py](download_materials.py), run `python download_materials.py`, and paste this anonymous repository's URL when prompted. The optional `--group` selects `figure`, `composite`, `baseline`, or `code`; the default downloads all groups. Downloads resume by skipping files that already pass integrity verification. Extract all downloaded ZIPs into the same folder.

Browse individual files:

| Materials | Files |
|---|---|
| Figure examples | [Eight captions, four methods each](01_figure_videos/figure_examples/) |
| Composite clips labelled QuanTopo | [180 captions and videos](01_figure_videos/composite180_quantopo/) |
| Full composite comparison | [180 captions × five methods](02_composite180/videos/) |
| Composite evaluation | [Captions](02_composite180/composite180.csv), [event DAGs](02_composite180/events_dag.json), [annotations](02_composite180/annotations.json), [scoring script](02_composite180/score.py), [results](02_composite180/results/) |
| Baseline reevaluation | [Fixed records](03_baseline_evaluation/fixed_records/), [runs](03_baseline_evaluation/runs/), [embeddings](03_baseline_evaluation/shared_embeddings/), [statistics](03_baseline_evaluation/statistics/), [results](03_baseline_evaluation/results/) |
| Implementation | [Code and runtime configuration](04_code/) |

The figure directory contains 32 example videos and 180 QuanTopo-labelled composite videos. The full composite directory contains 900 videos across five methods, arranged by caption. Open a caption folder, then select a video to view or download it. The ZIP links above provide grouped downloads; the directory links provide individual files.

Data provenance: event annotations are target-fitted examples rather than independent human ratings. QuanTopo-labelled videos include external-model previews or edited illustrations; ActionPlan clips were generated from its official checkpoint.

To calculate the composite event metrics locally, run `score.py` and `paired_score.py` from `02_composite180` after downloading that directory and installing its requirements.
