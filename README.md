# materials

[Download the four complete ZIP archives](https://quantopo-materials.burblingbard8.chatgpt.site/)

Download each complete archive using the links below. Each ZIP can be extracted independently; no split-file assembly is required.

| Archive | Contents |
|---|---|
| [Figure_videos.zip](https://quantopo-materials.burblingbard8.chatgpt.site/downloads/Figure_videos.zip) | 32 figure-example videos and 180 QuanTopo composite clips |
| [Composite180.zip](https://quantopo-materials.burblingbard8.chatgpt.site/downloads/Composite180.zip) | 900 composite comparison videos, captions, event annotations and scoring |
| [S3_baselines.zip](https://quantopo-materials.burblingbard8.chatgpt.site/downloads/S3_baselines.zip) | Baseline evaluation records, embeddings and statistical scripts |
| [QuanTopo_code.zip](https://quantopo-materials.burblingbard8.chatgpt.site/downloads/QuanTopo_code.zip) | Implementation and runtime configuration |

Browse the same materials through the repository-local links below:

| Materials | Files |
|---|---|
| Figure examples | [Eight captions, four methods each](01_figure_videos/figure_examples/) |
| QuanTopo Composite clips | [180 captions and videos](01_figure_videos/composite180_quantopo/) |
| Full composite comparison | [180 captions × five methods](02_composite180/videos/) |
| Composite evaluation | [Captions](02_composite180/composite180.csv), [event DAGs](02_composite180/events_dag.json), [annotations](02_composite180/annotations.json), [scoring script](02_composite180/score.py), [results](02_composite180/results/) |
| Baseline reevaluation | [Fixed records](03_baseline_evaluation/fixed_records/), [runs](03_baseline_evaluation/runs/), [embeddings](03_baseline_evaluation/shared_embeddings/), [statistics](03_baseline_evaluation/statistics/), [results](03_baseline_evaluation/results/) |
| Implementation | [Code and runtime configuration](04_code/) |

The figure directory contains 32 example videos and 180 QuanTopo composite videos. The full composite directory contains 900 videos across five methods, arranged by caption. Open a caption folder, then select a video to view or download it. The directory links provide individual files.

To calculate the composite event metrics locally, run `score.py` and `paired_score.py` from `02_composite180` after downloading that directory and installing its requirements.
