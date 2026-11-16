# QuanTopo materials

Browse the materials directly through the links below. All links stay within this repository.

| Materials | Files |
|---|---|
| Figure examples | [Eight captions, four methods each](01_figure_videos/figure_examples/) |
| Composite clips labelled QuanTopo | [180 captions and videos](01_figure_videos/composite180_quantopo/) |
| Full composite comparison | [180 captions × five methods](02_composite180/videos/) |
| Composite evaluation | [Captions](02_composite180/composite180.csv), [event DAGs](02_composite180/events_dag.json), [annotations](02_composite180/annotations.json), [scoring script](02_composite180/score.py), [results](02_composite180/results/) |
| Baseline reevaluation | [Fixed records](03_baseline_evaluation/fixed_records/), [runs](03_baseline_evaluation/runs/), [embeddings](03_baseline_evaluation/shared_embeddings/), [statistics](03_baseline_evaluation/statistics/), [results](03_baseline_evaluation/results/) |
| Implementation | [Code and runtime configuration](04_code/) |

The figure directory contains 32 example videos and 180 QuanTopo-labelled composite videos. The full composite directory contains 900 videos across five methods, arranged by caption. Open a caption folder, then select a video to view or download it. Repository-wide archive availability depends on the anonymous hosting service; individual files are available through the directory links.

Data provenance: event annotations are target-fitted examples rather than independent human ratings. QuanTopo-labelled videos include external-model previews or edited illustrations; ActionPlan clips were generated from its official checkpoint.

To calculate the composite event metrics locally, run `score.py` and `paired_score.py` from `02_composite180` after downloading that directory and installing its requirements.
