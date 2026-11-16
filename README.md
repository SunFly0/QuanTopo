# QuanTopo materials

[Open the downloads](https://github.com/SunFly0/QuanTopo/releases)

| Download | Contents |
|---|---|
| `Figure_videos.zip` | 32 figure-example videos plus all 180 QuanTopo-labelled composite clips, with captions; 212 videos for the full-video request |
| `Composite180.zip` | 180 captions × five methods: 900 videos, event annotations and scoring scripts |
| `S3_baselines.zip` | Baseline evaluation records, embeddings and statistical scripts |
| `QuanTopo_code.zip` | Implementation; also browsable in `04_code` |

Data provenance: the event annotations are target-fitted examples rather than
independent human ratings. QuanTopo-labelled videos include external-model
previews or edited illustrations; ActionPlan clips were generated from its
official checkpoint.

After extracting `Figure_videos.zip`, the videos are grouped under:

```text
01_figure_videos/
  figure_examples/          # eight caption examples, four methods each
  composite180_quantopo/    # one clip per composite caption
```

After extracting `Composite180.zip`, run `score.py` and `paired_score.py` from
`02_composite180` to calculate its event metrics.
