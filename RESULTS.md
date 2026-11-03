# Actual experimental results

Three training seeds: 42, 43, 44. Same fixed split: 18,900/4,050/4,050. Accuracy mean ± sample SD (%).

|Model|Clean|Brightness x0.6|Gaussian blur r=1|
|---|---|---|---|
|baseline|96.72 ± 0.12|37.42 ± 2.20|69.16 ± 0.43|
|se|96.40 ± 0.29|33.63 ± 1.94|69.68 ± 2.21|
|cbam|96.29 ± 0.14|34.21 ± 3.79|71.11 ± 2.09|

Attention did not improve clean accuracy in this setting. CBAM had a higher mean blurred-image accuracy, with larger seed variability. No significance claim is made. Complete checkpoints, logs, per-image predictions, and split manifest are included in the local/server deliverables. See evidence/ for run-level outputs.
