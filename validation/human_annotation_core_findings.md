# Human Annotation Core Findings

The returned 600-row pilot annotation file is analyzed as `human_annotation_real`. It is not merged with synthetic annotator outputs.

## Core directional alignment
- Fairness debt vs perceived unfairness: Spearman 0.613 [0.559, 0.662], Kendall 0.473.
- Fairness debt vs frustration risk: Spearman 0.551 [0.488, 0.606], Kendall 0.419.
- Rank--MMR hysteresis vs progression mismatch salience: Spearman 0.915 [0.900, 0.928], Kendall 0.786.
- Smurf-victimization exposure vs perceived unfairness: Spearman 0.472 [0.401, 0.533], Kendall 0.353.

## Balance-only comparison
Immediate win-probability deviation is not directly present in the returned annotation summaries. The script therefore reports a `win_prob_deviation_proxy` based on the available tail-unfairness field and marks the regression as exploratory. In this returned feature reconstruction, the balance proxy is not independent of fairness debt, so incremental model comparisons should be interpreted as a limitation rather than strong evidence.

## Inter-rater reliability
No repeated session overlap is present, so Krippendorff alpha, ICC, and average pairwise agreement are not estimated.