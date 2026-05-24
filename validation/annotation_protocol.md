# Annotation Protocol for Fairness-Debt Construct Validation

## Purpose

The protocol evaluates whether the fairness-debt diagnostic aligns with expert or experienced-player judgments of adverse exposure in synthetic match histories. It is intended for future external validation and is not reported as completed human-subject data in the current manuscript.

## Participants

Recommended sample: 5--10 annotators with experience in competitive ranked games, game analytics, matchmaking research, or online game design.

## Materials

Annotators receive `sample_synthetic_match_histories.csv`. Each row describes one synthetic player-session segment with:

- match balance
- loss streak
- large-gap exposure
- role mismatch
- smurf-like exposure indicator
- rank--MMR divergence
- waiting-time proxy
- visible-rank movement
- final outcome pattern
- fairness-debt diagnostic value

## Annotation Questions

Annotators complete `annotation_form.csv` with:

- perceived unfairness, 1--7
- frustration risk, 1--7
- progression mismatch salience, 1--7
- acceptable / concerning / unsure
- optional comment

## Planned Analysis

When real annotations are collected:

- Spearman correlation between fairness debt and perceived unfairness
- Kendall tau between fairness-debt ranking and perceived-unfairness ranking
- Bootstrap confidence intervals
- Inter-rater reliability using ICC or Krippendorff alpha when feasible
- Disagreement audit for high-fairness-debt but low-human-concern cases and the reverse

## Reporting Boundary

No human annotation results should be reported until annotators have actually completed the form. Empty forms and synthetic examples are not validation evidence by themselves.
