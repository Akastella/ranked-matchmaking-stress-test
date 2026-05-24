# Auxiliary Annotation Protocol for Synthetic-Session Plausibility Check

## Purpose

This protocol documents the auxiliary pilot annotation material used to assess whether displayed synthetic-session summaries and diagnostic channels are understandable to annotators. The annotation layer is supplementary plausibility evidence only. It does not validate a construct, real player experience, telemetry behavior, real affect, or retention. The main manuscript's claims rely on the synthetic simulation protocol, verification checks, risk-transfer experiment, component diagnostics, and robustness summaries.

The pilot questionnaire used anonymous ratings of synthetic session summaries. It did not use real player records and did not collect sensitive personal information. Because no session was rated by multiple annotators, inter-rater reliability such as ICC or Krippendorff alpha is not estimable in the current pilot design. Future overlapping annotation could be used to estimate reliability.

## Participants

Annotators were asked to rate synthetic session summaries. The questionnaire did not request sensitive personal information or real gameplay records.

## Materials

Annotators receive synthetic player-session summaries. Each row describes one synthetic player-session segment with:

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

The pilot annotation layer can be summarized with:

- Spearman correlation between fairness debt and perceived unfairness
- Kendall tau between fairness-debt ranking and perceived-unfairness ranking
- Bootstrap confidence intervals
- Disagreement audit for high-exposure but low-human-concern cases and the reverse

## Reporting Boundary

The annotation material is supplementary plausibility evidence over synthetic summaries. It is not used as primary simulation evidence and does not establish external behavioral validity.
