# Ranked Matchmaking Stress-Test Protocol

This repository-ready package contains the source code, configuration files, synthetic CSV summaries, figure-generation scripts, tables, and supplementary material needed to reproduce the manuscript:

**A Synthetic Stress-Testing Protocol for Ranked Matchmaking Systems**

The simulation data are synthetic. No commercial matchmaking logs, real player telemetry, or real player records are included. The auxiliary questionnaire materials rate synthetic session summaries and are provided only as supplementary plausibility material.

## Contents

- `src/`: simulation package source code.
- `scripts/`: selected experiment, aggregation, verification, and figure/table scripts.
- `config/`: simulation and performance configuration files.
- `results_csv/`: generated CSV summaries used by the manuscript.
- `figures/`: submission figures in PDF/PNG form.
- `tables/`: editable LaTeX tables.
- `supplement/`: supplementary notes, tables, and appendices.
- `docs/reproducibility.md`: concise reproduction notes.

## Reproduction Outline

1. Install Python dependencies listed in `requirements.txt`.
2. Run the desired simulation profile from `scripts/`.
3. Aggregate task-level outputs.
4. Regenerate tables and figures from CSV outputs.
5. Compile the manuscript with a LaTeX engine compatible with `elsarticle`.

Fixed seeds and configuration files are included to support deterministic reruns within the reported profiles.

## License and Citation

See `LICENSE_NOTICE.md` and `CITATION.cff` for reuse and citation metadata.

## Contact

Corresponding author: Junshan Mu, `motongmu@gmail.com`.
