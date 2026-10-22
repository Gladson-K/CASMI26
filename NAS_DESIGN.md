# CPU NAS design and interpretation

## Search algorithm

- Optimizer: Optuna TPE with a fixed seed and multivariate/group suggestions.
- Pruning: median pruner; a trial reports intermediate MRR@25 after query batches.
- Persistence: SQLite (`outputs/nas_cpu_10h.sqlite3`), so a restart continues the study.
- Execution: one Optuna worker; BLAS/OpenMP threads are capped by `project.num_threads` to control RAM and avoid oversubscription.
- Budget: `timeout_seconds: 36000` (10 hours); `n_trials` is only an upper bound.

## Tuned dimensions

- Precursor neutral-mass tolerance (ppm).
- Fragment m/z matching tolerance (Da).
- Relative-intensity floor and intensity exponent.
- Peak cap (64–256).
- Per-structure aggregation (`max` or mean of the top three spectral matches).
- Maximum number of mass-window spectra scored per query.

## Score protocol

The objective evaluates MRR@25 on a fixed spectrum-level split, with other spectra from the same structure potentially present in the reference library. This estimates direct spectral-library matching performance when a matching structure is represented by reference spectra. It is not an unseen-molecule holdout, so it will overestimate competition performance if test structures are novel. Do not tune or report this as a final public leaderboard metric.

For a stronger competition estimate, create a separate molecule/InChIKey-grouped holdout, and test external-library coverage and candidate recall independently. The next modelling layer should combine DreaMS embeddings, exact-mass/adduct constraints, analog propagation, and fragmentation explainability, then train a ranker using grouped folds. Cache all static embeddings/features outside Optuna trials.
