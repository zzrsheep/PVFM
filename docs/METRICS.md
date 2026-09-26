# Common-Q9 scoring

The trained head remains Q11. For **all probabilistic scores** select the exact
levels 0.1, 0.2, ..., 0.9 before calculation. The 0.05/0.95 outputs are excluded;
no model retraining or quantile interpolation occurs. Missing required levels
raise an error rather than silently changing the evaluation grid.

For a valid forecast/observation pair, pinball loss is
`rho_tau(y-q) = max(tau*(y-q), (tau-1)*(y-q))`.

- AQL: mean raw pinball loss over the common nine quantiles and valid targets.
- CRPS: `2 * integral_0^1 rho_tau(y-Q(tau)) d tau`, approximated by trapezoidal
  integration. First apply cumulative maximum to quantile predictions; add
  tau=0 and tau=1 with constant endpoint predictions. Evaluate pinball loss
  at those endpoints too. This is a finite-grid approximation, not an exact
  distributional CRPS. No absolute-power-sum normalization is used.
- MAE/RMSE: absolute/squared error of q50 in capacity-factor units.
- R²: `1 - SSE / SST`, with SST computed from **all valid window/horizon
  pairs at a station/task**, not a mean of window R². SST≤1e-6 yields undefined
  R²; the station is omitted only from the R² mean and its count is reported.

Within each task, first concatenate valid pairs per station, then take the
arithmetic mean of station scores. RMSE is computed per station before this
average. The same timestamp in overlapping windows remains a separate
forecast-origin/horizon pair. No time-point deduplication is introduced.
Case-study and ordinary-demo selections are summarized separately.

Q11 training loss and common-Q9 evaluation AQL are deliberately different
quantile sets. Evaluation AQL is not a reported Q11 training loss. Under this
trapezoidal/tail protocol CRPS is not defined as exactly twice AQL.

The functions in `pvfm/probabilistic.py` are extracted from the current
project scorer. The common-grid selection and station grouping are explicit
in `pvfm/metrics.py`. Non-finite predictions on valid targets fail loudly.
