# mlweights (Python)

Scaling and checking survey weights for two-level random-intercept models.

- `scale_level1(w, cluster, method="size")`: rescale level-1 weights within clusters.
- `check_level1_weights(w, cluster)`: report how far level-1 weights are from being scaled.
- `dhs_level_weights(...)`: split a DHS final weight into cluster and within-cluster weights, with rescaling.
- `approx_inflated_variance(sigma2, v, c)`: first-order cluster variance under inflated level-1 weights (a lower bound).
- `WeightedRILogit`: random-intercept logit model fitted by weighted pseudo-maximum likelihood.

See the repository README for examples.
