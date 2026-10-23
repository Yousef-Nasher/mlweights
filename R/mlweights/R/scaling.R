.check_w <- function(w, cluster) {
  w <- as.numeric(w)
  if (length(w) != length(cluster)) stop("weights and cluster must have the same length")
  if (any(!is.finite(w)) || any(w <= 0)) stop("weights must be finite and positive")
  w
}

#' Rescale level-1 weights within clusters
#'
#' @param w Level-1 (within-cluster) weights.
#' @param cluster Cluster identifiers.
#' @param method `"size"`: weights sum to the cluster sample size.
#'   `"effective"`: weights sum to the effective sample size
#'   \eqn{(\sum w)^2 / \sum w^2}.
#' @return The rescaled weights. Both methods return 1 for every unit when the
#'   weight is constant within the cluster.
#' @examples
#' scale_level1(c(4, 4, 4, 2, 6), c(1, 1, 1, 2, 2))
#' @export
scale_level1 <- function(w, cluster, method = c("size", "effective")) {
  method <- match.arg(method)
  w <- .check_w(w, cluster)
  s1 <- stats::ave(w, cluster, FUN = sum)
  target <- if (method == "size") stats::ave(w, cluster, FUN = length) else s1^2 / stats::ave(w^2, cluster, FUN = sum)
  w * target / s1
}

#' Inflation factor of level-1 weights
#'
#' @inheritParams scale_level1
#' @return For each unit, \eqn{c_j}: the sum of the level-1 weights in its
#'   cluster divided by the number of units. A value of 1 means scaled.
#' @export
cluster_inflation <- function(w, cluster) {
  w <- .check_w(w, cluster)
  stats::ave(w, cluster, FUN = sum) / stats::ave(w, cluster, FUN = length)
}

#' Is a weight constant within clusters?
#'
#' @inheritParams scale_level1
#' @param tol Relative tolerance.
#' @return `TRUE` or `FALSE`.
#' @export
is_constant_within <- function(w, cluster, tol = 1e-8) {
  w <- .check_w(w, cluster)
  r <- stats::ave(w, cluster, FUN = function(x) max(x) - min(x))
  all(r <= tol * abs(w))
}
