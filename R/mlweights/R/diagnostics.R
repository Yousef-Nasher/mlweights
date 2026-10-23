#' Latent-scale ICC and median odds ratio
#'
#' @param sigma2 Cluster (random-intercept) variance on the logit scale.
#' @return `icc_logit()`: \eqn{\sigma^2 / (\sigma^2 + \pi^2/3)}.
#'   `mor()`: \eqn{\exp(\sqrt{2\sigma^2}\,\Phi^{-1}(0.75))}.
#' @export
icc_logit <- function(sigma2) sigma2 / (sigma2 + pi^2 / 3)

#' @rdname icc_logit
#' @export
mor <- function(sigma2) exp(sqrt(2 * sigma2) * stats::qnorm(0.75))

#' Check whether level-1 weights are scaled
#'
#' @inheritParams scale_level1
#' @param w1 Level-1 weights as they would be given to the software.
#' @param tol Tolerance for calling a cluster's inflation factor different from 1.
#' @return A list with the mean, minimum and maximum inflation factor, the
#'   share of clusters not scaled, whether the weight is constant within
#'   clusters, and a message.
#' @export
check_level1_weights <- function(w1, cluster, tol = 0.05) {
  w1 <- .check_w(w1, cluster)
  c_j <- tapply(w1, cluster, sum) / tapply(w1, cluster, length)
  const <- is_constant_within(w1, cluster)
  off <- mean(abs(c_j - 1) > tol)
  msg <- if (off == 0) "Level-1 weights are scaled." else if (const)
    "Level-1 weights are constant within clusters but not equal to 1. Rescaled, they all become 1: set the level-1 weight to 1 and keep the cluster weight." else
    "Level-1 weights are not scaled. Rescale them within clusters before fitting."
  list(mean_c = mean(c_j), min_c = min(c_j), max_c = max(c_j), share_unscaled = off,
       constant_within = const, message = msg)
}

#' First-order cluster variance under inflated level-1 weights
#'
#' Solves \eqn{\sum_j w_{2j} [\sigma^2 + v_j(1 - 1/c_j) - s] / (s + v_j/c_j)^2 = 0}.
#' With equal \eqn{v} and \eqn{c} the answer is \eqn{\sigma^2 + (1 - 1/c)v}.
#' The normal approximation behind it ignores clusters in which all outcomes
#' are equal. For \eqn{c \ge 1} the real inflation is usually larger, so read
#' the result as a lower bound.
#'
#' @param sigma2 Cluster variance from the correctly scaled fit.
#' @param v Per-cluster sampling variance of the cluster effect on the logit
#'   scale, about \eqn{1 / \sum_i p_{ij}(1 - p_{ij})}.
#' @param c Inflation factor (scalar or one per cluster).
#' @param w2 Optional cluster weights.
#' @return The approximate cluster variance.
#' @examples
#' approx_inflated_variance(0.3, rep(0.5, 100), 4)
#' @export
approx_inflated_variance <- function(sigma2, v, c, w2 = NULL) {
  c <- rep_len(c, length(v))
  w2 <- if (is.null(w2)) rep(1, length(v)) else w2 / mean(w2)
  if (any(c <= 0) || any(v <= 0) || sigma2 < 0) stop("c and v must be positive and sigma2 non-negative")
  f <- function(s) sum(w2 * (sigma2 + v * (1 - 1 / c) - s) / (s + v / c)^2)
  if (f(0) <= 0) return(0)
  stats::uniroot(f, c(0, sigma2 + max(v) + 1), tol = 1e-12)$root
}
