#' Level-weights for DHS-type samples, with level-1 rescaling
#'
#' Splits the final weight into a cluster weight and a within-cluster weight
#' following the DHS Program approximation (de-normalise the final weight,
#' then split the variation factor with exponent `alpha`). It adds the step
#' that the published code leaves out: the within-cluster weight is rescaled
#' to sum to the cluster sample size.
#'
#' @param v005 Final weight with six implied decimals.
#' @param stratum,cluster Stratum and cluster codes.
#' @param A_h,M_h Named vectors (names are stratum codes): clusters in the
#'   frame and average households per cluster.
#' @param S_h Households selected per cluster (scalar or named vector).
#' @param M_total Households in the frame.
#' @param m_c Completed household interviews.
#' @param alpha Share of the variation factor given to the cluster weight.
#' @return A data frame with `d` (de-normalised weight), `w2` (cluster
#'   weight), `w1` (within-cluster weight before scaling) and `w1_scaled`
#'   (the one to use in the model).
#' @export
dhs_level_weights <- function(v005, stratum, cluster, A_h, M_h, S_h, M_total, m_c, alpha = 0.5) {
  key <- as.character(stratum)
  get <- function(m) if (length(m) == 1 && is.null(names(m))) rep(m, length(key)) else as.numeric(m[key])
  A <- get(A_h); M <- get(M_h); S <- get(S_h)
  if (anyNA(c(A, M, S))) stop("A_h, M_h and S_h must cover every stratum")
  a <- stats::ave(seq_along(key), key, FUN = function(i) length(unique(cluster[i])))
  d <- v005 / 1e6 * (M_total / m_c)
  f <- d / ((A / a) * (M / S))
  w2 <- (A / a) * f^alpha
  w1 <- d / w2
  data.frame(d = d, w2 = w2, w1 = w1, w1_scaled = scale_level1(w1, cluster))
}
