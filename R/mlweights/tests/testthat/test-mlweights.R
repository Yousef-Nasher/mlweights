test_that("size scaling sums to cluster size", {
  set.seed(1)
  cl <- rep(1:20, sample(2:8, 20, TRUE)); w <- runif(length(cl), .5, 6)
  s <- scale_level1(w, cl)
  expect_equal(as.numeric(tapply(s, cl, sum)), as.numeric(table(cl)))
})
test_that("effective scaling", {
  s <- scale_level1(c(1, 2, 3, 4, 4), c(1, 1, 1, 2, 2), "effective")
  expect_equal(sum(s[1:3]), 36 / 14); expect_equal(s[4:5], c(1, 1))
})
test_that("constant weights become one", {
  cl <- rep(c(10, 20, 30), 4:6); w <- rep(c(3.8, 4.1, .2), 4:6)
  expect_equal(scale_level1(w, cl), rep(1, 15))
  expect_true(is_constant_within(w, cl))
  expect_false(is_constant_within(c(w[-15], .3), cl))
  r <- check_level1_weights(w, cl)
  expect_true(r$constant_within); expect_match(r$message, "set the level-1 weight to 1")
  expect_equal(check_level1_weights(rep(1, 15), cl)$share_unscaled, 0)
})
test_that("inflation factor", {
  expect_equal(cluster_inflation(c(4, 4, 4, 1, 3), c(1, 1, 1, 2, 2)), c(4, 4, 4, 2, 2))
})
test_that("DHS split multiplies back and rescales to one", {
  cl <- rep(1:6, each = 5); st <- rep(c(1, 1, 1, 2, 2, 2), each = 5)
  v005 <- rep(c(8, 12, 10, 9, 15, 6) * 1e5, each = 5)
  out <- dhs_level_weights(v005, st, cl, A_h = c("1" = 300, "2" = 500), M_h = c("1" = 110, "2" = 95),
                           S_h = 25, M_total = 60000, m_c = 150)
  expect_equal(out$w1 * out$w2, out$d); expect_equal(out$w1_scaled, rep(1, 30))
  out1 <- dhs_level_weights(v005, st, cl, c("1" = 300, "2" = 500), c("1" = 110, "2" = 95), 25, 60000, 150, alpha = 1)
  expect_equal(out1$w1, ifelse(st == 1, 110, 95) / 25)
})
test_that("approximation limits and summaries", {
  v <- rep(.5, 100)
  expect_equal(approx_inflated_variance(.3, v, 1), .3, tolerance = 1e-8)
  expect_equal(approx_inflated_variance(.3, v, 4), .3 + .5 * .75, tolerance = 1e-8)
  expect_lt(approx_inflated_variance(.3, v, .5), .3)
  expect_equal(icc_logit(pi^2 / 3), .5); expect_equal(mor(0), 1)
  expect_error(scale_level1(c(1, -1), c(1, 1)))
})
