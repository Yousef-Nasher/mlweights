# Cross-check of zimbabwe_mr27.py with a second program (R package WeMix).
# Needs: haven, WeMix, and the mlweights R package. Set DHS_DATA_DIR.
suppressMessages({library(haven); library(WeMix); library(mlweights)})
dir <- Sys.getenv("DHS_DATA_DIR", "data")
d <- read_dta(file.path(dir, "ZWIR72FL.DTA"),
              col_select = c(v001, v005, v012, v022, v025, v106, v213, v218, v313, v501, v605))
d <- as.data.frame(zap_labels(d))
d <- subset(d, v501 == 1 & !(v213 == 1 | v605 == 7) & v005 > 0)
d$y <- as.numeric(d$v313 == 3)
d$age <- d$v012
d$secondary <- ifelse(d$v106 %in% 2:3, 1, ifelse(d$v106 %in% 0:1, 0, NA))
d$urban <- as.numeric(d$v025 == 1)
d$children <- factor(pmin(d$v218, 4))
d <- d[complete.cases(d[, c("y", "age", "secondary", "urban", "children")]), ]
A_h <- setNames(c(673, 3340, 162, 2451, 463, 2843, 839, 2298, 165, 1343, 218, 1280, 981, 2230, 372, 2907, 4920, 198, 1682), 1:19)
M_h <- setNames(c(108, 102, 98, 100, 99, 98, 101, 100, 113, 106, 97, 99, 99, 99, 100, 101, 104, 144, 98), 1:19)
w <- dhs_level_weights(d$v005, d$v022, d$v001, A_h, M_h, S_h = 28, M_total = 2976119, m_c = 10534, alpha = 0.5)
d <- cbind(d, w); d$one <- 1
f <- y ~ age + secondary + urban + children + (1 | v001)
fit <- function(w1, w2) {
  d$a <- d[[w1]]; d$b <- d[[w2]]
  m <- mix(f, data = d, weights = c("a", "b"), cWeights = TRUE, family = binomial(link = "logit"), nQuad = 15)
  c(m$coef, sigma2 = unname(m$vars[1]))
}
res <- cbind(as_published = fit("w1", "w2"), level1_rescaled = fit("w1_scaled", "w2"),
             final_weight_level2 = fit("one", "d"), unweighted = fit("one", "one"))
print(round(res, 3))
dir.create("results", showWarnings = FALSE)
write.csv(round(res, 4), "results/zimbabwe_mr27_wemix.csv")
