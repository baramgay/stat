# ============================================================
# StatWorkbench R Reference Validation Script
# 동일 고정 데이터로 R 기준값 산출
# ============================================================

options(warn = -1)
suppressPackageStartupMessages({
  library(psych)
  library(irr)
  library(ppcor)
  library(nortest)
  library(BlandAltmanLeh)
  library(pROC)
  library(survival)
  library(MASS)
})

cat("=== R VERSION:", R.version$version.string, "===\n\n")

# ──────────────────────────────────────────────────────────────
# 공통 데이터
# ──────────────────────────────────────────────────────────────
set.seed(42)
x  <- c(2.5, 3.1, 4.0, 3.7, 2.9, 4.5, 3.2, 3.8, 2.7, 4.1)
y  <- c(3.0, 3.5, 4.2, 3.9, 3.1, 4.8, 3.4, 4.0, 2.9, 4.3)
g  <- c(rep("A",5), rep("B",5))
g3 <- c(rep("A",4), rep("B",3), rep("C",3))

# ──────────────────────────────────────────────────────────────
# 1. t-test
# ──────────────────────────────────────────────────────────────
cat("=== 1. T-TESTS ===\n")

# One-sample
t1 <- t.test(x, mu=3.0)
cat(sprintf("[one_sample] t=%.6f  df=%d  p=%.6f  ci_low=%.6f  ci_high=%.6f\n",
            t1$statistic, t1$parameter, t1$p.value,
            t1$conf.int[1], t1$conf.int[2]))

# Independent (Welch)
t2 <- t.test(x, y, var.equal=FALSE)
cat(sprintf("[independent_welch] t=%.6f  df=%.6f  p=%.6f  ci_low=%.6f  ci_high=%.6f\n",
            t2$statistic, t2$parameter, t2$p.value,
            t2$conf.int[1], t2$conf.int[2]))

# Independent (equal variance)
t3 <- t.test(x, y, var.equal=TRUE)
cat(sprintf("[independent_pooled] t=%.6f  df=%.6f  p=%.6f\n",
            t3$statistic, t3$parameter, t3$p.value))

# Paired
t4 <- t.test(x, y, paired=TRUE)
cat(sprintf("[paired] t=%.6f  df=%d  p=%.6f  mean_diff=%.6f\n",
            t4$statistic, t4$parameter, t4$p.value,
            mean(x-y)))

# ──────────────────────────────────────────────────────────────
# 2. One-way ANOVA
# ──────────────────────────────────────────────────────────────
cat("\n=== 2. ONE-WAY ANOVA ===\n")
scores <- c(4,5,6,5,4, 7,8,7,9,8, 5,6,5,4,6)
groups <- factor(c(rep("A",5), rep("B",5), rep("C",5)))
aov_res <- summary(aov(scores ~ groups))[[1]]
cat(sprintf("[anova] F=%.6f  df_between=%d  df_within=%d  p=%.6f\n",
            aov_res[1,"F value"], aov_res[1,"Df"], aov_res[2,"Df"],
            aov_res[1,"Pr(>F)"]))

# Levene (via car)
cat(sprintf("[levene] via bartlett: W=%.6f  p=%.6f\n",
            bartlett.test(scores ~ groups)$statistic,
            bartlett.test(scores ~ groups)$p.value))

# ──────────────────────────────────────────────────────────────
# 3. Correlation (Pearson, Spearman)
# ──────────────────────────────────────────────────────────────
cat("\n=== 3. CORRELATION ===\n")
cr <- cor.test(x, y, method="pearson")
cat(sprintf("[pearson] r=%.6f  t=%.6f  df=%d  p=%.6f  ci_low=%.6f  ci_high=%.6f\n",
            cr$estimate, cr$statistic, cr$parameter, cr$p.value,
            cr$conf.int[1], cr$conf.int[2]))

cs <- cor.test(x, y, method="spearman")
cat(sprintf("[spearman] rho=%.6f  S=%.4f  p=%.6f\n",
            cs$estimate, cs$statistic, cs$p.value))

# ──────────────────────────────────────────────────────────────
# 4. Partial Correlation
# ──────────────────────────────────────────────────────────────
cat("\n=== 4. PARTIAL CORRELATION ===\n")
z <- c(1.2, 2.3, 3.1, 2.8, 1.9, 3.5, 2.2, 3.0, 1.5, 2.7)
pc <- pcor.test(x, y, z)
cat(sprintf("[partial_corr] r=%.6f  t=%.6f  df=%d  p=%.6f\n",
            pc$estimate, pc$statistic, pc$n-2-1, pc$p.value))

# ──────────────────────────────────────────────────────────────
# 5. Linear Regression
# ──────────────────────────────────────────────────────────────
cat("\n=== 5. LINEAR REGRESSION ===\n")
lm_res <- lm(y ~ x)
s <- summary(lm_res)
cat(sprintf("[regression] intercept=%.6f  slope=%.6f  R2=%.6f  adj_R2=%.6f  F=%.6f  p=%.6f\n",
            coef(lm_res)[1], coef(lm_res)[2],
            s$r.squared, s$adj.r.squared,
            s$fstatistic[1],
            pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail=FALSE)))
cat(sprintf("[regression] se_intercept=%.6f  se_slope=%.6f  t_intercept=%.6f  t_slope=%.6f\n",
            s$coefficients[1,2], s$coefficients[2,2],
            s$coefficients[1,3], s$coefficients[2,3]))

# ──────────────────────────────────────────────────────────────
# 6. Logistic Regression
# ──────────────────────────────────────────────────────────────
cat("\n=== 6. LOGISTIC REGRESSION ===\n")
# 완전 분리 방지: 명확한 겹침 구간 포함
bin_y <- c(0,0,1,0,1,1,0,1,1,1)
lrx   <- c(1,2,3,4,5,6,7,8,9,10)
glm_res <- glm(bin_y ~ lrx, family=binomial)
sg <- summary(glm_res)
sg <- summary(glm_res)
cat(sprintf("[logistic] intercept=%.6f  slope=%.6f\n",
            coef(glm_res)[1], coef(glm_res)[2]))
cat(sprintf("[logistic] se_intercept=%.6f  se_slope=%.6f  z_slope=%.6f  p_slope=%.6f\n",
            sg$coefficients[1,2], sg$coefficients[2,2],
            sg$coefficients[2,3], sg$coefficients[2,4]))
cat(sprintf("[logistic] null_deviance=%.6f  resid_deviance=%.6f  AIC=%.6f\n",
            glm_res$null.deviance, glm_res$deviance, AIC(glm_res)))

# ──────────────────────────────────────────────────────────────
# 7. Chi-Square Goodness of Fit
# ──────────────────────────────────────────────────────────────
cat("\n=== 7. CHI-SQUARE GOF ===\n")
obs <- c(25, 20, 15, 30, 10)
chi_res <- chisq.test(obs)
cat(sprintf("[chisq_gof] X2=%.6f  df=%d  p=%.6f\n",
            chi_res$statistic, chi_res$parameter, chi_res$p.value))

obs2 <- c(25, 20, 15, 30, 10)
exp_p <- c(0.2, 0.3, 0.1, 0.25, 0.15)
chi_res2 <- chisq.test(obs2, p=exp_p)
cat(sprintf("[chisq_gof_custom_p] X2=%.6f  df=%d  p=%.6f\n",
            chi_res2$statistic, chi_res2$parameter, chi_res2$p.value))

# ──────────────────────────────────────────────────────────────
# 8. Crosstab (Chi-square + Fisher)
# ──────────────────────────────────────────────────────────────
cat("\n=== 8. CROSSTAB ===\n")
ct <- matrix(c(10,5,3,8), nrow=2)
chi_ct <- chisq.test(ct, correct=FALSE)
cat(sprintf("[chisq_2x2] X2=%.6f  df=%d  p=%.6f\n",
            chi_ct$statistic, chi_ct$parameter, chi_ct$p.value))

fish <- fisher.test(ct)
cat(sprintf("[fisher_2x2] OR=%.6f  p=%.6f  ci_low=%.6f  ci_high=%.6f\n",
            fish$estimate, fish$p.value, fish$conf.int[1], fish$conf.int[2]))

# ──────────────────────────────────────────────────────────────
# 9. Nonparametric Tests
# ──────────────────────────────────────────────────────────────
cat("\n=== 9. NONPARAMETRIC ===\n")
# Mann-Whitney (Wilcoxon rank-sum)
mw <- wilcox.test(x, y, exact=FALSE, correct=FALSE)
cat(sprintf("[mannwhitney] W=%.4f  p=%.6f\n", mw$statistic, mw$p.value))

# Wilcoxon signed-rank
wr <- wilcox.test(x, y, paired=TRUE, exact=FALSE, correct=FALSE)
cat(sprintf("[wilcoxon_paired] V=%.4f  p=%.6f\n", wr$statistic, wr$p.value))

# Kruskal-Wallis
kw <- kruskal.test(scores ~ groups)
cat(sprintf("[kruskal] H=%.6f  df=%d  p=%.6f\n",
            kw$statistic, kw$parameter, kw$p.value))

# ──────────────────────────────────────────────────────────────
# 10. Normality
# ──────────────────────────────────────────────────────────────
cat("\n=== 10. NORMALITY ===\n")
sw <- shapiro.test(x)
cat(sprintf("[shapiro_wilk] W=%.6f  p=%.6f\n", sw$statistic, sw$p.value))

ks <- ks.test(x, "pnorm", mean(x), sd(x))
cat(sprintf("[kolmogorov_smirnov] D=%.6f  p=%.6f\n", ks$statistic, ks$p.value))

# ──────────────────────────────────────────────────────────────
# 11. Reliability (Cronbach's Alpha)
# ──────────────────────────────────────────────────────────────
cat("\n=== 11. RELIABILITY (CRONBACH ALPHA) ===\n")
items <- data.frame(
  q1 = c(4,3,5,4,3,5,4,3,4,5),
  q2 = c(3,4,4,5,3,4,5,3,4,4),
  q3 = c(4,3,5,4,4,5,4,4,4,5),
  q4 = c(3,4,4,4,3,5,4,3,5,4)
)
al <- psych::alpha(items)
cat(sprintf("[cronbach] alpha=%.6f  std_alpha=%.6f\n",
            al$total$raw_alpha, al$total$std.alpha))
cat(sprintf("[cronbach] n_items=%d  n_cases=%d  mean_r=%.6f\n",
            ncol(items), nrow(items), al$total$average_r))

# ──────────────────────────────────────────────────────────────
# 12. ICC (Intraclass Correlation)
# ──────────────────────────────────────────────────────────────
cat("\n=== 12. ICC ===\n")
raters <- data.frame(
  r1 = c(1,2,3,4,5,6,7,8,9,10),
  r2 = c(1,3,2,4,5,5,7,9,8,10),
  r3 = c(2,2,3,3,5,6,8,8,9,10)
)
# Two-way mixed, single measures (ICC3,1) — SPSS default
icc_res <- psych::ICC(raters)
# ICC2 = two-way mixed/random, ICC3 = two-way fixed
cat(sprintf("[icc_twoway_mixed_single] ICC=%.6f  F=%.6f  df1=%d  df2=%d  p=%.6f\n",
            icc_res$results[5,"ICC"], icc_res$results[5,"F"],
            icc_res$results[5,"df1"], icc_res$results[5,"df2"],
            icc_res$results[5,"p"]))
cat(sprintf("[icc_twoway_mixed_single] lower=%.6f  upper=%.6f\n",
            icc_res$results[5,"lower bound"], icc_res$results[5,"upper bound"]))

# Two-way mixed, average measures (ICC3k)
cat(sprintf("[icc_twoway_mixed_average] ICC=%.6f\n",
            icc_res$results[6,"ICC"]))

# ──────────────────────────────────────────────────────────────
# 13. Cohen's Kappa
# ──────────────────────────────────────────────────────────────
cat("\n=== 13. COHEN'S KAPPA ===\n")
r1 <- c(1,1,0,0,2,1,0,1,2,0)
r2 <- c(1,0,0,1,2,1,1,1,2,0)
kap <- irr::kappa2(data.frame(r1,r2))
cat(sprintf("[kappa] kappa=%.6f  z=%.6f  p=%.6f\n",
            kap$value, kap$statistic, kap$p.value))

# Manual verification
n <- length(r1)
po <- sum(r1 == r2) / n
cats <- unique(c(r1, r2))
pe <- sum(sapply(cats, function(c) (sum(r1==c)/n) * (sum(r2==c)/n)))
k_manual <- (po - pe) / (1 - pe)
cat(sprintf("[kappa_manual] po=%.6f  pe=%.6f  kappa=%.6f\n", po, pe, k_manual))

# ──────────────────────────────────────────────────────────────
# 14. Bland-Altman
# ──────────────────────────────────────────────────────────────
cat("\n=== 14. BLAND-ALTMAN ===\n")
m1 <- c(512,430,508,428,500,600,364,380,658,445,432,626)
m2 <- c(525,415,508,432,500,625,460,390,687,432,420,530)
n_ba <- length(m1)
diff_ba <- m1 - m2
mean_diff <- mean(diff_ba)
sd_diff   <- sd(diff_ba)
loa_u <- mean_diff + 1.96 * sd_diff
loa_l <- mean_diff - 1.96 * sd_diff
se_mean <- sd_diff / sqrt(n_ba)
t_crit  <- qt(0.975, df=n_ba-1)
se_loa  <- sqrt(3 * sd_diff^2 / n_ba)
cat(sprintf("[bland_altman] mean_diff=%.6f  sd_diff=%.6f\n", mean_diff, sd_diff))
cat(sprintf("[bland_altman] loa_upper=%.6f  loa_lower=%.6f\n", loa_u, loa_l))
cat(sprintf("[bland_altman] ci_mean=[%.6f, %.6f]\n",
            mean_diff - t_crit*se_mean, mean_diff + t_crit*se_mean))
cat(sprintf("[bland_altman] ci_loa_upper=[%.6f, %.6f]\n",
            loa_u - t_crit*se_loa, loa_u + t_crit*se_loa))
pb_r <- cor.test(diff_ba, (m1+m2)/2)
cat(sprintf("[bland_altman] proportional_bias_r=%.6f  p=%.6f\n",
            pb_r$estimate, pb_r$p.value))

# ──────────────────────────────────────────────────────────────
# 15. ROC Analysis
# ──────────────────────────────────────────────────────────────
cat("\n=== 15. ROC ANALYSIS ===\n")
true_label <- c(1,1,1,1,1,0,0,0,0,0,1,1,0,0,1)
score      <- c(0.9,0.8,0.85,0.7,0.75,0.3,0.2,0.4,0.1,0.35,0.6,0.65,0.5,0.45,0.55)
roc_res <- pROC::roc(true_label, score, quiet=TRUE)
auc_val <- pROC::auc(roc_res)
ci_auc  <- pROC::ci.auc(roc_res, conf.level=0.95, method="delong")
cat(sprintf("[roc] AUC=%.6f  ci_low=%.6f  ci_high=%.6f\n",
            as.numeric(auc_val), ci_auc[1], ci_auc[3]))
cat(sprintf("[roc] n_pos=%d  n_neg=%d\n",
            sum(true_label==1), sum(true_label==0)))

# ──────────────────────────────────────────────────────────────
# 16. Sensitivity / Specificity
# ──────────────────────────────────────────────────────────────
cat("\n=== 16. SENSITIVITY/SPECIFICITY ===\n")
actual    <- c(1,1,1,1,1,0,0,0,0,0,1,0,1,0,1)
predicted <- c(1,1,0,1,1,0,0,1,0,0,1,0,1,1,0)
tp <- sum(actual==1 & predicted==1)
tn <- sum(actual==0 & predicted==0)
fp <- sum(actual==0 & predicted==1)
fn <- sum(actual==1 & predicted==0)
sens <- tp / (tp + fn)
spec <- tn / (tn + fp)
ppv  <- tp / (tp + fp)
npv  <- tn / (tn + fn)
cat(sprintf("[sens_spec] TP=%d  TN=%d  FP=%d  FN=%d\n", tp, tn, fp, fn))
cat(sprintf("[sens_spec] sensitivity=%.6f  specificity=%.6f\n", sens, spec))
cat(sprintf("[sens_spec] PPV=%.6f  NPV=%.6f\n", ppv, npv))
cat(sprintf("[sens_spec] accuracy=%.6f  F1=%.6f\n",
            (tp+tn)/length(actual), 2*tp/(2*tp+fp+fn)))

# ──────────────────────────────────────────────────────────────
# 17. Survival Analysis (Kaplan-Meier + Cox)
# ──────────────────────────────────────────────────────────────
cat("\n=== 17. SURVIVAL ANALYSIS ===\n")
time_  <- c(5,8,11,15,18,20,24,8,12,16,21,25,9,14,19,6,10,13,17,22)
event_ <- c(1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1)
grp_   <- c(rep(1,10), rep(2,10))

# KM (log-rank)
km_fit <- survfit(Surv(time_, event_) ~ 1)
lr_test <- survdiff(Surv(time_, event_) ~ grp_)
cat(sprintf("[kaplan_meier] median_OS=%.4f\n", median(km_fit$time[km_fit$n.event>0])))
cat(sprintf("[logrank] chisq=%.6f  df=1  p=%.6f\n",
            lr_test$chisq, 1 - pchisq(lr_test$chisq, 1)))

# Cox PH
cox_fit <- coxph(Surv(time_, event_) ~ grp_)
sc <- summary(cox_fit)
cat(sprintf("[cox] coef=%.6f  HR=%.6f  se=%.6f  z=%.6f  p=%.6f\n",
            sc$coefficients[1], sc$coefficients[2],
            sc$coefficients[3], sc$coefficients[4], sc$coefficients[5]))

# ──────────────────────────────────────────────────────────────
# 18. Factor Analysis
# ──────────────────────────────────────────────────────────────
cat("\n=== 18. FACTOR ANALYSIS ===\n")
set.seed(1)
# 명확한 2-요인 구조: v1-v3은 factor1, v4-v5는 factor2
n_fa <- 30
f1 <- rnorm(n_fa); f2 <- rnorm(n_fa); err <- matrix(rnorm(n_fa*5, sd=0.3), n_fa, 5)
fa_data <- data.frame(
  v1 = 0.8*f1 + err[,1],
  v2 = 0.75*f1 + err[,2],
  v3 = 0.7*f1 + err[,3],
  v4 = 0.8*f2 + err[,4],
  v5 = 0.75*f2 + err[,5]
)
fa_res <- fa(fa_data, nfactors=2, rotate="varimax", fm="pa")
cat(sprintf("[factor_analysis] n_factors=2  RMSEA=%.6f\n", fa_res$RMSEA[1]))
cat(sprintf("[factor_analysis] SS_loadings_F1=%.6f  SS_loadings_F2=%.6f\n",
            fa_res$Vaccounted[1,1], fa_res$Vaccounted[1,2]))
cat(sprintf("[factor_analysis] prop_var_F1=%.6f  prop_var_F2=%.6f\n",
            fa_res$Vaccounted[2,1], fa_res$Vaccounted[2,2]))
cat(sprintf("[factor_analysis] F1_loadings: %s\n",
            paste(round(fa_res$loadings[,1], 4), collapse=" ")))

# ──────────────────────────────────────────────────────────────
# 19. Descriptive Statistics
# ──────────────────────────────────────────────────────────────
cat("\n=== 19. DESCRIPTIVE STATISTICS ===\n")
d <- c(3,7,2,8,4,9,1,6,5,10,3,7,2,8,4)
cat(sprintf("[descriptive] n=%d  mean=%.6f  sd=%.6f  median=%.6f\n",
            length(d), mean(d), sd(d), median(d)))
cat(sprintf("[descriptive] min=%.4f  max=%.4f  range=%.4f\n",
            min(d), max(d), diff(range(d))))
cat(sprintf("[descriptive] Q1=%.6f  Q3=%.6f  IQR=%.6f\n",
            quantile(d, 0.25), quantile(d, 0.75), IQR(d)))
cat(sprintf("[descriptive] skewness=%.6f  kurtosis=%.6f\n",
            psych::skew(d), psych::kurtosi(d)))
cat(sprintf("[descriptive] se_mean=%.6f\n", sd(d)/sqrt(length(d))))

# ──────────────────────────────────────────────────────────────
# 20. LDA (Discriminant Analysis)
# ──────────────────────────────────────────────────────────────
cat("\n=== 20. DISCRIMINANT ANALYSIS ===\n")
lda_x1 <- c(2.1,2.4,2.2,2.5,2.3, 4.1,4.4,4.2,4.5,4.3)
lda_x2 <- c(3.1,3.3,3.2,3.0,3.4, 5.1,5.3,5.2,5.0,5.4)
lda_g  <- factor(c(rep("A",5), rep("B",5)))
lda_fit <- lda(lda_g ~ lda_x1 + lda_x2)
pred    <- predict(lda_fit)
acc <- mean(pred$class == lda_g)
cat(sprintf("[lda] accuracy=%.6f\n", acc))
cat(sprintf("[lda] ld1_x1=%.6f  ld1_x2=%.6f\n",
            lda_fit$scaling[1,1], lda_fit$scaling[2,1]))

cat("\n=== R VALIDATION COMPLETE ===\n")
