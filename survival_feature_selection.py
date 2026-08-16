library(survival)
library(glmnet)

# ---------- Load training data ----------
train <- read.csv("train.csv", check.names = FALSE)
features <- names(train)[5:ncol(train)]

# ---------- Z-score using training data ----------
train_mean <- sapply(train[, features], mean)
train_sd   <- sapply(train[, features], sd)

train[, features] <- sweep(train[, features], 2, train_mean, "-")
train[, features] <- sweep(train[, features], 2, train_sd, "/")

# ---------- Univariate Cox ----------
uni_p <- sapply(features, function(f) {
  fit <- coxph(Surv(DFS, DFS_censor) ~ train[[f]], data = train)
  summary(fit)$coefficients[1, "Pr(>|z|)"]
})

uni_features <- features[uni_p < 0.05]

# ---------- LASSO-Cox ----------
x <- as.matrix(train[, uni_features])
y <- Surv(train$DFS, train$DFS_censor)

set.seed(100)

cvfit <- cv.glmnet(
  x, y,
  family = "cox",
  alpha = 1,
  nfolds = 10,
  type.measure = "C",
  standardize = FALSE
)

coef_lasso <- as.matrix(coef(cvfit, s = "lambda.min"))

selected_features <- rownames(coef_lasso)[coef_lasso[, 1] != 0]

# ---------- Save selected features ----------
write.csv(
  data.frame(
    Feature = selected_features,
    Coefficient = coef_lasso[selected_features, 1]
  ),
  "selected_features.csv",
  row.names = FALSE
)

write.csv(
  train[, c(1:3, selected_features)],
  "train_selected.csv",
  row.names = FALSE
)
