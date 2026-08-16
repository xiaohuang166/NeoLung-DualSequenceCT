# ---------- Load validation cohorts ----------
internal_validation <- read.csv(
  "internal_validation.csv",
  check.names = FALSE
)

external_validation_1 <- read.csv(
  "external_validation_1.csv",
  check.names = FALSE
)

external_validation_2 <- read.csv(
  "external_validation_2.csv",
  check.names = FALSE
)

external_validation_3 <- read.csv(
  "external_validation_3.csv",
  check.names = FALSE
)


# ---------- Apply training-derived normalization ----------
apply_train_scale <- function(data) {
  data[, features] <- sweep(data[, features], 2, train_mean, "-")
  data[, features] <- sweep(data[, features], 2, train_sd, "/")
  data
}

internal_validation <- apply_train_scale(internal_validation)
external_validation_1 <- apply_train_scale(external_validation_1)
external_validation_2 <- apply_train_scale(external_validation_2)
external_validation_3 <- apply_train_scale(external_validation_3)


# ---------- Save training-selected features ----------
save_validation <- function(data, file) {
  output <- data[, c(1:3, selected_features), drop = FALSE]
  write.csv(output, file, row.names = FALSE)
}

save_validation(
  internal_validation,
  "internal_validation_selected.csv"
)

save_validation(
  external_validation_1,
  "external_validation_1_selected.csv"
)

save_validation(
  external_validation_2,
  "external_validation_2_selected.csv"
)

save_validation(
  external_validation_3,
  "external_validation_3_selected.csv"
)
