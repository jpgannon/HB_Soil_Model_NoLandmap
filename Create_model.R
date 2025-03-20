# Ensemble ML (Random Forest + SVM + Multinomial) for Soil Type Prediction

library(terra)
library(randomForest)
library(e1071)
library(nnet)
library(tidyverse)

# 1. Load covariate rasters
#originally included: tpi20, tpi100, tpi200, mrvbf, hbuaab, EDb, twid
cov_stack <- c(
  rast("output/tpi20saga.tif"),
  rast("output/tpi100saga.tif"),
  rast("output/tpi200saga.tif"),
  rast("output/mrvbf.tif"),
  rast("output/hydem5m_TWId.tif"),
  rast("output/uaab_norm2.tif"),
  rast("output/EDb.tif")
)

names(cov_stack) <- c("tpi20", "tpi100", "tpi200", 
                      "valley_flatness", "twi_downslope",
                      "br_catcharea", "EDb"
                      )

# 2. Load soil observations
pts <- vect("Data/HBEF_NEW_biscuit_Pedon_LAST.shp")
pts_df <- extract(cov_stack, pts)#, bind=TRUE)
data <- cbind(as.data.frame(pts), pts_df)
data$HPU <- factor(data$hpu)
data <- dplyr::select(data, HPU, tpi20, 
                      tpi100, tpi200, valley_flatness, 
                      twi_downslope, br_catcharea, EDb)

tm_shape(rast("output/tpi100saga.tif"))+
  tm_raster(style = "cont", palette = "viridis", title = "TPI 100") +
  tm_shape(pts) +
  tm_symbols(col = "red", size = 0.3, shape = 19)

# 3. Train-test split
set.seed(123)
#train_idx <- unlist(lapply(split(seq_len(nrow(data)), data$HPU), 
#                          function(x) sample(x, size = 0.8 * length(x))))
#train_set <- data[train_idx, ]
#test_set <- data[-train_idx, ]
train_set <- data
test_set <- data

# 4. Train Models
# Random Forest
rf_model <- randomForest(HPU ~ ., data=train_set, ntree=500)

# SVM
svm_model <- svm(HPU ~ ., data=train_set, probability=TRUE)

# Multinomial Logistic Regression
multinom_model <- multinom(HPU ~ ., data=train_set, trace=FALSE)

# 5. Spatial predictions
# Predict RF
rf_probs <- predict(cov_stack, rf_model, type="prob")

# Predict SVM (custom function)
# Corrected SVM prediction function for terra::predict
svm_pred_fun <- function(model, data) {
  pred <- attr(predict(model, data, probability=TRUE), "probabilities")
  # Ensure predictions are returned as a matrix with rows matching input data rows
  if (is.vector(pred)) {
    pred <- matrix(pred, nrow=1)
  }
  return(pred)
}

# Perform prediction
svm_probs <- predict(cov_stack, svm_model, fun=svm_pred_fun, na.rm=TRUE)

# Predict Multinomial
multinom_pred_fun <- function(model, data) {
  pred <- predict(model, newdata=data, type="probs")
  return(pred)
}
multinom_probs <- predict(cov_stack, multinom_model, fun=multinom_pred_fun)

# 6. Ensemble averaging
ensemble_probs <- (rf_probs + svm_probs + multinom_probs) / 3
#names(ensemble_probs) <- paste0("HPU", 1:8)

# 7. Final categorical map (most probable HPU)
ensemble_class <- which.max(ensemble_probs)

# Assign descriptive labels
#levels(ensemble_class) <- data.frame(ID=1:8, 
  #                                   HPU=paste("HPU", 1:8))

# 8. Save Outputs
writeRaster(ensemble_class, "ensemble_HPU_class.tif", overwrite=TRUE)
writeRaster(ensemble_probs, "ensemble_HPU_probs.tif", overwrite=TRUE)

# 9. Validate ensemble on test data
rf_test_probs <- predict(rf_model, test_set, type="prob")
svm_test_probs <- attr(predict(svm_model, test_set, probability=TRUE), "probabilities")
multinom_test_probs <- predict(multinom_model, test_set, type="probs")

ensemble_test_probs <- (rf_test_probs + svm_test_probs + multinom_test_probs) / 3
ensemble_test_pred <- apply(ensemble_test_probs, 1, function(x) levels(data$HPU)[which.max(x)])

cm <- table(Observed=test_set$HPU, Predicted=ensemble_test_pred)
accuracy <- sum(diag(cm)) / sum(cm)
print(cm)
cat("Ensemble Test Accuracy:", round(accuracy*100, 2), "%\n")
