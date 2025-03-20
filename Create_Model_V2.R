library(terra)    # modern spatial raster/vector package
library(mlr)      # machine learning framework for ensemble modeling
library(ranger)
library(kernlab)
library(nnet)

# Example: load raster covariates as a SpatRaster
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

# Load training point data (must contain coordinates and class labels)
pts <- vect("Data/HBEF_NEW_biscuit_Pedon_LAST.shp")
pts_df <- extract(cov_stack, pts)#, bind=TRUE)
data <- cbind(as.data.frame(pts), pts_df)
data$HPU <- factor(data$hpu)
training_data <- dplyr::select(data, x = easting, y = northing, class = HPU, tpi20, 
                      tpi100, tpi200, valley_flatness, 
                      twi_downslope, br_catcharea, EDb)

# Function to rotate coordinates by a given angle (in radians) around origin (0,0)
rotate_coords <- function(x, y, angle) {
  xr <- x * cos(angle) - y * sin(angle)
  yr <- x * sin(angle) + y * cos(angle)
  cbind(xr, yr)
}
# Define angles (in radians) for oblique coordinates – 14 angles from 0 to 180 degrees
angles <- seq(0, 180, length.out = 14) 
angles_rad <- angles * pi/180

# Generate oblique coordinate features for each angle
for (ang in angles_rad) {
  rot_xy <- rotate_coords(training_data$x, training_data$y, ang)
  ang_deg <- round(ang * 180/pi, 1)        # angle in degrees (one decimal)
  colnames(rot_xy) <- c(paste0("rX_", ang_deg), paste0("rY_", ang_deg))
  training_data <- cbind(training_data, rot_xy)
}
# Remove original x and y columns (already represented by rX_0 and rY_0)
training_data$x <- NULL
training_data$y <- NULL

# Prepare an mlr classification task (factor target for multi-class or binomial)
task <- makeClassifTask(data = training_data, target = "class")

# Define base learners for the ensemble
lrn_rf  <- makeLearner("classif.ranger",   predict.type = "prob")      # Random Forest
lrn_svm <- makeLearner("classif.svm",      predict.type = "prob")      # Support Vector Machine
lrn_log <- makeLearner("classif.multinom", predict.type = "prob")      # Multinomial Logistic Regression
base_learners <- list(lrn_rf, lrn_svm, lrn_log)

# Define a stacked ensemble learner (using logistic regression as meta-learner)
ensemble <- makeStackedLearner(base.learners = base_learners, 
                               predict.type  = "prob", 
                               method        = "stack.cv",    # cross-validated stacking
                               super.learner = "classif.glmnet")  # penalized logistic regression meta-model

# Train the ensemble model on the training task
ensemble_model <- train(ensemble, task)

saveRDS(ensemble_model, paste0("soilmodel-", date(),".rds"))

####
# Predict ensemble model across entire raster space (cov_stack)
####

# Generate coordinates raster matching cov_stack
coords_rast <- as.data.frame(terra::xyFromCell(cov_stack, 1:ncell(cov_stack)))
names(coords_rast) <- c("x", "y")

# Extract covariate values for all raster cells
cov_values <- as.data.frame(cov_stack)
new_data <- cbind(coords_rast, cov_values)

# Remove rows with any NA values to avoid prediction mismatch
valid_idx <- complete.cases(new_data)
valid_data <- new_data[valid_idx, ]

# Compute oblique coordinates (using the same angles as training)
for (ang in angles_rad) {
  rot_xy <- rotate_coords(valid_data$x, valid_data$y, ang)
  ang_deg <- round(ang * 180/pi, 1)
  colnames(rot_xy) <- c(paste0("rX_", ang_deg), paste0("rY_", ang_deg))
  valid_data <- cbind(valid_data, rot_xy)
}
valid_data$x <- NULL; valid_data$y <- NULL  # remove original coords

# Predict using the trained ensemble model on valid data
ensemble_pred <- predict(ensemble_model, newdata = valid_data)
probabilities <- ensemble_pred$data[, -1]  # probabilities for each class
predicted_class <- ensemble_pred$data$response

# Explicitly define the raster stack as numeric (float)
prob_rasters <- rast(nrows = nrow(cov_stack), 
                     ncols = ncol(cov_stack), 
                     ext = ext(cov_stack), 
                     crs = crs(cov_stack),
                     nlyrs = ncol(probabilities)) 

names(prob_rasters) <- colnames(probabilities)

# Also explicitly numeric classification raster
class_raster <- rast(prob_rasters[[1]])
names(class_raster) <- "predicted_class"

# Ensure predictions match raster cell number:
prob_matrix_full <- matrix(NA_real_, nrow = ncell(cov_stack), ncol = ncol(probabilities))
class_vector_full <- rep(NA_integer_, ncell(cov_stack))

prob_matrix_full[valid_idx, ] <- as.matrix(probabilities)
class_vector_full[valid_idx] <- as.numeric(predicted_class)

# Confirm numeric type explicitly for probabilities
prob_matrix_full <- apply(prob_matrix_full, 2, as.numeric)

# Explicit numeric assignment loop
for(i in seq_len(ncol(probabilities))){
  values(prob_rasters[[i]]) <- prob_matrix_full[, i]
}


# Assign classification raster values (this already works but kept for completeness)
values(class_raster) <- class_vector_full

# Write output rasters
writeRaster(prob_rasters, "ensemble_probabilities.tif", overwrite=TRUE)
writeRaster(class_raster, "ensemble_classified.tif", overwrite=TRUE)
