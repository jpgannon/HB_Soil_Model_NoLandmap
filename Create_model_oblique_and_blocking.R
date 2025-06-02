library(terra)
library(mlr)
library(ranger)
library(kernlab)
library(nnet)
library(blockCV)
library(dplyr)
library(sf)
library(gstat)
library(caret)

# Load raster covariates
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
                      "br_catcharea", "EDb")

# Load training point data
pts <- vect("Data/HBEF_NEW_biscuit_Pedon_LAST.shp")
pts_df <- extract(cov_stack, pts)
data <- cbind(as.data.frame(pts), pts_df)
data$HPU <- factor(data$hpu)

training_data <- select(data, x = easting, y = northing, class = HPU, tpi20, 
                        tpi100, tpi200, valley_flatness, 
                        twi_downslope, br_catcharea, EDb)

# Rotate coordinates function
rotate_coords <- function(x, y, angle) {
  xr <- x * cos(angle) - y * sin(angle)
  yr <- x * sin(angle) + y * cos(angle)
  cbind(xr, yr)
}

angles <- seq(0, 180, length.out = 14)
angles_rad <- angles * pi/180

for (ang in angles_rad) {
  rot_xy <- rotate_coords(training_data$x, training_data$y, ang)
  ang_deg <- round(ang * 180/pi, 1)
  colnames(rot_xy) <- c(paste0("rX_", ang_deg), paste0("rY_", ang_deg))
  training_data <- cbind(training_data, rot_xy)
}

# Variogram to determine effective range for spatial blocking
training_sf <- st_as_sf(training_data, coords = c("rX_0", "rY_0"), crs = crs(cov_stack))
vgm_empirical <- variogram(as.numeric(class) ~ 1, data = training_sf)
vgm_model <- fit.variogram(vgm_empirical, model = vgm("Exp"))
effective_range <- vgm_model$range[2]

# Spatial blocking using effective range
# Convert training data to sf spatial object for blockCV
training_sf <- st_as_sf(training_data, coords = c("rX_0", "rY_0"), crs = crs(cov_stack))

# Use cv_spatial for spatial cross-validation blocking
# Convert training data to sf spatial object for blockCV
training_sf <- st_as_sf(training_data, coords = c("rX_0", "rY_0"), crs = crs(cov_stack))

# Spatial blocking with cv_spatial using effective range
# Convert to sf object explicitly using original coordinates
training_sf <- st_as_sf(training_data, coords = c("rX_0", "rY_0"), crs = crs(cov_stack))

# Calculate empirical variogram to determine spatial blocking range
vgm_empirical <- variogram(as.numeric(class) ~ 1, data = training_sf)
vgm_model <- fit.variogram(vgm_empirical, model = vgm("Exp"))
effective_range <- vgm_model$range[2]

# Perform spatial blocking using cv_spatial from blockCV
# Ensure 'training_data' has original coordinates for spatial object creation
training_sf <- st_as_sf(training_data, coords = c("rX_0", "rY_0"), crs = crs(cov_stack))

# Calculate empirical variogram and effective spatial range
vgm_empirical <- variogram(as.numeric(class) ~ 1, data = training_sf)
vgm_model <- fit.variogram(vgm_empirical, model = vgm("Exp"))
effective_range <- vgm_model$range[2]

# Perform spatial blocking using effective range
library(blockCV)
set.seed(123)
block <- cv_spatial(
  x = training_sf,
  column = "class",
  size = effective_range,
  k = 5,
  selection = "random"
)

# Extract numeric indices explicitly from the folds returned by cv_spatial
train_idx <- block$folds_list[[1]][[1]]
test_idx <- block$folds_list[[1]][[2]]


# Explicitly subset data using numeric indices
train_set <- training_data[train_idx, ]
test_set <- training_data[test_idx, ]

# Drop any empty factor levels explicitly
train_set$class <- droplevels(train_set$class)
test_set$class <- droplevels(test_set$class)

# Sanity-check that train_set is correctly populated
if(nrow(train_set) == 0) stop("train_set is empty, check data or spatial blocking.")
if(nlevels(train_set$class) < 2) stop("train_set has fewer than 2 classes, cannot proceed.")

# Proceed safely to create mlr task
task <- makeClassifTask(data = train_set, target = "class")


# Base learners
lrn_rf  <- makeLearner("classif.ranger", predict.type = "prob")
lrn_svm <- makeLearner("classif.svm", predict.type = "prob")
lrn_log <- makeLearner("classif.multinom", predict.type = "prob")
base_learners <- list(lrn_rf, lrn_svm, lrn_log)

# Stacked ensemble
ensemble <- makeStackedLearner(base.learners = base_learners, 
                               predict.type  = "prob", 
                               method        = "stack.cv",
                               super.learner = "classif.glmnet")

# Train model
ensemble_model <- train(ensemble, task)
saveRDS(ensemble_model, paste0("soilmodel-", date(),".rds"))

# Predict on test set and generate confusion matrix
ensemble_test_pred <- predict(ensemble_model, newdata = test_set)
conf_matrix <- confusionMatrix(
  factor(ensemble_test_pred$data$response, levels = levels(test_set$class)),
  test_set$class
)
print(conf_matrix)

# Spatial prediction preparation
coords_rast <- as.data.frame(xyFromCell(cov_stack, 1:ncell(cov_stack)))
names(coords_rast) <- c("x", "y")

cov_values <- as.data.frame(cov_stack)
new_data <- cbind(coords_rast, cov_values)

valid_idx <- complete.cases(new_data)
valid_data <- new_data[valid_idx, ]

for (ang in angles_rad) {
  rot_xy <- rotate_coords(valid_data$x, valid_data$y, ang)
  ang_deg <- round(ang * 180/pi, 1)
  colnames(rot_xy) <- c(paste0("rX_", ang_deg), paste0("rY_", ang_deg))
  valid_data <- cbind(valid_data, rot_xy)
}
#valid_data$x <- NULL; valid_data$y <- NULL

# Predict
ensemble_pred <- predict(ensemble_model, newdata = valid_data)
probabilities <- ensemble_pred$data[, -1]
predicted_class <- ensemble_pred$data$response

prob_rasters <- rast(nrows = nrow(cov_stack), ncols = ncol(cov_stack), ext = ext(cov_stack), 
                     crs = crs(cov_stack), nlyrs = ncol(probabilities))
names(prob_rasters) <- colnames(probabilities)

class_raster <- rast(prob_rasters[[1]])
names(class_raster) <- "predicted_class"

prob_matrix_full <- matrix(NA_real_, nrow = ncell(cov_stack), ncol = ncol(probabilities))
class_vector_full <- rep(NA_integer_, ncell(cov_stack))

prob_matrix_full[valid_idx, ] <- apply(as.matrix(probabilities), 2, as.numeric)
class_vector_full[valid_idx] <- as.numeric(predicted_class)

for(i in seq_len(ncol(probabilities))){
  values(prob_rasters[[i]]) <- prob_matrix_full[, i]
}
values(class_raster) <- class_vector_full

writeRaster(prob_rasters, paste0("ensemble_probabilities", date(),".tif"), overwrite=TRUE)
writeRaster(class_raster, paste0("ensemble_classified", date(), ".tif"), overwrite=TRUE)
