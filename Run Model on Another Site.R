library(terra)    # modern spatial raster/vector package
library(mlr)      # machine learning framework for ensemble modeling
library(ranger)
library(kernlab)
library(nnet)
library(tidyverse)

data_dir <- "BlackPond/"

#make COVSTACK; stack of covariate rasters
cov_stack <- c(
  rast(paste0(data_dir,"output/tpi20saga.tif")),
  rast(paste0(data_dir,"output/tpi100saga.tif")),
  rast(paste0(data_dir,"output/tpi200saga.tif")),
  rast(paste0(data_dir,"output/mrvbf.tif")),
  rast(paste0(data_dir,"output/hydem5m_TWId.tif")),
  rast(paste0(data_dir,"output/uaab_norm2.tif")),
  rast(paste0(data_dir,"output/EDb.tif"))
)

names(cov_stack) <- c("tpi20", "tpi100", "tpi200", 
                      "valley_flatness", "twi_downslope",
                      "br_catcharea", "EDb")

ensemble_model <- readRDS("soilmodel-Thu Mar 20 12:59:58 2025.rds")

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
writeRaster(prob_rasters, 
            paste0(data_dir,"ensemble_probabilities.tif"), overwrite=TRUE)
writeRaster(class_raster, 
            paste0(data_dir,"ensemble_classified.tif"), overwrite=TRUE)
