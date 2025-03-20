library(terra)

goodmodel <- rast("modelout2024-02-21.tif")
newmodel <- rast("ensemble_classified.tif")

diffs <- goodmodel - newmodel
is.na(diffs[diffs == 0]) <- TRUE

plot(diffs)
