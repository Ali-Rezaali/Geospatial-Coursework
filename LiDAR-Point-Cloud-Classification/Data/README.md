# Data Directory

## Structure
- `sample_points/`: Contains the raw `.las` files used for training and testing.

## Required Files
Due to privacy and size constraints, the original `.las` files are not committed to this repository.
To run the code, place the following files in `sample_points/`:
- `nonground_training.las` (Main dataset)
- `BuildingGableRoof_Train.las`
- `BuildingFlatRoof_Train.las`
- `Vegetation_Train.las`
- `Ground_Train.las`

## Notes
- The scripts in `src/` expect this exact folder structure.
- Ensure the `.las` files are correctly georeferenced and aligned before running.