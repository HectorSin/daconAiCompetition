# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trade data analysis and forecasting project for the Dacon Competition (3rd Kookmin University AI Big Data Analysis Contest):

1. **Task 1: Comovement Detection** - Identify lead-lag relationships between 100 trade items using CCF, Granger Causality, DTW, and FDR correction
2. **Task 2: Trade Volume Forecasting** - Predict future trade volumes using LightGBM with time-series cross-validation

**Data**: 43 months of trade data across 100 items in long format (date, item_code, value)

## Common Commands

```bash
# Environment setup
conda create -n daconai python=3.10 -y
conda activate daconai
pip install -r requirements.txt
python verify_installation.py

# Testing (Windows-specific paths required)
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/ -v
/c/Users/SMART/anaconda3/envs/daconai/python.exe -m pytest tests/test_preprocess.py -v

# Training pipeline
python src/train.py

# Jupyter notebooks for exploration
jupyter lab

# MLflow experiment tracking
mlflow ui  # Access at http://localhost:5000
```

## Architecture

### Pipeline Overview

The codebase follows a standard ML pipeline for time-series forecasting:

1. **Raw Data** (`data/raw/`) → 2. **Preprocessing** (`src/preprocess.py`) → 3. **Feature Engineering** (`src/features.py`) → 4. **Modeling** (`src/train.py`) → 5. **Prediction** (`src/predict.py`)

Parallel to forecasting: **Comovement Detection** (`src/comovement.py`) for Task 1

### Key Modules

**`src/preprocess.py`** - Data transformation and quality checks
- `preprocess_pipeline()` - Main entry point: loads long format CSV, pivots to wide format (dates × items)
- `check_stationarity_all_items()` - Runs ADF & KPSS tests on all 100 items
- `decompose_all_items()` - STL decomposition (trend, seasonal, residual)
- Quality checks: negative values, outliers (IQR), missing values

**`src/comovement.py`** - Lead-lag relationship detection
- `calculate_ccf_matrix()` - Cross-correlation for all item pairs
- `calculate_granger_matrix()` - Granger causality tests (directional)
- `calculate_dtw_matrix()` - Dynamic Time Warping distances
- `apply_fdr_to_results()` - Benjamini-Hochberg correction for 9,900 pairs
- `detect_comovement_comprehensive()` - Runs all three methods with FDR

**`src/train.py`** - Baseline training pipeline
- Simple lag features and rolling statistics
- LightGBM with time-based train/val split (80/20)
- Model saved to `models/dummy_model.pkl`

**`config.py`** - Centralized configuration
- All paths, seeds, model hyperparameters, thresholds
- **CRITICAL**: Always import `Config` class, never hardcode values

### Project Status (from PLAN.md)

- ✅ **Phase 1**: Environment setup and project structure
- ✅ **Phase 2**: Dummy data generation and initial pipeline
- ⚠️ **Phase 3**: EDA and comovement detection (IN PROGRESS - real data uploaded, 12.2% missing, needs aggregation)
- ⏳ **Phase 4**: Feature engineering and full modeling pipeline
- ⏳ **Phase 5**: Hyperparameter tuning with Optuna, MLflow integration

Current test status: 24/25 passing (96%)

## Critical Development Patterns

### Time-Series Constraints (MUST FOLLOW)

- **NO random shuffling** - Always use time-based splits (`train_test_split` with temporal ordering)
- **NO data leakage** - Features must ONLY use past information (e.g., lag features, not future values)
- **Validate stationarity** - Use ADF/KPSS tests before modeling; apply differencing if non-stationary
- **Account for seasonality** - Monthly patterns expected in trade data

### Multiple Testing Correction (Task 1 Critical)

Testing 9,900 item pairs (100 choose 2) requires FDR correction:
- **NEVER use raw p-values** for comovement detection
- **ALWAYS apply** `apply_fdr_to_results()` with `FDR_ALPHA = 0.05` from config
- Without correction, false positive rate will be ~50% instead of 5%

### Code Organization

- **Notebooks** (`notebooks/`) - EDA, visualization, prototyping only
- **Scripts** (`src/`) - All production code with docstrings
  - Import all parameters from `Config` class
  - Add unit tests to `tests/` for new functions
- **Windows paths**: Use full conda env paths for pytest (see commands above)
- **Logs in Korean**: Intentional for competition context, don't translate

## Known Limitations & Warnings

- **Limited data**: Only 43 months → high overfitting risk → be conservative with features
- **Real data issues**: 12.2% missing values, multiple transactions per month require aggregation
- **Validation**: Use `sktime.SlidingWindowSplitter` for time-series CV (planned for Phase 4)
- **Reproducibility**: All random operations must use `Config.SEED = 42`
- **Test paths**: pytest requires full Windows conda paths on this machine
