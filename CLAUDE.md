# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a trade data analysis and forecasting project for the Dacon Competition (3rd Kookmin University AI Big Data Analysis Contest). The project involves:

1. **Task 1: Comovement Detection** - Identifying lead-lag relationships between 100 trade items using CCF, Granger Causality, DTW, and FDR correction
2. **Task 2: Trade Volume Forecasting** - Predicting future trade volumes using LightGBM with time-series cross-validation

## Development Environment

### Setup Commands

```bash
# Create conda environment
conda create -n daconai python=3.10 -y

# Activate environment
conda activate daconai

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file with verbose output
pytest tests/test_preprocess.py -v
```

### Training Pipeline

```bash
# Generate dummy data first (run notebook)
jupyter lab  # Then run notebooks/00_dummy_data_generator.ipynb

# Run initial training pipeline
python src/train.py
```

### Experiment Tracking

```bash
# Start MLflow UI
mlflow ui
# Access at http://localhost:5000
```

## Architecture & Pipeline

### Data Flow

1. **Raw Data** (`data/raw/`) - Long format CSV with columns: `date`, `item_code`, `value`
   - 43 months of data across 100 trade items
   - Dummy data intentionally includes comovement pairs for testing

2. **Preprocessing** (`src/preprocess.py`) - Transforms data from long to wide format
   - `load_data()` - Loads CSV and converts date column
   - `pivot_data()` - Converts to wide format (dates × items)
   - `check_negative_values()` - Validates non-negative trade values
   - `log_outliers()` - Detects outliers using IQR method
   - `check_missing_values()` - Identifies missing data patterns
   - `preprocess_pipeline()` - Orchestrates full preprocessing workflow

3. **Feature Engineering** (Planned in `src/features.py`)
   - Lag features (1-12 months)
   - Rolling statistics (3, 6, 12 month windows)
   - Growth rates (MoM, YoY)
   - Fourier features for seasonality
   - STL decomposition components (trend, seasonal, residual)
   - Interaction features between related items

4. **Comovement Detection** (Planned in `src/comovement.py`)
   - Cross-Correlation Function (CCF) with max lag of 6 months
   - Granger Causality Test for directional relationships
   - Dynamic Time Warping (DTW) for non-linear patterns
   - False Discovery Rate (FDR/Benjamini-Hochberg) correction for 9,900 pairs

5. **Modeling** (`src/train.py`)
   - Main model: LightGBM with MAE objective
   - Benchmarks: SARIMA, Facebook Prophet
   - Time-series cross-validation using sliding windows (sktime)
   - Metrics: RMSE, MAPE

6. **Model Persistence** - Trained models saved to `models/`

7. **Prediction** (Planned in `src/predict.py`) - Inference script for submissions

### Configuration Management

All settings are centralized in `config.py`:

- **Paths**: `DATA_RAW`, `DATA_PROCESSED`, `MODELS_DIR`, `OUTPUT_DIR`
- **Reproducibility**: `SEED = 42`
- **Validation Strategy**: `VAL_START_DATE`, `VAL_END_DATE`, `PRED_DATE`
- **Model Parameters**: `LGBM_PARAMS` dictionary
- **Comovement Parameters**: `CCF_LAG_MAX`, `CCF_THRESHOLD`, `GRANGER_PVAL_THRESHOLD`, `DTW_THRESHOLD`
- **Feature Engineering**: `LAG_FEATURES`, `ROLLING_WINDOWS`
- **MLflow Settings**: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`

Always import and use `Config` class rather than hardcoding values.

### Project Phases (from PLAN.md)

The project follows a 5-phase implementation plan:

- **Phase 1**: Environment setup and project structure (COMPLETED)
- **Phase 2**: Dummy data generation and initial pipeline (COMPLETED)
- **Phase 3**: EDA, stationarity tests (ADF/KPSS), and comovement detection (IN PROGRESS)
- **Phase 4**: Feature engineering and full modeling pipeline
- **Phase 5**: Hyperparameter tuning with Optuna, MLflow integration, documentation

Refer to `PLAN.md` for detailed checklist and `TECHSPEC_PLAN.md` for technical specifications.

## Key Development Patterns

### Time-Series Considerations

- **Preserve temporal order** - Use time-based splits, not random shuffling
- **Prevent data leakage** - Features must only use past information
- **Stationarity** - Test with ADF/KPSS, apply differencing if needed
- **Seasonality** - Account for monthly patterns in trade data

### Testing Strategy

- Unit tests in `tests/` directory verify individual function correctness
- Focus on testing preprocessing and feature engineering functions
- Use `pytest` for all test execution

### Multiple Testing Correction

When testing 9,900 item pairs for comovement:
- Raw p-values will produce excessive false positives
- Apply FDR (Benjamini-Hochberg) correction using `FDR_ALPHA = 0.05` from config
- This is critical for Task 1 submission validity

### Notebooks vs Scripts

- **Notebooks** (`notebooks/`) - Exploratory analysis, visualization, prototyping
  - `00_dummy_data_generator.ipynb` - Creates synthetic data with intentional patterns
  - `01_eda_and_preprocessing.ipynb` - Stationarity tests, STL decomposition
  - `02_comovement_detection.ipynb` - CCF, Granger, DTW analysis
  - `03_forecasting_model.ipynb` - Model comparison and benchmarking

- **Scripts** (`src/`) - Production code, reusable functions, CI/CD
  - Functions should be well-documented with docstrings
  - Import from `config.py` for all parameters
  - Add corresponding tests in `tests/`

## Commands Reference

### Development Workflow

```bash
# Start interactive development
jupyter lab

# Run training pipeline
python src/train.py

# Run tests
pytest tests/ -v

# Check specific module
pytest tests/test_preprocess.py -v
```

### Data Validation

The preprocessing pipeline includes quality checks that log warnings:
- Negative values (trade volumes should be ≥ 0)
- Outliers beyond IQR × 1.5
- Missing value patterns

These are informational and don't halt execution but should be investigated.

## Important Notes

- **Short time series**: Only 43 months available, risk of overfitting is high
- **Validation strategy**: Use time-series CV (sliding windows), not random splits
- **Feature count**: Be conservative with feature engineering given limited data
- **Reproducibility**: Always set random seeds using `Config.SEED`
- **Korean comments**: Some docstrings and logs are in Korean - this is intentional for the Korean competition context
