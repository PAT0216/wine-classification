# Makefile for Wine Classification Project
# ==========================================
# This Makefile automates the entire data analysis pipeline for 
# classifying wines as red or white based on chemical properties.
#
#Used the help of Agentic LLM to create this Makefile to ensure all aspects of the project are covered and to fix bugs.
#
# Authors: Prabuddha Tamhane, Harrison Li, Shihan Xu, Wesley Beard
# Date: 2025-12-12
#
# Usage:
#   make all     - Run the entire analysis pipeline from start to finish
#   make clean   - Remove all generated data and files to start fresh
#   make test    - Run all unit tests with pytest
#
# Pipeline Overview:
#   1. Download raw data from UCI ML Repository
#   2. Validate, split, and transform data 
#   3. Perform exploratory data analysis (EDA)
#   4. Train and evaluate classification models
#   5. Render the final Quarto report
#
# Dependencies:
#   - Python 3.x with packages: pandas, scikit-learn, altair, click, etc.
#   - Quarto CLI for report rendering
#   - See environment.yml for full dependency list

# ==============================================================================
# PHONY TARGETS
# ==============================================================================
# These targets don't correspond to actual files
.PHONY: all clean test data eda analysis report

# ==============================================================================
# VARIABLES
# ==============================================================================
# Paths
RAW_DATA_DIR = data/raw
PROCESSED_DATA_DIR = data/processed
RESULTS_DIR = results
TABLES_DIR = $(RESULTS_DIR)/tables
FIGURES_DIR = $(RESULTS_DIR)/figures
MODELS_DIR = $(RESULTS_DIR)/models
REPORTS_DIR = reports

# Scripts
SCRIPTS_DIR = scripts

# Random seed for reproducibility
SEED = 123

# ==============================================================================
# MAIN TARGETS
# ==============================================================================

# Default target: run entire pipeline
all: report

# Convenience targets for partial runs
data: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
eda: $(TABLES_DIR)/data_summary.csv
analysis: $(TABLES_DIR)/cross_val_results.csv
report: $(REPORTS_DIR)/wine_classifier.html

# Run all tests
test:
	python -m pytest tests/ -v

# ==============================================================================
# DATA PIPELINE
# ==============================================================================

# Step 1: Download raw data from UCI ML Repository
# Output: data/raw/wine-raw.csv
$(RAW_DATA_DIR)/wine-raw.csv:
	python $(SCRIPTS_DIR)/download_data.py \
		--path=$(RAW_DATA_DIR)

# Step 2: Validate, split, and transform data
# Input: data/raw/wine-raw.csv
# Output: data/processed/* (6 CSV files) + results/models/wine_preprocessor.pickle
# Note: Using a single representative target to avoid running the script multiple times
$(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv: $(RAW_DATA_DIR)/wine-raw.csv
	python $(SCRIPTS_DIR)/validate_split_transform.py \
		--raw-path=$(RAW_DATA_DIR) \
		--raw-filename=wine-raw.csv \
		--processed-path=$(PROCESSED_DATA_DIR) \
		--preprocessor-path=$(MODELS_DIR) \
		--seed=$(SEED)

# Other processed files depend on the main target (no recipe needed, just dependencies)
$(PROCESSED_DATA_DIR)/scaled-wine-features-test.csv: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
$(PROCESSED_DATA_DIR)/wine-target-train.csv: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
$(PROCESSED_DATA_DIR)/wine-target-test.csv: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
$(PROCESSED_DATA_DIR)/wine-train.csv: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
$(PROCESSED_DATA_DIR)/wine-test.csv: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
$(MODELS_DIR)/wine_preprocessor.pickle: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv

# ==============================================================================
# EXPLORATORY DATA ANALYSIS
# ==============================================================================

# Step 3: Generate EDA tables and figures
# Input: data/processed/wine-train.csv
# Output: results/tables/data_summary.csv, data_info.csv
#         results/figures/hist_univariate_distributions.png, 
#         distributions_of_features.png, pairwise_correlations.png
$(TABLES_DIR)/data_summary.csv: $(PROCESSED_DATA_DIR)/wine-train.csv
	python $(SCRIPTS_DIR)/eda.py \
		--clean-data=$(PROCESSED_DATA_DIR)/wine-train.csv \
		--plot-to=$(RESULTS_DIR)

# Other EDA outputs depend on the main target  
$(TABLES_DIR)/data_info.csv: $(TABLES_DIR)/data_summary.csv
$(FIGURES_DIR)/hist_univariate_distributions.png: $(TABLES_DIR)/data_summary.csv
$(FIGURES_DIR)/distributions_of_features.png: $(TABLES_DIR)/data_summary.csv
$(FIGURES_DIR)/pairwise_correlations.png: $(TABLES_DIR)/data_summary.csv

# ==============================================================================
# MODEL ANALYSIS
# ==============================================================================

# Step 4: Train models and generate evaluation metrics
# Input: Scaled training/testing features and targets
# Output: results/tables/cross_val_results.csv, test_metrics.csv
#         results/figures/confusion_matrix.png
$(TABLES_DIR)/cross_val_results.csv: $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv \
                                      $(PROCESSED_DATA_DIR)/scaled-wine-features-test.csv \
                                      $(PROCESSED_DATA_DIR)/wine-target-train.csv \
                                      $(PROCESSED_DATA_DIR)/wine-target-test.csv
	python $(SCRIPTS_DIR)/model_analysis.py \
		--train-features=$(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv \
		--train-target=$(PROCESSED_DATA_DIR)/wine-target-train.csv \
		--test-features=$(PROCESSED_DATA_DIR)/scaled-wine-features-test.csv \
		--test-target=$(PROCESSED_DATA_DIR)/wine-target-test.csv \
		--results-dir=$(RESULTS_DIR) \
		--seed=$(SEED)

# Other model analysis outputs depend on the main target
$(TABLES_DIR)/test_metrics.csv: $(TABLES_DIR)/cross_val_results.csv
$(FIGURES_DIR)/confusion_matrix.png: $(TABLES_DIR)/cross_val_results.csv

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

# Step 5: Render the final Quarto report (HTML and PDF)
# Input: All tables and figures from EDA and model analysis
# Output: reports/wine_classifier.html and reports/wine_classifier.pdf
$(REPORTS_DIR)/wine_classifier.html $(REPORTS_DIR)/wine_classifier.pdf: $(TABLES_DIR)/data_summary.csv \
                                      $(TABLES_DIR)/data_info.csv \
                                      $(TABLES_DIR)/cross_val_results.csv \
                                      $(TABLES_DIR)/test_metrics.csv \
                                      $(FIGURES_DIR)/hist_univariate_distributions.png \
                                      $(FIGURES_DIR)/distributions_of_features.png \
                                      $(FIGURES_DIR)/pairwise_correlations.png \
                                      $(FIGURES_DIR)/confusion_matrix.png \
                                      $(REPORTS_DIR)/wine_classifier.qmd
	cd $(REPORTS_DIR) && quarto render wine_classifier.qmd --to html
	cd $(REPORTS_DIR) && quarto render wine_classifier.qmd --to pdf

# ==============================================================================
# CLEAN TARGET
# ==============================================================================

# Remove all generated files to start fresh
# WARNING: This will delete all data, results, and reports!
clean:
	@echo "Cleaning generated files..."
	# Remove raw data
	rm -f $(RAW_DATA_DIR)/wine-raw.csv
	# Remove processed data
	rm -f $(PROCESSED_DATA_DIR)/scaled-wine-features-train.csv
	rm -f $(PROCESSED_DATA_DIR)/scaled-wine-features-test.csv
	rm -f $(PROCESSED_DATA_DIR)/wine-target-train.csv
	rm -f $(PROCESSED_DATA_DIR)/wine-target-test.csv
	rm -f $(PROCESSED_DATA_DIR)/wine-train.csv
	rm -f $(PROCESSED_DATA_DIR)/wine-test.csv
	# Remove model artifacts
	rm -f $(MODELS_DIR)/wine_preprocessor.pickle
	# Remove results tables
	rm -f $(TABLES_DIR)/data_summary.csv
	rm -f $(TABLES_DIR)/data_info.csv
	rm -f $(TABLES_DIR)/cross_val_results.csv
	rm -f $(TABLES_DIR)/test_metrics.csv
	# Remove results figures
	rm -f $(FIGURES_DIR)/hist_univariate_distributions.png
	rm -f $(FIGURES_DIR)/distributions_of_features.png
	rm -f $(FIGURES_DIR)/pairwise_correlations.png
	rm -f $(FIGURES_DIR)/confusion_matrix.png
	# Remove rendered report (keep .qmd source)
	rm -f $(REPORTS_DIR)/wine_classifier.html
	rm -f $(REPORTS_DIR)/wine_classifier.pdf
	rm -rf $(REPORTS_DIR)/wine_classifier_files
	@echo "Clean complete!"
