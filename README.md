# floating-wind-SLS: Submarine-Landslide-Susceptibility-Constrained Spatial Planning for Floating Offshore Wind

Official code and data for the manuscript:

> **Spatial planning of floating offshore wind constrained by submarine landslide susceptibility: A case study of the Calabrian Margin**  
> Qilin Jiang, Yide Wang, Zihang Fei, Zhuoxing Du, Xiaolei Liu  
> Manuscript submitted to *Ocean Engineering* (citation details to be added after publication).

This repository provides the implementation of a two-stage, susceptibility-constrained spatial planning framework for floating offshore wind (FOW) in the Calabrian Margins. The framework first maps regional submarine landslide susceptibility (SLS) using multiple machine-learning models and SHAP interpretation, and then integrates the resulting susceptibility layer into a hard-constrained and CAE-DEC-based FOW suitability assessment.

The repository includes the model code, data preprocessing and alignment utilities, representative input rasters and sample tables, and selected output tables and maps used in the manuscript.

---

## 1. Overview

The expansion of floating offshore wind into deep and distant waters requires spatial planning that considers not only wind resources and engineering conditions, but also submarine geohazards. This repository implements a regional decision-support framework with two connected workflows:

### Stage A: Regional submarine landslide susceptibility assessment

- An inventory of mapped submarine landslides and nine topographic, geological and seismic predictors are used.
- Eleven machine-learning models are trained and compared under five-fold stratified cross-validation with grid-search hyperparameter optimization:
  - Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, MLP, CNN, RNN and Transformer.
- CatBoost is selected for full-domain susceptibility mapping.
- SHAP is used to quantify model-based feature contributions.
- The susceptibility map is reclassified into five levels and used as the geological-constraint layer in Stage B.

### Stage B: Susceptibility-constrained FOW spatial planning

- Six planning indicators are used:
  - submarine landslide susceptibility level;
  - water depth;
  - distance to shoreline;
  - distance to shipping lanes;
  - mean annual wind speed;
  - wind power density.
- Hard constraints are applied to delineate candidate areas.
- Entropy weighting provides a prior suitability score.
- A convolutional autoencoder combined with deep embedded clustering (CAE-DEC) learns patch-scale multi-factor spatial representations and produces a comprehensive suitability index (CSI).
- Ablation experiments and sensitivity analyses are used to evaluate clustering structure and spatial robustness.

### Key features

- Interpretable multi-model machine-learning framework for regional SLS mapping.
- SHAP-based quantification of predictor contributions.
- Explicit integration of SLS as a geological constraint in FOW spatial planning.
- Patch-based CAE-DEC representation of local multi-factor spatial configurations.
- Entropy-weighted prior constraints and hard-constraint screening.
- Ablation and sensitivity analyses for clustering and spatial stability.
- Reproducible random seeds and command-line interfaces for the main workflows.

---

## 2. Reported performance

The following values are reported in the associated manuscript using the complete experimental dataset and the settings described in the paper. They are provided for reference; re-running the code may produce small numerical differences due to hardware, library versions and non-deterministic GPU operations.

### Stage A: CatBoost SLS model

| Metric | Value |
|---|---:|
| Test AUC | 0.8905 |
| Accuracy | 0.8146 |
| F1-score | 0.7072 |
| Precision | 0.6068 |
| Recall | 0.8475 |

SHAP analysis identified slope gradient, distance to faults and water depth as the three most influential predictors, with normalized mean absolute SHAP contribution shares of 31.3%, 28.7% and 17.7%, respectively; together they account for more than 77% of the total normalized contribution.

### Stage B: CAE-DEC clustering

| Method | Silhouette ↑ | CH ↑ | DBI ↓ |
|---|---:|---:|---:|
| DEC | 0.3735 | 203.12 | 0.9575 |
| PCA-DEC | 0.4660 | 281.28 | 0.7284 |
| **CAE-DEC (Ours)** | **0.6726** | **1016.28** | **0.4828** |

High relative-suitability areas are concentrated mainly along the eastern Ionian margin, where relatively low landslide susceptibility coincides with the prescribed water-depth range and favourable wind-resource conditions.

**Note:** These metrics were obtained with the complete dataset and parameter settings used in the manuscript. The outputs included in this repository are intended to support inspection, code testing and methodological reproduction; they should not be interpreted as a formal validation of project-level engineering feasibility or economic performance.

---

## 3. Repository structure

The repository currently contains two main workflows. The folder names may contain spaces and Chinese characters; please quote paths in shell commands.

```text
floating-wind-SLS/
│
├── CAE-DEC/
│   ├── main.py
│   ├── 输入/
│   │   ├── 水深.tif
│   │   ├── 离岸距离.tif
│   │   ├── 距航道距离.tif
│   │   ├── 风速.tif
│   │   ├── 风功率密度.tif
│   │   └── 易发性评价结果重分类.tif
│   └── 输出/
│       ├── entropy_weights.csv
│       ├── clustering_ablation.csv
│       ├── lambda_sensitivity.csv
│       ├── hyperparam_sensitivity.csv
│       ├── Table2_lambda.csv
│       └── CSI_continuous.tif
│
├── Submarine landslide susceptibility/
│   ├── main.py
│   ├── 输入/
│   │   └── 样本点.csv
│   └── 输出/
│       ├── 1_五张指标对比.csv
│       ├── 2_最优模型结果.csv
│       └── 3_SHAP特征重要性排序.csv
│
├── README.md
└── requirements.txt

    4. Environment and installation
Requirements
Python >= 3.9

PyTorch >= 2.0 (GPU recommended for CAE-DEC)

CUDA 11.8 or a compatible CUDA environment for GPU acceleration

GDAL/rasterio-compatible system environment for GeoTIFF processing

Other dependencies listed below

A minimal requirements.txt:

numpy>=1.23
pandas>=1.5
scipy>=1.9
scikit-learn>=1.2
rasterio>=1.3
torch>=2.0
xgboost>=1.7
lightgbm>=3.3
catboost>=1.2
shap>=0.42

