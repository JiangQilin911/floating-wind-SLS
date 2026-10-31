# floating-wind-SLS: Submarine-Landslide-Susceptibility-Constrained Spatial Planning for Floating Offshore Wind

Official code and data for the manuscript:

> **Spatial planning of floating offshore wind constrained by submarine landslide susceptibility: A case study of the Calabrian Margin**  
> Qilin Jiang, Yide Wang, Zihang Fei, Zhuoxing Du, Xiaolei Liu  
> Manuscript submitted to *Ocean Engineering* (citation details to be added after publication).

This repository provides the implementation of a two-stage, susceptibility-constrained spatial planning framework for floating offshore wind (FOW) in the Calabrian Margins. The framework first maps regional submarine landslide susceptibility (SLS) using multiple machine-learning models and SHAP interpretation, and then integrates the resulting susceptibility layer into a hard-constrained and CAE-DEC-based FOW suitability assessment.

The repository includes the model code, data preprocessing and alignment utilities, representative input rasters and sample tables, and selected output tables and maps used in the manuscript.

---

## 1. Overview

The expansion of floating offshore wind into deep and distant waters requires spatial planning that considers not only wind resources and engineering conditions, but also submarine geohazards. This repository implements a regional decision-support framework with two connected workflows.

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
```

---

## 4. Environment and installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0 (GPU recommended for CAE-DEC)
- CUDA 11.8 or a compatible CUDA environment for GPU acceleration
- GDAL/rasterio-compatible system environment for GeoTIFF processing
- Other dependencies listed below

A minimal `requirements.txt`:

```text
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
```

### Installation

```bash
git clone https://github.com/JiangQilin911/floating-wind-SLS.git
cd floating-wind-SLS
```

Create and activate a virtual environment (optional but recommended):

```bash
conda create -n floating-wind-sls python=3.10 -y
conda activate floating-wind-sls
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Please install a PyTorch version compatible with your operating system, GPU and CUDA environment if it is not already available.

---

## 5. Data availability and preparation

### 5.1 Data included in this repository

The repository provides the derived raster layers and sample tables used by the two workflows, together with selected output tables and a continuous CSI raster.

The main input files are:

| File | Description |
|---|---|
| `样本点.csv` | Training samples for SLS modelling, including label and nine reclassified predictors |
| `水深.tif` | Water depth |
| `离岸距离.tif` | Distance to shoreline |
| `距航道距离.tif` | Distance to shipping lanes |
| `风速.tif` | Mean annual wind speed |
| `风功率密度.tif` | Wind power density |
| `易发性评价结果重分类.tif` | Reclassified SLS level (1–5) |

The SLS sample table contains the following columns:

```text
OID, OBJECTID, CID, label, 水深, 坡度, 坡向, 曲率, 距断层距离,
地形起伏度, 地面粗糙度, 沉积物类型, PGA
```

where `label = 1` indicates a landslide sample and `label = 0` indicates a non-landslide sample.

### 5.2 Data sources

The predictor layers and planning indicators were derived from public datasets and regional inventories, including:

- EMODnet Bathymetry for water depth and bathymetric derivatives;
- GEM Global Active Faults Database for distance to active faults;
- EMODnet Geology for sediment type;
- USGS earthquake catalogue and the UMA05 ground-motion prediction model for PGA;
- MaGIC submarine landslide inventory for the Calabrian continental margins;
- NOAA Digital Coast for distance to shoreline;
- EMODnet Human Activities / EMSA route density for distance to shipping lanes;
- Global Wind Atlas for mean annual wind speed and wind power density.

Users should obtain original third-party datasets from their respective providers and comply with the corresponding data licences. The repository does not grant permission to redistribute original third-party survey data beyond the terms of their providers.

### 5.3 Input preprocessing and alignment

The CAE-DEC workflow reads the six planning GeoTIFFs and aligns them to a common grid:

- target resolution: **100 m**;
- target CRS: **EPSG:32633**;
- resampling: bilinear for continuous layers, nearest for the reclassified susceptibility layer;
- water depth is used as absolute depth;
- distances in metres are converted to kilometres in the hard-constraint screening.

Hard constraints used in the manuscript and implemented in `CAE-DEC/main.py` are:

| Indicator | Hard constraint |
|---|---:|
| Submarine landslide susceptibility level | ≤ 3 |
| Water depth | 50–1000 m |
| Distance to shoreline | ≥ 12 km |
| Distance to shipping lanes | ≥ 5 km |
| Mean annual wind speed | ≥ 6.0 m/s |
| Wind power density | ≥ 200 W/m² |

The thresholds are regional screening criteria rather than universal engineering limits.

For custom datasets, all input rasters should be spatially co-registered and should follow the same channel definitions and file naming conventions, or the file list in `CAE-DEC/main.py` should be modified accordingly.

---

## 6. Running the code

### 6.1 Stage A: Submarine landslide susceptibility assessment

```bash
cd "Submarine landslide susceptibility"
python main.py \
  --input_csv "输入/样本点.csv" \
  --output_dir "输出" \
  --encoding gbk
```

If the CSV is encoded as UTF-8, use:

```bash
python main.py --input_csv "输入/样本点.csv" --output_dir "输出" --encoding utf-8
```

The script trains and compares eleven machine-learning models using five-fold stratified cross-validation and grid search. The best model is selected by mean cross-validation AUC, and SHAP analysis is performed on the selected model.

Outputs:

| File | Content |
|---|---|
| `1_五张指标对比.csv` | AUC, accuracy, F1, precision and recall for all models |
| `2_最优模型结果.csv` | Full-sample susceptibility probability predicted by the best model |
| `3_SHAP特征重要性排序.csv` | Mean absolute SHAP feature importance ranking |

### 6.2 Stage B: Susceptibility-constrained FOW spatial planning

```bash
cd "CAE-DEC"
python main.py \
  --tif_dir "输入" \
  --save_dir "输出" \
  --sensitivity balanced
```

The `--sensitivity` argument controls the ablation and sensitivity workload:

| Option | Description |
|---|---|
| `off` | Run the main CAE-DEC workflow only; no sensitivity analysis |
| `fast` | Reduced sensitivity settings |
| `balanced` | Default settings used for the reported sensitivity tables |
| `full` | Extended sensitivity settings; computationally expensive |

The CAE-DEC script uses CUDA if available. To select a specific GPU:

**Linux / macOS:**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --tif_dir "输入" --save_dir "输出" --sensitivity balanced
```

**Windows (PowerShell):**

```powershell
$env:CUDA_VISIBLE_DEVICES=0
python main.py --tif_dir "输入" --save_dir "输出" --sensitivity balanced
```

Outputs:

| File | Content |
|---|---|
| `entropy_weights.csv` | Entropy weights of the six planning indicators |
| `clustering_ablation.csv` | Clustering metrics for DEC, PCA-DEC and CAE-DEC |
| `lambda_sensitivity.csv` | Sensitivity results for the prior-constraint coefficient λ |
| `hyperparam_sensitivity.csv` | Sensitivity results for cluster number K and sharpness coefficient γ |
| `Table2_lambda.csv` | Formatted λ-sensitivity table for the manuscript |
| `CSI_continuous.tif` | Continuous comprehensive suitability index (CSI) raster |

The continuous CSI map can be classified into low, moderate and high relative suitability using the Jenks natural-breaks method, following the manuscript.

### 6.3 Custom paths and local configuration

The scripts contain default paths that reflect the authors' local environment. For a new environment, either pass the command-line arguments shown above or edit the default paths in the corresponding `main.py`.

For paths containing spaces or Chinese characters, always quote the path in the shell:

```bash
python main.py --tif_dir "CAE-DEC/输入" --save_dir "CAE-DEC/输出"
```

---

## 7. Method summary

### 7.1 SLS modelling

- Nine initial predictors: water depth, slope gradient, slope aspect, curvature, distance to faults, topographic relief, seafloor roughness, sediment type and PGA.
- Strongly correlated terrain predictors are removed; slope gradient is retained as the representative terrain predictor.
- Eleven models are compared under a unified five-fold stratified cross-validation and grid-search protocol.
- CatBoost is selected for full-domain prediction.
- SHAP is used to interpret global feature contributions.

### 7.2 FOW suitability assessment

- Hard constraints screen out unsuitable candidate areas.
- Entropy weighting provides objective prior scores for the six soft indicators.
- Patch extraction uses overlapping windows; pixels outside the hard-constraint candidate area are filled with valid-patch means to preserve complete spatial structure for the CAE.
- CAE-DEC learns a 32-dimensional latent representation of six-channel patches and performs soft clustering.
- Cluster-level suitability anchors are combined with soft cluster memberships to compute CSI.
- Ablation experiments compare DEC, PCA-DEC and CAE-DEC.
- Sensitivity analyses perturb λ, K and γ and evaluate IoU, Moran's I, CH and silhouette metrics.

---

## 8. Output interpretation and limitations

The CSI represents **relative spatial development suitability within the study area** under the adopted indicators, hard constraints and prior rules. It does not represent a final engineering approval, an economically optimal project location, or a project-level failure probability.

The SLS map is a regional statistical susceptibility map derived from the available inventory and predictor set. It should be used for early-stage regional screening and should be followed by high-resolution geophysical surveys, geotechnical investigations and site-specific numerical modelling.

The CAE-DEC clustering metrics measure intrinsic latent-space separation and compactness. They do not measure engineering prediction accuracy, energy yield or economic performance.

The planning framework does not explicitly model turbine power curves, capacity factors, wake losses, mooring-system design, submarine-cable routing or onshore grid-connection costs. These should be incorporated in project-level studies.

---

## 9. Reproducibility

- Random seed: `42` for Python, NumPy and PyTorch where applicable.
- The SLS workflow uses stratified five-fold cross-validation and grid search.
- The CAE-DEC workflow uses the same random seed for the baseline model and sensitivity runs unless otherwise specified.
- GPU operations may introduce small non-deterministic differences.
- Library versions, CUDA/cuDNN versions and hardware can affect numerical results.

To maximise reproducibility, record the Python, PyTorch, CUDA, scikit-learn, XGBoost, LightGBM, CatBoost and rasterio versions used in your run.

---

## 10. Citation

If you use this code or data in your research, please cite the associated manuscript:

```bibtex
@article{Jiang2026floatingwindSLS,
  title   = {Spatial planning of floating offshore wind constrained by submarine landslide susceptibility: A case study of the Calabrian Margin},
  author  = {Jiang, Qilin and Wang, Yide and Fei, Zihang and Du, Zhuoxing and Liu, Xiaolei},
  journal = {Ocean Engineering},
  year    = {2026},
  note    = {Manuscript submitted; citation details to be added after publication}
}
```

Please also cite the original data providers where appropriate, including EMODnet, GEM, USGS, Global Wind Atlas, EMSA/EMODnet Human Activities, NOAA and the MaGIC submarine landslide database.

---

## 11. License

No open-source license has been specified for this repository yet. Please add a `LICENSE` file before public release. Until a license is provided, default copyright applies.

The source-code license, if added, does not automatically grant permission to redistribute original third-party survey data or derived data products. Users should independently verify the applicable data-use restrictions before redistributing or reusing the data.

---

## 12. Contact

For questions about the code, data format or reproducibility, please open an issue in this repository:

https://github.com/JiangQilin911/floating-wind-SLS

For research correspondence, please contact the corresponding author:

**Xiaolei Liu** — xiaolei@ouc.edu.cn

---

## Acknowledgements

This work is a contribution to the United Nations Decade of Ocean Science for Sustainable Development (2021–2030) through the endorsed Ocean Decade Project "Submarine Geological Hazards and Digitization" (No. 57.8). Funding information is provided in the associated manuscript.
