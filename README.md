# Q-Learning FCS Clustering System for HIV Diagnosis

A reinforcement learning approach to automated HIV status diagnosis using flow cytometry data. This project demonstrates the application of Q-learning to medical diagnostics through clinically-informed feature discretization and two-phase progressive training.

## 🎯 Project Overview

This system uses Q-learning (a reinforcement learning algorithm) to learn optimal clustering strategies for HIV diagnosis based on flow cytometry standard (FCS) data. The approach is unique in that it:

1. **Uses feature-based discretization**: Feature bins are configurable per marker rather than arbitrary quantiles
2. **Employs two-phase progressive training**:
   - Phase 1: Quality optimization on HIV+ samples only
   - Phase 2: Diagnostic refinement on labeled mixed samples
3. **Supports GPU acceleration**: Can use RAPIDS cuML for faster clustering on compatible GPUs
4. **Provides comprehensive visualizations**: Ideal for science fair presentations and educational purposes

## 🔬 Scientific Motivation

Flow cytometry measures channel/header intensities from raw FCS files (e.g., FS Lin, SS Log, IgG1-FITC, CD45-ECD). This system learns clustering strategies from these measured channel patterns rather than relying on fixed clinical thresholds.

Traditional diagnosis uses fixed thresholds, but this system learns optimal clustering strategies through reinforcement learning.

## 📋 Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies:

- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `fcsparser>=0.2.0` - FCS file parsing
- `matplotlib>=3.4.0`, `seaborn>=0.11.0` - Visualization
- `PyYAML>=5.4.0` - Configuration management
- `tqdm>=4.62.0` - Progress bars

### Optional (GPU Acceleration):

- `cuml>=23.0.0` - RAPIDS cuML for GPU clustering
- `cudf>=23.0.0` - RAPIDS cuDF for GPU DataFrames

**Note**: cuML requires NVIDIA GPU with CUDA support. The system automatically falls back to CPU if GPU is unavailable.

## 🚀 Installation

### Standard Installation (CPU)

```bash
# Clone the repository
git clone https://github.com/mabanana/q-table-fcs-clustering.git
cd q-table-fcs-clustering

# Install dependencies
pip install -r requirements.txt
```

### GPU Installation (Optional)

For GPU acceleration with RAPIDS cuML:

```bash
# Install RAPIDS (requires CUDA-enabled GPU)
# See https://rapids.ai/start.html for installation instructions
conda create -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia \
    cuml=23.12 python=3.10 cudatoolkit=11.8

conda activate rapids-23.12
pip install -r requirements.txt
```

## 📊 Data Preparation

Place your FCS files in the appropriate directories:

```
data/
├── positive/           # HIV+ samples for Phase 1 training
│   ├── patient001.fcs
│   ├── patient002.fcs
│   └── ...
├── mixed_training/     # Labeled mixed samples for Phase 2
│   ├── patient010_positive.fcs
│   ├── patient011_negative.fcs
│   └── ...
└── mixed/             # Unlabeled samples for testing
    ├── sample001.fcs
    ├── sample002.fcs
    └── ...
```

**Important**: For labeled training data (Phase 2), filenames must contain "positive" or "negative" to indicate HIV status.

## 📊 Presentation Graphics

Running `--train-full` or `--make-poster-plots` automatically generates the following presentation-ready artifacts under `output/plots/`.

### CLI Options

```bash
# After training: generate all poster-quality plots from a saved Q-table
python main.py --make-poster-plots --q-table output/q_table.pkl

# Also generate a 2D scatter comparison using patient FCS data
python main.py --make-poster-plots --q-table output/q_table.pkl --input data/mixed/
```

### Generated Outputs

| File | Description |
|------|-------------|
| `output/performance_summary.csv` | Performance table: Accuracy, F1, MCC, Mean k for each method |
| `output/plots/performance_summary.png` | Same table rendered as a PNG figure |
| `output/plots/reward_vs_episodes.png` | Combined reward vs episode for Phase 1 (silhouette) & Phase 2 (F1), with phase boundary and moving-average smoothing |
| `output/plots/phase2_f1_vs_episodes.png` | F1 score over Phase 2 episodes with moving-average smoothing |
| `output/plots/q_value_heatmap.png` | Q-table heatmap with `k=2…10` x-axis labels and state-encoding subtitle |
| `output/plots/action_distribution.png` | Bar chart of cluster-count selections with k-value x-axis labels |
| `output/plots/phase1_learning_curve.png` | Phase 1 cumulative & average silhouette reward |
| `output/plots/phase2_learning_curve.png` | Phase 2 cumulative & average F1 reward |
| `output/plots/scatter_comparison.png` | Side-by-side 2D scatter of baseline-k vs Q-chosen-k clustering (requires `--input`) |

### Style

All plots use consistent 10×8 inch figures, 300 DPI, readable fonts, labelled axes, and legends where applicable.  The Q-table heatmap includes a subtitle describing the state encoding (e.g. *"125 states, 3-feature state encoding"*).  The scatter comparison shows two panels — *Baseline (Fixed k)* and *Q-Table* — for the same sample, with colour-coded cluster assignments.

---

## 🎮 Usage

### Complete Two-Phase Training

Train both phases sequentially and generate visualizations:

```bash
python main.py --train-full --episodes 2000 --use-gpu
```

### Phase-by-Phase Training

**Phase 1 Only** (Quality learning on HIV+ data):

```bash
python main.py --train-phase1 --episodes 1000
```

**Phase 2 Only** (Diagnostic refinement - requires Phase 1 Q-table):

```bash
python main.py --train-phase2 --episodes 1000 --q-table output/q_table.pkl
```

### Testing/Inference

Apply the learned policy to unlabeled data:

```bash
python main.py --test --input data/mixed/ --output output/predictions.csv --q-table output/q_table.pkl
```

### Generate Visualizations

Create plots from existing Q-table:

```bash
python main.py --visualize --q-table output/q_table.pkl
```

### Advanced Options

```bash
python main.py --help  # Show all options

# Custom configuration file
python main.py --train-full --config my_config.yaml

# Adjust logging verbosity
python main.py --train-full --log-level DEBUG

# Custom episode counts
python main.py --train-full --episodes 5000
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Data Paths

```yaml
data:
  positive: "data/positive/"
  mixed_training: "data/mixed_training/"
  mixed_test: "data/mixed/"
```

### Feature Discretization Thresholds

```yaml
markers:
  use_for_clustering:
    [
      "FS Lin",
      "SS Log",
      "IgG1-FITC",
      "IgG1-PE",
      "CD45-ECD",
      "IgG1-PC5",
      "IgG1-PC7",
    ]

discretization:
  state_features: ["FS Lin", "SS Log", "CD45-ECD"]
  feature_bins:
    FS Lin: [0, 1, 2, 3, 4, 999999]
    SS Log: [0, 1, 2, 3, 4, 999999]
    CD45-ECD: [0, 1, 2, 3, 4, 999999]
```

**Important**: Marker names in `markers.use_for_clustering`, `discretization.state_features`, and `discretization.feature_bins` must match raw FCS column headers. The loader enforces strict matching and raises an error if requested headers are missing.

### Q-Learning Parameters

```yaml
q_learning:
  epsilon_start: 0.3 # Initial exploration rate
  epsilon_end: 0.05 # Minimum exploration rate
  epsilon_decay: 0.995 # Decay per episode
  learning_rate: 0.1 # Alpha (α)
  discount_factor: 0.9 # Gamma (γ)
  n_clusters_range: [2, 10]
```

### Training Parameters

```yaml
training:
  stage1_episodes: 1000 # Phase 1 iterations
  stage2_episodes: 1000 # Phase 2 iterations
  batch_size: 10 # Samples per iteration
```

## 🧠 How It Works

### 1. Feature Discretization

Continuous flow cytometry measurements are discretized into bins defined per feature (marker). Bin counts are configurable in config.yaml.

### 2. State Space

States encode discretized feature combinations:

- **2 features**: $b^2$ states
- **3 features**: $b^3$ states

Where $b$ is the number of bins per feature (e.g., $b=5$ gives 25 or 125 states).

### 3. Action Space

Actions represent cluster count selection: k ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}

### 4. Q-Learning Algorithm

**Update Rule:**

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

Where:

- s: current state
- a: selected action (cluster count)
- r: reward (silhouette score or F1-score)
- α: learning rate (0.1)
- γ: discount factor (0.9)

**Exploration Strategy:** ε-greedy with decay

- Start: ε = 0.3 (30% random exploration)
- End: ε = 0.05 (5% exploration)
- Decay: multiply by 0.995 each episode

### 5. Two-Phase Progressive Training

**Phase 1: Quality Optimization**

- Data: HIV+ samples only
- Objective: Learn cluster counts that maximize silhouette score
- Reward: Silhouette score (cluster quality metric)
- Purpose: Understand structure of positive samples

**Phase 2: Diagnostic Refinement**

- Data: Mixed labeled samples (HIV+ and HIV-)
- Objective: Fine-tune for diagnostic accuracy
- Reward: F1-score or Matthews Correlation Coefficient
- Purpose: Optimize cluster-to-diagnosis mapping

## 📈 Expected Results

After successful training, you should see:

1. **Learning Curves**: Rewards increasing over iterations
2. **Q-Value Convergence**: Q-table stabilizing with clear policies
3. **Action Distribution**: Preference for certain cluster counts
4. **Diagnostic Performance**:
   - Accuracy: 70-90% (depends on data quality)
   - F1-Score: 0.65-0.85
   - Silhouette Score: 0.3-0.6 (higher = better separation)

## 📊 Output Files

### Generated During Training

```
output/
├── q_table.pkl              # Trained Q-table (can be reloaded)
├── q_table_phase1.pkl       # After Phase 1 only
├── training.log             # Detailed execution log
├── plots/
│   ├── phase1_learning_curve.png
│   ├── phase2_learning_curve.png
│   ├── q_value_heatmap.png
│   ├── action_distribution.png
│   ├── feature_distributions.png
│   └── training_summary.html
└── results.csv              # Predictions on test data
```

### Prediction Output Format

`results.csv` contains:

- `filename`: Original FCS filename
- `encoded_state`: Discrete state representation
- `selected_cluster_count`: Chosen k value
- `assigned_cluster`: Which cluster the sample belongs to
- `predicted_hiv_status`: "positive" or "negative"

## 🧪 Testing

Run unit tests to validate components:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_discretizer.py -v
pytest tests/test_q_learning.py -v

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🛠️ Troubleshooting

### Problem: "No FCS files found"

**Solution**: Ensure FCS files are in correct directories and have `.fcs` extension

### Problem: "cuML not available"

**Solution**: Either install RAPIDS cuML or remove `--use-gpu` flag (will use CPU)

### Problem: "No labeled data found"

**Solution**: For Phase 2, filenames must contain "positive" or "negative"

### Problem: Poor performance

**Solutions**:

- Increase training episodes: `--episodes 5000`
- Adjust learning parameters in `config.yaml`
- Ensure sufficient labeled training data (>50 samples recommended)
- Check FCS file quality and exact header availability (names must match raw FCS columns)

### Problem: "ImportError: fcsparser"

**Solution**: `pip install fcsparser`

## 🎓 Science Fair Presentation Tips

1. **Explain the Clinical Context**: Start with why HIV diagnosis matters and what flow cytometry measures

2. **Demonstrate Reinforcement Learning**: Use visualizations to show how the Q-table evolves during training

3. **Highlight Innovation**: Emphasize clinically-informed discretization vs. arbitrary binning

4. **Show Results**: Display learning curves, confusion matrices, and example predictions

5. **Discuss Trade-offs**: Simplicity (discretization) vs. accuracy, exploration vs. exploitation

6. **Future Work**: Mention potential improvements (deep Q-learning, more markers, larger datasets)

## 📚 References

### Scientific Background

1. **WHO HIV Staging**: [WHO Guidelines](https://www.who.int/hiv/pub/guidelines/HIVstaging150307.pdf)
2. **Flow Cytometry**: Understanding immune cell populations
3. **FlowCAP Challenge**: Community benchmark for flow cytometry analysis

### Technical References

1. **Q-Learning**: Watkins & Dayan (1992) - "Q-learning"
2. **Reinforcement Learning**: Sutton & Barto - "Reinforcement Learning: An Introduction"
3. **FCS Format**: ISAC - Flow Cytometry Standard

## 📝 License

This is an educational project for science fair purposes. The code is provided as-is for learning and demonstration.

## 🤝 Contributing

This is a science fair project, but suggestions and improvements are welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions about this project, please open a GitHub issue.

## 🙏 Acknowledgments

- WHO for HIV staging criteria
- ISAC for FCS standard specification
- RAPIDS team for cuML GPU acceleration
- FlowCAP community for flow cytometry benchmarks

---

**Note**: This system is for educational/research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.
