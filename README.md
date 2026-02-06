# Q-Learning FCS Clustering System for HIV Diagnosis

A reinforcement learning approach to automated HIV status diagnosis using flow cytometry data. This project demonstrates the application of Q-learning to medical diagnostics through clinically-informed feature discretization and two-phase progressive training.

## 🎯 Project Overview

This system uses Q-learning (a reinforcement learning algorithm) to learn optimal clustering strategies for HIV diagnosis based on flow cytometry standard (FCS) data. The approach is unique in that it:

1. **Uses clinically-informed discretization**: Feature bins are based on WHO HIV staging criteria rather than arbitrary quantiles
2. **Employs two-phase progressive training**: 
   - Phase 1: Quality optimization on HIV+ samples only
   - Phase 2: Diagnostic refinement on labeled mixed samples
3. **Supports GPU acceleration**: Can use RAPIDS cuML for faster clustering on compatible GPUs
4. **Provides comprehensive visualizations**: Ideal for science fair presentations and educational purposes

## 🔬 Scientific Motivation

Flow cytometry measures cell surface markers (CD4, CD8, CD3, HLA-DR, etc.) that are crucial for immune system health. HIV specifically attacks CD4+ T cells, leading to:
- **Decreased CD4 counts**: WHO uses CD4 thresholds to stage HIV disease
- **Inverted CD4/CD8 ratio**: Healthy individuals have ratios >1.0; HIV patients often <1.0
- **Immune activation**: Markers like HLA-DR indicate chronic immune activation

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

### Clinical Discretization Thresholds
```yaml
discretization:
  cd4_bins: [0, 200, 350, 500, 999999]  # WHO HIV staging
  cd4_cd8_ratio_bins: [0, 1.0, 1.5, 2.5, 999999]
  activation_bins: [0, 500, 1500, 3000, 999999]
```

### Q-Learning Parameters
```yaml
q_learning:
  epsilon_start: 0.3      # Initial exploration rate
  epsilon_end: 0.05       # Minimum exploration rate
  epsilon_decay: 0.995    # Decay per episode
  learning_rate: 0.1      # Alpha (α)
  discount_factor: 0.9    # Gamma (γ)
  n_clusters_range: [2, 10]
```

### Training Parameters
```yaml
training:
  stage1_episodes: 1000   # Phase 1 iterations
  stage2_episodes: 1000   # Phase 2 iterations
  batch_size: 10          # Samples per iteration
```

## 🧠 How It Works

### 1. Clinically-Informed Discretization

Continuous flow cytometry measurements are discretized into bins based on clinical criteria:

**CD4 Count** (WHO HIV Staging):
- Bin 0: <200 cells/μL (Severe immunodeficiency)
- Bin 1: 200-350 (Advanced immunodeficiency)
- Bin 2: 350-500 (Mild immunodeficiency)
- Bin 3: >500 (Normal)

**CD4/CD8 Ratio** (Clinical Significance):
- Bin 0: <1.0 (Inverted - immunocompromised)
- Bin 1: 1.0-1.5 (Low)
- Bin 2: 1.5-2.5 (Normal)
- Bin 3: >2.5 (High)

**Activation Markers** (HLA-DR):
- Bin 0: <500 (Low)
- Bin 1: 500-1500 (Moderate)
- Bin 2: 1500-3000 (High)
- Bin 3: >3000 (Very high)

### 2. State Space

States encode discretized feature combinations:
- **2 features** (CD4 + CD4/CD8): 4×4 = 16 states
- **3 features** (+ activation): 4×4×4 = 64 states

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
- Check FCS file quality and marker availability

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
