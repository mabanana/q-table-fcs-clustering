# Data Flow Architecture

## Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  RAW FCS FILES (data/raw/)                                  │
│  ├─ 147.fcs (uncompensated)                                 │
│  ├─ 148.fcs (uncompensated)                                 │
│  └─ 149.fcs (uncompensated)                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  COMPENSATION MATRIX (data/compensation/0001.csv)           │
│  + METADATA (data/compensation/HEUvsUE.csv)                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
          [src/preprocessing.py: Apply Compensation]
          [Uses fcswrite to write compensated FCS files]
          [Adds $COMP='Applied' metadata flag]
                         ↓
          [MetadataRouter: Route by Label]
                         ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    HEU (HIV+)        UE            NA (unlabeled)
        ↓               ↓               ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ data/        │ │ data/        │ │ data/        │
│ positive/    │ │ mixed_       │ │ mixed/       │
│              │ │ training/    │ │              │
│ 147_HEU_     │ │ 147_HEU_     │ │ 149_CPG.fcs  │
│ CPG.fcs      │ │ CPG.fcs      │ │              │
│ (compensated)│ │ 148_UE_      │ │              │
│ $COMP=Applied│ │ PIC.fcs      │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
        ↓               ↓               ↓
┌─────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (main.py)                                │
│  ├─ FCSLoader checks $COMP flag                             │
│  ├─ Phase 1: Quality learning on positive/                  │
│  └─ Phase 2: Diagnostic refinement on mixed_training/       │
└─────────────────────────────────────────────────────────────┘
```

## Compensation Workflow Options

### Option 1: Preprocessing (Recommended)

**Process**: Compensation applied once during preprocessing, written to FCS files

**Advantages**:
- Compensation applied only once (faster for large datasets)
- Intermediate files available for inspection
- Better reproducibility
- Original files preserved in data/raw/

**Command**:
```bash
python preprocess_data.py --raw-dir data/raw/ \
                          --compensation data/compensation/0001.csv \
                          --metadata data/compensation/HEUvsUE.csv
python main.py --train-full --episodes 2000
```

### Option 2: On-the-fly During Training

**Process**: Compensation applied during data loading in training pipeline

**Advantages**:
- Quick experiments without preprocessing
- Easy to test different compensation matrices
- Saves disk space (no intermediate files)

**Command**:
```bash
python main.py --train-full --episodes 2000 \
               --compensation data/compensation/0001.csv \
               --apply-compensation
```

## Key Components

### 1. CompensationMatrixLoader (src/preprocessing.py)
- Loads compensation matrix from CSV
- Validates matrix dimensions and values
- Provides marker names

### 2. FCSPreprocessor (src/preprocessing.py)
- Applies compensation to FCS data
- Writes compensated data using fcswrite library
- Adds $COMP='Applied' metadata flag
- Fallback to copy method if fcswrite unavailable

### 3. FCSLoader (src/fcs_loader.py)
- Loads FCS files with optional compensation
- Checks $COMP flag to avoid double-compensation
- Factory method for loading with compensation matrix
- Integration with FCSPreprocessor for on-the-fly compensation

### 4. MetadataRouter (src/preprocessing.py)
- Routes files based on metadata labels
- Creates organized directory structure
- Adds labels to filenames

## Data Consistency Guarantees

1. **No Double-Compensation**: FCSLoader checks $COMP metadata flag
2. **Metadata Preservation**: Original FCS metadata preserved in output files
3. **Matrix Consistency**: Same compensation matrix used throughout pipeline
4. **Fallback Safety**: Graceful degradation if fcswrite unavailable

## File Metadata Flags

### $COMP Flag
- **'Applied'**: File has been compensated
- **Not set or other value**: File is uncompensated

FCSLoader uses this flag to determine whether to apply compensation during loading.

## Example Data Flow

```
Input: 147.fcs (uncompensated, HEU sample)

Preprocessing:
1. Load: 147.fcs → raw data
2. Apply: compensation matrix → compensated data
3. Write: 147_HEU_CPG.fcs with $COMP='Applied'
4. Route: → data/positive/ and data/mixed_training/

Training:
1. Load: 147_HEU_CPG.fcs
2. Check: $COMP='Applied' → skip compensation
3. Extract: CD4, CD8 markers
4. Discretize: → state bins
5. Train: Q-learning agent
```

## References

- FCS 3.0/3.1 standard: https://isac-net.org/page/Data-Standards
- fcswrite documentation: https://pypi.org/project/fcswrite/
- FlowCap II challenge: http://flowcap.flowsite.org/
