# Data Directory

This directory contains subdirectories for FCS (Flow Cytometry Standard) files used in the Q-learning clustering project for HIV diagnosis.

## Data Preprocessing

### Raw Data Structure

Before preprocessing, organize your data as follows:

```
data/
├── compensation/
│   ├── 0001.csv              # Compensation matrix (spillover matrix)
│   ├── HEUvsUE.csv           # Full dataset metadata
│   └── HEUvsUETraining.csv   # Training split metadata
└── raw/
    ├── 1.fcs
    ├── 2.fcs
    ├── 3.fcs
    └── ... (all raw FCS files)
```

### Running Preprocessing

```bash
# Basic preprocessing with default settings
python preprocess_data.py

# Custom paths and multiple metadata files
python preprocess_data.py \
  --raw-dir data/raw/ \
  --compensation data/compensation/0001.csv \
  --metadata data/compensation/HEUvsUE.csv data/compensation/HEUvsUETraining.csv \
  --output-base data/

# Dry run to preview operations
python preprocess_data.py --dry-run

# Overwrite existing processed files
python preprocess_data.py --overwrite
```

### What Preprocessing Does

1. **Loads compensation matrix** from CSV file (e.g., 0001.csv)
2. **Applies spectral compensation** to correct fluorophore spillover
3. **Reads metadata** to determine sample labels (HEU, UE, or NA)
4. **Routes files** to appropriate directories:
   - HEU samples → `/data/positive/` (for Phase 1 training)
   - HEU/UE samples → `/data/mixed_training/` (for Phase 2 training)
   - NA samples → `/data/mixed/` (for final testing)
5. **Renames files** to include label and stimulation (e.g., `147.fcs` → `147_UE_CPG.fcs`)

**Note:** The current implementation copies FCS files and saves compensated data to CSV format due to limitations in FCS writing libraries. For production use with proper FCS format output, consider using FlowKit or fcswrite libraries.

### Compensation Matrix Format

The compensation matrix CSV should have:
- First row: empty cell, then marker names
- Subsequent rows: marker name, then spillover coefficients

Example (0001.csv):
```csv
"","FITC.A","PE.A","PerCP.Cy5.5.A","PE.Cy7.A","APC.A","APC.Cy7.A","Pacific.Blue.A","Alex.700.A"
"FITC-A",1,0.3937,0.01775,0.00288,0.00056,-0.00053,0.00249,0.00016
"PE-A",0.00337,1,0.04413,0.00616,0.00014,0.00040,-0.00041,-0.00009
...
```

### Metadata Format

Metadata CSV files should have these columns:
- `FCSFileName`: Numeric ID matching raw FCS filename (e.g., "147" for "147.fcs")
- `Label`: Sample classification ("HEU", "UE", or "NA")
- `Stimulation`: TLR stimulation condition ("unstim", "PAM", "LPS", "R848", "PG", "PIC", "CPG")
- `SampleNumber`: Subject/patient identifier

Example (HEUvsUE.csv):
```csv
FCSFileName,Label,Stimulation,SampleNumber
147,UE,CPG,1
146,UE,PIC,1
63,HEU,CPG,3
...
```

## Subdirectories

### `/data/positive/`
Contains FCS files from HIV positive patients only. These files are used for initial training and understanding the characteristics of positive samples.

### `/data/mixed_training/`
Contains prelabeled FCS files (both positive and negative) used for supervised training of the machine learning model. Each file should be properly labeled to indicate whether it's from a positive or negative patient.

### `/data/mixed/`
Contains unlabeled FCS files for final diagnosis testing. The trained model will analyze these files to predict HIV status.

## File Format

All files in these subdirectories should be in FCS (Flow Cytometry Standard) format with extensions `.fcs` or `.FCS`.

## Important Notes

- **These files are NOT tracked by Git** due to their large size and sensitive nature
- FCS files must be provided separately and placed in the appropriate subdirectories
- The directories themselves (with `.gitkeep` files) are tracked to maintain the project structure
- Thousands of FCS files may be stored here during project execution
