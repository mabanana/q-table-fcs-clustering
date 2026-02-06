# Data Directory

This directory contains subdirectories for FCS (Flow Cytometry Standard) files used in the Q-learning clustering project for HIV diagnosis.

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
