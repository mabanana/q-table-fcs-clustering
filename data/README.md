# Data Directory

This directory contains subdirectories for FCS (Flow Cytometry Standard) files used in the Q-learning clustering project for dendritic cell activation and cytokine analysis.

## Subdirectories

### `/data/positive/`
Contains FCS files from a homogeneous condition (e.g., stimulated dendritic cells). These files are used for initial training and understanding the characteristics of samples from this condition.

### `/data/mixed_training/`
Contains prelabeled FCS files (both positive and negative) used for supervised training of the machine learning model. Each file should be properly labeled to indicate its classification (e.g., HEU vs UE, or stimulated vs unstimulated).

### `/data/mixed/`
Contains unlabeled FCS files for final classification testing. The trained model will analyze these files to predict their class.

## File Format

All files in these subdirectories should be in FCS (Flow Cytometry Standard) format with extensions `.fcs` or `.FCS`.

## Expected Markers

The FCS files should contain dendritic cell and cytokine markers:
- **Dendritic cell markers**: CD123, MHCII, CD14, CD11c
- **Cytokine measurements**: IFNa, IL6, IL12, TNFa
- **Housekeeping**: FSC-A, SSC-A, Time

## Compensation Status

The FCS files are expected to have compensation already applied (metadata should indicate `APPLY COMPENSATION: TRUE`). The loader is configured with `compensation: false` to avoid reapplying compensation.

## Important Notes

- **These files are NOT tracked by Git** due to their large size and sensitive nature
- FCS files must be provided separately and placed in the appropriate subdirectories
- The directories themselves (with `.gitkeep` files) are tracked to maintain the project structure
- Thousands of FCS files may be stored here during project execution
