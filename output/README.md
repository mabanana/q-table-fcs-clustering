# Output Directory

This directory contains all files generated during the execution of the Q-learning clustering analysis.

## Generated Files

When `main.py` is run, the following types of files will be created in this directory:

### Q-tables
- Saved as pickle (`.pkl`) or JSON (`.json`) files
- Contain the learned Q-values from the reinforcement learning process
- Used for model persistence and loading trained models

### Plots
- Visualization files showing clustering results
- May include scatter plots, heatmaps, and other graphical representations
- Typically saved as `.png`, `.jpg`, or `.pdf` files

### Data Files
- CSV and text files containing metrics and results
- Performance statistics, cluster assignments, and diagnostic predictions
- Useful for analysis and reporting

## Important Notes

- **All contents of this directory are NOT tracked by Git**
- Files are generated at runtime and should not be committed to version control
- The directory structure itself (with `.gitkeep` file) is tracked to maintain the project organization
- Generated files may be large and numerous depending on the dataset size
- You can safely delete files from this directory; they will be regenerated on the next run
