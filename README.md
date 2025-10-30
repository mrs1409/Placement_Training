# Student Performance Prediction Analysis

This project analyzes the Student Performance Dataset and builds regression models to predict final grades (G3).

Tasks performed:
- Load and clean `student-por.csv`
- Exploratory data analysis (studytime, failures, absences, parental education)
- Train linear regression and random forest models
- Compare models with/without family support variables
- Produce actionable insights
- Generate professional PDF report

## Quick Start

Run the analysis:

```cmd
python run_analysis.py
```

Generate PDF report:

```cmd
python generate_report.py
```

## Outputs

**Main Report:**
- `Student_Performance_Analysis_Report.pdf` - Professional report with all findings, statistics, and recommendations

**Analysis Files** (saved in `outputs/`):
- `studytime_vs_G3.png` - Visualization
- `model_comparison.txt` - Numeric summary
- `*.joblib` - Trained models

## Requirements

See `requirements.txt` for all dependencies (pandas, scikit-learn, matplotlib, seaborn, reportlab)
