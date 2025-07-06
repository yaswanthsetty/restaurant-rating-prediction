# Quick Setup Guide

## 🚀 Quick Start

### Option 1: Run the Python Script (Fast)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the model
python restaurant_rating_predictor.py
```

### Option 2: Run the Full Jupyter Notebook (Complete Analysis)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Open "Restaurant ratings.ipynb"
```

## 📊 Expected Output

The Python script will show:
- Data preprocessing steps
- Model training progress  
- Final performance metrics:
  - **R² Score**: ~0.6395
  - **RMSE**: ~0.3300
  - **MAE**: ~0.2405
- Top 10 most important features

## 🔧 Troubleshooting

### Common Issues:

1. **FileNotFoundError**: Make sure `Dataset .csv` is in the same folder
2. **Module not found**: Run `pip install -r requirements.txt`
3. **Jupyter not starting**: Install with `pip install jupyter`

### System Requirements:
- Python 3.8+
- 4GB+ RAM recommended
- ~10MB disk space for the dataset

## 📁 Project Files

| File | Purpose |
|------|---------|
| `restaurant_rating_predictor.py` | Standalone script for quick results |
| `Restaurant ratings.ipynb` | Full analysis with visualizations |
| `Dataset .csv` | Zomato restaurant data |
| `requirements.txt` | Python dependencies |

---

⚡ **Quick tip**: For the full experience with plots and detailed analysis, use the Jupyter notebook!
