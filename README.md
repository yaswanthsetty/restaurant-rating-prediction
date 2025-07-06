# 🍽️ Restaurant Rating Prediction Project

A machine learning project to predict restaurant ratings using Zomato data with multiple algorithms including XGBoost, LightGBM, and traditional ML methods.

## 📊 Project Overview

This project implements a comprehensive machine learning pipeline to predict restaurant ratings based on various features like location, cuisine type, cost, and customer engagement metrics.

### 🎯 Key Results
- **Best Model**: XGBoost 
- **R² Score**: 0.6395 (explains ~64% of variance)
- **RMSE**: 0.1089
- **MAE**: 0.2405

## 🗂️ Project Structure

```
Restaurant ML Project/
├── Restaurant ratings.ipynb          # Main analysis notebook
├── Zomato_Rating_Prediction_Documentation.ipynb  # Documentation
├── Dataset .csv                       # Raw dataset
├── requirements.txt                   # Python dependencies
├── README.md                         # Project documentation
├── .gitignore                        # Git ignore file
└── images/                           # Generated visualizations (optional)
```

## 📋 Features

### Data Processing
- ✅ **Data Cleaning**: Missing value handling, duplicate removal
- ✅ **Feature Engineering**: Log transformation, multi-hot encoding for cuisines
- ✅ **Categorical Encoding**: One-hot encoding for location and currency
- ✅ **Feature Selection**: Rare category grouping to prevent overfitting

### Machine Learning Models
- 📊 **Baseline**: Mean predictor
- 🔵 **Linear Regression**: Basic linear model
- 🌲 **Decision Tree**: Tree-based model with depth tuning
- 🌳 **Random Forest**: Ensemble method
- 🟠 **Gradient Boosting**: Advanced boosting algorithm
- 🧠 **XGBoost**: Best performing model
- 💡 **LightGBM**: Fast gradient boosting

### Model Interpretation
- 📈 **Feature Importance**: XGBoost feature ranking
- 🔍 **SHAP Values**: Model explainability analysis
- 📊 **Cross-Validation**: 5-fold CV for robust evaluation

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/restaurant-rating-prediction.git
cd restaurant-rating-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook "Restaurant ratings.ipynb"
```

4. **Run all cells** to reproduce the analysis

## 📊 Dataset

The dataset contains restaurant information from Zomato with the following key features:

| Feature | Description |
|---------|-------------|
| `Aggregate rating` | Target variable (0-5 scale) |
| `Average Cost for two` | Average cost for two people |
| `Votes` | Number of customer votes |
| `Price range` | Price category (1-4) |
| `Cuisines` | Types of food served |
| `City` | Restaurant location |
| `Has Table booking` | Online booking availability |
| `Has Online delivery` | Delivery service availability |

## 🏆 Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Mean Baseline | 0.5495 | 0.4476 | 0.0000 |
| Linear Regression | 0.1281 | 0.2639 | 0.5757 |
| Decision Tree | 0.1425 | 0.3471 | 0.5282 |
| Random Forest | - | - | - |
| Gradient Boosting | 0.1156 | 0.2493 | 0.6172 |
| **XGBoost** ⭐ | **0.1089** | **0.2405** | **0.6395** |
| LightGBM | 0.1107 | 0.2425 | 0.6334 |

## 📈 Key Insights

### Top Predictive Features
1. **Votes** - Customer engagement strongly correlates with quality
2. **Average Cost** - Price point influences ratings
3. **Price Range** - Category-based pricing impacts perception
4. **Location** - Geographic factors matter
5. **Cuisine Type** - Certain cuisines tend to rate higher

### Business Insights
- Restaurants with higher customer engagement (votes) tend to have better ratings
- Price-quality relationship exists but isn't linear
- Location plays a significant role in rating patterns
- Certain cuisine combinations perform better than others

## 🔧 Technical Details

### Data Preprocessing
- **Missing Value Treatment**: Median/mode imputation
- **Outlier Handling**: Statistical outlier detection
- **Feature Engineering**: Log transformation for skewed distributions
- **Encoding**: Multi-hot for cuisines, one-hot for categories

### Model Selection Process
1. **Baseline Establishment**: Mean predictor for comparison
2. **Linear Models**: Start with interpretable models
3. **Tree-Based Models**: Capture non-linear relationships
4. **Ensemble Methods**: Combine multiple learners
5. **Gradient Boosting**: Advanced optimization techniques

### Validation Strategy
- **Train-Test Split**: 80-20 stratified split
- **Cross-Validation**: 5-fold CV for model selection
- **Performance Metrics**: RMSE, MAE, R² for comprehensive evaluation

## 🚀 Future Improvements

### Model Enhancements
- [ ] **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV
- [ ] **Feature Interaction**: Polynomial and interaction terms
- [ ] **Ensemble Methods**: Stacking and blending
- [ ] **Deep Learning**: Neural networks for complex patterns

### Data Enhancements
- [ ] **Temporal Features**: Time-based patterns and seasonality
- [ ] **Geospatial Data**: Distance to landmarks, population density
- [ ] **Text Analysis**: Review sentiment analysis
- [ ] **External Data**: Weather, events, economic indicators

### Production Readiness
- [ ] **API Development**: REST API for model serving
- [ ] **Model Monitoring**: Performance tracking and drift detection
- [ ] **A/B Testing**: Model comparison in production
- [ ] **Automated Retraining**: Continuous model updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Zomato** for providing the restaurant data
- **Scikit-learn** for machine learning algorithms
- **XGBoost** and **LightGBM** teams for excellent gradient boosting implementations
- **SHAP** for model interpretability tools

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/restaurant-rating-prediction](https://github.com/yourusername/restaurant-rating-prediction)

---

⭐ **Star this repository if you found it helpful!**
