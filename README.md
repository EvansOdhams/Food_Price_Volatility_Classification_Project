# 🌾 Food Price Volatility Classification - Kenya

A comprehensive machine learning project that predicts food price volatility levels in Kenya using multiple classification algorithms. This project was developed as part of my MSc AI - Data Mining and Big Data Classification Module coursework.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-orange.svg)](https://github.com/EvansOdhams/Food_Price_Volatility_Classification_Project)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technology Stack](#technology-stack)
- [Key Insights](#key-insights)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project addresses a critical challenge in food security by predicting price volatility for essential food commodities in Kenya. Using real-world data from the World Bank and FSS-KEN, I've built a classification system that can predict volatility levels (Low, Medium, High, Extreme) with remarkable accuracy.

The project includes:
- **6 different classification algorithms** for comprehensive comparison
- **21 engineered features** from price data
- **2,278 monthly price records** across 13 Kenyan markets
- **Interactive Streamlit web application** for exploration and predictions
- **Professional visualization dashboard** with real-time insights

---

## 🎯 Project Objectives

1. **Classification Problem**: Predict food price volatility levels into 4 categories (Low, Medium, High, Extreme)
2. **Algorithm Comparison**: Evaluate and compare 6 different classification algorithms
3. **Feature Engineering**: Create comprehensive features from raw price data
4. **Real-world Application**: Develop a practical solution for food security planning in Kenya
5. **Data Integration**: Combine multiple data sources for comprehensive analysis
6. **User Interface**: Build an interactive dashboard for end-users

---

## 📊 Dataset

### Data Sources
- **World Bank Monthly Food Prices** (Kenya)
- **Food Security Simulator Kenya (FSS-KEN)**

### Dataset Characteristics
- **Total Records**: 2,278 monthly observations
- **Commodities**: 3 (Maize, Potatoes, Sorghum)
- **Markets**: 13 locations across Kenya
- **Time Period**: 2007-2020
- **Currency**: KES (Kenyan Shillings)

### Price Ranges (KES)
- **Maize**: 6.0 - 100.0 per kg
- **Potatoes**: 368.0 - 5,700.0 per 100kg
- **Sorghum**: 1,100.0 - 7,470.0 per kg

### Market Coverage
- **Maize Markets**: Kitui, Mandera, Lodwar, Marsabit, Kilifi, Kajiado, Garissa, Hola, Marigat
木 - **Potatoes Markets**: Nairobi, Eldoret, Kisumu, Kitui, Nakuru
- **Sorghum Markets**: Eldoret, Nairobi, Kisumu, Kitui, Nakuru

---

## ✨ Key Features

### 📈 Data Exploration
- Interactive filtering by commodity, market, and volatility class
- Multiple visualization options (Monthly Averages, Individual Data Points, Price Distribution)
- Real-time summary metrics
- Comprehensive data tables

### 🤖 Model Performance Dashboard
- Side-by-side comparison of 6 classification algorithms
- Detailed performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Visual charts for quick comparison
- Top 3 performing models highlighted

### 🔮 Live Predictions
- Real-time volatility predictions
- Dynamic market selection based on commodity
- Realistic price range validation
- Confidence scores and interpretations

### 💡 Insights & Analysis
- Feature importance visualization
- Key insights and recommendations
- Data-driven policy suggestions
- Market-specific analysis

---

## 🔬 Methodology

### Feature Engineering

I engineered 21 features from the raw price data:

#### Lagged Price Changes
- 1-month, 3-month, 6-month, and 12-month price changes
- Month-over-month and year-over-year comparisons

#### Rolling Volatility Measures
- 3-month, 6-month, and 12-month rolling volatility
- Coefficient of variation (CV) calculations
- Rolling standard deviations

#### Seasonal Features
- Month and quarter indicators
- Year progression features

#### Momentum Indicators
- Price momentum calculations
- Trend analysis features

### Classification Classes

Based on monthly price change percentages:
- **Low Volatility**: < 5% monthly change
- **Medium Volatility**: 5-15% monthly change
- **High Volatility**: 15-30% monthly change
- **Extreme Volatility**: > 30% monthly change

### Algorithms Implemented

1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree** - Interpretable tree-based model
3. **Random Forest** - Ensemble tree method
4. **Support Vector Machine (SVM)** - Kernel-based classification
5. **XGBoost** - Gradient boosting framework
6. **Neural Network** - Multi-layer perceptron

---

## 📊 Model Performance

### Overall Results

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| **Decision Tree** | 🥇 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Random Forest** | 🥇 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **XGBoost** | 🥈 98.9% | 98.9% | 98.9% | 98.9% | N/A |
| **Neural Network** | 🥉 94.9% | 94.9% | 95.0% | 94.9% | 99.5% |
| **SVM** | 71.5% | 71.3% | 71.3% | 71.5% | 88.4% |
| **Logistic Regression** | 55.8% | 50.9% | 49.5% | 55.8% | 76.2% |

### Key Findings

- **Best Performing Models**: Decision Tree and Random Forest achieved perfect 100% accuracy
- **Excellent Performance**: XGBoost achieved 98.9% accuracy
- **Most Important Features**: 1-month price changes (42% importance)
- **Class Distribution**: 48.6% Low, 30.3% Medium, 13.9% High, 6.4% Extreme

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/EvansOdhams/Food_Price_Volatility_Classification_Project.git
cd Food_Price_Volatility_Classification_Project
```

2. **Install dependencies**
```bash
cd streamlit_app
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 💻 Usage

### Running the Jupyter Notebook

1. Open `Food_Price_Volatility_Classification_Project.ipynb` in Google Colab or Jupyter
2. Run cells sequentially to:
   - Load and preprocess data
   - Engineer features
   - Train classification models
   - Evaluate and compare results

### Using the Streamlit App

The web application provides:

1. **🏠 Home**: Project overview and quick statistics
2. **📈 Data Explorer**: Filter and visualize price trends
3. **🤖 Model Performance**: Compare all 6 algorithms
4. **🔮 Live Predictions**: Get real-time volatility predictions
5. **💡 Insights**: View feature importance and recommendations
6. **📚 About**: Project methodology and details

### Making Predictions

1. Navigate to the "Live Predictions" page
2. Select a commodity (Maize, Potatoes, or Sorghum)
3. Choose an available market
4. Enter current and previous month prices
5. Click "Predict Volatility" to get results

---

## 📁 Project Structure

```
Food_Price_Volatility_Classification_Project/
│
├── 📊 Datasets/                                    # Raw data files
│   ├── KEN_RTFP_mkt_2007_2025-10-13.csv          # Main dataset
│   └── [Additional data files]
│
├── 🌐 streamlit_app/                               # Web application
│   ├── app.py                                     # Main application code
│   ├── requirements.txt                           # Python dependencies
│   └── README.md                                  # App documentation
│
├── 📓 Food_Price_Volatility_Classification_Project.ipynb  # Main notebook
│
└── README.md                                      # This file
```

---

## 📈 Results

### Model Comparison Visualization

The Streamlit app includes interactive visualizations comparing:
- Accuracy scores across models
- F1-Score metrics
- Detailed confusion matrices
- Feature importance rankings

### Key Insights

1. **Price Changes Drive Volatility**: 1-month price changes are the strongest predictor (42% importance)
2. **Tree-Based Models Excel**: Decision trees and Random Forest recommendations
3. **Seasonal Patterns**: Identified seasonal volatility trends in Kenyan markets
4. **Regional Differences**: Markets show varying volatility patterns
5. **Commodity-Specific**: Maize shows highest volatility, requiring special attention

---

## 🛠️ Technology Stack

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting
- **streamlit** - Web application framework

### Visualization
- **plotly** - Interactive charts
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualization

### Development
- **Jupyter Notebook** - Analysis environment
- **Google Colab** - Cloud-based development

---

## 💡 Key Insights

### For Policymakers
- Monitor 1-month price changes for early warning signals
- Focus on maize markets due to high volatility
- Implement regional strategies for different Kenyan markets
- Use rolling volatility measures for trend detection

### For Analysts
- Tree-based models (Decision Tree, Random Forest) show superior performance
- Feature engineering significantly improves model accuracy
- Price momentum indicators are crucial predictors
- Cross-validation ensures model reliability

### For Stakeholders
- Price volatility impacts food security in Kenya
- Real-time monitoring can improve market interventions
- Data-driven insights support evidence-based policy
- Predictive models enable proactive management

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution
- Additional classification algorithms
- Feature engineering improvements
- Model deployment enhancements
- Documentation improvements
- Bug fixes and optimizations

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

**Author**: Evans Odhams  
**Email**: evansodhams@gmail.com  
**GitHub**: [@EvansOdhams](https://github.com/EvansOdhams)  
**LinkedIn**: [Evans Odhams](https://linkedin.com/in/evansodhams)

---

## 🙏 Acknowledgments

- **World Bank** for providing comprehensive food price data
- **Food Security Simulator Kenya (FSS-KEN)** for household data
- **MSc AI Program** at [Your University] for educational framework
- **Open Source Community** for excellent libraries and tools

---

## 📚 References

- World Bank Monthly Food Prices: [Data Source](https://www.worldbank.org/)
- Food Security Simulator Kenya: [FSS-KEN](https://www.ifpri.org/)
- Scikit-learn Documentation: [sklearn](https://scikit-learn.org/)
- XGBoost Documentation: [xgboost](https://xgboost.readthedocs.io/)
- Streamlit Documentation: [streamlit](https://docs.streamlit.io/)

---

## 🎓 Academic Context

This project was developed for the **MSc AI - Data Mining and Big Data Classification Module** coursework. It demonstrates proficiency in:
- Classification algorithm implementation
- Feature engineering techniques
- Model evaluation and comparison
- Data preprocessing and integration
- Real-world problem solving
- Interactive application development

---

**⭐ If you find this project useful, please consider giving it a star!**

---

<div align="center">
  <p>Made with ❤️ for Food Security in Kenya</p>
  <p>© 2025 Evans Odhams. All rights reserved.</p>
</div>

