# Food Price Volatility Classification System
## Streamlit Web Application

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 🎯 Features

### 1. **Home Dashboard** 🏠
- Project overview and objectives
- Quick statistics and key metrics
- Volatility distribution visualization
- Getting started guide

### 2. **Data Explorer** 📈
- Interactive data filtering
- Price trend analysis
- Volatility distribution charts
- Market and commodity comparison

### 3. **Model Performance** 🤖
- Comparison of all 6 models
- Accuracy and F1-Score metrics
- Detailed performance visualization
- Best model identification

### 4. **Live Predictions** 🔮
- Real-time volatility prediction
- Interactive input form
- Confidence scoring
- Interpretable results

### 5. **Insights & Analysis** 💡
- Feature importance ranking
- Key insights and findings
- Recommendations
- Model analysis

### 6. **About** 📚
- Project information
- Methodology documentation
- Technology stack
- Academic context

---

## 🎨 Features Included

### ✅ Core Features
- [x] Interactive data exploration
- [x] Model performance comparison
- [x] Live prediction interface
- [x] Feature importance visualization
- [x] Professional styling and layout
- [x] Responsive design
- [x] Multiple visualizations (Plotly charts)
- [x] Comprehensive documentation

### 🎯 User Experience
- **Navigation**: Easy sidebar navigation
- **Visuals**: Clean, modern interface
- **Interactivity**: Real-time filtering and predictions
- **Information**: Clear explanations and insights
- **Performance**: Fast loading and smooth interactions

---

## 📊 Data Integration

To connect with your actual trained models and data:

1. **Load Models**: Modify `app.py` to load your trained models from pickle files
2. **Add Real Data**: Replace sample data with your actual dataset
3. **Database**: Optionally connect to Supabase for live data
4. **Authentication**: Add user authentication if needed

### Example Integration:

```python
# Load trained model
@st.cache_resource
def load_model(model_name):
    with open(f'models/{model_name}.pkl', 'rb') as f:
        return pickle.load(f)

# Use model for predictions
model = load_model('random_forest')
prediction = model.predict(features)
```

---

## 🎓 Presentation Tips

### For Your Presentation:

1. **Start with Home**: Show project overview and statistics
2. **Data Explorer**: Demonstrate interactive filtering
3. **Model Performance**: Highlight model comparison results
4. **Live Predictions**: Make real-time predictions
5. **Insights**: Explain key findings and feature importance
6. **About**: Discuss methodology and learning outcomes

### Key Talking Points:
- ✅ Real-world application of classification
- ✅ 6 algorithms compared and evaluated
- ✅ 100% accuracy achieved with tree-based models
- ✅ Feature importance insights
- ✅ Food security policy applications

---

## 🛠️ Customization

### Adding Your Own Data:

1. **Replace Sample Data**: Update `load_sample_data()` function
2. **Connect to Database**: Add Supabase connection
3. **Real Models**: Load your trained models
4. **Custom Charts**: Add your own visualizations

### Styling:

- Modify the CSS in the `main()` function
- Change colors, fonts, layouts
- Add your branding

---

## 📝 Notes

- Sample data is used for demonstration
- Connect to your Google Colab notebook for real models
- Add authentication for production deployment
- Deploy to Streamlit Cloud for public access

---

## 🚀 Deployment

### Local Development:
```bash
streamlit run app.py
```

### Streamlit Cloud:
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Docker:
```bash
docker build -t food-price-app .
docker run -p 8501:8501 food-price-app
```

---

## 📧 Support

For questions or issues, please contact the development team.

**Built with ❤️ for MSc AI Data Mining Classification Module**

