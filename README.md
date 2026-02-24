# 🌾 AgriPredict Pro: Intelligent Crop Yield Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agripredict-pro.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AgriPredict Pro** is a premium, AI-driven dashboard designed to optimize agricultural output and food security through precise crop yield forecasting. Leveraging advanced machine learning (XGBoost) and historical climatic data, it provides farmers and planners with actionable insights into seasonal harvests across India.

---

##  Key Features

- **Precision AI Forecasting**: High-accuracy predictions using an Ensemble XGBoost Regressor ($R^2 \approx 0.94$).
- **Premium User Experience**: A sleek, modern dashboard built with Streamlit, featuring customized emerald-themed aesthetics and micro-animations.
- **Dynamic Variable Analysis**: Interactive simulation of environmental factors like rainfall and temperature.
- **Strategic Insights**: Automated advice on irrigation and water management based on selected environmental parameters.
- **Scalable Architecture**: Robust preprocessing pipeline handling categorical state/crop data and numerical climate metrics.

---

##  Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Dashboarding & UI)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [XGBoost](https://xgboost.ai/), [Scikit-Learn](https://scikit-learn.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Visuals**: [Matplotlib](https://matplotlib.org/), [Vega-Lite](https://vega.github.io/vega-lite/) (via Streamlit)

---

## Project Structure

```text
├── app.py                # Main Streamlit Dashboard Application
├── train.py              # Model Training & Pipeline Serialization
├── check.py              # Data Validation Utility
├── models/               # Directory for serialized ML models
│   └── crop_yield_pipeline.pkl
├── Final_Dataset.csv     # Historical Agriculture & Climate Dataset
└── README.md             # Project Documentation
```

---

##  Model Architecture

The core engine uses a **Log-Transformed Target Regression** approach to handle the non-linear distribution of crop yields.

1. **Preprocessing**: 
    - One-Hot Encoding for `State_Name`, `Crop_Type`, and `Crop`.
    - Feature Engineering: Interaction terms (`Rainfall * Temp`) and Polynomial features.
    - Log transformation to normalize highly skewed yield data.
2. **Model**: 
    - XGBoost Regressor with 600 estimators.
    - L1/L2 Regularization to prevent overfitting.
    - Validated with an 80/20 train-test split.

---

##  Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/agripredict-pro.git
   cd agripredict-pro
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional)**:
   *The pre-trained model is included in the `models/` folder. To retrain:*
   ```bash
   python train.py
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

##  Usage

1. **Select Context**: Use the sidebar to set your state, season, and crop type.
2. **Adjust Environment**: Use the sliders to input expected rainfall and temperature levels.
3. **Execute AI Projection**: Click the "EXECUTE AI PROJECTION" button to generate a yield forecast.
4. **Review Insights**: Analyze the generated metrics and strategic advice for optimal harvest planning.

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Developed for Sustainable Agriculture Planning</b><br>
  © 2026 AgriPredict Pro Analytics Division
</p>
