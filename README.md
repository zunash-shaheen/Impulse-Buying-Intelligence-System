# IBIS — Impulse Buying Intelligence System ⚡

IBIS is a sleek, cyberpunk-styled analytics dashboard built with Streamlit. It is designed to analyze the online browsing behaviors of e-commerce shoppers and predict purchase probabilities using a Machine Learning Logistic Regression model.

## Features

- **Cyberpunk Dark Theme:** Custom CSS styling with a beautiful, responsive dark mode UI.
- **Data Visualizations:** Comprehensive insights utilizing matplotlib, diving deep into monthly trends, temporal patterns, and traffic sources.
- **Statistical Analysis:** Visual breakdowns of discrete and continuous probability distributions (Binomial, Poisson, Normal) applied directly to the dataset.
- **Machine Learning Integration:** A logistic regression model trained on 12,330 e-commerce sessions to classify users into buyers and non-buyers, boasting dynamic prediction capabilities based on user inputs.

## Technologies Used

- **Python 3**
- **Streamlit** - Web framework
- **Pandas & NumPy** - Data manipulation
- **Scikit-Learn** - Machine learning pipeline (Logistic Regression, Standardization)
- **Matplotlib** - Custom styled data visualizations
- **SciPy** - Statistical distributions

## How to Run Locally

### Prerequisites
Make sure you have Python installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### Starting the Dashboard
Navigate to the project directory and run the Streamlit app:
```bash
python -m streamlit run ibis_app.py
```
*(If your PATH is configured correctly, `streamlit run ibis_app.py` will also work).*

The app will open automatically in your browser at `http://localhost:8501`.

## Deployment

This app is fully prepared for deployment on [Streamlit Community Cloud](https://share.streamlit.io/). 
1. Push this repository to GitHub.
2. Connect your GitHub to Streamlit Cloud.
3. Select `ibis_app.py` as the main file and click "Deploy".
