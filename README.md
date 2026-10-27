# 5G Network Traffic Forecasting and Congestion Prediction

## Project Overview

This project uses Machine Learning and Deep Learning techniques to forecast future network traffic and estimate potential network congestion using historical telecommunications data.

The system analyzes internet activity patterns across different geographical regions and predicts future network demand.

## Dataset

**Dataset:** Telecom Italia Milan Telecommunications Activity Dataset

- Location: Milan, Italy
- Period: November–December 2013
- Time Interval: 10 minutes
- Prediction Target: Internet Activity

## Project Workflow

1. Data Exploration
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Machine Learning Model Training
6. Deep Learning
7. Congestion Prediction
8. Streamlit Dashboard

## Feature Engineering

Features used for traffic forecasting include:

- Lag features
- Rolling mean and standard deviation
- Hour
- Weekday
- Weekend indicator
- Minute slot

## Machine Learning Models

Models explored:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LSTM
- GRU

## Congestion Prediction

The system estimates traffic-load categories:

- Normal
- Moderate
- High
- Congested

These categories represent estimated traffic-load conditions, not directly measured network congestion.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib, Seaborn and Plotly
- Jupyter Notebook
- Streamlit

## Dataset and Model Files

Large datasets and trained model files are excluded from the Git repository.

Instructions for obtaining the dataset and generating model files should be added to make the project reproducible.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard using the Streamlit entry-point file inside the `app/` folder.

For example:

```bash
streamlit run app/app.py
```

Update the command above if your actual dashboard filename differs.

## Future Enhancements

- Real-time 5G traffic monitoring
- Improved congestion-risk forecasting
- Integration with network performance metrics
- Advanced deep learning models
- Automated network resource optimization

## Note

This project uses historical telecommunications data as a foundation for a 5G-oriented forecasting framework. It does not directly measure live 5G network performance.