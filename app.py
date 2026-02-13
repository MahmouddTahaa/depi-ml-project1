import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import r2_score, make_scorer
import pandas as pd


st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide")


data = pd.read_csv("./data/processed/boston.csv")

prices = data['MEDV']
features = data.drop('MEDV', axis=1)


def performance_metric(y_true, y_predict):
    return r2_score(y_true, y_predict)


def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    regressor = DecisionTreeRegressor(random_state=0)

    params = {'max_depth': range(1, 11)}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor,
                        param_grid=params,
                        scoring=scoring_fnc,
                        cv=cv_sets)

    grid = grid.fit(X, y)

    return grid.best_estimator_


model = fit_model(features, prices)

# Training performance
train_pred = model.predict(features)
train_r2 = r2_score(prices, train_pred)
median_price = float(prices.median())

feature_importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)


st.title("Boston Housing Price Predictor")
st.markdown("Predict home selling price in Boston using a trained Decision Tree model. Adjust inputs and press **Predict**.")

with st.sidebar:
    st.header("About")
    st.write("A small demo app that predicts Boston home prices from a trained Decision Tree model.")
    st.markdown("---")
    st.subheader("Model")
    st.write(f"Trained model: DecisionTreeRegressor (max_depth={model.get_params().get('max_depth', 'auto')})")
    st.metric("Training R²", f"{train_r2:.3f}")
    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(data.head(5))
    st.markdown("---")
    st.caption("Dataset: Boston housing (processed) — features and target `MEDV` in $1000s")


col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("Input Parameters")
    rm = st.slider("Number of Rooms (RM)", min_value=1.0, max_value=15.0, value=5.0, step=0.1, help="Average number of rooms per dwelling")
    lstat = st.slider("Neighborhood Poverty Level (%) (LSTAT)", min_value=0.0, max_value=50.0, value=15.0, step=0.1, help="% lower status of the population")
    ptratio = st.slider("Student-Teacher Ratio (PTRATIO)", min_value=5.0, max_value=30.0, value=15.0, step=0.1, help="Pupil-teacher ratio by town")
    st.markdown("\n")
    if st.button("Predict"):
        prediction = model.predict([[rm, lstat, ptratio]])
        predicted_price = float(prediction[0])
        diff = predicted_price - median_price
        with col2:
            st.success("Prediction")
            st.subheader(f"Estimated Selling Price: ${predicted_price:,.2f}k")
            st.caption(f"Median dataset price: ${median_price:,.2f}k — Difference: ${diff:,.2f}k")
        st.info("Note: This is a demo model trained on a processed dataset. Use with caution for real decisions.")

with col2:
    st.header("Model Insights")
    st.write("Top features contributing to the prediction")
    fi_df = pd.DataFrame({'feature': feature_importances.index, 'importance': feature_importances.values})
    st.bar_chart(fi_df.set_index('feature'))
    with st.expander("Show full dataset statistics"):
        st.dataframe(data.describe().transpose())

with col3:
    st.header("Data Distribution")
    st.write("Target (`MEDV`) distribution")
    st.bar_chart(prices.value_counts(bins=20).sort_index())
    st.markdown("---")
    st.write("Quick help")
    st.write("- Adjust sliders on the left and press Predict.\n- Use the sidebar to inspect sample data and model R².")

with st.expander("Model Details and Limitations"):
    st.write("This app uses a Decision Tree regressor tuned with grid search over `max_depth` (1-10). Results shown are training-set metrics and are provided for demonstration only.")
