# Used Car Resale Value Prediction

A machine-learning valuation engine for estimating the fair market resale value of pre-owned cars from historical dealership sales data.

## Objective

Predict a vehicle's resale value from the following attributes:

- Make and model
- Model year
- Mileage
- Fuel type
- Transmission
- Engine capacity
- Accident history

The project cleans and encodes the historical data, trains a `RandomForestRegressor`, evaluates predictions using Mean Absolute Error (MAE), and identifies the vehicle attributes that contribute most to the valuation.

## Machine-Learning Workflow

1. **Load historical sales data** from a CSV or another tabular source.
2. **Inspect and clean the data** by handling missing values, correcting data types, removing duplicates, and checking for invalid values or extreme outliers.
3. **Prepare the features**:
   - Keep numerical fields such as model year, mileage, and engine capacity numeric.
   - Encode categorical fields such as make, model, fuel type, transmission, and accident history.
   - Separate the target column, `resale_value`, from the input features.
4. **Split the data** into training and test sets. A typical split is 80% training and 20% testing.
5. **Train a Random Forest Regressor** on the processed training data.
6. **Evaluate the model** on unseen test data using MAE and, optionally, additional regression metrics.
7. **Inspect feature importance** to understand which vehicle characteristics influence predicted value.
8. **Save the trained pipeline** so that the same preprocessing and model can be used for future valuations.

## Expected Dataset

The input data should contain one row per historical vehicle sale. Example columns:

| Column | Type | Description |
| --- | --- | --- |
| `make` | Categorical | Vehicle manufacturer |
| `model` | Categorical | Vehicle model |
| `model_year` | Numeric | Manufacturing or registration year |
| `mileage` | Numeric | Distance driven |
| `fuel_type` | Categorical | Petrol, diesel, hybrid, electric, etc. |
| `transmission` | Categorical | Manual, automatic, or other type |
| `engine_capacity` | Numeric | Engine size, normally in cc or litres |
| `accident_history` | Categorical or binary | Whether the vehicle has accident history |
| `resale_value` | Numeric | Historical sale price and prediction target |

The target column and units should be kept consistent throughout training and evaluation. Prices should use one currency and mileage and engine capacity should use consistent units.

## Recommended Implementation

A scikit-learn `Pipeline` and `ColumnTransformer` should be used so training and prediction apply identical preprocessing:

```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numeric_features = ["model_year", "mileage", "engine_capacity"]
categorical_features = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "accident_history",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
```

The exact hyperparameters should be tuned using cross-validation on the training set. The test set should remain untouched until final evaluation.

## Evaluation

Mean Absolute Error is the primary evaluation metric:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

MAE expresses the average absolute difference between the actual and predicted resale values in the same currency as the target. Lower values indicate better performance and are straightforward for dealership stakeholders to interpret.

The final report should include:

- Test-set MAE
- Number of training and test records
- A comparison of actual and predicted values
- Validation strategy and Random Forest hyperparameters
- Any known limitations or data-quality concerns

## Feature Importance

After training, extract feature importance values from the fitted Random Forest and rank them from highest to lowest. Because categorical variables are one-hot encoded, importances for individual encoded categories may be grouped back to their original feature for easier interpretation.

Feature importance can help answer questions such as:

- How strongly does vehicle age affect value?
- Does mileage have a larger effect than engine capacity?
- How much does accident history reduce expected value?
- Which makes or models retain value best?

Feature importance indicates model association, not causation. Use domain knowledge and validation checks before making pricing policy decisions.

## Example Prediction

Once the pipeline is trained, a new vehicle can be valued using the same input columns:

```python
new_vehicle = {
    "make": "Example Make",
    "model": "Example Model",
    "model_year": 2021,
    "mileage": 45000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "engine_capacity": 1500,
    "accident_history": "No",
}

predicted_value = model.predict([new_vehicle])[0]
print(f"Estimated resale value: {predicted_value:,.2f}")
```

For production use, serialize the complete fitted pipeline with a versioned model artifact, for example with `joblib`. Storing the preprocessing and estimator together prevents training-serving mismatches.

## Business Considerations

- The model estimates a market value; it does not replace a professional inspection or appraisal.
- Sale prices may reflect location, season, dealer incentives, demand, service history, optional equipment, and negotiation effects that are not included in the listed features.
- Monitor performance over time because market prices and buyer preferences change.
- Review predictions for unusual vehicles and out-of-range inputs before using them in pricing decisions.
- Protect customer, dealer, and transaction data when storing or deploying the model.

## Suggested Project Structure

```text
RESALE-CAR-VALUE/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── resale_value_analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── resale_value_pipeline.joblib
├── requirements.txt
└── README.md
```

## Getting Started

1. Place the historical sales dataset in `data/raw/`.
2. Install Python dependencies, including `pandas`, `scikit-learn`, `numpy`, `matplotlib`, and `joblib`.
3. Run the data-cleaning and training workflow.
4. Record the final test MAE and feature-importance results.
5. Save the fitted pipeline and use it for new vehicle valuations.

## Limitations and Next Steps

Possible improvements include hyperparameter tuning, cross-validation, price normalization by location, adding service history and vehicle condition, monitoring data drift, and comparing the Random Forest with gradient boosting models. Any alternative model should be selected using the same held-out evaluation process.
