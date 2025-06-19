## 📊 Restaurant Ratings Prediction
This project builds a machine learning pipeline to predict restaurant ratings using structured data and a Gradient Boosting model. It includes data preprocessing, model training, evaluation, and feature importance analysis.

## 🚀 Project Highlights
- **🔍 Goal**: Predict the Aggregate rating of a restaurant.
- **🧹 Preprocessing**: Imputes missing values, scales numerics, and one-hot encodes categories.
- **🌲 Model**: GradientBoostingRegressor from scikit-learn.
- **📈 Evaluation**: RMSE and R² score.
- **🧠 Interpretation**: Uses permutation importance to understand the most influential features.

## 📁 Dataset
- File Location: Replace the local dataset path (Dataset .csv) if running on a different machine.
- Target Column: Aggregate rating
- Dropped Columns: Rating color, Rating text (considered leaky features)

## 🛠️ Libraries Used
- pandas
- numpy
- scikit-learn

## 🧪 Steps in the Pipeline
- Data Loading:
   - Reads the dataset and separates features (X) and target (y).
- Data Splitting:
   - Splits data into training and testing sets (80/20 ratio).
- Preprocessing Pipeline:
   - Numeric: Median imputation + standard scaling
   - Categorical: Most frequent imputation + one-hot encoding
- Model Training:
   - Trains a GradientBoostingRegressor.
- Evaluation:
   - Prints:
       - Root Mean Squared Error (RMSE)
       - R² Score
- Feature Importance
   - Calculates and prints the top 15 most important features using permutation importance.

## 📊 Sample Output
- Model Performance on Test Set:
  - RMSE: 0.235
  - R² Score: 0.912
- Top 15 Most Influential Features:
...

## 📌 Usage
- Update the dataset path in the script if needed.
- Run the script using Python:
  ```bash
     python Restaurant_Ratings.py

## 📃 Author
- Developed as part of the Cognifyz Internship Program.
