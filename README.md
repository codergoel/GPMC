# GSTN_30

## Overview

This repository contains our final submission for the hackathon. Our project focuses on building a predictive model using the provided datasets. The main objective is to evaluate the performance of our trained model on custom test datasets supplied by the jury members.

## Quick Start Guide

To evaluate our model using your custom test dataset, please follow these steps:

### Prerequisites

Ensure you have _Python 3.x_ installed, along with the following Python libraries:

- _pandas_
- _joblib_
- _scikit-learn_
- _matplotlib_

You can install the required libraries using the following command:

bash
pip install pandas joblib scikit-learn matplotlib

### Steps

1.  _Open main_testing.ipynb_

    - This is the primary notebook for evaluating the saved model with your test data.
    - You can open it using Jupyter Notebook or any compatible environment.

2.  _Update the Test Dataset Path_

    - Locate the code block where the test dataset path is specified.
    - Replace the placeholder path with the path to your custom test dataset:

      python
      if **name** == "**main**": # Path to the saved model
      model_file = 'final_gst_model.joblib'

          # Path to your custom test dataset (replace with the actual file path)
          custom_test_file = "path/to/your/X_Test_Data_Input.csv"

          # Evaluate the model and make predictions
          predictions, probabilities = evaluate_saved_model(model_file, custom_test_file)

3.  _Run the Notebook_

    - Execute all cells in main_testing.ipynb.
    - The notebook will load the saved model and generate predictions on your test dataset.
    - Outputs will include predicted labels, probabilities, and performance metrics.

## Dependencies

The main_testing.ipynb notebook requires the following Python libraries:

- _pandas_: For data manipulation and analysis.
- _joblib_: For loading the saved model.
- _scikit-learn_: For evaluation metrics and machine learning utilities.
- _matplotlib_: For plotting graphs and visualizations.

Install the dependencies using:

bash
pip install pandas joblib scikit-learn matplotlib

## Additional Resources

The repository includes additional notebooks that document our development process. These are for reference and do not need to be run for evaluation.

### 1. combination_testing.ipynb

- _Purpose_: Tests the top three combinations of datasets to determine which yields the best performance metrics.
- _Details_: Executed on Google Colab.
- _Note_: Jury members can review the implementation but do not need to run this notebook.

### 2. top_3_combination.ipynb

- _Purpose_: Calculates meta values and identifies datasets similar to ours using Euclidean distance.
- _Details_: Meta value calculations were performed on Google Colab; the rest executed locally.
- _Note_: Provides insights into dataset similarities.

### 3. model_save.ipynb

- _Purpose_: Contains the code for training the model and saving it using the joblib library.
- _Details_: Includes performance metrics of our final model.
- _Note_: Can be run locally if you update the dataset paths accordingly.

## Code Overview for main_testing.ipynb

The main_testing.ipynb notebook includes the following key components:

- _Import Necessary Libraries_

  python
  import pandas as pd
  import joblib # For loading the saved model
  from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
  roc_auc_score, confusion_matrix as sklearn_cm, log_loss, balanced_accuracy_score, roc_curve)
  import matplotlib.pyplot as plt

- _Load Custom Test Data_

  python
  def load_custom_test_data(file_path): # Loads the custom test dataset from the provided file path. # Assumes 'target' column is present in the dataset.
  test_data = pd.read_csv(file_path)
  if 'target' in test_data.columns:
  y_test = test_data['target']
  X_test = test_data.drop(columns=['target', 'ID'], axis=1)
  else:
  y_test = None # No target column provided
  X_test = test_data.drop(columns=['ID'], axis=1)
  return X_test, y_test

- _Evaluate Saved Model_

  python
  def evaluate_saved_model(model_file, custom_test_file): # Evaluates the saved model on the provided custom test dataset and displays performance metrics. # Load the saved model
  model = joblib.load(model_file)
  print(f"Model loaded from {model_file}")

      # Load the custom test dataset
      X_test, y_test = load_custom_test_data(custom_test_file)

      # Make predictions
      y_pred = model.predict(X_test)
      y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for class 1

      # If the test dataset has target labels, calculate performance metrics
      if y_test is not None:
          # Metrics calculations
          accuracy = accuracy_score(y_test, y_pred)
          precision = precision_score(y_test, y_pred, zero_division=0)
          recall = recall_score(y_test, y_pred, zero_division=0)
          f1 = f1_score(y_test, y_pred, zero_division=0)
          auc = roc_auc_score(y_test, y_pred_proba)
          cm = sklearn_cm(y_test, y_pred)
          logloss = log_loss(y_test, y_pred_proba)
          balanced_acc = balanced_accuracy_score(y_test, y_pred)
          fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

          # Display metrics
          print(f"Accuracy: {accuracy}")
          print(f"Precision: {precision}")
          print(f"Recall: {recall}")
          print(f"F1 Score: {f1}")
          print(f"AUC-ROC: {auc}")
          print(f"Log Loss: {logloss}")
          print(f"Balanced Accuracy: {balanced_acc}")
          print(f"Confusion Matrix:\n{cm}")

          # Plot AUC-ROC curve
          plt.figure(figsize=(8, 6))
          plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC (area = {auc:.2f})')
          plt.plot([0, 1], [0, 1], color='red', linestyle='--')
          plt.xlabel('False Positive Rate (FPR)')
          plt.ylabel('True Positive Rate (TPR)')
          plt.title('Receiver Operating Characteristic (ROC) Curve')
          plt.legend(loc='lower right')
          plt.grid()
          plt.show()
      else:
          print("No target column found in the custom dataset. Predictions have been generated, but no evaluation metrics can be calculated.")

      # Return predictions for further analysis
      return y_pred, y_pred_proba

- _Example Usage_

  python
  if **name** == "**main**": # Path to the saved model
  model_file = 'final_gst_model.joblib'

      # Path to the custom test dataset (replace with the path of your test dataset)
      custom_test_file = "path/to/your/X_Test_Data_Input.csv"  # Replace with actual file path

      # Evaluate the model and make predictions
      predictions, probabilities = evaluate_saved_model(model_file, custom_test_file)

## Notes

- _Model File_

  - The saved model file final_gst_model.joblib is included in the repository.
  - Ensure this file is in the same directory as main_testing.ipynb when running the notebook.

- _Test Dataset Format_

  - The custom test dataset should be in CSV format and structured similarly to the training data.
  - Ensure that the feature columns match those expected by the model.
  - The dataset should contain an ID column and may include a target column if available.

- _Dependencies_

  - The notebooks use standard Python libraries as listed above.
  - Install any missing libraries before running the notebooks.

## Contact Information

If you have any questions or need assistance, please contact:

- _Team ID_: GSTN_30

Thank you for considering our submission. We look forward to your feedback.

---
