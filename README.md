# Prophet-ML 🏠

A lightweight machine learning pipeline using **Linear Regression** to predict median California house values based on regional demographic and architectural features. 

This repository serves as a practical introduction to data preprocessing, model training, performance evaluation, and data visualization using `scikit-learn`.

---

## 🚀 Features & Project Structure

The project builds an end-to-end machine learning pipeline using a clean subset of the California Housing dataset:
* **Selected Predictors:** Trains on 3 critical features:
  * `MedInc`: Median income in block group
  * `HouseAge`: Median house age in block group
  * `AveRooms`: Average number of rooms per household
* **Pipeline Steps:** Preprocessing ➡️ Train/Test Data Splitting (80/20) ➡️ Linear Model Training ➡️ Prediction Evaluation ➡️ Results Visualization.

---

## 📊 Model Performance & Results

Upon testing against hidden evaluation data, the model yielded the following baseline statistics:

* **Root Mean Squared Error (RMSE):** `0.8117`  
* **R-squared ($R^2$) Score:** `0.4972` *(The model successfully accounts for ~50% of the pricing variations)*

### Custom Evaluation Example
* **Predicted Valuation for a Custom House Profile:** `$249,389.23`

---

## 🛠️ Installation & Setup

Ensure you have Python 3 installed on your machine. Install the required dependencies using `pip3`:

```bash
pip3 install numpy pandas scikit-learn matplotlib
