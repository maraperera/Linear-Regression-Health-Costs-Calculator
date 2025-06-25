# Health-Costs-Calculator (Linear Regression)

This project aims to predict individual medical **expenses** based on various demographic and lifestyle factors. It uses a regression model built with **TensorFlow Keras** to learn patterns from a dataset of healthcare consumers.

---

## Project Overview

The goal of this project is to develop a machine learning model that can accurately predict healthcare costs. The dataset includes features such as `age`, `sex`, `bmi`, `children`, `smoker`, and `region`. The model is trained to minimize the **Mean Absolute Error (MAE)**, with a target MAE of under $3500 to pass the challenge.

---

## Dataset

The dataset used for this project is `insurance.csv`, sourced from FreeCodeCamp. It contains the following columns:

* `age`: Age of the primary beneficiary.
* `sex`: Gender of the primary beneficiary (male/female).
* `bmi`: Body Mass Index, providing an understanding of body, relative to height and weight, objectively.
* `children`: Number of children covered by health insurance / number of dependents.
* `smoker`: Whether the beneficiary smokes (yes/no).
* `region`: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
* `expenses`: Individual medical costs billed by health insurance.

---

## Project Structure

The project is implemented as a **Google Colaboratory notebook**, which can be easily run in your browser. The notebook covers the following steps:

1.  **Import Libraries**: Essential libraries like TensorFlow, Keras, Pandas, Matplotlib, and NumPy are imported.
2.  **Data Loading**: The `insurance.csv` dataset is downloaded and loaded into a Pandas DataFrame.
3.  **Data Preprocessing**:
    * Categorical features (`sex`, `smoker`, `region`) are converted into numerical representations using **label encoding** (as shown in the provided code).
    * The dataset is split into training (80%) and testing (20%) sets.
    * The "expenses" column is separated from both training and testing sets to serve as the labels for model training and evaluation.
4.  **Model Definition**: A **sequential Keras model** is built with dense layers and ReLU activation functions. The output layer has a single neuron for regression.
5.  **Model Training**: The model is trained using the **RMSprop optimizer** and `mean_squared_error` (`mse`) as the loss function, with `mae` and `mse` as metrics. Training progress is visualized (implicitly, via `tfdocs.modeling.EpochDots()`).
6.  **Model Evaluation**: The trained model's performance is evaluated on the unseen test dataset. The key metric for success is the **Mean Absolute Error (MAE)**.
7.  **Prediction and Visualization**: The model makes predictions on the test dataset, and these predictions are plotted against the true values to visually assess the model's accuracy.

---

## Getting Started

To run this project:

1.  **Open the Notebook**: Click on the following Google Colab link:
    * [Link to Google Colab Notebook (Once you create it and share publically)](https://colab.research.google.com/github/maraperera/Linear-Regression-Health-Costs-Calculator/blob/main/Linear%20Regression%20Health%20Costs%20Calculator.ipynb)
2.  **Run All Cells**: In Google Colab, go to `Runtime` -> `Run all`.

The notebook will download the data, preprocess it, train the model, and evaluate its performance. The final cell will display the MAE and indicate whether the challenge was passed.

---

## Results

The model aims to achieve a Mean Absolute Error (MAE) of **less than $3500**. A lower MAE indicates better prediction accuracy. The final cell of the notebook will output the MAE on the test set and confirm if the challenge's target MAE was met.



---

## Technologies Used

* **Python 3**
* **TensorFlow**
* **Keras**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **scikit-learn**

---

## License

This project is open-source and available under the [MIT License](LICENSE). *(It's good practice to include a LICENSE file in your repository. You can create one by selecting "Add file" -> "Create new file" on GitHub and naming it `LICENSE`.)*
