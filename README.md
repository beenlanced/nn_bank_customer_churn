# Predicting Bank Churn Using Neural Networks (NNs)

<p>
  <img alt="Bank Churn Figure for Neural Network" src="imgs/bank_churn_banker.gif"/>
</p>

[img source: Vic López](https://dribbble.com/shots/5246002-Oppressive-banker)

## Project Description

This project came out of the [Post Graduate program in Artificial Intelligence and Machine Learning (PGAIML): Business Applications University of Texas at Austin Course](https://la.utexas.edu/greatlearning/uta-artificial-intelligence-machine-learning.html).

This project aims to develop a predictive model to identify potential customer churn for abanking context. The analysis utilizes neural network-based classification algorithms to predict whether a customer will leave the bank within the next six months.

### The Problem

Businesses such as banks providing services must concern themselves with the issue of `"Customer Churn"`. Churn is when customers switch to another service provider. For businesses, it is crucial to comprehend the factors that influence a customer's decision to bounce. By understanding these factors, businesses management can guide their management's efforts to enhance business services to retain customers and understand which services might have higher priorities over others.

**As the bank's data scientist build a neural network based classifier that can determine whether a customer will leave the bank within the next 6 months.**

### Some Questions to Keep in Mind

- How many active members are there with the bank?
- How do the rates of churn compare by gender?
- How are rates of churn correlated to geography?
- How are the different customer attributes correlated to each other?
- What is the distribution of the credit score of customers?
  - Are there any noticeable patterns or outliers in the distribution?

### What this Project Does Specifically

The project:

- Loads and inspects the banking data
- Preprocesses/cleans the data
- Performs exploratory data analysis (EDA)
  - Statistical summary of the data
  - Normalize the data
  - Feature engineering
    - Categorical encoding
  - Univariate analysis
  - Bivariate analysis
- Tests for balanced and imbalanced data sets
- Use different methods mentioned below to improve the model by finding the optimal threshold using
  **Area Under the Receiver-Operating Characteris (ROC) Curve [(AUC)]** curves for each of the methods
- Builds a NN model with SGD optimizer
- Builds a NN model with Adam optimizer
- Builds a NN Network model with Dropout and SGD optimizer
- Builds a NN Network model with Dropout and Adam optimizer
- Builds a model with balanced data by applying SMOTE and SGD optimizer and Droput
- Builds a model with balanced data by applying SMOTE and Adam optimizer and Dropout
- Assess the model performance for each of the created models
- Chooses the best model among the created models
- Creates a Deep Learning Keras built neural network to build the prediction model
- Conducts analysis of the predictive model results

### Summary, Actionable Insights, and Business Recommendations

The best model is the `simple dropout model with a 12.72% FN rate (Model with SMOTE, ADAM, and DROPOUT)`, and this is the model to take to production.

This model produced the following

**Reduced overfitting:** Seen in validation loss figures amongst all of the models tested.

**Accuracy:** The model achieved an accuracy of approximately 81%, indicating that it correctly classified 81% of the instances in the test dataset.

**Precision:** For predicting churn (class 1), the precision is 53%, implying that when the model predicts churn, it is correct 53% of the time. For non-churn (class 0), the precision is 94%.

**Recall:** The recall for churn (class 1) is 69%, meaning that the model identified 69% of the actual churn cases correctly. For non-churn (class 0), the recall is 84%.

**F1-score:** The F1-score, which is the harmonic mean of precision and recall, is 0.60 for churn (class 1) and 0.88 for non-churn (class 0).

#### Business Recommendations

**Precision and Recall Balance:** The model demonstrates a trade-off between precision and recall. While it achieves a high precision for non-churn customers, the precision for churn prediction is relatively lower. However, it manages to capture a significant portion of actual churn cases with a reasonable recall. This indicates that the model can effectively identify potential churners, albeit with some misclassifications.

**Focus on Recall Improvement:** To improve the model's performance further, particularly in identifying churn cases, focus should be placed on improving recall without significantly sacrificing precision. This can be achieved through strategies such as feature engineering, exploring different model architectures, adjusting class weights, or collecting more diverse data.

**Utilizing Predictions for Customer Retention:** Despite its limitations, the model can still provide valuable insights for the bank's customer retention strategies. By leveraging the model predictions, the bank can prioritize efforts towards retaining customers who are predicted to churn, offering tailored incentives, personalized communication, or targeted marketing campaigns to mitigate churn risk.

**Continuous Model Monitoring, Improvement, and Refreshment:** It's essential to continuously monitor the model's performance and retrain it periodically with updated data. As customer behavior evolves over time, the model needs to adapt to capture new patterns and trends accurately. Regular evaluation and refinement of the model will ensure its effectiveness in supporting the bank's customer retention efforts.

**Conclusion:** While this Model provides a solid foundation for predicting customer churn, there's room for improvement to enhance its predictive power and applicability in real-world business scenarios. By focusing on refining the model and integrating its predictions into strategic decision-making processes, the bank can proactively address churn risk and foster long-term customer relationships.

---

## Objective

The project contains the key elements:

- `Area Under the Receiver-Operating Characteris (ROC) Curve [(AUC)]` scoring for classifer identification,
- `Deep Learning` for neural networks building,
- `Dropout`, regularization technique appilied to prevent overfitting by randomly turning off a fraction of the neurons during training,
- `Git` (version control),
- `imblearn` Python library to perform oversampling and undersampling for balancing data sets,
- `Jupyter` Python coded notebooks,
- `Keras` to build nodes and layers,
- `Matplotlib` visualization of data,
- `Numpy` for arrays and numerical operations,
- `Pandas` for dataframe usage,
- `Python` the standard modules,
- `Seaborn` visualization of data,
- `Scikit-Learn` to get training and test datasets,
- `SMOTE` to help with oversampling and balancing data sets,
- `TensorFlow` to build nodes and layers,
- `uv` package management including use of `ruff` for linting and formatting

## Tech Stack

![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

---

## Getting Started

Here are some instructions to help you set up this project locally.

---

## Installation Steps

The Python version used for this project is `Python 3.11` to be compatible with TensorFlow.

Follow the requirements for using TensorFlow [here](https://www.tensorflow.org/install/pip#macos)

use `uv pip install tensorflow`

- Make sure to use python versions `Python 3.9–3.12
- pip version 19.0 or higher for Linux (requires manylinux2014 support) and Windows. pip version 20.3 or higher for macOS.
- Windows Native Requires Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

### Clone the Repo

1. Clone the repo (or download it as a zip file):

   ```bash
   git clone https://github.com/beenlanced/nn_bank_customer_churn.git
   ```

2. Create a virtual environment named `.venv` using `uv` Python version 3.11:

   ```bash
   uv venv --python=3.11
   ```

3. Activate the virtual environment: `.venv`

   On macOS and Linux:

   ```bash
   source .venv/bin/activate #mac
   ```

   On Windows:

   ```bash
    # In cmd.exe
    venv\Scripts\activate.bat
   ```

4. Install packages using `pyproject.toml` or (see special notes section)

   ```bash
   uv pip install -r pyproject.toml
   ```

### View Notebooks to see Exploratory Data Analysis and Predicative Model Construction

---

## Dataset

The case study is from an open-source dataset from Kaggle: [Bank Churn Preditiion Using Neural Networks](https://www.kaggle.com/code/chandasaisanthosh/bank-churn-prediction-using-neural-network).

The dataset contains 10,000 sample points with 14 distinct features such as

- CustomerId,
- CreditScore,
- Geography,
- Gender,
- Age,
- Tenure,
- Balance

Deriving a Python Data Dictionary from the data set with the following keys:

- **customer_dd:** Unique ID which is assigned to each customer
- **surname:** Last name of the customer
- **credit_score:** Reflects the customer's credit history
- **geography:** Customer location
- **gender:** Customer gender
- **age:** Age of the customer
- **tenure:** Number of years the customer has been with the bank
- **num_of_products:** Number of products that a customer has purchased through the bank
- **balance:** Account balance
- **Has_credict_card:** Customer is or is not a credit card holder
  - **0=no**: Label (Customer does not have a credit card)
  - **1=yes** Label (Customer does have a credit card)
- **estimated_salary:** Estimated salary
- **is_active_member:** Is the customer active and regularly engaged with bank services
  - **0=no**: Label (Customer not active)
  - **1=yes** Label (Customer was active)
- **exited:** whether or not the customer left the bank within six month. It can take two values
  - **0=no**: Label (Customer did not leave the bank)
  - **1=yes** Label (Customer left the bank) -- they churned!

---

### Final Words

Thanks for visiting.

Give the project a star (⭐) if you liked it or if it was helpful to you!

You've `beenlanced`! 😉

---

## Acknowledgements

I would like to extend my gratitude to all the individuals and organizations who helped in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, have been invaluable. Thank you.

Specifically, I would like to acknowledge:

- The folks who host the [Post Graduate program in Artificial Intelligence and Machine Learning (PGAIML): Business Applications University of Texas at Austin Course](https://la.utexas.edu/greatlearning/uta-artificial-intelligence-machine-learning.html). Go Longhorns!

- Guidance from [Joseph Reeves](https://github.com/jreves/AIML_Projects/blob/main/Bank%20Churn%20Neural%20Network%20Modeling.ipynb) and [Arena Hernandez](https://github.com/ArenaHernandez/NeuralNetworks_BankChurnPrediction).

- [Hema Kalyan Murapaka](https://www.linkedin.com/in/hemakalyan) and [Benito Martin](https://martindatasol.com/blog) for sharing their README.md templates upon which I have derived my README.md.

- The folks at Astral for their UV [documentation](https://docs.astral.sh/uv/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details
