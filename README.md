
# Dengue Case Prediction

**Overview :**                                                            This project demonstrates a comprehensive approach to predicting dengue outbreaks by leveraging historical weather data and machine learning techniques. Dengue fever is a mosquito-borne disease that poses significant public health challenges, particularly in tropical and subtropical regions. Timely and accurate predictions of potential outbreaks can empower health authorities to take preventive actions, allocate resources more effectively, and mitigate the impact of the disease.

The project includes a complete pipeline from data collection, preprocessing, exploratory data analysis (EDA), model training, and evaluation. It showcases skills in data cleaning, feature engineering, visualization, and modeling with a focus on real-world applications of machine learning.


## Project Objectives
1.**Predict Dengue Outbreaks:** Use historical weather data (temperature, humidity, precipitation, etc.) to predict the number of dengue cases in a specific region.   

2.**Apply Machine Learning:** Build, train, and evaluate machine learning models to classify and predict dengue case severity.

3.**Feature Analysis:** Identify key weather patterns and environmental factors that most influence dengue outbreaks.

4.**Model Comparison:** Compare multiple machine learning algorithms to determine the most accurate model for predicting dengue cases.

## Key Features
**Data Preprocessing and Cleaning**   
- **Duplicate Handling:** Identifies and removes duplicate records to maintain data integrity.

- **Missing Value Imputation:** Detects and addresses missing values, ensuring that the dataset is complete for model training.

- **Feature Selection:** Removes irrelevant features (such as serial numbers) and selects key predictors like temperature, humidity, and wind speed.

**Exploratory Data Analysis (EDA)**   
- **Weather Pattern Analysis:** Visualizes the relationship between weather variables (temperature, precipitation) and dengue cases using seaborn and matplotlib.

- **Correlation Matrix:** Displays correlations between the different environmental factors and the target variable (dengue cases) to identify which factors have the highest predictive power.

**Machine Learning Modeling**  
- **Supervised Learning:** Models like Random Forest, Decision Trees, and Logistic Regression are applied to predict the dengue case labels (normal or outbreak).
- **Classification and Regression:** Both classification models (for outbreak labels) and regression models (for predicting the number of cases) are explored.

**Model Evaluation**  
- **Performance Metrics:** Evaluates models based on metrics such as accuracy, precision, recall, and F1 score to ensure the robustness of predictions.

- **Cross-Validation:** Applies techniques like K-fold cross-validation to prevent overfitting and ensure generalization across different data subsets.

- **Confusion Matrix:** Visualizes the true positives, false positives, true negatives, and false negatives to interpret model performance.
## Technologies Used
- **Programming Language:** Python
- **Libraries:**
    - **NumPy:** For numerical operations and handling multi-dimensional arrays.
    - **Pandas:** For data manipulation, cleaning, and preparation.
    - **Matplotlib and Seaborn:** For data visualization and exploratory analysis.
    - **Scikit-learn:** For machine learning algorithms, model training, and evaluation.
## Dataset Description
The dataset includes historical weather data combined with the number of dengue cases over a period. Each record contains multiple weather-related features and corresponding dengue case data.

- **Temperature:** Maximum, minimum, and average daily temperatures.
- **Humidity:** Daily humidity levels.
- **Precipitation:** Amount of rainfall.
- **Solar Radiation:** Measures of solar energy received.
- **Dengue Case Count:** The number of reported dengue cases for a given day.
- **Label:** A binary classification indicating whether the number of cases is considered a normal day or an outbreak day.
## How to Run the Project
**Steps to Run**  

1.**Clone the Repository:**  
https://github.com/mukusss/Dengue-Case-Prediction.git

2.**Open the Jupyter Notebook:**  
jupyter notebook Dengue_Case_Prediction.ipynb
## Future Enhancements
- **Incorporating More Features:** Including additional factors like population density and sanitation levels for a more holistic model.
- **Geographical Scaling:** Expanding the model to work across different regions with varying climate patterns.

- **Real-Time Predictions:** Integrating real-time weather APIs for dynamic dengue outbreak predictions.