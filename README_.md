# Customer Churn Prediction with AdaBoost  

This project explores the application of **AdaBoost** for customer churn prediction. The model aims to identify churn customers, helping prevention measures to be taken in advance.  

## Key Features  

- **Data Preprocessing:**  
  - Removal of redundant columns  
  - Encoding categorical variables (frequency & ordinal encoding)  
  - Handling missing values with KNN imputation  
  - Data normalization using Min-Max scaling  

- **Model Selection: AdaBoost**  
  - Boosting technique that combines weak learners for improved accuracy  
  - Less prone to overfitting compared to complex ensemble models  
  - Balanced trade-off between interpretability and performance  

- **Handling Imbalanced Data:**  
  - Applied **SMOTENC** oversampling to improve prediction on minority classes  
  - Ensured oversampling only applied to training data to prevent data leakage  

- **Hyperparameter Tuning:**  
  - Used **GridSearchCV** to optimize key parameters  
  - Evaluated performance based on F1-score  

## Results & Insights  
- AdaBoost effectively balances interpretability and accuracy for default prediction  
- SMOTENC significantly improved model sensitivity toward minority classes  
- Hyperparameter tuning optimized AdaBoost’s performance on the dataset  

## Future Improvements  
- Exploring other boosting models like XGBoost and CatBoost for comparison  
- Implementing feature selection to enhance model efficiency 
