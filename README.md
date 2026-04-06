
Vehicle Fuel Consumption


TABLE OF CONTENTS

No.	 Contents 	 Page no.
1.	Introduction	2
2.	Dataset Description	3
3.	Dataset pre-processing	5
4.	Feature Scaling 	6
5.	Dataset splitting	7
6	Model Training & testing	7
7.	Model Selection/Comparison Analysis	11
8.	Conclusion	17







1. Introduction: 
The "Vehicle Fuel Consumption Prediction" project aims to explore vehicle characteristics, analyze fuel efficiency, and assess environmental impacts. By leveraging a comprehensive dataset, this project identifies critical factors influencing fuel consumption and carbon emissions. It applies machine learning models to predict metrics such as miles per gallon (MPG) and carbon dioxide emissions, providing actionable insights for stakeholders including consumers, automotive industry professionals, and policymakers.
This initiative is motivated by the need for sustainable transportation solutions and informed decision-making regarding fuel-efficient technologies.
2. Dataset Description:

Link: fuel dataset
Reference: https://www.kaggle.com/datasets/tanishqdublish/vehcile-fuel-consumption
The dataset has 38,113 data points and a total of 81 features. It seems extensive, with a mix of numerical and categorical data types, spanning vehicle characteristics and fuel consumption metrics.

Regression or Classification:
❖	The dataset includes various continuous numerical outputs like city_mpg_ft1, highway_mpg_ft1, city_electricity_consumption, and more. These suggest that the primary problem could be regression-focused, aimed at predicting fuel efficiency or consumption metrics based on vehicle features. However, we used classify scores and wrapped the labels within 3 classes (High, Low, Medium). And then dealt with a classification problem. 



Features in the Dataset

❖	Total 81 Features including categorical (e.g., vehicle make, model, transmission) and numerical (e.g., MPG, carbon emissions)  and 38,113 data points before preprocessing.
❖	Input Features: Inputs likely include categorical variables such as make, model, class, drive, transmission, fuel_type, along with numerical inputs like year, engine_cylinders, engine_displacement.
❖	Output Features: Potential outputs (targets for prediction) could be variables like city_mpg_ft1, highway_mpg_ft1, which measure fuel efficiency in different conditions.
❖	In the Dataset, the distribution of the “High” Class. Represent using a bar chart of 3 classes:
 

3. Dataset preprocessing  

Fault 1: NULL VALUE 
●	Several features have over 10,000 missing values, making them unreliable for analysis.
●	Examples include annual_fuel_cost and engine_displacement.
●	These missing values reduce the dataset's quality and could bias the model if left unaddressed.
Solution (Handling Null Values):
●	Columns with more than 10,000 missing values were dropped as the level of missingness made them unsuitable for imputation.
●	For columns with fewer null values, imputation techniques were applied by using “DT.dropna”. 
●	These steps reduced the dataset dimensions from 81 to 68 features, retaining only the most useful attributes.
Fault 2:
Categorical Features:
●	Columns like make and model have high cardinality (a large number of unique values).
●	High cardinality can introduce complexity and overfitting in machine learning models, as these variables might dominate others during training.

Solution (Encoding Categorical Features):
●	Categorical columns such as fuel_type and transmission were converted into numerical format using Label Encoding, assigning unique integer values to each category.
●	This transformation ensures compatibility with machine learning models that require numerical inputs. Post-encoding validation ensured all categorical data were successfully converted into numerical formats.

4. Feature Scaling
Feature scaling is a crucial preprocessing step in this project to ensure that all numerical features contribute equally to the model's performance. In this project, Min-Max Scaling was used as the feature scaling technique.

Why Min-Max Scaling Was Used:
Min-Max Scaling normalizes the range of numerical features to a specific range, typically [0, 1]. This is particularly useful when features have different scales, which could disproportionately influence the model's performance. By scaling the data, the project ensures that:
1.	Features with larger ranges do not dominate those with smaller ranges.
2.	Gradient-based optimization algorithms (e.g., in neural networks) converge faster.
3.	Distance-based models (e.g., k-Nearest Neighbors) perform optimally.
5. Dataset splitting 
The code snippet uses the train_test_split function from scikit-learn to divide the dataset new_DT into training and testing sets. Specifically, it separates the features and the target column ('NEw col') into x_train, x_test, y_train, and y_test. It allocates 70% of the data to the training set and 30% to the testing set, as specified by test_size=0.3. The random_state=42 ensures that the split is reproducible, meaning the same split will occur every time the code is run. This division helps in training machine learning models on the x_train and y_train datasets, and subsequently testing them on the x_test and y_test datasets to evaluate model performance. The shapes of the resulting training and testing sets are then printed to confirm their dimensions. 

6. Model Training & testing
❖	K-Nearest Neighbors (KNN) classifier using scikit-learn, it initializes a KNN classifier with 3 neighbors and fits this model to the training data (x_train, y_train). After training, the model predicts the labels for the training data itself, and the accuracy of these predictions is calculated using the accuracy_score function. The accuracy, which represents how well the model performs on the training set, is then printed. This process helps in understanding the model's learning effectiveness on the training data before it's tested on unseen data.
 
 

         
❖	In Support Vector Machine (SVM) classifier using scikit-learn in Python. The classifier is trained on a dataset, and its accuracy is evaluated on both the training and test sets, showing around 93.95%  accuracy for the training set. However, there appears to be a misuse in the process, as the classifier is retrained on the test set, which is a methodological error, as the test set should only be used for evaluation, not training.
 
 

❖	We use the Gaussian Naive Bayes classifier. This model is trained on the training data and evaluated on both the training and test data. The classifier shows a training accuracy of approximately 93.09% and a test accuracy of about 93.82%. However, there’s an error in the workflow: a new model instance is created and mistakenly fitted on the test data, which is a fundamental mistake in machine learning practice since the test set should only be used for final evaluation to prevent data leakage and overfitting
❖	 
 







7. Model selection/ comparison Analysis 
For our project, we used three models to predict a Student Stress Level; KNN (K-Nearest Neighbors), SVC (Support Vector Classifier), and GaussianNB (Gaussian Naive Bayes), 
❖	The bar chart visualizes the accuracy levels of three different machine learning models: KNN (K-Nearest Neighbors), SVC (Support Vector Classifier), and GaussianNB (Gaussian Naive Bayes). Each model's bar is colored differently—red for KNN, green for SVC, and blue for GaussianNB—to easily distinguish between them. The y-axis represents the accuracy level, ranging from 0 to 1 (0% to 100%). The x-axis lists the different models. This chart provides a clear comparison of the performance of the three models on a certain task, likely the fuel consumption classification mentioned in the title "Bar Chart of Fuel Consumption". The actual accuracy values are not visible in this description, but from the chart, it looks like the SVC model might have the highest accuracy, followed closely by GaussianNB, with KNN having the lowest accuracy among the three. 
 



 


K-Nearest Neighbors (KNN): 
 
 
 
13 KNN classifies data points based on the majority class among their k nearest neighbors. It measures the distance between points in a multi-dimensional space, assigning the class most common among its k nearest neighbors. 


Support Vector Classifier (SVC): 
 
 
 
SVC finds the optimal hyperplane in a high-dimensional space that separates classes with the widest margin. It transforms data into a higher-dimensional space and finds the hyperplane by maximizing the margin between classes. 

Gaussian Naive Bayes (GaussianNB): 
 
 
 
Using Bayes ' theorem, GaussianNB calculates the probability of a data point belonging to each class based on the feature values. It assumes that features are independent and follow a Gaussian distribution, hence "naive", and then selects the class with the highest probability


8. Conclusion
In our project, we tested three different classification models to predict the Vehicle Fuel Consumption. All the models we used in the project work in a different mechanism, but the purpose of all of them here is to determine which model efficiently uses fuel . Every model could do that in their own way. We used the same three metrics to count accuracy percentages (Precision, Recall, and F-1 Score) for all models, so that they all can be compared on the same scale. The dataset presents a comprehensive collection of vehicle attributes and fuel consumption metrics, suitable for exploratory data analysis and predictive modeling. However, challenges such as null values and categorical data require preprocessing techniques like imputation and encoding for effective analysis. With proper handling, this dataset offers valuable insights into factors influencing fuel efficiency and can support the development of robust predictive models.
