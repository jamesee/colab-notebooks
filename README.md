
# 1. AIAP Technical Assessment

Submitted by :	Ee Chee Hong <br> 
email: james.ee.sg@gmail.com

# 2. Overview of the submitted folder and its structure.

```bash
.
├── README.md
├── data
│   └── news_popularity.db
├── eda_part1.ipynb
├── eda_part2.ipynb
├── images
│   ├── pairplot1.png
│   ├── pairplot2.png
│   ├── pairplot3.png
│   └── pairplot4.png
├── requirements.txt
├── run.sh
└── src
    ├── classification.py
    ├── myeda.py
    ├── myeda_constants.py
    ├── myml.py
    ├── mymodels.py
    ├── mymodule.pyc
    ├── myscript.py
    ├── mysql.py
    └── regression.py

```


# 3. Execute the script

```
./run.sh
```

# 4. Creating the ML pipeline

This machine learning pipeline relied heavily on the Pandas DataFrame pipe function <br>
(see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html).

* Step 1
	* Develop a load_data() function and point it to data/news_popularity.db in custom module src/mysql.py

* Step 2
	* Do the EDA on a jupyter notebook to develop functions for use in Pandas pipe function.

* Step 3
	* Copy those functions into a custom module file (in this case src/myeda.py) for use in machine learning programs (in this case src/regression.py and src/classification.py) 
	* All constants (list, class etc) used in EDA (eda.ipynb) is managed centrally with a file (src/myeda_constants.py).

* Step 4
	* Develop ML functions (e.g. my_machine_learning() in src/myml.py) using an editor (in my case I used jupyter-notebook. See task2_regression_workingcopy.ipynb and task2_classification_workingcopy.ipynb) and copy it together with its evaluation functions and constants to custom module src/myml.py.

* Step 5
	* Copy the model settings into src/mymodels.py

* Step 6
	* Port over the ML functions into python scripts (src/regression.py and src/classification.py)

* Step 7
	* Modify the run.sh bash script to point to the respective python programs.


The above described the steps to develop a machine learning pipeline. <br>

Once the custom modules are developed, the ML programs will import those modules and directly pull raw data from the database and process the EDA steps on the fly, by-passing the eda.ipynb.


# 5. Overview of key findings in Task1

Key findings from the EDA are as follows:

## 5.1 Skewness of columns
* We observed that all 3 tables are heavily positive-skewed. It means that there are extreme values (outliers). Outlier treatment with all the numeric columns of all 3 tables are needed.
* We noticed the skewness of can be as high as 198. Desirable skewness should be between -1 and 1. This give us the hints that we need to perform log transform on these highly skewed columns.
* From ploting the boxplot charts, we realised that just by doing IQR outlier treatment alone is insufficient. There are still many data points outside the upper and lower whisker. We need to perform log transform on the following columns:
	* n_tokens_content
	* n_unique_tokens
	* n_non_stop_words
	* n_non_stop_unique_tokens
	* num_hrefs
	* num_imgs
	* n_comments
	* average_token_length
	* self_reference_min_shares
	* self_reference_max_shares
	* self_reference_avg_shares

## 5.2 Missing data
* Only table "description" has missing data which requires missing data handling. The other 2 tables have no missing data.
* Replaced the missing data with the new mean of outlier-treated column.

## 5.3 Outlier treatment
* I have prepared IQR and Z-score outlier identification methods. Tried the data on both and realised that IQR outlier identification gave a better outcome in term of reducing the number of data points outside the whiskers.
* Decided to replace outlier values with median of column as the mean of column is affected by these outliers.
* Observed that the skewness improved greatly after performing the IQR-based outlier treatment

## 5.4 Correlation
* Investigated the correlation of 'timedelta' column and target "shares' column. Both seems to have low correlation.
* "n_unique_tokens" and "n_non_stop_unique_tokens" have strong negative correlation with "n_tokens_content"
* "self_reference_min_shares" has strong positive correlation with "self_reference_max_shares"
* "kw_avg_max", "kw_min_avg", "kw_max_avg" all have strong correlation
* "kw_avg_min" and "kw_min_min" have strong correlation with "timedelta"

Correlation information is useful when doing feature selection. For example, if we have highly correlated variables, we can choose one and drop the others when we want to reduce the number of features.

## 5.5 One-Hot-Encoding
* Column "weekend" and "data_channel" are both object datatype. Both needed to perform one-hot-encoding before feeding into ML models.

## 5.6 Misc
* The number of rows of table "keywords" is 35680 which is 3964 short of that of table "article" and "description" (39644). Need to handle this difference when the 3 tables merge into one.
* Column "url" will be dropped as it cannot be used for ML models
* To answer the sample question on when were the data collected, the url links in the "url" column gave us clues that the data was acquired between 7th Jan 2013 and 27th Dec 2014
* The statistics of column 'shares' are as follows:
	* count 39644.000000
	* mean 3395.380184
	* std 11626.950749
	* min 1.000000
	* 25% 946.000000
	* 50% 1400.000000
	* 75% 2800.000000
	* max 843300.000000


# 6. Evaluation of models developed in Task2

## 6.1. Regression Results
```bash
************************************** Results
                            Mean Squared Error  R2 score  Mean Absolute Error
randomforest_estimator_10             0.569575  0.341203             0.544461
randomforest_estimator_50             0.520850  0.397561             0.518998
randomforest_estimator_100            0.513972  0.405516             0.515407
LinearRegression                      0.642584  0.256758             0.600683
**************************************
```

1. The above are just some metrics for evaluating regression model. The R2 score, which ranges between 0 and 1, is the most commonly use metric as it provide some indications on the goodness of fit of the model to the data. The better regression fit the closer the R2 score to 1. On the other hand, the other 2 metrics (MSE and MAE) are just for relative comparison as there are no benchmark to tell whether the model is good or not just based on a standalone metric reading.<br>

1. For our case, by all metrics (i.e. R2, MSE and MAE),  Random Forest model is definitely a better model for our data set as compared to just simple linear regression model. Random Forest with estimator of 100 is marginally better than Random Forest with estimator of 50, and estimator of 50 is better than estimator of 10. However, the R2 metric tells us that the use of Random Forest model is below expectation as even with estimator of 100, as the R2 score is just 0.4.  

1. We can improve our R2 score just by increasing the number of estimators albeit computationally expensive. However, it will come a point where there will be diminishing return on the R2 score improvement with just pure increase in estimators. The process of finding the optimal number of estimators is called hyper-parameters tuning.

1. Due to the time constraint, I was unable to perform such hyper-parameters tuning using the Scikit Learn GridSearchCV or the more recent Keras-tuner.  

Reference:
[link to Google!](http://google.com)
[GitHub](http://github.com)
https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
![Classification Accuracy is Not Enough: More Performance Measures You Can Use](https://www.sicara.ai/blog/hyperparameter-tuning-keras-tuner)
<br>

 
## 6.2. Classification Results
```bash
************************************** Results
                                     Accuracy  Precision    Recall  F1 score   ROC AUC
RandomForestClassifier_estimator10   0.699178   0.752520  0.644581  0.694381  0.702684
RandomForestClassifier_estimator50   0.724496   0.757851  0.705903  0.730955  0.725690
RandomForestClassifier_estimator100  0.727205   0.756184  0.716476  0.735794  0.727894
KNeighborsClassifier_n3              0.614070   0.640364  0.620617  0.630336  0.613649
LogisticRegression                   0.690863   0.699494  0.730925  0.714864  0.688290
SupportVectorClassification          0.704129   0.716655  0.730925  0.723720  0.702408
**************************************
```

1. AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) is the most important evaluation metrics for any classification model's performance. An excellent model has AUC near to 1, which means it has good measure of separability. For a model with AUC under 0.5 indicate that the classifier performs worse than a random classifier. For our case, all our models have AUC above 0.5. The Random Forest models seemed (I am being careful here :) ) a better choice for our data as we can achieve more than 0.7. Again, the more number of estimators the better the AUC-ROC score. Hyper-parameter tuning is needed to find the most optimal number of estimators. 
 
![AUC-ROC Curve](images/AUC-ROC-Curve.png)

2. In this simple exercise, it is non-conclusive that Random Forest model is a better choice than Support Vector Classifier or KNN Classifier or Logistic Regression. This is because we still have not done the hyper-parameters tuning for these models. SVC, KNN and Logistic Regression models may perform better after hyper-parameters tuning. Further investigations are needed.

3. Jason Brownlee of Machine Learning Mastery had an excellent post stating that Accuracy alone is not enough for a classification problem. We need more performance measures like Precision, Recall and F1 score. Precision is the number of True Positives(TP) divided by the number of True Positives(TP) and False Positives(FP). Recall is the number of True Positives divided by the number of TP and False Negatives (FN). The Precision and Recall metrics are of inverse relationship. The F1 Score is the 2*((precision*recall)/(precision+recall)). F1 score conveys the balance between precision and recall. Depending on the situation, we may want a higher Recall metric and thus sacrifice the Precision metrics. For example, in the breast cancer dataset cited by Jason, a lower Recall means higher False Negatives which we cannot afford for diagnosis of recurrence of breast cancer. 

4. For our case, I assessed that a balanced Recall and Precision metrics (i.e. a appropriate high F1 score) will be good for our objective. We can see that the precision and recall metrics are quite close for all our models. This is because I have intentionally chosen a bin divider (in this case the 50 percentile) to avoid the large imbalance class issue.


Reference:
* [test](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
* [Classification Accuracy is Not Enough: More Performance Measures You Can Use -  by Jason Brownlee](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/#:~:text=F1%20Score,the%20precision%20and%20the%20recall)




# 7. Other considerations for models developed

* If given more time, I would like to do the followings:
	1. Using deep learning models on the dataset
	1. Perform hyper-parameters tuning using keras-tuner
	1. Feature selections

