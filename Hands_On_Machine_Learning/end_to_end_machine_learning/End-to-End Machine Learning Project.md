# Key Terms
1. **Block Group** - the smallest geographical unit for which the US Census Bureau publishes sample data. Also known as *districts*.
2. **Pipelines** - a sequence of data processing components.
	1. components are typically run asynchronously, where each component is relatively self-contained and as the pipeline runs each component pulls in data, processes it and returns a result to the data store. The data store operates as the interface in this system.
3. **Multiple Regression** - multiple features are used to make a prediction.
4. **Univariate Regression** - when the prediction made is a single value. 
5. **Multivariate Regression** - when the prediction made contains multiple values. 
6. **Standard Deviation** - measures how dispersed the values are. 
	1. For more info: [[Descriptive Statistics#Standard Deviation| Standard Deviation]]
7. **Percentiles** - indicate a value below which a given percentage of observations in a group of observations fall. 
	1. For more info: [[Descriptive Statistics#Percentiles todo|Percentiles]]
8. **Linear Correlation** - is a number that tells you strongly two things move together. 
9. **Imputation** - the process of replacing missing data with substituted values to maintain dataset integrity for machine learning models.
# The Problem
The current estimation methods for median house price by district is a manual human-lead process that is not only time-consuming and costly but often has error margins as large as 30%. Machine Learning House Corp would like us to use California Census data to build a model of housing prices across the state that can predict the median house price of a given district.
## The Data
Organized into "Block Groups"<sup>1</sup> that contain metrics such as population, median income, and median housing price. Here we are going to call "Block Groups" *districts*.

In our data, each *row* represents a single district, and each *column* represents a single feature of the data. 
- Note: features == attributes

Below we can view how many entries there are, the attributes of the data, and what the data type of each attribute is. 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   longitude           20640 non-null  float64
 1   latitude            20640 non-null  float64
 2   housing_median_age  20640 non-null  float64
 3   total_rooms         20640 non-null  float64
 4   total_bedrooms      20433 non-null  float64
 5   population          20640 non-null  float64
 6   households          20640 non-null  float64
 7   median_income       20640 non-null  float64
 8   median_house_value  20640 non-null  float64
 9   ocean_proximity     20640 non-null  object 
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
```

### View the Data
Below are snapshots of how we can view the data for the attribute *longitude*. 
#### Descriptive Statistics<sup>6,7</sup>

![[homl_e2e_descriptive_stats.png]]
#### Histogram
![[homl_e2e_histogram.png]]

## The System
Based on the data we observe that because the data is labeled we have a [[The Machine Learning Landscape#Supervised Learning | supervised learning]] problem.

Further, because the output is a single value generated using multiple inputs we have a *multiple regression*<sup>3</sup> and *univariate regression*<sup>4</sup> problem. 

Lastly, because we aren't dealing with a continuous flow of data entering our system, we don't need to make adjustments to changing data often. The data is also relatively small and can fit easily into memory, thus we will use [[The Machine Learning Landscape#Batch Learning|batch learning]] to train our model. 

# Model Training
Since we know we're going with a *univariate multiple regression* model our next steps would be to set up how we're going to [[The Machine Learning Landscape#Model Training| train the model]]. 
## Performance Measure
A common choice for a performance measure for a regression model is a *Loss Function*. We want to chose this type of function because we want to be able to have an idea of how much error the system makes in its predictions. 

For our regression we will use [[Linear Regression#Root Mean Squared Error|RMSE]] or better know as the Root Mean Squared Error. This performance measure allows us to view the difference between our predicted values and our labeled (actual) values. 
- **Note:** RMSE is taking the difference between our actual labeled value $y_i$ and the predicted value $\hat {y_i}$ to calculated the error the model made on the $i^{th}$ instance. (See [[Linear Regression]] for more info). 

$RMSE(X,y,h) = \sqrt{\dfrac{1}{m} \displaystyle\sum_{i=1}^{m}(y_i - h(x_i))^2}$
where,
- $m$ - number of instances in the dataset we are measuring the RMSE on.
- $x_i$ - a vector of all the feature values, excluding the labeled actual value, of the $i^{th}$ instance of the dataset.
- $y_i$ - the actual value of the $i^{th}$ instance of the dataset.
- $X$ - the matrix containing all the feature values, excluding the labeled actual values, of all the instances in the dataset. 
	-  the $i^{th}$ row  of this matrix is the transpose of $x_i$ denoted as $(x_i)^T$. 
- $h$ - the systems prediction function or *hypothesis*.
	- when our system outputs a prediction based on the feature vector $x_i$, that predicted value is $h(x_i)$ or better known as $\hat y_i$ .
#### But what if there are many outliers?
RMSE is rather sensitive to outliers and as such other performance metrics may evaluate the model better, such as:
- MSE - [[Linear Regression#The Loss Function - Mean Squared Error| Mean Squared Error]]
- $R^2$ - [[Linear Regression#$R 2$ | R-squared]]
- Adjusted $R^2$ - [[Linear Regression#Adjusted $R 2$ | Adjusted R-squared]]

## Train, Test Sets
>[!tip] Spend time creating your test set so that when you evaluate your model based on it, your evaluation results are as precise as they can be.

We must split our data into 2 distinct sets. *Train* and *Test*. The data is usually split $80:20$, train : test. 
- to avoid *snooping bias*, we must not view the test set because our brains are really good at detecting patterns where there are none. If we are biased then we can be too optimistic with our generalization error estimate. 
#### How do we get the split to be the same every time?
1. Set the *seed* in the random number generator.
#### What happens if the data gets updated?
The train-test split will be different than the previous and this the model will be run on a different data. 

We can handle this by ensuring that every update the test set will contain 20% of the new instances but will *not* contain any instance that was previously in the test set:
1. Compute a hash using each instances's identifier -- this assumes that each instance has a unique and immutable identifier.
2. Find the maximum hash value of the data set. 
3. Place the instance in the test set $iff$ the hash is lower than or equal to 20% of the maximum hash value.
#### Sampling Methods
**Randomized** - data is collected randomly from the population #todo 
**Stratified** - the population is separated in to homogenous subgroups called *strata*, and the data is collected such that it is representative of the population.

## Data Exploration
The goal of data exploration is to better understand the dataset and its features. To accomplish this we can:
- create plots to better visualize the features of the data
- transform the data
- observed how features might relate to one another
- watch for correlated features
### Tools
#### Standard Correlation Coefficient
This is also known as *Pearson's r*. It measures *linear correlations*<sup>8</sup> "as x goes up y goes up/down".

| $-1.0$   | $0.0$ | $1.0$    |
| -------- | ----- | -------- |
| negative | none  | positive |
| up/down  | N/A   | up/up    |
*NOTE* correlation doesn't necessarily have a relationship to the slope of the line. We can see that in the second line of graphs. The third line of graphs are *non-linearly correlated*
![[chp2_figure_2-16.jpg]]

#### Feature Combinations
Some features aren't as useful at describing the data or the relationship between it and the target by themselves as they are when combined with another feature. 

A simple way to combine features is to get the ratio of one to another. 

We can then compare our new combined features to our individual features against our target and see if we have notable changes and thus should keep the new feature. 
- one option is comparing them through via calculating the [[End-to-End Machine Learning Project#Standard Correlation Coefficient| standard correlation coeff.]] of the new feature to the target.
- we seek [[Linear Regression#Linearity between target and features|Linearity]] between our target and our features. 
**NOTE** features that are too linearly correlated can cause issues with some models. This phenomenon is called [[Linear Regression#No Multicollinearity between features | multicollinearity]].
- simple weighted sums often produce collinearity into the new feature.
- [[Linear Regression]] models are particularly sensitive to this. 

### So what do we gain?
- identify quirks in the data that may need to be cleaned
- identify interesting correlations between features and our target.
- identify features that have skewness that will likely need to be transformed to better improve our model's ability to learn. 
- identify feature combinations that better describe the relationship between the individual features and the target. 
- if a model has been selected, we can test the relationships between the features against the ==assumptions== the model needs to be true in order to succeed. From there we can identify features that might fail those assumptions and clean them up. 
## Clean and Prepare the Data
1. Remove the target feature from the training set. 
2. Remove any new features you created solely for the purpose of data exploration. 
3. Identify and handle missing data in features.
	1. remove rows with the missing feature
	2. remove the attribute entirely
	3. set the missing values to some value such as 0, the mean, median or more etc... through *imputation*<sup>9</sup>. The value chosen depends on the datatype of the feature. 