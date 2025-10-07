## So what is it?
Linear Regression is used to predict(find) the linear relationship between the independent variable(s) and the dependent variable of a dataset where these variables are **continuous** aka numerical.

## Key Terms
### Continuous Variable
A variable that can take on a infinite number of distinct values, possibly within a range of values.
- For example, numerical type data
### Bias
The amount to which a models predictions differ from the target values on the training data, i.e. *Training Error*
### Variance 
An indicator of how much the models performance would differ if different training data were used. i.e. a small change in the dataset has a *high* impact on the models performance. 
- In the case of Linear Regression, as we will see below, we want to understand the standard error around our estimated feature weights, $\hat \theta$.

$variance(\hat \theta) = \dfrac{1}{n} \displaystyle\sum_{i=1}^{n}(\hat \theta_i - \hat \theta_{mean})^2$
### Standard Error
The uncertainty of an estimator across repeated samples.
- It is the square root of the variance in $\hat \theta$

$SE(\hat \theta) = \sqrt{\dfrac{1}{n} \displaystyle\sum_{i=1}^{n}(\hat \theta_i - \hat \theta_{mean})^2}$
### Generalization
How well a model has learned the patterns of a dataset during train and then applies what it learns to unseen(test) data. 
### Overfitting
The scenario where the model becomes very accurate to the training set but *cannot* generalize on the test dataset.
	- Low bias
	- High variance
	- Data is more complex and has too many features

### Underfitting
The scenarios where a model performs poorly on both the training and test datasets. 
	- High bias
	- Low variance
	- Data is less complex and doesn't have enough features for the model to learn patterns. 

![[IMG_3581.png]]
### Correlation vs. Causation
1. **Correlation** - There is a relationship or pattern between two variables, $A, B$
	1. Positive - when $A$ increases so does $B$
	2. Negative - when $A$ increases $B$ decreases
	3. zero - when $A$ changes $B$ does not
2. **Causation** - One event results from another event

Now as we have heard so so many times *Correlation* $\neq$ *Causation*. But what does that mean?
- When this occurs there is the possibility of a 3rd, possibly unknown, variable that is affecting the relationship between $A$ and $B$. This 3rd variable is called a **Latent Variable**.
### Latent Variables
These are variables that are not directly observed, but rather are inferred via the way the model is applied to other known variables. 
### Residuals
The error in the model defined further as the difference between the predicted target, $\hat Y$, and the actual target, $Y$.
## Formula
### General Model Function
**$g(X) = y$**
- $g$ = model
- $X$ = feature matrix
- $y$ = target vector
single observation is $g(x_i) = y_i$ 
- $x_i = (x_{i1}, x_{i2}, ..., x_{i,n})$  - is a single feature vector with values from $x_{i1}$ to $x_{i,n}$ 
- $y_i$ - a single value that is dependent on the feature $x_i$

### Linear Regression Formula
>[!tip] $Y = \theta^T \cdot X + W$ 

* $Y$ = target vector
	* size $n \times 1$
* $X$ = feature matrix
	* a 2D-matrix of size $n \times m + 1$
* $\theta$ = weight vector
	*  size $n \times 1$
* $W$ = residual vector
	* size $n \times 1$
*Note*: $n$ = sample size, $m$ = number of features

![[Screenshot 2025-09-26 at 17.43.13.png]]
#### The breakdown
$g(x_i) = \theta_0 + x_{i1} \cdot \theta_1 + x_{i2} \cdot \theta_2 + ... + x_{in} \cdot \theta_n$.

Simplified to:
$g(x_i) = \theta_0 + \displaystyle\sum_{j=1}^{n} \theta_j \cdot x_{ij}$
- $g(x_i) \approx y_i$ at record *i* in our data
	- $x_{ij}$ = all the values of the feature(s) for that record
- $\theta_0$ = bias term
	- this term technically has an $x_{i0}$  for $j = 0$ dotted with it, but that value will always be `1`
		- Why? Well the bias term is learned when there are no features affecting the target. 
- $\theta_j$ = weight for feature  $x_{ij}$
	- The weight of a feature is how 'important' the feature is in predicting the target value. 

##### OMG it's a Line of Best Fit
BUT WAIT! LOOK! The summation part looks just like a vector-vector multiplication, so we can further write our Linear Regression function as:

$g(x_i) = \theta_0 + x_i \cdot \theta_i^T$ 

- we transpose(T) one of the vectors because remember vectors are of shape $n \times 1$ so we need to ensure that we are multiplying our vectors such that:
	- the number of rows of $x_i$ = the number of cols of $w_i$

Now...what does that look like? The *line of best fit*! 
$y = mx + b$
- $y = (g(x_i) \approx y_i)$
- $m = \theta_i$, the slope at a give point 
- $x = x_i$
- $b = \theta_0$, the y-intercept
	- when the $x$ coordinate = 0, aka when our line intersects with the y-axis

Therefore, all we're trying to do is find the line of best fit for our linearly related dependent variable and independent variables aka our target and feature vector. 
![[Screenshot 2025-09-26 at 16.21.56.png]]
#### Let's look at a Basic Example
From our [[Car Price Prediction ]] we saw that there were features that we could use to predict car price, or MSRP. We will use `['engine_hp', 'city_mpg', 'popularity']` as our features.
 *Note*: these values come from the lecture video.
- $x_i$ = `[435, 11, 86]`
	- Each feature in our $x_i$ will have an associate weight in our weight vector $w_i$
-  $g(x_i) = \theta_0 + x_1 \cdot \theta_1 + x_2 \cdot \theta_2 + x_3 \cdot \theta_3$
- $\theta_0 = 7.17$
- $\theta_i = [0.01, 0.04, 0.002]$ 

So with our values, we can plug them into $g(x_i)$
$7.17 + (453 * 0.01 )+ (11 * 0.04) + (86 * 0.002) = 12.312$

**Observe**
Since we know something about the car, the features, these will affect the predicted price of the car. 
- Therefore, each weight dictates how much of an affect that feature will have on the car. 
	- So, the `city_mpg(11)` with weight `0.04` has the most affect on our car price, and we can look at it intuitively:
		- A car with a better mpg probably has a better engine and might even structurally be better built thus increasing the price.



## Assumptions
### Linearity between target and features
The relationship between the independent variable(s) and the dependent variable must be linear aka they are *Linearly Dependent*.
- Okay but does it mean to be **Linearly Dependent**?
	- A straight line relationship exists between the variables that preserves the following rules:
		1. Additivity - $f(x + y) = f(x) + f(y)$
		2. Homogeneity - $f(cx) = cf(x)$, where $c = constant$

**Fail State** - if the regression fails either tenant of the rules of Linear Dependence, it *may not* be able to identify a pattern in the data and thus will make a poor prediction on unseen data. 
### No Multicollinearity between features
Independent variables must be linearly independent aka no correlation between individual features exists.
- Multicollinearity **reduces the precision of the estimated feature weights** because linearly related features provide 'repeated' information to the model. This directly affects the way the model views the significance of a feature. 
### Homoscedascity
The Linear Regression model assumes that the error terms have Homoscedascity.
- If Homoscedacity is violated, the reliability of the model decreases. 
#### Well, what is Homoscedascity? 
The variance of the error(residuals) associated with each target should be equally distributed along the best fit line. The variance in the error terms is constant. 

### Normal Distribution of Error Terms
Linear Regression assumes that the error terms follow the normal distribution due to the Central Limit Theorem. 
- If the error terms *do not* follow the normal distribution then the Confidence Intervals become too wide or too narrow make it difficult to estimate the feature weights and thus the model can no longer make stable, reliable predictions.


### Endogeneity
A phenomenon where the features are correlated to the residuals of the model. 
- If *Endogeneity* is present, the optimization process will lead to biased weights(parameters) of the model, which adversely affects model performance.
	- The presence of *Latent Variables* is a good indicator of Endogeneity.
## Training the Linear Regression
### Get the Weight Vector
*Note*: we won't be doing all the math here, just a high level. 

**Goal**: Solve our Linear Regression formula for the weight vector, $\theta$ via a set of linear equations. 
**Recall**: $Y = \theta^T X + W$
- Now, on our initial 'run' of a Linear Regression we don't know our residuals $W$ so we simplify to:
		$Y = \theta^T X$
- We also must remember that the feature matrix, $X$, is of size $n \times m+1$ which means that it is *rectangular* matrix. This makes getting a set of linear equations with $\theta$ more challenging. 
**Steps**
1. We must first *transform* our feature matrix, $X$, into a *square matrix* by multiplying it with its *Transpose*. 
		$X^T \cdot X$
	- Multiplying by the transpose of  gives us a square matrix of size $(n + 1 \times n+1)$ better known as the **Gram Matrix**.
2. We apply the transformation in *step 1* to BOTH sides of our equation.
		$(X^TX) \theta^T = X^TY$
3. We can now multiply both sides by the *Inverse* of the Gram Matrix $X^TX$. 
		$(X^TX) \cdot (X^TX)^{-1} \theta^T = X^TY\cdot (X^TX)^{-1}$
	- Multiplying a matrix by its inverse 'cancels' out both matrices and results in a scalar of $1$.
		
>[!tip] $\hat\theta = (X^TX)^{-1}X^TY$ 
- we don't need to have $\theta$ be transposed anymore because we are no longer multiplying our weight vector by the feature matrix $X$.
- **Remember** that we can't actually get the real values of the weights so they are just an estimate, hence the ^ on $\theta$. 

**Recall**: Since we're training initially on just the training set we have no concept of what our error, $W$, is. So the above weight vector won't take those into account. 
#### Why is this Important?
1. The weight of a feature tells us about the importance of a feature to the prediction.
2. Since the target is the result of the dot product between the feature matrix and the weight vector the models predications are directly affected by how well we estimate the weight matrix. 
### Prepare the Dataset
#### Ensure Assumptions are Upheld
##### Linear Dependence Between Target and Features
**Test**
- Plot the residuals and the fitted values and ensure that residuals do not form a strong pattern. 
	- The residuals should be randomly and uniformly scattered on the x-axis.
**Fix**
- Transform features and/or target if they are non-linear.
##### Multicollinearity
**Test**
1. Heatmap of correlations to determine if any of our features are correlated. 
2. Variance Inflation Factor - measures the inflation in the variances of the weight estimates due to collinearities that exist among the features. The score is represented as the correlation between a feature and all other features.
	- VIF = 1, then there is no multicollinearity.
	- VIF $\geq$ 5 or is close to exceeding 5, we say there is moderate multicollinearity. 
	- VIF $\geq$ 10 or exceeds 10, it shows signs of high multicollinearity.
**Fix**
- Remove or merge correlated variables.
##### Homoscedascity
**Test**
- We can plot the residuals vs. predicted values with a line of best fit and observe whether or not the variance around the line is constant.
	-  If the residuals are *symmetrically* distributed across the regression line, then the data is said to be homoscedastic.
	- If the residuals are *not symmetrically* distributed across the regression line, then the data is said to be heteroscedastic. 
		- In this case, the residuals can form a funnel shape or any other non-symmetrical shape.
**Fix**
1. Non-linear Transformation of the target vector
2. Feature Engineering via adding augmented feature vectors to the model. 
##### Normally Distributed Residuals
**Test**
1. Plot residuals using a histogram and observe if they are bell-shaped or not.
2. A normal distribution has a mean $\bar x = 0$ 
**Fix**
- Non-linear transformations of feature vector(s) or target vector
#### Get only the continuous features
Okay, so what does that mean again? 
- *Remember*, Linear Regression is done on continuous variables *only*. Meaning that we can only do a Linear Regression on features that are *numerical*. 

*Fortunately*, getting the subset of features that are continuous is fairly simple in code!
```python
continous_features = list(df.dtypes[df.dtypes != 'object'])
```
*Note*: changing categorical, aka non-numerical, features into continuous features requires a bit more and will be covered under Feature Engineering. 

#### Missing Value Handling
There are several ways to fill in missing values in our records.
1. Fill with $0$ - to make our model 'ignore' these features when predicting the target for that record.
	- `df.fillna({'col_name': 0})`
2. Fill with mean.
	- `df.fillna({'col_name': col_mean})`
#### Putting it Together
```python
def prepare_X(df):
	# Get the continuous features only 
	df_num = df[continous_features]
	# fill in missing values
	df_num = df_num.fillna(0)
	# get the updated feature matrix
	X = df_num.values
	
	return X
```
#### Standardize
##### *Log Transformation*
`nplog1p()`
##### *Z-Score* 
- Standardize features by removing the mean and scaling to unit variance.
```python
	# To scale the data using z-score
	import sklearn
	from sklearn.preprocessing import StandardScaler
	
	scaler = StandardScaler()
	data_scaled= scaler.fit_transform(data)
```
#### Setup Train, Test, and Validation Datasets
##### Method 1
- **Breakdown**:
	- Train ~ 60%
	- Test ~ 20%
	- Validate ~ 20%
	 ```python
		# set split sizes
		n = len(df)
		n_val = int(n * 0.2)
		n_test = int(n * 0.2)
		n_train = n - (n_val + n_test)
	  ```
- **Randomize**
	```python
	# generate a sequence of numbers that represents our indices
	idx = np.arange(n)
	# shuffle the indices
	np.random.seed(2) # ensures same shuffle everytime
	np.random.shuffle(idx)
	```
	- *Note*: we can also use the following:
			- pandas: `df.sample(frac=1, random_state=42).reset_index(drop=True)` 
- **Split & fix indices**
	```python
		# split the data
		df_train = df.iloc[idx[n_train:]]
		df_val = df.iloc[idx[n_train:(n_val + n_train)]]
		df_test = df.iloc[idx[(n_val + n_train):]]
		
		# reset the indices
		df_train = df_train.reset_index(drop=True)
		df_val = df_val.reset_index(drop=True)
		df_test = df_test.reset_index(drop=True)```
- **Get the target variable**  
	- `y_train = df_train.msrp.values`
- **Remove the target variable** from the train, test, and validation DataFrames
	- `del df_train['msrp']`
		- Apply to df_val and df_test respectively	
##### Method 2
- Splits into train and test, no validation set.
```python
		from sklearn.model_selection import train_test_split
		X = # DataFrame of features
		Y = # Series of output vals
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify = Y)
```
### Coded Linear Regressions
#### A simple coded Linear Regression
```python
w_0 = 0 # weight_0, arbitrary val for examples sake
w = [1, 1, 1] # weight arbitrary array[val] for examples sake
def linear_reg(x_i):
	n = len(x_i)
	
	pred = w_0
	
	for j in range(n):
		pred = pred + w[j]*x_i[j]
		
	return pred
```
#### A simple coded Univariate Linear Regression in Vector Form
```python
def dot(xi, w):
	'''A basic dot product'''
	
	n = len(xi)
	res = 0.0
	
	for j in range(n):
		res = res + xi[j] * w[j]
	
	return res

def linear_reg(xi, w0, w):	
	xi = xi + [1]
	w = w + [w0]
	return dot(xi, w)
```
#### A simple coded Multivariate Linear Regression in Matrix Form
Okay, so we know that a Linear Regression isn't always done on a single feature vector, $x_i$, but rather a matrix of features $X$. 
- So what we do instead is the matrix-vector multiplication between the feature matrix, $X$, and the transposed feature weight vector, $W$, to get our prediction. 
```python
def linear_reg(X, W):
	return X.dot(W.T)
```
### Let's Create a Coded Training Method
```python
X = '2D matrix of features'
y = 'target vector'

def train_linear_reg(X, y):
# add bias term col to X
ones = np.ones(X.shape[0])
X = np.column_stack([ones, X])

# calculate weight vector (normal equation)
XTX = X.T.dot(X) # gram matrix
XTX_inverse = np.linalg.inv(XTX) # inverse of the gram matrix
w_full = XTX_inverse.dot(X.T).dot(y) # weight vector of X

# get bias term, and remove from w
w_0 = w_full[0]
w = w_full[1:]

return w_0, w
```
## Performance Assessments
### The Loss Function - Mean Squared Error
This function represents the *COST* of making wrong predictions. The goal of training the Linear Regression is to *minimize* the Loss Function, $L$.
* **Error** - is defined as the difference between our predicted values, $\hat y$ and our actual values $y$. These are also know as our **Residual** vector.

>[!tip] $L_{min} = \dfrac{1}{n} min\displaystyle\sum_{i=1}^{n}(Y_i - \hat\theta^TX_i)^2$
#### Getting the MSE for Linear Regression
**Recall**: $Y_i = \theta_0 + x_{i1} \cdot \theta_1 + x_{i2} \cdot \theta_2 + ... + x_{in} \cdot \theta_n = \theta^T X_i$.

$Residual = Y_i - \hat Y_i = Y_i - \hat\theta^T X_i$

$MSE = \dfrac{1}{n} \displaystyle\sum_{i=1}^{n}(Y_i - \hat\theta^TX_i)^2$
### Root Mean Squared Error
The RMSE is calculated by taking the square root of the Loss Function, $L$. In the case of Linear Regression this means $\sqrt{MSE}$.

$RMSE = \sqrt{\dfrac{1}{n} \displaystyle\sum_{i=1}^{n}(Y_i - \hat\theta^TX_i)^2}$

==*Important:*== A **Lower** RMSE == a better model
- We use the RMSE by calculating the value for our *train* and our *validation* dataset and comparing the two. 
	- $RMSE_{train} \approx RMSE_{val}$ then our model predicts the same on *unseen* data
		- Otherwise, something is wrong with our training.
##### In Code
```python
def rmse(y, y_pred):
	error = y - y_pred
	sqr_error = error ** 2
	mse = sqr_error.mean()
	
	return np.sqrt(mse)
```


### $R^2$ 
Informs about the quality of the model's fit by explaining the variance in the dependent(target) variable by the independent(feature) variables. Aka how good is the model at predicting unseen data. 

$R^2 = 1 - \dfrac{Sum\,of\,Squared\,Residuals}{Sum\,of\,Squared\,Means}$

$SSResiduals = min\sum_{i=1}^{n}(y_i - \hat y_i)^2$

$SSMeans = min\sum_{i=1}^{n}(y_i - \bar y_i)^2$

==Important:== The **higher** the value of $R^2$ the better the model is at predicting the target. 

### Adjusted $R^2$
A version of $R^2$ that takes into account the number of independent variables present in the model and grows only if the independent variable adds value to the model.
- $R^2$ naturally grows with each independent variable added to the model, therefore it can fail to capture whether or not the model is predicting well. 

$Adjusted\,R^2 = 1 - \dfrac{(1 - R^2)(n - 1)}{n - k -1}$
- $k\,=\,number\,of\,independent\,variables$
- $n\,=\,sample\,size$

==Important:== The **higher** the value of $Adjusted\,R^2$ the better the model is at predicting the target.
## Cross Validation Techniques
### Simple
We seek a model that is able to generalize over unseen data while maintaining as many features as possible *without* overfitting the data. 
1. Split the data into two or three subsets.
	- Training and Test => 70:30
	- Training, Test and Validation => 60:20:20
2. Then we train our model on the *Training* subset.
3. We validate our model on the *Validation* subset by making predictions on the unseen data in the subset.
4. We then calculate the same *Performance Metric* for the *training* and *validation* predictions and see if our model was able to generalize. 

**Advantages**
- easy to implement 
- easy to understand
**Drawbacks**
- *Data Waste* is inevitable in training because the data is split into 2-3 subsets.
- The error on validation subset can be highly random and cannot adequately tell us if our model is generalizing well. 
### Leave-One-Out-Cross-Validation
On a dataset of $size\,=\,n$, 
1. we train our model on $n-1$ data points, via *iterating* over our dataset $n$ number of times such that the **Left Out** data point is 'unique' on each iteration. 
2. We then validate each iteration on the **Left Out** data point. 
3. For each iterations model we collect the *Loss Function* $L(y_i, \hat y_i)$ for both *Training* and *Validation*
4. Upon completion of the iteration, we calculate the mean of the *Loss Function* for both *Training* and *Validation* and determine whether or not our model was able to generalize well.

**Advantages**
- *No Variability* - every data point has to be present at least once in the validation set thus there is no randomness is what data the model is being validated on.
- *No Data Wastage* - every data point is used to train the model at some point.
**Drawbacks**
- *Costly* - training is don $n$-times which means more computational needs and more time to run the validation.
- *Prediction Errors are Highly Dependent* - each model has many record in common, $n-1$, when trained so the errors in the models will be similar if not the same.

*NOTE:* we can use LOOCV to select the number of features to use in our model via running LOOCV on different feature combinations and choosing the model-features combination that gives the lowest LOOCV error value. 
### K-fold Cross Validation
A more *practical* version of LOOCV.
Chose a value $k$ to be the number of folds (groups),
1. Randomly divide the data in to $k$ number of groups
2. For every group:
	1. train on $n-1$ records
	2. validate on the hold out record
3. Repeat 1 & 2, $k$ times and record the error, $L_k(y_i, \hat y_i)$, for each iteration.
4. After the iteration is complete, calculate the mean of the recorded *Loss Functions* for both *Training* and *Validation* and determine whether or not our model was able to generalize well.

**Advantages**
- *Less Costly* - Validation is run on only $k$ number of groups vs. the length of the data.
	- There exists a threshold $k$ where as the number of folds increase so do the error. Therefore, only a set number of folds will have to be run for optimal validation. 
**Drawbacks**
- *Inherent Variance* - different models trained on different samples from the same population have different prediction errors.
	- This may sound great, but *some dependency* between the different prediction errors is a good thing because the *variance* within the dataset is handled. 
#### Bootstrapping to handle Variance
A statistical procedure that resamples a single dataset to create many simulated datasets via random sampling with replacement. 
1. Determine number of Bootstrap Samples.
	- Can use K-Fold CV for this.
2. Determine the Bootstrap Sample size, $m$. Where $m \leq n$.
3. Randomly select a data point from $n$ and place it in a Bootstrap Sample. 
4. Return the data point back to $n$.
5. Repeat 2 & 3 until each Bootstrap sample is filled and the desired number of Bootstrap samples is met.
6. Run a Cross-Validation technique on Bootstrap Sample, where the *validation* set is the samples not selected for that Bootstrap Sample. 

**Advantages**
- *Reduced Variance* - variance in the data is handled because the dependency on a single record is removed via the creation of simulated dataset that allow for record duplication. 
	- Thus, we get some dependency within the prediction errors.
- *Standard Error Calculation* - we can use this more stabilized variance to calculate the standard error of our weight, $\hat \theta$, estimates.
	- Therefore, we can provide better confidence intervals for our models prediction estimates. 

## Model Improvement 
Now that we've seen some ways to see how good our model is, how do we **Improve** it?
#### Feature Engineering
Augmenting existing features to create new features to improve model performance.
##### Augmented Feature Vectors
Now, we're not creating a new feature from nothing. Think of it more as giving an existing feature a *new form* by changing it in some way. 
1. **Simple** changes are enacted by performing simple mathematics on a feature.
	- For example, say we have a feature *established_year* which represents the year a restaurant was opened in a neighborhood. Now, instead of just the year we want to know how long this restaurant's been around, its *age*. 
		- Well we know that to get *age* we would do the following: $age = 2025 - established\_year$
		- We would apply this to the entire *established_year* column, and then replace this column with the new feature column *age*.
		```python
		# get a deep copy of our original dataframe to not modify the original 
		df_copy = df.copy()
		# add new column age
		df_copy['age'] = 2025 - df_copy['established_year']
		# remove previous variable 
		del df_copy['established_year]
		```
		- ==Important:== we DON'T want to keep both features because we then introduce multicollinearity into our model.
2. **Transform or Combine** one or more features by applying non-linear vector/matrix transformations. 
	- Dot product of two features: $x_i \cdot x_j$
		- Creates a brand new feature so multicollinearity with the pre-existing features $x_i$ and $x_j$ is minimal. 
	- Logarithm of a feature: $log(x_i)$
		- Since this is a *transformation* of an existing feature some multicollinearity could be introduced, so it is critical to check if this is true.  
##### "One Hot" Encoding of Categorical Variables
Okay, but what if we believe a *categorical variable* is critical to our prediction? Well, we can do a binary encoding of it where the values of this variable become *True or False* aka $1 || 0$.

Each categorical variable has a set of values, and we can take those values and assign each it's own column. Then for each record in each column we say *True* if the value is in that record and *False* otherwise. For example:

| **Weather** | **Rain** | **Snow** | **Sun** |
| ----------- | :------- | -------- | ------- |
| Rain        | True     | False    | False   |
| Sun         | False    | False    | True    |
| Sun         | False    | False    | True    |
| Snow        | False    | True     | False   |
| Rain        | True     | False    | False   |
Now we have weather encoded into *continuous* variables, thus we can use it in our Linear Regression! 
- **In Code**
```python
# drop_first: removes the 1st col, in example (Rain), because we can assume that if Snow and Sun are False then we have Rain.
df.get_dummies(data=df, columns=['Weather'], drop_first=True)
```
#### Regularization
A technique to help avoid overfitting the model when adding new non-linear features. 
##### Ridge
In order for the model to generalize better over unseen data, a **Ridge Regularization** adds a penalty to the linear regression for features with large weights. Thus, the weight's effect is reduced via minimization without removing the affect of the feature on the model.

$SSResiduals = min_{\theta}[\sum_{i=1}^{n}(y_i - \hat y_i)^2 + \alpha\sum_{j=1}^{m}\hat \theta_j^2]$
- $\alpha$ = regularization parameter
- n = total number of records/observations
- m = total number of features

**Therefore** and increase in $\alpha$ increase the penalty applied to the weight vector, $\hat \theta$, of the features, and thus minimizes the impact of said weights.
##### Lasso
In order to reduce the variance in a models' predictions, a **Lasso Regularization's** penalty prefers small, near $0$, weights. Thus, a feature's weight can be reduced to $0$ removing that feature's affect on the model entirely. 
- Here, bias in the features is introduced to better reduce variance. 

$SSResiduals = min_{\theta}[\sum_{i=1}^{n}(y_i - \hat y_i)^2 + \alpha\sum_{j=1}^{m}|\hat \theta_j|]$
- $\alpha$ = regularization parameter
- n = total number of records/observations
- m = total number of features


![[Screenshot 2025-09-28 at 11.24.47.png]]




