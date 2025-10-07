## Packages
```python
# import libraries for data manipulation
import numpy as np
import pandas as pd

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# disable pop_up display of graphs to display them in notebook.
%matplotlib inline
```
## Methods
```python
'''PANDAS METHODS'''
# how many unique values in dataframe/col
nunique()
# get the number of times a value appears in a col
value_counts()
# get data organized by a specific feature
groupby()
# apply a function to the data
apply()

```
## Plots
- histogram - `sns.hisplot()`
- bar graph -`sns.countplot()`
- box plot - `sns.boxplot()`
- heatmap - `sns.heatmap()`
- pointplot - `sns.pointplot()`
- scatterplot - `sns.scatterplot()`
- line graph - `sns.lineplot()`
- linear model plot - `sns.lmplot()`
	- fits a line of best fit to the data
- joint plot - `sns.jointplot()`
	- visualize bivariate and univariate profiles in the same plot
- violin plot - `sns.violinplot()`
	- a box plot on it's side that shows the density/distribution of numeric variables 
- strip plot - `sns.stripplot()`
	- a scatter plot that better visualizes data distributions
- swarm plot - `sns.swarmplot()`
	- a scatter plot that better visualizes data distributions w/out overlapping points
- cat plot - `sns.catplot()`
	- displays the relationship between a numerical variable and one or more categorical variables
