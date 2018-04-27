
# coding: utf-8

# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns

sns.set(style="ticks", color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[73]:


#Read in aggregated data
liquor = pd.read_csv('data/aggregatedData.csv', index_col='County')
liquor.head()


# ## Data Processing
# 
# ##### After some investigation, it was determined that Scotch had a value of nan for Friedmont County. We replaced this with a 0 representing no Scotch Sales

# In[87]:


liquor['SCOTCH'][liquor['SCOTCH'].isnull()] = 0


# ##### We also convert sales of each liquor into percentages instead of whole sales so that the population/total number of sales doesn't bias these values

# In[75]:


liquor['BOURBON'] = liquor['BOURBON']/liquor['nSales']; 
liquor['BRANDIES'] = liquor['BRANDIES']/liquor['nSales']; 
liquor['COCKTAILS'] = liquor['COCKTAILS']/liquor['nSales']; 
liquor['GINS'] = liquor['GINS']/liquor['nSales']; 
liquor['LIQUEUR'] = liquor['LIQUEUR']/liquor['nSales']; 
liquor['OTHER'] = liquor['OTHER']/liquor['nSales'];
liquor['RUM'] = liquor['RUM']/liquor['nSales']; 
liquor['SCHNAPPS'] = liquor['SCHNAPPS']/liquor['nSales']; 
liquor['SCOTCH'] = liquor['SCOTCH']/liquor['nSales']; 
liquor['SPIRITS'] = liquor['SPIRITS']/liquor['nSales']; 
liquor['TEQUILA'] = liquor['TEQUILA']/liquor['nSales']; 
liquor['VODKA'] = liquor['VODKA']/liquor['nSales']; 
liquor['WHISKEY'] = liquor['WHISKEY']/liquor['nSales']; 


# ## Linear Regression Model Fitting
# 
# ##### I am attempting to find something interesting by building linear regression models. I want to see if median household income is a good predictor of anything. 
# 
# ##### In my first model, I want to see if we can use income to predict the volume of alcohol purchased. Since population clearly has a large effect on volume of alcohol sold, I used volume per capita. A better way of doing this might be to use population as a variable, but I thought that would give us a better idea of the exact impact of income

# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LinearRegression(normalize = True)
X = liquor['Median_Household_Income'].reshape(-1, 1)
Y = liquor['Volume Per Capita']

X_train = X[:-20]
X_test = X[-20:]

# Split the targets into training/testing sets
y_train = Y[:-20]
y_test = Y[-20:]

reg.fit(X_train, y_train)
reg.score(X_train, y_train)
# Make predictions using the testing set
y_pred = reg.predict(X_test)

print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Score: %.2f' % reg.score(X_train, y_train))


# ##### These metrics aren't very good. The score of .01 means that we are pretty much plotting the same volume no matter what the median household income. 
# 
# ##### Let's plot the results to verify

# In[89]:


# Plot outputs
plt.scatter(X_test, y_test,  color='black');
plt.scatter(X_test, y_pred, color='blue');
plt.plot(X_test, y_pred, color='blue', linewidth=3);
plt.xlabel('Median Household Income');
plt.ylabel('Volume Per Capita');
plt.xticks();
plt.yticks();


# ### Linear Model 2 
# 
# ##### This time I wanted to see if income affected price. This seems like it would have a higher correlation. It turns out it also isn't that significant as we again have a pretty low score.

# In[90]:


reg = linear_model.LinearRegression(normalize = True)
X = liquor['Median_Household_Income'].reshape(-1, 1)
Y = liquor['Price per Liter']

X_train = X[:-20]
X_test = X[-20:]

# Split the targets into training/testing sets
y_train = Y[:-20]
y_test = Y[-20:]

reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Score: %.2f' % reg.score(X_train, y_train))

# Plot outputs
plt.scatter(X_test, y_test,  color='black');
plt.scatter(X_test, y_pred, color='blue');
plt.plot(X_test, y_pred, color='blue', linewidth=3);
plt.xlabel('Median Household Income');
plt.ylabel('Price Per Liter');
plt.xticks();
plt.yticks();


# In[76]:


liquor.head()


# # Next Steps
# 
# ## I think we should categorize all of our variables. I think we don't have enough data points to capture the nuances in large values like median income and number of total sales. This would also allow us to start doing association rules
