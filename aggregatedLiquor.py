
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import sklearn.cluster as sk
import numpy
import seaborn as sns

sns.set(style="ticks", color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read in aggregated data
liquor = pd.read_csv('data/aggregatedData.csv', index_col='County')
liquor.head()


# ## Data Processing
# 
# ##### After some investigation, it was determined that Scotch had a value of nan for Friedmont County. We replaced this with a 0 representing no Scotch Sales


# In[3]:


liquor['SCOTCH'][liquor['SCOTCH'].isnull()] = 0


# ##### We also convert sales of each liquor into percentages instead of whole sales so that the population/total number of sales doesn't bias these values


# In[4]:


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

liquor.head()

# ## Linear Regression Model Fitting
# 
# ##### I am attempting to find something interesting by building linear regression models. I want to see if median household income is a good predictor of anything. 
# 
# ##### In my first model, I want to see if we can use income to predict the volume of alcohol purchased. Since population clearly has a large effect on volume of alcohol sold, I used volume per capita. A better way of doing this might be to use population as a variable, but I thought that would give us a better idea of the exact impact of income


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


liquor.loc['MEAN'] = liquor.apply(numpy.mean)
liquor.loc['MEDIAN'] = liquor.apply(numpy.median)
liquor.head()


# # Next Steps
# 
# ## I think we should categorize all of our variables. I think we don't have enough data points to capture the nuances in large values like median income and number of total sales. This would also allow us to start doing association rules


# In[9]:


categoryPercents = liquor[['BOURBON', 'BRANDIES', 'COCKTAILS', 'GINS', 'LIQUEUR', 'OTHER', 'RUM', 'SCHNAPPS', 'SCOTCH', 'SPIRITS', 'TEQUILA', 'VODKA', 'WHISKEY']]
categoryPercents.head()


# In[10]:


# Pretty much worthless
#fig,axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
#fig.set_figheight(15)
#fig.set_figwidth(15)
#axes_list = [item for sublist in axes for item in sublist]
#for county in categoryPercents.index:
#    ax = axes_list.pop(0)
#    categoryPercents.loc[county].plot(kind='bar', ax=ax)


# In[11]:


# Pretty much worthless
#fig,axes = plt.subplots(nrows=3, ncols=5)
#fig.set_figheight(15)
#fig.set_figwidth(15)
#axes_list = [item for sublist in axes for item in sublist]
#for cat in list(categoryPercents):
#    ax = axes_list.pop(0)
#    categoryPercents[cat].plot(kind='bar', ax=ax)


# In[20]:


averaged = pd.DataFrame(index=categoryPercents.index, columns=list(categoryPercents))
lean = pd.DataFrame(index=categoryPercents.index, columns=list(categoryPercents))
for county in categoryPercents.index:
    overs = 0
    for cat in list(categoryPercents):
        newVal = categoryPercents.loc[county, cat] - categoryPercents.loc['MEDIAN', cat]
        averaged.loc[county, cat] = newVal
        overs += int(100 * abs(newVal) / 2)
    if overs >= 3:
        lean.loc[county] = averaged.loc[county]
    else:
        lean = lean.drop(county, axis=0)
    
averaged = averaged[averaged.columns].astype(float)
lean = lean[lean.columns].astype(float)
lean


# In[41]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(lean, annot=True, fmt="d", linewidths=.5, ax=ax, cmap=sns.color_palette("coolwarm", 21), vmin=-0.105, vmax=0.105)


# In[49]:


lean.plot(kind='scatter', x='VODKA', y='WHISKEY')


# In[50]:


klean = lean[['VODKA', 'WHISKEY']] 
kaveraged = averaged[['VODKA', 'WHISKEY']] 
kmeans = sk.KMeans(n_clusters=3, random_state=0).fit(klean)
plt.scatter(klean['VODKA'], klean['WHISKEY'],c=kmeans.labels_)


# In[51]:


kmeans = sk.KMeans(n_clusters=2, random_state=0).fit(klean)
plt.scatter(klean['VODKA'], klean['WHISKEY'],c=kmeans.labels_)


# In[52]:


kmeans = sk.KMeans(n_clusters=3, random_state=0).fit(kaveraged)
plt.scatter(kaveraged['VODKA'], kaveraged['WHISKEY'],c=kmeans.labels_)


# In[53]:


kmeans = sk.KMeans(n_clusters=4, random_state=0).fit(kaveraged)
plt.scatter(kaveraged['VODKA'], kaveraged['WHISKEY'],c=kmeans.labels_)


# In[54]:


averaged['Group'] = kmeans.labels_
averaged.loc[averaged['Group'] == 0]

