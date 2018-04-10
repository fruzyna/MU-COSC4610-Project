
# coding: utf-8

# In[83]:


# COSC 4610 Final Project
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import re

sns.set(style="ticks", color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


# Read in liquor sales dataset (go make some coffee)
liquor = pd.read_csv('data/Iowa_Liquor_Sales.csv', parse_dates=True, index_col='Invoice/Item Number', low_memory = False).sort_values('Date')


# In[91]:


# Makes all county names uniform (all caps, no "COUNTY")
def correctCounty(county):
    if isinstance(county, str):
        return county.upper().replace(' COUNTY', '')
    return str(county)

# Filter into fewer categories
cats = ['VODKA', 'TEQUILA', 'COCKTAILS?', 'GINS?', 'BRAND[(IES)Y]', 'WHISK[(IES)Y]', 'SCHNAPPS?', 'CREME', 'RUM', 'SCOTCH', 'ANISETTE', 'AMARETTO', 'BEER', 'BOURBON', 'TRIPLE SEC', 'DECANTERS', 'LIQUEURS?', 'SPIRITS?', 'SPECIAL']
catNames = ['VODKA', 'TEQUILA', 'COCKTAILS', 'GINS', 'BRANDIES', 'WHISKIES', 'SCHNAPPS', 'CREME', 'RUM', 'SCOTCH', 'ANISETTE', 'AMARETTO', 'BEER', 'BOURBON', 'TRIPLE SEC', 'DECANTERS', 'LIQUEURS', 'SPIRITS', 'SPECIAL']
def filterCategories(category):
    for i in range(len(cats)):
        cat = cats[i]
        name = catNames[i]
        if re.search(cat, category.upper()):
            return name
    return category.upper()


# In[92]:


# Trim unnecessary rows
if 'liquor' in locals():
    liquor = liquor.drop('Address', axis=1)
    liquor = liquor.drop('City', axis=1)
    liquor = liquor.drop('Zip Code', axis=1)
    liquor = liquor.drop('County Number', axis=1)
    liquor = liquor.drop('Volume Sold (Gallons)', axis=1)
    liquor = liquor.drop('Pack', axis=1)


# In[93]:


# Make county names uniform
liquor['County'] = liquor['County'].apply(correctCounty)
liquor['Category Name'] = liquor['Category Name'].astype(str)
liquor['Gen Category'] = liquor['Category Name'].apply(filterCategories)
# Remove dollar signs
liquor['Sale (Dollars)'] = liquor['Sale (Dollars)'].str[1:]
liquor['Sale (Dollars)'] = liquor['Sale (Dollars)'].astype(float)
liquor['State Bottle Cost'] = liquor['State Bottle Cost'].str[1:]
liquor['State Bottle Cost'] = liquor['State Bottle Cost'].astype(float)
liquor['State Bottle Retail'] = liquor['State Bottle Retail'].str[1:]
liquor['State Bottle Retail'] = liquor['State Bottle Retail'].astype(float)
liquor


# In[6]:


# Get range of dates from data
earliest = liquor['Date'].min().split('/')
latest = liquor['Date'].max().split('/')
earliest = date(int(earliest[2]), int(earliest[0]), int(earliest[1]))
latest = date(int(latest[2]), int(latest[0]), int(latest[1]))
delta = (latest - earliest).days / 365
dateRange = str(earliest.strftime('%m/%d/%Y')) + ' to ' + str(latest.strftime('%m/%d/%Y'))
delta


# In[7]:


# Read in poverty dataset and trim to median incomes by county
poverty = pd.read_csv('data/est16-ia.csv')
poverty['County'] = poverty['County'].apply(correctCounty)
poverty = poverty.set_index('County')
income = poverty['Median_Household_Income'].sort_values(ascending=False)
#income


# In[10]:


#histogram of all total incomes
plt.hist(income)
plt.savefig('plots/DistIncome.png')


# In[13]:


# Plot median household income by county
income.plot(kind='bar', figsize=(15,7))
plt.savefig('plots/CountyMedianIncome.png')


# In[95]:


# Plot count of each general category
catCounts = liquor['Gen Category'].value_counts()
catCounts.plot(kind='bar', figsize=(15,7))
plt.savefig('plots/Categories.png')


# In[14]:


# The liquor sales have fake? counties, find them
overlap = liquor['County'].isin(income.index)
overlap.value_counts()


# In[15]:


# Investigate the fake counties
fakeCounties = ~liquor['County'].isin(income.index)
liquor[fakeCounties].County.unique()
# El PASO County is in Colorado so I'm not sure what it is doing in this dataset 


# In[16]:


fakeCountiesmapper = {
    'OBRIEN': "O'BRIEN",
    'BUENA VIST': 'BUENA VISTA',
    'POTTAWATTA': 'POTTAWATTAMIE',
    'CERRO GORD':'CERRO GORDO'
}
liquor = liquor.replace(fakeCountiesmapper)
overlap = liquor['County'].isin(income.index)
overlap.value_counts()


# In[17]:


# Remove leftover "fake" counties
liquor = liquor[overlap]
liquor['County'].isin(income.index).value_counts()


# In[18]:


# Count sales by county
pd.set_option("max_rows",105)
countySales = liquor['County'].value_counts()

# Plot that
title = 'Total Sales by County ' + dateRange
plot = countySales.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Sales')
plt.savefig('plots/CountyTotalSales.png')


# In[19]:


# Histogram
fig1 = plt.hist(countySales, bins = 15)
plt.title("Distribution of Alcohol Sales")
plt.xlabel = 'Count of Alcohol Sales'
plt.ylabel = 'Number of Counties'
plt.savefig('plots/AlcoholSalesDistribution.png')


# In[20]:


# Let's Zoom in on the left end of the distribution where most of our data lies
fig1 = plt.hist(countySales[countySales.values < 250000], bins = 15)
plt.title("Distribution of Alcohol Sales")
plt.xlabel = 'Count of Alcohol Sales'
plt.ylabel = 'Number of Counties'
plt.savefig('plots/ZoomedAlcoholSalesDistribution.png')


# In[21]:


# Estimate populations by county
populations = (poverty['People_of_All_Ages_in_Poverty'] * 100 / poverty['Percent_of_People_of_All_Ages_in_Poverty']).astype(int).sort_values(ascending=False)
#populations


# In[22]:


# Calculate and plot liquor sales per capita per year
salesPerCapita = (countySales / populations).sort_values(ascending=False)
salesPerCapita = salesPerCapita.divide(delta)
title = 'Avg Annual Per Capita Sales by County ' + dateRange
plot = salesPerCapita.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Sales')
plt.savefig('plots/SalesPerCapita.png')


# In[23]:


salesPerCapita['DICKINSON']
salesPerCapita['FREMONT']


# In[24]:


# Get the total volume sold by county
def getCountyVolume(county):
    return liquor.loc[liquor['County'] == county, 'Volume Sold (Liters)'].sum()

countyVolumes = pd.Series()
for county in income.index:
    countyVolumes = countyVolumes.set_value(county, getCountyVolume(county))

countyVolumes = countyVolumes.sort_values(ascending=False)


# In[25]:


# Plot volume per capita by county per year
volumePerCapita = (countyVolumes / populations).sort_values(ascending=False)
volumePerCapita = volumePerCapita.divide(delta)
title = 'Avg Annual Per Capita Volume Sold by County ' + dateRange
plot = volumePerCapita.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Volume (L)')
plt.savefig('plots/VolumePerCapita.png')


# In[26]:


print('Dickneson: ', round(volumePerCapita['DICKINSON'],2))
print('Fremont: ', round(volumePerCapita['FREMONT'],2))


# In[27]:


# Get the total spent by county
def getCountySpent(county):
    return liquor.loc[liquor['County'] == county, 'Sale (Dollars)'].sum()

countySpent = pd.Series()
for county in income.index:
    countySpent = countySpent.set_value(county, getCountySpent(county))

countySpent = countySpent.sort_values(ascending=False)


# In[28]:


# Plot total spent per capita by county per year
spentPerLiter = (countySpent / countyVolumes).sort_values(ascending=False)
title = 'Avg Cost Per Liter by County ' + dateRange
plot = spentPerLiter.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Dollars Spent')
plt.savefig('plots/CostPerLiter.png')


# In[29]:


spentPerLiter.describe()
round(spentPerLiter['FREMONT'],2)


# In[30]:


# Volume per sale by county, these numbers seem fishy
volumePerSale = (countyVolumes / countySales).sort_values(ascending=False)
title = 'Avg Volume Per Sale by County ' + dateRange
plot = volumePerSale.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Volume (L)')
plt.savefig('plots/VolumePerSale.png')


# In[105]:


# Cost per sale by county, these numbers seem fishy
spentPerSale = (countySpent / countySales).sort_values(ascending=False)
title = 'Avg Cost Per Sale by County ' + dateRange
plot = spentPerSale.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Total Cost')
plt.savefig('plots/CostPerSale.png')


# In[106]:


# Merge all county data
counties = pd.concat([income, populations, salesPerCapita, volumePerCapita, spentPerLiter, volumePerSale, spentPerSale], axis=1)
counties.columns = ['Median Household Income', 'Estimated Population', 'Sales Per Capita', 'Volume Per Capita', 'Spent Per Liter', 'Volume Per Sale', 'Spent Per Sale']
counties.to_csv('clean_data/county_stats.csv')
counties


# In[31]:


# This is a waste of time for now
# Get the median income of a county by name
#def getIncome(county):
#    return income.loc[county]

# Apply the income of the county to each sale
#liquor['Median Household Income'] = liquor['County'].apply(getIncome)


# In[32]:


# Investigating Fremont
fremont = liquor.loc[liquor['County'] == 'FREMONT']
fremont['Store Name'].value_counts()


# In[33]:


# Highest amount of purchases
liquor['Store Name'].value_counts().head()


# In[34]:


#Lowest amount of purchases
liquor['Store Name'].value_counts().tail()


# In[35]:


# Investigating Casey's
casey = liquor.loc[liquor['Store Name'].str.contains('Casey\'s')]
casey['Store Name'].value_counts().describe()


# In[68]:


# Plot sales and income
fig = sns.lmplot(y='Sales Per Capita', x='Median Household Income',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Median Household Income vs Liquor Sales Per Capita by County')
fig.savefig('plots/SaleIncome.png')


# In[67]:


# Plot volume and income
fig = sns.lmplot(y='Volume Per Capita', x='Median Household Income',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Median Household Income vs Liquor Volume Per Capita by County')
fig.savefig('plots/VolumeIncome.png')


# In[65]:


# Plot sale size and income
fig = sns.lmplot(y='Volume Per Sale', x='Median Household Income',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Median Household Income vs Volume Per Sale by County')
fig.savefig('plots/SaleVolumeIncome.png')


# In[63]:


# Plot cost per liter and income
fig = sns.lmplot(y='Spent Per Liter', x='Median Household Income',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Median Household Income vs Cost Per Liter by County')
fig.savefig('plots/CostIncome.png')


# In[109]:


# Plot cost per sale and income
fig = sns.lmplot(y='Spent Per Sale', x='Median Household Income',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Median Household Income vs Spent Per Sale by County')
fig.savefig('plots/SpentIncome.png')


# In[62]:


# Plot cost per liter and sales
fig = sns.lmplot(y='Spent Per Liter', x='Sales Per Capita',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Sales Per Capita vs Cost Per Liter by County')
fig.savefig('plots/CostSales.png')


# In[61]:


# Plot population and sales per capita
fig = sns.lmplot(x='Estimated Population', y='Sales Per Capita',data=counties,fit_reg=True)
ax = plt.gca()
ax.set_title('Sales Per Capita vs Estimated Population by County')
fig.savefig('plots/PopSales.png')


# In[72]:


# Most common types of drinks
liquor['Category Name'].value_counts()

