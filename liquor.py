
# coding: utf-8

# In[1]:


# COSC 4610 Final Project
import pandas as pd


# In[2]:


# Read in liquor sales dataset (go make some coffee)
liquor = pd.read_csv('data/Iowa_Liquor_Sales.csv', parse_dates=True, index_col='Invoice/Item Number').sort_values('Date')


# In[3]:


# Makes all county names uniform (all caps, no "COUNTY")
def correctCounty(county):
    if isinstance(county, str):
        return county.upper().replace(' COUNTY', '')
    return str(county)


# In[4]:


# Trim unnecessary rows
liquor = liquor.drop('Address', axis=1)
liquor = liquor.drop('City', axis=1)
liquor = liquor.drop('Zip Code', axis=1)
liquor = liquor.drop('County Number', axis=1)
liquor = liquor.drop('Volume Sold (Gallons)', axis=1)
liquor = liquor.drop('Pack', axis=1)
# Make county names uniform
liquor['County'] = liquor['County'].apply(correctCounty)


# In[5]:


earliest = liquor['Date'].min().split('/')
latest = liquor['Date'].max().split('/')
from datetime import date
earliest = date(int(earliest[2]), int(earliest[0]), int(earliest[1]))
latest = date(int(latest[2]), int(latest[0]), int(latest[1]))
delta = (latest - earliest).days / 365
dateRange = str(earliest.strftime('%m/%d/%Y')) + ' to ' + str(latest.strftime('%m/%d/%Y'))
delta


# In[6]:


liquor


# In[7]:


# Read in poverty dataset and trim to median incomes by county
poverty = pd.read_csv('data/est16-ia.csv')
poverty['County'] = poverty['County'].apply(correctCounty)
poverty = poverty.set_index('County')
income = poverty['Median_Household_Income'].sort_values(ascending=False)
#income


# In[8]:


# Plot median household income by county
income.plot(kind='bar', figsize=(15,7))


# In[9]:


# The liquor sales have fake? counties, find them
overlap = liquor['County'].isin(income.index)
overlap.value_counts()


# In[10]:


# Remove "fake" counties
liquor = liquor[overlap]
liquor['County'].isin(income.index).value_counts()


# In[18]:


# Count sales by county
pd.set_option("max_rows",105)
countySales = liquor['County'].value_counts()

# Plot that
get_ipython().run_line_magic('matplotlib', 'inline')
title = 'Total Sales by County ' + dateRange
plot = countySales.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Sales')


# In[12]:


# Estimate populations by county
populations = (poverty['People_of_All_Ages_in_Poverty'] * 100 / poverty['Percent_of_People_of_All_Ages_in_Poverty']).astype(int).sort_values(ascending=False)
#populations


# In[19]:


# Calculate and plot liquor sales per capita per year
salesPerCapita = (countySales / populations).sort_values(ascending=False)
salesPerCapita = salesPerCapita.divide(delta)
title = 'Avg Annual Per Capita Sales by County ' + dateRange
plot = salesPerCapita.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Sales')


# In[14]:


# Get the total volume sold by county
def getCountyVolume(county):
    return liquor.loc[liquor['County'] == county, 'Volume Sold (Liters)'].sum()

countyVolumes = pd.Series()
for county in income.index:
    countyVolumes = countyVolumes.set_value(county, getCountyVolume(county))

countyVolumes = countyVolumes.sort_values(ascending=False)


# In[20]:


# Plot volume per capita by county per year
volumePerCapita = (countyVolumes / populations).sort_values(ascending=False)
volumePerCapita = volumePerCapita.divide(delta)
title = 'Avg Annual Per Capita Volume Sold by County ' + dateRange
plot = volumePerCapita.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Volume (L)')


# In[21]:


# Volume per sale by county, these numbers seem fishy
volumePerSale = (countyVolumes / countySales).sort_values(ascending=False)
title = 'Avg Volume Per Sale by County ' + dateRange
plot = volumePerSale.plot(kind='bar', figsize=(15,7), title=title)
plot.set_xlabel('County')
plot.set_ylabel('Liquor Volume (L)')


# In[17]:


# Get the median income of a county by name
def getIncome(county):
    return income.loc[county]

# Apply the income of the county to each sale
liquor['Median Household Income'] = liquor['County'].apply(getIncome)

