#!/usr/bin/env python
# coding: utf-8

# # Forecasting Net Prophet
# 
# You’re a growth analyst at [MercadoLibre](http://investor.mercadolibre.com/investor-relations). With over 200 million users, MercadoLibre is the most popular e-commerce site in Latin America. You've been tasked with analyzing the company's financial and user data in clever ways to make the company grow. So, you want to find out if the ability to predict search traffic can translate into the ability to successfully trade the stock.
# 
# Instructions
# 
# This section divides the instructions for this Challenge into four steps and an optional fifth step, as follows:
# 
# * Step 1: Find unusual patterns in hourly Google search traffic
# 
# * Step 2: Mine the search traffic data for seasonality
# 
# * Step 3: Relate the search traffic to stock price patterns
# 
# * Step 4: Create a time series model with Prophet
# 
# * Step 5 (optional): Forecast revenue by using time series models
# 
# The following subsections detail these steps.
# 
# ## Step 1: Find Unusual Patterns in Hourly Google Search Traffic
# 
# The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.
# 
# To do so, complete the following steps:
# 
# 1. Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?
# 
# 2. Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?
# 
# ## Step 2: Mine the Search Traffic Data for Seasonality
# 
# Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.
# 
# To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:
# 
# 1. Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).
# 
# 2. Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# 
# 3. Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?
# 
# ## Step 3: Relate the Search Traffic to Stock Price Patterns
# 
# You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.
# 
# To do so, complete the following steps:
# 
# 1. Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.
# 
# 2. Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?
# 
# 3. Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
# 
#     * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
# 
#     * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
# 
# 4. Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?
# 
# ## Step 4: Create a Time Series Model with Prophet
# 
# Now, you need to produce a time series model that analyzes and forecasts patterns in the hourly search data. To do so, complete the following steps:
# 
# 1. Set up the Google search data for a Prophet forecasting model.
# 
# 2. After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?
# 
# 3. Plot the individual time series components of the model to answer the following questions:
# 
#     * What time of day exhibits the greatest popularity?
# 
#     * Which day of the week gets the most search traffic?
# 
#     * What's the lowest point for search traffic in the calendar year?
# 
# ## Step 5 (Optional): Forecast Revenue by Using Time Series Models
# 
# A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!
# 
# Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.
# 
# To do so, complete the following steps:
# 
# 1. Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.
# 
# 2. Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
# 
# 3. Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.
# 

# ## Install and import the required libraries and dependencies

# In[1]:


# Install the required libraries
get_ipython().system('pip install pystan')
get_ipython().system('pip install prophet')
get_ipython().system('pip install hvplot')
get_ipython().system('pip install holoviews')


# In[55]:


# Import the required libraries and dependencies
import pandas as pd
import holoviews as hv
from prophet import Prophet
import hvplot.pandas
from pathlib import Path
import numpy as np
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Step 1: Find Unusual Patterns in Hourly Google Search Traffic
# 
# The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.
# 
# To do so, complete the following steps:
# 
# 1. Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?
# 
# 2. Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?
# 

# #### Step 1: Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?

# In[58]:


# Upload the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame
# Set the "Date" column as the Datetime Index.

#from google.colab import files
#uploaded = files.upload()
google_hourly_search_trends_path = Path('Resources/google_hourly_search_trends.csv')
google_hourly_search_trends = pd.read_csv(
    google_hourly_search_trends_path,
    index_col='Date',
    parse_dates=True,
    infer_datetime_format=True
).dropna()

# Review the first and last five rows of the DataFrame
display(google_hourly_search_trends.head())
display(google_hourly_search_trends.tail())


# In[59]:


# Review the data types of the DataFrame using the info function
google_hourly_search_trends.info(verbose=True)


# In[60]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Slice the DataFrame to just the month of May 2020
df_may_2020 = google_hourly_search_trends.loc["2020-05-01":"2020-05-31"]

# Use hvPlot to visualize the data for May 2020
df_may_2020.hvplot()


# #### Step 2: Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?

# In[11]:


# Calculate the sum of the total search traffic for May 2020
traffic_may_2020 = df_may_2020["Search Trends"].sum()

# View the traffic_may_2020 value
traffic_may_2020


# In[61]:


# Calcluate the monhtly median search traffic across all months 
# Group the DataFrame by index year and then index month, chain the sum and then the median functions
median_monthly_traffic = google_hourly_search_trends.groupby(by=[google_hourly_search_trends.index.year, google_hourly_search_trends.index.month]).median()
# View the median_monthly_traffic value
median_monthly_traffic


# In[62]:


# Compare the seach traffic for the month of May 2020 to the overall monthly median value

Compare_may = df_may_2020.mean()


# **Question:** Did the Google search traffic increase during the month that MercadoLibre released its financial results?
# 
# **Answer:** # YOUR ANSWER HERE

# ## Step 2: Mine the Search Traffic Data for Seasonality
# 
# Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.
# 
# To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:
# 
# 1. Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).
# 
# 2. Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# 
# 3. Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?
# 

# #### Step 1: Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

# In[63]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Group the hourly search data to plot (use hvPlot) the average traffic by the day of week 
group_level = google_hourly_search_trends.index.dayofweek 
google_hourly_search_trends.groupby(group_level).mean().hvplot()


# #### Step 2: Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

# In[64]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the hour of the day and day of week search traffic as a heatmap.
google_hourly_search_trends.hvplot.heatmap(
    x='index.hour',
    y='index.dayofweek',
    C='Search Trends',
    cmap='reds'
).aggregate(function=np.mean)


# ##### Answer the following question:

# **Question:** Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# 
# **Answer:** # YOUR ANSWER HERE
# Monday

# #### Step 3: Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

# In[65]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Group the hourly search data to plot (use hvPlot) the average traffic by the week of the year
google_hourly_search_trends.groupby(by=[google_hourly_search_trends.index.year, google_hourly_search_trends.index.weekofyear]).mean().plot(
    title="Search Mean Volume per Week",
    figsize=[10, 5]
)


# ##### Answer the following question:

# **Question:** Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?
# 
# **Answer:** # YOUR ANSWER HERE

# ## Step 3: Relate the Search Traffic to Stock Price Patterns
# 
# You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.
# 
# To do so, complete the following steps:
# 
# 1. Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.
# 
# 2. Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?
# 
# 3. Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
# 
#     * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
# 
#     * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
# 
# 4. Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?
# 

# #### Step 1: Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

# In[71]:


# Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the Datetime Index.
#from google.colab import files
#uploaded = files.upload()

df_mercado_stock = Path('Resources/mercado_stock_price.csv')
df_mercado = pd.read_csv(
    df_mercado_stock,
    index_col='date',
    parse_dates=True,
    infer_datetime_format=True
).dropna()

# Review the first and last five rows of the DataFrame
display(df_mercado.head())
display(df_mercado.tail())


# In[73]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the closing price of the df_mercado_stock DataFrame
df_mercado.hvplot()


# In[74]:


# Concatenate the df_mercado_stock DataFrame with the df_mercado_trends DataFrame
# Concatenate the DataFrame by columns (axis=1), and drop and rows with only one column of data
mercado_stock_trends_df = pd.concat([df_mercado, google_hourly_search_trends], axis=1).dropna()

# View the first and last five rows of the DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())


# #### Step 2: Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

# In[75]:


# For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06) 
first_half_2020 = mercado_stock_trends_df.loc["2020-01-01":"2020-06-30"]

# View the first and last five rows of first_half_2020 DataFrame
display(first_half_2020.head())
display(first_half_2020.tail())


# In[76]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the close and Search Trends data
# Plot each column on a separate axes using the following syntax
# `hvplot(shared_axes=False, subplots=True).cols(1)`
first_half_2020.hvplot(shared_axes=False, subplots=True).cols(1)


# ##### Answer the following question:

# **Question:** Do both time series indicate a common trend that’s consistent with this narrative?
# 
# **Answer:** # Yes

# #### Step 3: Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
# 
# * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
# 
# * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
# 

# In[77]:


# Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
# This column should shift the Search Trends information by one hour
mercado_stock_trends_df['Lagged Search Trends'] = mercado_stock_trends_df['Search Trends'].shift(1)


# In[78]:


# Create a new column in the mercado_stock_trends_df DataFrame called Stock Volatility
# This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
mercado_stock_trends_df['Stock Volatility'] = mercado_stock_trends_df['close'].std()


# In[79]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the stock volatility
mercado_stock_trends_df.hvplot()


# **Solution Note:** Note how volatility spiked, and tended to stay high, during the first half of 2020. This is a common characteristic of volatility in stock returns worldwide: high volatility days tend to be followed by yet more high volatility days. When it rains, it pours.

# In[82]:


# Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
# This column should calculate hourly return percentage of the closing price
mercado_stock_trends_df['Hourly Stock Return'] = mercado_stock_trends_df['close'].pct_change()


# In[83]:


# View the first and last five rows of the mercado_stock_trends_df DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())


# #### Step 4: Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

# In[85]:


# Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
mercado_stock_trends_df[["Lagged Search Trends", "Search Trends", "Hourly Stock Return"]].corr()


# ##### Answer the following question:
# 

# **Question:** Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?
# 
# **Answer:** # YOUR ANSWER HERE

# ## Step 4: Create a Time Series Model with Prophet
# 
# Now, you need to produce a time series model that analyzes and forecasts patterns in the hourly search data. To do so, complete the following steps:
# 
# 1. Set up the Google search data for a Prophet forecasting model.
# 
# 2. After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?
# 
# 3. Plot the individual time series components of the model to answer the following questions:
# 
#     * What time of day exhibits the greatest popularity?
# 
#     * Which day of the week gets the most search traffic?
# 
#     * What's the lowest point for search traffic in the calendar year?
# 

# #### Step 1: Set up the Google search data for a Prophet forecasting model.

# In[87]:


# Using the df_mercado_trends DataFrame, reset the index so the date information is no longer the index
mercado_prophet_df = df_mercado_trends.reset_index()

# Label the columns ds and y so that the syntax is recognized by Prophet
mercado_prophet_df.columns = ['ds', 'y']

# Drop an NaN values from the prophet_df DataFrame
mercado_prophet_df = mercado_prophet_df.dropna()

# View the first and last five rows of the mercado_prophet_df DataFrame
display(mercado_prophet_df.head())
display(mercado_prophet_df.tail())


# In[91]:


# Call the Prophet function, store as an object
model_mercado_trends =  Prophet()
model_mercado_trends


# In[92]:


# Fit the time-series model.
model_mercado_trends.fit(mercado_prophet_df)


# In[93]:


# Create a future dataframe to hold predictions
# Make the prediction go out as far as 2000 hours (approx 80 days)
future_mercado_trends =  model_mercado_trends.make_future_dataframe(periods=2000, freq='H')


# View the last five rows of the future_mercado_trends DataFrame
display(future_mercado_trends.tail(5))


# In[110]:


# Make the predictions for the trend data using the future_mercado_trends DataFrame
forecast_mercado_trends = model_mercado_trends.predict(future_mercado_trends)

# Display the first five rows of the forecast_mercado_trends DataFrame
display(forecast_mercado_trends.head())


# #### Step 2: After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?

# In[97]:


# Plot the Prophet predictions for the Mercado trends data
model_mercado_trends.plot(forecast_mercado_trends)


# ##### Answer the following question:

# **Question:**  How's the near-term forecast for the popularity of MercadoLibre?
# 
# **Answer:** # YOUR ANSWER HERE
# 

# #### Step 3: Plot the individual time series components of the model to answer the following questions:
# 
# * What time of day exhibits the greatest popularity?
# 
# * Which day of the week gets the most search traffic?
# 
# * What's the lowest point for search traffic in the calendar year?
# 

# In[111]:


# Set the index in the forecast_mercado_trends DataFrame to the ds datetime column
forecast_mercado_trends[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[112]:


forecast_mercado_trends = forecast_mercado_trends.set_index('ds')
forecast_mercado_trends.head()


# In[115]:


# View the only the yhat,yhat_lower and yhat_upper columns from the DataFrame

forecast_mercado_trends[['yhat', 'yhat_lower', 'yhat_upper']].head()


# Solutions Note: `yhat` represents the most likely (average) forecast, whereas `yhat_lower` and `yhat_upper` represents the worst and best case prediction (based on what are known as 95% confidence intervals).

# In[113]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# From the forecast_mercado_trends DataFrame, use hvPlot to visualize
#  the yhat, yhat_lower, and yhat_upper columns over the last 2000 hours 
forecast_mercado_trends[['yhat', 'yhat_lower', 'yhat_upper']].hvplot()


# In[117]:


# Reset the index in the forecast_mercado_trends DataFrame
forecast_mercado_trends = forecast_mercado_trends.reset_index()

# Use the plot_components function to visualize the forecast results 
# for the forecast_mercado_trends DataFrame 
figures_mercado_trends = model_mercado_trends.plot_components(forecast_mercado_trends)


# ##### Answer the following questions:

# **Question:** What time of day exhibits the greatest popularity?
# 
# **Answer:** # YOUR ANSWER HERE

# **Question:** Which day of week gets the most search traffic? 
#    
# **Answer:** # YOUR ANSWER HERE

# **Question:** What's the lowest point for search traffic in the calendar year?
# 
# **Answer:** # YOUR ANSWER HERE
# 

# ## Step 5 (Optional): Forecast Revenue by Using Time Series Models
# 
# A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!
# 
# Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.
# 
# To do so, complete the following steps:
# 
# 1. Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data. The daily sales figures are quoted in millions of USD dollars.
# 
# 2. Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
# 
# 3. Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.
# 
# 
# 

# #### Step 1: Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.

# In[ ]:


# Upload the "mercado_daily_revenue.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the DatetimeIndex
# Sales are quoted in millions of US dollars
from google.colab import files
uploaded = files.upload()

df_mercado_sales = # YOUR CODE HERE

# Review the DataFrame
# YOUR CODE HERE


# In[ ]:


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the daily sales figures 
# YOUR CODE HERE


# In[ ]:


# Apply a Facebook Prophet model to the data.

# Set up the dataframe in the neccessary format:
# Reset the index so that date becomes a column in the DataFrame
mercado_sales_prophet_df = # YOUR CODE HERE

# Adjust the columns names to the Prophet syntax
mercado_sales_prophet_df.columns = # YOUR CODE HERE

# Visualize the DataFrame
# YOUR CODE HERE


# In[ ]:


# Create the model
mercado_sales_prophet_model = # YOUR CODE HERE

# Fit the model
# YOUR CODE HERE


# In[ ]:


# Predict sales for 90 days (1 quarter) out into the future.

# Start by making a future dataframe
mercado_sales_prophet_future = # YOUR CODE HERE

# Display the last five rows of the future DataFrame
# YOUR CODE HERE


# In[ ]:


# Make predictions for the sales each day over the next quarter
mercado_sales_prophet_forecast = # YOUR CODE HERE

# Display the first 5 rows of the resulting DataFrame
# YOUR CODE HERE


# #### Step 2: Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

# In[ ]:


# Use the plot_components function to analyze seasonal patterns in the company's revenue
# YOUR CODE HERE


# ##### Answer the following question:

# **Question:** For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
# 
# **Answer:** # YOUR ANSWER HERE

# #### Step 3: Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.

# In[ ]:


# Plot the predictions for the Mercado sales
# YOUR CODE HERE


# In[ ]:


# For the mercado_sales_prophet_forecast DataFrame, set the ds column as the DataFrame Index
mercado_sales_prophet_forecast = # YOUR CODE HERE

# Display the first and last five rows of the DataFrame
# YOUR CODE HERE


# In[ ]:


# Produce a sales forecast for the finance division
# giving them a number for expected total sales next quarter.
# Provide best case (yhat_upper), worst case (yhat_lower), and most likely (yhat) scenarios.

# Create a forecast_quarter Dataframe for the period 2020-07-01 to 2020-09-30
# The DataFrame should include the columns yhat_upper, yhat_lower, and yhat
mercado_sales_forecast_quarter = # YOUR CODE HERE

# Update the column names for the forecast_quarter DataFrame
# to match what the finance division is looking for 
mercado_sales_forecast_quarter = # YOUR CODE HERE

# Review the last five rows of the DataFrame
# YOUR CODE HERE


# In[ ]:


# Displayed the summed values for all the rows in the forecast_quarter DataFrame
# YOUR CODE HERE


# ### Based on the forecast information generated above, produce a sales forecast for the finance division, giving them a number for expected total sales next quarter. Include best and worst case scenarios, to better help the finance team plan.
# 
# **Answer:** # YOUR ANSWER HERE

# In[ ]:




