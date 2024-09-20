import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(train_file, test_file, store_file):
    """Load data from csv files"""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    store_df = pd.read_csv(store_file)
    return train_df, test_df, store_df

def exploratory_data_analysis(df_train, df_test, df_store):
    """Perform exploratory data analysis"""
    # Check distribution of sales in training and test sets
    sns.set()
    plt.figure(figsize=(10,6))
    sns.histplot(df_train['Promo'], label='Training Set', kde=True)
    sns.histplot(df_test['Promo'], label='Test Set', kde=True)
    plt.title('Distribution of Promotion in Training and Test Sets')
    plt.legend()
    plt.show()

    # Check sales behavior before, during, and after holidays
    holiday_sales = df_train[df_train['StateHoliday']!= 0]['Sales']
    non_holiday_sales = df_train[df_train['StateHoliday'] == 0]['Sales']
    sns.set()
    plt.figure(figsize=(10,6))
    sns.histplot(holiday_sales, label='Holiday Sales', kde=True)
    sns.histplot(non_holiday_sales, label='Non-Holiday Sales', kde=True)
    plt.title('Comparison of Sales Before, During, and After Holidays')
    plt.legend()
    plt.show()

    # Check seasonal purchase behaviors
    df_train['Month'] = df_train['Date'].dt.month
    df_train['Year'] = df_train['Date'].dt.year  
    seasonal_sales = df_train.groupby('Month')['Sales'].sum()
    sns.set()
    plt.figure(figsize=(10,6))
    sns.lineplot(x=seasonal_sales.index, y=seasonal_sales.values)
    plt.title('Seasonal Purchase Behaviors')
    plt.show()

    # Check correlation between sales and number of customers
    correlation = df_train['Sales'].corr(df_train['Customers'])
    logger.info("Correlation between sales and number of customers: %s", correlation)

    # Check promo effect on sales
    promo_sales = df_train[df_train['Promo'] == 1]['Sales']
    non_promo_sales = df_train[df_train['Promo'] == 0]['Sales']
    sns.set()
    plt.figure(figsize=(10,6))
    sns.histplot(promo_sales, label='Promo Sales', kde=True)
    sns.histplot(non_promo_sales, label='Non-Promo Sales', kde=True)
    plt.title('Effect of Promo on Sales')
    plt.legend()
    plt.show()

    # Check trends of customer behavior during store opening and closing times
    opening_sales = df_train[df_train['Open'] == 1]['Sales']
    closing_sales = df_train[df_train['Open'] == 0]['Sales']
    sns.set()
    plt.figure(figsize=(10,6))
    sns.histplot(opening_sales, label='Opening Sales', kde=True)
    sns.histplot(closing_sales, label='Closing Sales', kde=True)
    plt.title('Customer Behavior During Store Opening and Closing Times')
    plt.legend()
    plt.show()

    # Check which stores are open on all weekdays
    weekday_stores = df_train.groupby('Store')['DayOfWeek'].nunique()
    open_all_weekdays = weekday_stores[weekday_stores == 5].index
    sns.set()
    plt.figure(figsize=(10,6))
    sns.countplot(x='Store', data=df_train[df_train['Store'].isin(open_all_weekdays)]) 
    plt.title('Stores Open on All Weekdays')
    plt.show()

    # Check how assortment type affects sales
    df_merged = pd.merge(df_train, df_store, on='Store')
    assortment_sales = df_merged.groupby('Assortment')['Sales'].sum()
    sns.set()
    plt.figure(figsize=(10,6))
    sns.barplot(x=assortment_sales.index, y=assortment_sales.values)
    plt.title('Effect of Assortment Type on Sales')
    plt.show()

    # Grouping sales by the distance to the next competitor
    distance_sales = df_merged.groupby('CompetitionDistance')['Sales'].sum()

    # Setting the seaborn style
    sns.set()

    # Creating the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=distance_sales.index, y=distance_sales.values)
    plt.title('Effect of Distance to Competitor on Sales')
    plt.xlabel('Distance to Competitor')
    plt.ylabel('Total Sales')
    plt.show()


    # Grouping sales by the year competitors opened
    new_competitor_sales = df_merged.groupby('CompetitionOpenSinceYear')['Sales'].sum()

    # Setting the seaborn style
    sns.set()

    # Creating the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=new_competitor_sales.index, y=new_competitor_sales.values)
    plt.title('Effect of New Competitors on Sales')
    plt.xlabel('Year Competitors Opened')
    plt.ylabel('Total Sales')
    plt.show()



