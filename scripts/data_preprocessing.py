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

def promo_distribution(df_train, df_test):    
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Create a figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Plot the distribution of promotions in the training set
    sns.countplot(x='Promo', data=df_train, ax=axes[0], hue='Promo', palette='Blues', dodge=False)
    axes[0].set_title('Promo Distribution in Training Set')
    axes[0].set_xlabel('Promotion Indicator')
    axes[0].set_ylabel('Count')
    axes[0].legend().set_visible(False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Promo', 'Promo'])

    # Plot the distribution of promotions in the test set
    sns.countplot(x='Promo', data=df_test, ax=axes[1], hue='Promo', palette='Oranges', dodge=False)
    axes[1].set_title('Promo Distribution in Test Set')
    axes[1].set_xlabel('Promotion Indicator')
    axes[1].legend().set_visible(False)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['No Promo', 'Promo'])

    # Add a main title
    fig.suptitle('Comparison of Promotion Distribution in Training and Test Sets', fontsize=16)

    plt.show()

def sales_behaviour(df_train):
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

def seasonal_purchase(df_train):
    # Check seasonal purchase behaviors
    df_train['Month'] = df_train['Date'].dt.month
    df_train['Year'] = df_train['Date'].dt.year  
    seasonal_sales = df_train.groupby('Month')['Sales'].sum()
    sns.set()
    plt.figure(figsize=(10,6))
    sns.lineplot(x=seasonal_sales.index, y=seasonal_sales.values)
    plt.title('Seasonal Purchase Behaviors')
    plt.show()

def sales_vs_customer(df_train):
    # Check correlation between sales and number of customers
    correlation = df_train['Sales'].corr(df_train['Customers'])
    logger.info("Correlation between sales and number of customers: %s", correlation)
    plt.figure(figsize=(15,10))
    sns.scatterplot(x=df_train.Sales, y=df_train.Customers)           
    plt.title("Correlation between sales and number of customers")
    
def promo_effect_on_sales(df_train):
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

def customer_behaviour(df_train):
    # Check trends of customer behavior during store opening times
    opening_sales = df_train[df_train['Open'] == 1]['Sales']

    # Plotting
    sns.set()
    plt.figure(figsize=(10, 6))
    sns.histplot(opening_sales, label='Opening Sales', kde=True, color='blue')
    plt.title('Customer Behavior During Store Opening Times')
    plt.legend()
    plt.show()

def stores_open_all_weekdays(df_train):
    # Check which stores are open on all weekdays 
    weekday_stores = df_train.groupby('Store')['DayOfWeek'].nunique()
    open_all_weekdays = weekday_stores[weekday_stores == 7].index  

    sns.set()
    plt.figure(figsize=(10, 6))
    
    # Plotting the count of stores open on all weekdays
    sns.countplot(x='Store', data=df_train[df_train['Store'].isin(open_all_weekdays)])
    plt.title('Stores Open on All Weekdays')
    plt.xticks(rotation=90)  # Rotate x-axis labels if there are many stores
    plt.show()

def effect_of_assortment_on_sales(df_train, df_store):
    # Check how assortment type affects sales
    df_merged = pd.merge(df_train, df_store, on='Store')
    assortment_sales = df_merged.groupby('Assortment')['Sales'].sum()
    sns.set()
    plt.figure(figsize=(10,6))
    sns.barplot(x=assortment_sales.index, y=assortment_sales.values)
    plt.title('Effect of Assortment Type on Sales')
    plt.show()

def effect_of_competitor_distance_on_sales(df_train, df_store):
    df_store = df_store[df_store['CompetitionDistance'] != 0]
    df_store = df_store[df_store['Store'] != 0]
    df_merged = pd.merge(df_train, df_store, on='Store')
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

def effect_of_new_competitors(df_train, df_store):
    df_store = df_store[df_store['CompetitionOpenSinceYear'] != 0]
    df_store = df_store[df_store['Store'] != 0]
    df_merged = pd.merge(df_train, df_store, on='Store')
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



