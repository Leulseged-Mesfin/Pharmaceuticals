import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def percentage_missing_values(df):
    total_number_cells = df.shape[0]
    countMissing = df.isnull().sum()
    # totalMissing = countMissing.sum()
    return f"The telecom contains {round(((countMissing/total_number_cells) * 100), 2)}% missing values."


# def fill_null_values(df):
#     for column in df.columns:
#         if df[column].dtype == 'object' and df[column].dtype == 'category':
#             # Fill missing values with the previous value (forward fill)
#             df[column].fillna(method='ffill', inplace=True)
#         elif df[column].dtype == 'float64' and df[column].dtype == 'int64':
#             # Fill missing values with 0
#             df[column].fillna(0, inplace=True)
#     return df

def fill_null_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            # Fill missing values with the previous value (forward fill)
            df[column].fillna(method='ffill', inplace=True)
        elif df[column].dtype == 'float64' or df[column].dtype == 'int64':
            # Fill missing values with 0
            df[column].fillna(0, inplace=True)
    return df.info()



def promo_distribution(train_df, test_df):
    # Calculate promotion distribution for train_dfing set
    train_df_promo_dist = train_df['Promo'].value_counts(normalize=True)

    # Calculate promotion distribution for test_df set
    test_df_promo_dist = test_df['Promo'].value_counts(normalize=True)

    # Print the distributions
    print("Train_dfing set promo distribution:\n", train_df_promo_dist)
    print("Test_df set promo distribution:\n", test_df_promo_dist)

    # Plot the distributions for visual comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    train_df['Promo'].value_counts(normalize=True).plot(kind='bar', ax=ax[0], color=['lightblue', 'lightgreen'])
    ax[0].set_title('Train_df Set Promo Distribution')
    ax[0].set_xlabel('Promo')
    ax[0].set_ylabel('Proportion')

    test_df['Promo'].value_counts(normalize=True).plot(kind='bar', ax=ax[1], color=['lightblue', 'lightgreen'])
    ax[1].set_title('Test_df Set Promo Distribution')
    ax[1].set_xlabel('Promo')
    ax[1].set_ylabel('Proportion')

    plt.tight_layout()
    plt.show()

# 
def compare_sale_behavior(train_df):
    # Convert 'Date' column to datetime
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # Identify days before, during, and after holidays
    train_df['BeforeHoliday'] = train_df['StateHoliday'].shift(-1).notna() & (train_df['StateHoliday'] == '0')  # Days before holiday
    train_df['DuringHoliday'] = train_df['StateHoliday'] != '0'  # Days during holiday
    train_df['AfterHoliday'] = train_df['StateHoliday'].shift(1).notna() & (train_df['StateHoliday'] == '0')  # Days after holiday

    # Create a new column to label the period (Before, During, After)
    train_df['HolidayPeriod'] = 'Regular'
    train_df.loc[train_df['BeforeHoliday'], 'HolidayPeriod'] = 'Before Holiday'
    train_df.loc[train_df['DuringHoliday'], 'HolidayPeriod'] = 'During Holiday'
    train_df.loc[train_df['AfterHoliday'], 'HolidayPeriod'] = 'After Holiday'

    # Calculate average sales for each period
    sales_behavior = train_df.groupby('HolidayPeriod')['Sales'].mean()

    # Print out the results
    print(sales_behavior)

    # Plot the sales behavior
    sales_behavior.plot(kind='bar', color='lightblue')
    plt.title('Average Sales Before, During, and After Holidays')
    plt.ylabel('Average Sales')
    plt.xlabel('Holiday Period')
    plt.xticks(rotation=0)
    plt.show()


def seasonal_puchacse_behavior(train_df):
    # Convert 'Date' to datetime format
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # Add a 'Month' and 'Day' column for easier filtering
    train_df['Month'] = train_df['Date'].dt.month
    train_df['Day'] = train_df['Date'].dt.day

    # Mark Christmas (usually December 24th - 26th), Easter (around April, can vary by year), etc.
    train_df['Season'] = 'Regular'

    # Christmas (December 24th to 26th)
    train_df.loc[(train_df['Month'] == 12) & (train_df['Day'].between(24, 26)), 'Season'] = 'Christmas'

    # Easter (approx. April, we'll assume April 10th to 20th for simplicity)
    train_df.loc[(train_df['Month'] == 4) & (train_df['Day'].between(10, 20)), 'Season'] = 'Easter'

    # Add other seasonal events if necessary (e.g., Thanksgiving, Black Friday, etc.)
    # Example for Easter: Update the date range accordingly to actual Easter dates for your data.

    # Calculate average sales per season
    seasonal_sales = train_df.groupby('Season')['Sales'].mean()

    # Print out the seasonal behavior
    print(seasonal_sales)

    # Visualize the seasonal sales trends
    seasonal_sales.plot(kind='bar', color=['lightblue', 'lightgreen', 'orange'])
    plt.title('Average Sales During Christmas, Easter, and Regular Periods')
    plt.ylabel('Average Sales')
    plt.xlabel('Season')
    plt.xticks(rotation=0)
    plt.show()


def correlation_sales_customer(train_df):
    # Create a scatter plot to visualize the correlation between sales and customers
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Customers', y='Sales', data=train_df, color='blue', alpha=0.5)

    # Add a trend line (optional, but helps to see the relationship better)
    sns.regplot(x='Customers', y='Sales', data=train_df, scatter=False, color='red')

    plt.title('Correlation Between Sales and Number of Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

    # Calculate the correlation coefficient
    correlation = train_df['Sales'].corr(train_df['Customers'])
    print(f"Correlation between Sales and Customers: {correlation:.2f}")


def affect_promo_on_sales(train_df):
    # Calculate the average sales and customers with and without promotion
    promo_sales = train_df.groupby('Promo').agg({'Sales': 'mean', 'Customers': 'mean'})
    promo_sales['Sales per Customer'] = promo_sales['Sales'] / promo_sales['Customers']

    # Print the results
    print("Average sales, customer count, and sales per customer during and without promotions:")
    print(promo_sales)

    # Plot the comparison
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Sales comparison
    sns.barplot(x=promo_sales.index, y=promo_sales['Sales'], ax=ax[0], palette='Blues')
    ax[0].set_title('Average Sales During and Without Promotion')
    ax[0].set_xlabel('Promo')
    ax[0].set_ylabel('Average Sales')

    # Customer count comparison
    sns.barplot(x=promo_sales.index, y=promo_sales['Customers'], ax=ax[1], palette='Greens')
    ax[1].set_title('Average Number of Customers During and Without Promotion')
    ax[1].set_xlabel('Promo')
    ax[1].set_ylabel('Average Customers')

    # Sales per customer comparison
    sns.barplot(x=promo_sales.index, y=promo_sales['Sales per Customer'], ax=ax[2], palette='Oranges')
    ax[2].set_title('Average Sales per Customer During and Without Promotion')
    ax[2].set_xlabel('Promo')
    ax[2].set_ylabel('Sales per Customer')

    plt.tight_layout()
    plt.show()


def deploy_promo(train_df, store_df):
    
    # Merge store info with train_df dataset
    train_df = pd.merge(train_df, store_df, on='Store')

    # Group by 'Store' and 'Promo' to evaluate sales and customer behavior for each store during and without promotions
    store_promo_analysis = train_df.groupby(['Store', 'Promo']).agg({
        'Sales': 'mean',
        'Customers': 'mean'
    }).reset_index()

    # Calculate the sales and customer growth percentage during promotions
    store_promo_analysis['Sales Growth (%)'] = store_promo_analysis.groupby('Store')['Sales'].pct_change() * 100
    store_promo_analysis['Customer Growth (%)'] = store_promo_analysis.groupby('Store')['Customers'].pct_change() * 100

    # Filter for stores where promotions are active (Promo = 1)
    promo_effective_stores = store_promo_analysis[store_promo_analysis['Promo'] == 1]

    # Display stores with high sales growth during promotions
    high_sales_growth_stores = promo_effective_stores.sort_values(by='Sales Growth (%)', ascending=False)

    # Plot the top 10 stores with the highest sales growth during promotions
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store', y='Sales Growth (%)', data=high_sales_growth_stores.head(10), palette='Blues')
    plt.title('Top 10 Stores with Highest Sales Growth During Promotions')
    plt.xlabel('Store')
    plt.ylabel('Sales Growth (%)')
    plt.show()

    # Show top stores for customer growth as well
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store', y='Customer Growth (%)', data=promo_effective_stores.sort_values(by='Customer Growth (%)', ascending=False).head(10), palette='Greens')
    plt.title('Top 10 Stores with Highest Customer Growth During Promotions')
    plt.xlabel('Store')
    plt.ylabel('Customer Growth (%)')
    plt.show()

    # Analyze stores with little to no growth during promotions
    low_sales_growth_stores = promo_effective_stores[promo_effective_stores['Sales Growth (%)'] < 10]
    print("Stores with minimal sales growth during promotions:", low_sales_growth_stores['Store'].unique())


def customer_behavior_trend(train_df):
    # Convert 'Date' to datetime format
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # Extract hour from 'Date'
    train_df['Hour'] = train_df['Date'].dt.hour

    # Group by hour and calculate average customers
    hourly_customers = train_df.groupby('Hour')['Customers'].mean().reset_index()

    # Plotting the trends of customer behavior during opening and closing times
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Hour', y='Customers', data=hourly_customers, marker='o', color='blue')

    plt.title('Average Number of Customers by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Number of Customers')
    plt.xticks(range(0, 24))  # Show all hours
    plt.grid()
    plt.axvline(x=9, color='red', linestyle='--', label='Store Opens')
    plt.axvline(x=21, color='green', linestyle='--', label='Store Closes')
    plt.legend()
    plt.show()


def affect_weekend_sales(train_df):
    # Convert 'Date' to datetime format and extract day of the week
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df['DayOfWeek'] = train_df['Date'].dt.dayofweek  # Monday=0, Sunday=6

    # Check which stores are open on all weekdays
    weekdays = [0, 1, 2, 3, 4]  # Monday to Friday
    open_weekdays = train_df[train_df['Open'] == 1].groupby('Store')['DayOfWeek'].unique()

    # Identify stores open on all weekdays
    stores_open_all_weekdays = [store for store, days in open_weekdays.items() if set(days) == set(weekdays)]

    # Filter weekend sales for these stores
    weekend_sales_open_all = train_df[(train_df['Store'].isin(stores_open_all_weekdays)) & (train_df['DayOfWeek'].isin([5, 6]))]
    weekend_sales_not_open_all = train_df[~train_df['Store'].isin(stores_open_all_weekdays) & (train_df['DayOfWeek'].isin([5, 6]))]

    # Calculate average weekend sales for both groups
    avg_sales_open_all = weekend_sales_open_all['Sales'].mean()
    avg_sales_not_open_all = weekend_sales_not_open_all['Sales'].mean()

    # Create a summary DataFrame for plotting
    sales_comparison = pd.DataFrame({
        'Store Group': ['Open on All Weekdays', 'Not Open on All Weekdays'],
        'Average Weekend Sales': [avg_sales_open_all, avg_sales_not_open_all]
    })

    # Plot the comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Store Group', y='Average Weekend Sales', data=sales_comparison, palette='pastel')
    plt.title('Average Weekend Sales: Stores Open on All Weekdays vs. Not')
    plt.ylabel('Average Weekend Sales')
    plt.xticks(rotation=10)
    plt.show()


def affects_assortment_sales(train_df):
    # Group by 'Assortment' and calculate average sales
    assortment_sales = train_df.groupby('Assortment')['Sales'].mean().reset_index()

    # Sort the results for better visualization
    assortment_sales = assortment_sales.sort_values(by='Sales', ascending=False)

    # Plotting the average sales by assortment type
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sales', y='Assortment', data=assortment_sales, palette='viridis')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Average Sales')
    plt.ylabel('Assortment Type')
    plt.grid(axis='x')
    plt.show()
