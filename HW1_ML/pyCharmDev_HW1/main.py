# About Dataset
# Customer Personality Analysis involves a thorough examination of a company's optimal customer profiles. This analysis facilitates a deeper understanding of customers, enabling businesses to tailor products to meet the distinct needs, behaviors, and concerns of various customer types.
# By conducting a Customer Personality Analysis, businesses can refine their products based on the preferences of specific customer segments. Rather than allocating resources to market a new product to the entire customer database, companies can identify the segments most likely to be interested in the product. Subsequently, targeted marketing efforts can be directed toward those particular segments, optimizing resource utilization and increasing the likelihood of successful product adoption.
# Details of Features are as below:
# •	Id: Unique identifier for each individual in the dataset.
# •	Year_Birth: The birth year of the individual.
# •	Education: The highest level of education attained by the individual.
# •	Marital_Status: The marital status of the individual.
# •	Income: The annual income of the individual.
# •	Kidhome: The number of young children in the household.
# •	Teenhome: The number of teenagers in the household.
# •	Dt_Customer: The date when the customer was first enrolled or became a part of the company's database.
# •	Recency: The number of days since the last purchase or interaction.
# •	MntWines: The amount spent on wines.
# •	MntFruits: The amount spent on fruits.
# •	MntMeatProducts: The amount spent on meat products.
# •	MntFishProducts: The amount spent on fish products.
# •	MntSweetProducts: The amount spent on sweet products.
# •	MntGoldProds: The amount spent on gold products.
# •	NumDealsPurchases: The number of purchases made with a discount or as part of a deal.
# •	NumWebPurchases: The number of purchases made through the company's website.
# •	NumCatalogPurchases: The number of purchases made through catalogs.
# •	NumStorePurchases: The number of purchases made in physical stores.
# •	NumWebVisitsMonth: The number of visits to the company's website in a month.
# •	AcceptedCmp3: Binary indicator (1 or 0) whether the individual accepted the third marketing campaign.
# •	AcceptedCmp4: Binary indicator (1 or 0) whether the individual accepted the fourth marketing campaign.
# •	AcceptedCmp5: Binary indicator (1 or 0) whether the individual accepted the fifth marketing campaign.
# •	AcceptedCmp1: Binary indicator (1 or 0) whether the individual accepted the first marketing campaign.
# •	AcceptedCmp2: Binary indicator (1 or 0) whether the individual accepted the second marketing campaign.
# •	Complain: Binary indicator (1 or 0) whether the individual has made a complaint.
# •	Z_CostContact: A constant cost associated with contacting a customer.
# •	Z_Revenue: A constant revenue associated with a successful campaign response.
# •	Response: Binary indicator (1 or 0) whether the individual responded to the marketing campaign.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.decomposition import PCA

##graph 1
def bar_chart_age_count(data_frame):
    # Sum the amounts for each food category across all customers
    current_year = 2024
    data_frame['age'] = current_year - data_frame['Year_Birth']

    # Define age categories
    bins = [0, 18, 35, 55, 75, data_frame['age'].max()]
    labels = ['0-18', '18-35', '35-55', '55-75', '75+']

    # Add a new column for age category
    data_frame['age_category'] = pd.cut(data_frame['age'], bins=bins, labels=labels, right=False)

    # Create a bar plot
    plt.figure(figsize=(4, 3))
    sns.countplot(x='age_category', data=data_frame, palette='viridis')
    plt.title('Number of People by Age Category')
    plt.xlabel('Age Category')
    plt.ylabel('Number of People')
    plt.show()

    #**Insight** -  The Insight we can take from Graph 1 is that most Customers are between the Ages of 35 and 75, We have Some elderly over the age of 75 and some who are young (18 to 35) We Also do not have under age Customers.
##graph 2
def pie_chart_food_categories(data_frame):
    #Sum the amounts for each food category across all customers
    category_totals = data_frame[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']].sum()

    #dictionary to map original category names to custom labels
    category_labels = {
        'MntWines': 'Wines',
        'MntFruits': 'Fruits',
        'MntFishProducts': 'Fish',
        'MntMeatProducts': 'Meats',
        'MntSweetProducts': 'Sweets',
        'MntGoldProds': 'Gold'
    }

    #Map original category names to my labels
    custom_labels = [category_labels.get(category, category) for category in category_totals.index]

    # Plot a pie chart with custom labels
    plt.figure(figsize=(6, 6))
    plt.pie(category_totals, labels=custom_labels, autopct='%1.1f%%', startangle=140,
            colors=['lightcoral', 'skyblue', 'lightgreen', 'gold', 'lightpink','black'])
    plt.title('Amount Spent Per Food Category')
    plt.show()


    ##**Insight** -  In Graph 2 we can understand that the most amount of money spent by customers is on Wines,
    # Above 50% of all spendings!, After that comes in Meat products and lastly are Fish Products, which customers spend the least on.
    # We can see the Distributions of Ratios in This Pie Chart.

    ##graph 3
def response_count_by_education(data_frame):
        # Group the data by education level and calculate the counts of responders and non-responders
   # education_counts = data_frame.groupby(['Education', 'Response']).size().unstack().fillna(0)
        # Group the data by education level and calculate the counts of responders and non-responders
        # Group the data by education level and calculate the counts of responders and non-responders
        education_counts = data_frame.groupby(['Education', 'Response']).size().unstack().fillna(0)

        # Reset the index to make 'Education' and 'Response' regular columns
        education_counts = education_counts.reset_index()

        # Melt the DataFrame
        melted_data = pd.melt(education_counts, id_vars='Education', var_name='Response', value_name='Count')

        # Plot a bar chart
        plt.figure(figsize=(6, 4))
        sns.barplot(x='Education', y='Count', hue='Response', data=melted_data)

        # Set legend labels and title
        plt.legend(title='Response', labels=['Non-Responder - Orange', 'Responder - Blue'])

        plt.title('Responder Distribution by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Count')
        plt.show()



        ## ---Q2: K-MEANS---

# Scale Data Using MinMax Scaler and Encode it
def min_max_scale(data_frame):
    # Get the size of the DataFrame
    rows, columns = data_frame.shape
    # Scale the data using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    # fit and transform the data

    # Identify numerical and non-numerical columns
    numerical_columns = ['Year_Birth', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                         'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumStorePurchases', 'NumWebVisitsMonth']
    non_numerical_columns = [col for col in data_frame_2.columns if col not in numerical_columns]

    # Separate the data into numerical and non-numerical DataFrames
    numerical_columns = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                         'MntSweetProducts', 'MntGoldProds', 'NumCatalogPurchases', 'NumDealsPurchases',
                         'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    non_numerical_columns = [col for col in data_frame_2.columns if col not in numerical_columns]

    # Separate the data into numerical and non-numerical DataFrames
    numerical_data = data_frame_2[numerical_columns]
    non_numerical_data = data_frame_2[non_numerical_columns]

    # Scale the numerical data using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    scaled_numerical_data = pd.DataFrame(min_max_scaler.fit_transform(numerical_data), columns=numerical_data.columns)

    # Concatenate the scaled numerical data with the non-numerical data
    scaled_data_frame_2 = pd.concat([non_numerical_data, scaled_numerical_data], axis=1)

    print("Original Data:")
    print(data_frame_2.head())
    print("\nScaled Data:")
    print(scaled_data_frame_2.head())
    # Encode categorical variables

        #**Insight** - We can perceive form Graph 3 that the most prevalent level of education is "Graduation" level and the least one is "Basic",
        # The No-Response:Response Ratio of all Levels is very high, meaning that most people DO NOT respond




def one_hot_encode(df):
        categorical_vars = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4',
                            'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue',
                            'Response']
        # Encode categorical variables
        encoder = OneHotEncoder()
        categorical_data = df[categorical_vars]
        # Fit and transform the categorical data
        encoded_categorical_data = encoder.fit_transform(categorical_data)
        print("Encoded variables: \n")
        print(encoded_categorical_data)

        # Create a DataFrame with the one-hot encoded features
        encoded_df = pd.DataFrame(encoded_categorical_data)

        # Display the final DataFrame
        print("Encoded Data:")
        print(encoded_df.head())



def scale_and_encode(df):

    rows, columns = df.shape
    # Scale the data using MinMaxScaler
    min_max_scaler = MinMaxScaler()
    # fit and transform the data

    # Identify numerical and non-numerical columns and other ones
    #13
    numerical_columns = ['Year_Birth', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                         'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumStorePurchases', 'NumWebVisitsMonth']
    #13
    categorical_columns = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4',
                           'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue',
                           'Response']
    other_columns = ['ID','Dt_Customer']
    #non_numerical_columns = [col for col in df.columns if col not in numerical_columns]
  #  non_numerical_data = df[non_numerical_columns]
    other_data = df[other_columns]
    numerical_data = df[numerical_columns]
    categorical_data = df[categorical_columns]

    scaled_numerical_data = pd.DataFrame(min_max_scaler.fit_transform(numerical_data), columns=numerical_data.columns)

    # Concatenate the scaled numerical data with the non-numerical data
    scaled_df = pd.concat([other_data, scaled_numerical_data], axis=1)



    # Encode categorical variables
    encoder = OneHotEncoder()
    # 13
    categorical_columns = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3', 'AcceptedCmp4',
                           'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue',
                           'Response']
    # Fit and transform the categorical data
   # encoded_categorical_data = encoder.fit_transform(categorical_data)
    encoded_categorical_data = encoder.fit_transform(categorical_data).toarray()

    print("Categorical data: Num of cols " + str(categorical_data.shape[1]))
    print(categorical_data)

    print("ENCODED Categorical data: Num of cols " + str(encoded_categorical_data.shape[1]))
    print(encoded_categorical_data)

    # Get the feature names
 #   feature_names = encoder.get_feature_names(input_features=categorical_columns)
    categories = [f"{col}_{category}" for col, cats in zip(categorical_data.columns, encoder.categories_) for category
                  in cats]
   # feature_names_array = np.array(encoder.categories_).ravel()
    feature_names_array = np.array(encoder.categories_, dtype=object).ravel()
    print("Feature names: " , feature_names_array)

    # Create a DataFrame with the one-hot encoded features and feature names
   # encoded_df = pd.DataFrame(encoded_categorical_data, columns=feature_names)
    encoded_df = pd.DataFrame(encoded_categorical_data, columns=categories)
    print("Encoded variables: \n")
    print(encoded_categorical_data)
    print("encoded_df: Num of cols " + str(encoded_df.shape[1]))
    #encoded_df = pd.get_dummies(df, columns=['categorical_columns', ])
    #print("encoded df size "  + str(encoded_df.size()))


    # Concatenate the encoded categorical variables with the scaled data
    final_data = pd.concat([scaled_df, encoded_df], axis=1)
   # final_data.to_csv('pre_processed_customer_segmentation_v2.csv', index=False)
    # Display the final DataFrame
    print("Final Data:")
    print(final_data.head())
    return final_data


def k_means_n5_wines_meats(df):
    # Apply k-Means on the 'MntWines' and 'MntMeatProducts' features with n_clusters=5
    wines_products_cols = ['MntWines', 'MntMeatProducts']
    extracted_data_set = df[wines_products_cols]
    res = KMeans(n_clusters=5, random_state=0, n_init='auto')
    res.fit(extracted_data_set)
    # Visualize the clusters
    sns.scatterplot(data=extracted_data_set, x='MntWines', y='MntMeatProducts', hue=res.labels_)
if __name__ == '__main__':
    print("Hello world")

    ## Q1: ----- Graph PLOTTING AND Insights -----

    ## ---GRAPH 1: Bar Graph Showing Customer Ages---

    # load the data
    # Provide the path to your Excel file
    path = 'C:\\Users\\user\\Dropbox\\Semester E\\Machine Learning\\HW\\HW1_ML\\customer_segmentation.csv'
    df = pd.read_csv(path)
    df_2 = pd.read_csv(path)

    #**Insight** -  The Insight we can take from Graph 1 is that most Customers are between the Ages of 35 and 75, We have Some elderly over the age of 75 and some who are young (18 to 35) We Also do not have under age Customers.


    ## ---GRAPH 2: Pie Chart Showing Customer Preferences By Food Categories---
   # pie_chart_food_categories(data_frame)

   # response_count_by_education(df)

    # Year_Birth, Income, Recency,Nums and Mnts
  #  min_max_scale(df)
 #   one_hot_encode(df_2)
    scale_and_encode(df_2)

    k_means_n5_wines_meats(scale_and_encode(df_2))




    ####Elbow-Method:
    # Define the number of clusters to test (you can choose a range)
    k_vals = range(1, 10)
    df_copy = final_data.copy()
    df_copy_no_date = df_copy.drop(columns=['Dt_Customer'])
    df_copy_no_date.fillna(0, inplace=True)
    # Initialize an empty list to store the variance explained for each k
    variance_per_k = []

    # Fit KMeans with different values of k and calculate variance explained
    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df_copy_no_date)
        variance_per_k.append(kmeans.inertia_)  # kmeans.inertia_ gives the variance explained by the model

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, variance_per_k, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Variance Explained')
    plt.show()


    ### Silhouette
    # Specify the range of clusters (k) to try
    k_values = range(2, 11)

    # List to store silhouette scores for each k
    silhouette_scores = []

    # Iterate over different values of k
    for k in k_values:
        # Initialize KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        # Fit the model and obtain cluster labels
        cluster_labels = kmeans.fit_predict(df_copy_no_date)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(df_copy_no_date, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores for each k
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Values of k')
    plt.text(3, -2, 'Best K using the Sillhouette Method: K=2', fontsize=12, ha='center')
    plt.show()



#PCA
# Adjust n_components as needed

df_no_date_copy2 = df_copy_no_date.copy()
#df_no_date_copy = df_no_date.drop(columns=['ID'])
#df_no_id_no_date_copy = df_copy_no_date.copy()
n_components = 2
# #scale ID too so it wont mess up our pca
# column_to_scale = df_no_date_copy2['ID'].values.reshape(-1, 1)
# Create a DataFrame with the principal components
pca = PCA(n_components)
pca_df = pca.fit_transform(df_no_date_copy2)


# Plotting the PCA
# Step 2: Visualize the PCA
plt.figure(figsize=(8, 6))
plt.scatter(pca_df[:, 0], pca_df[:, 1], cmap='viridis', edgecolor='k', s=60)
plt.title('PCA of Customer Segmentation Pre-Processed Dataset: \n')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 3: Find the variance explained in this PCA
explained_variance_ratio = pca.explained_variance_ratio_
total_variance_explained = np.sum(explained_variance_ratio)

# Display explained variance
print(f"Variance explained by PC1: {explained_variance_ratio[0]*100:.2f}%")
print(f"Variance explained by PC2: {explained_variance_ratio[1]*100:.2f}%")
print(f"Total variance explained by both components: {total_variance_explained*100:.2f}%")



#**A**: The Variance Explained by this 2 component PCA means that PC1 is responsible for 16.15% of the TOTAL variance observed in the data samples, while PC2 is responsible for the remaining 14.73%.
However, these 2 components explain only 30.88% of the total variance in the data.
