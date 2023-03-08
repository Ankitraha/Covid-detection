"""
First I deleted all columns except the covid results and columns AH to AO as the model would be a non clinical way to predict data and column AH to AO contains non clinical data.  I checked which columns had 50% of the data missing and removed them.
I converted all the true and false value to 1 and 0 respectively.I then removed rows having value other than Negative and Positive and converted Negative and Positive to 0 and 1 respectively. 
I then coumted the number of negative and positive results and found the number of positive results to be too low. As a result I upscaled the number of positive results to be approxiamtely
equal to 1000 less than the number of negative results.Then I shuffled the rows in the dataset so that the train and validate data would have been selected
randomly. Then I split the data into training and validation data and, trained the model with knn classifier with no. of closest neighbours set to 3.The accuracy cameout Then I performed the same preprocessing to the test data. Then I inserted the 
columns that were present in the pre processed training dataset but not in the pre processed test dataset.I then predicted the reults using the trained model and created 
a seperate csv file containg the id and results. 
"""
 






import tensorflow as tf
from tensorflow import keras
from scipy.stats.stats import ModeResult
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def remove_missing_columns(df, threshold):
    """
    Remove columns from a pandas DataFrame that have at least `threshold`
    proportion of their values missing.
    """
    missing = df.isnull().mean()
    missing = missing[missing >= threshold]
    for col in missing.index:
        del df[col]
    return df

# Example usage
df = pd.read_csv('/kaggle/input/dataset/DataSet_Test.csv')
df = df.drop(columns=["swab_type", "test_name", "age", "high_risk_exposure_occupation", "high_risk_interactions" , "diabetes" , "chd" , "htn" , "cancer" , "asthma" , "copd" , "autoimmune_dis" , "smoker"])
df = remove_missing_columns(df, 0.5)
def check_missing_values(df, threshold):
    """
    Check which columns in a pandas DataFrame have at least `threshold`
    percent of their values missing.
    """
    missing = df.isnull().mean()
    missing = missing[missing >= threshold]
    return missing

df.drop('batch_date', axis=1, inplace=True)
missing = check_missing_values(df, 0.5)
print(missing)

df = df.applymap(lambda x: 1 if x==True else (0 if x==False else x))

def remove_rows(df):
    """
    Removes rows having data value in the 'results' column other than 'positive' and 'negative'
    """
    df = df[df.covid19_test_results.isin(['Positive', 'Negative'])]
    return df
df = remove_rows(df)
# Count the number of rows with 'negative' in the 'covid_results' column
negative_count = df[df['covid19_test_results'] == 'Negative'].shape[0]
# Calculate the desired number of rows for the 'positive' category
desired_positive_count = negative_count-1000

# Randomly select the desired number of rows from the 'positive' rows
positive_rows = df[df['covid19_test_results'] == 'Positive']
positive_rows = positive_rows.sample(n=desired_positive_count, replace=True, random_state=1)

# Add the selected rows to the original DataFrame
df = pd.concat([df, positive_rows])
positive_count = df[df['covid19_test_results'] == 'Positive'].shape[0]
print("Number of rows with 'negative' in the 'covid_results' column:", negative_count)
print("Number of rows with 'positive' in the 'covid_results' column:", positive_count)
categories = df["covid19_test_results"].value_counts().keys()
print(categories)
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns.difference(['covid19_test_results']))
df.dropna(axis=0, inplace=True)
missing_value_count = df.shape[0] - df.count()
print(missing_value_count)
df['covid19_test_results'] = df['covid19_test_results'].replace({'Positive':1, 'Negative':0})
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("modified_data1.csv", index=False)

# Split the training data into features and labels
X_train = df.drop(columns=['covid19_test_results'])
y_train = df['covid19_test_results']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the KNeighborsClassifier class
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier using the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_val)

# Print the accuracy of the classifier
print(knn.score(X_val, y_val))
# Print the classification report
print(classification_report(y_val, y_pred))

def remove_missing_columns(df1, threshold):
    """
    Remove columns from a pandas DataFrame that have at least `threshold`
    proportion of their values missing.
    """
    missing = df1.isnull().mean()
    missing = missing[missing >= threshold]
    for col in missing.index:
        del df1[col]
    return df1

# Example usage
df1 = pd.read_csv('/kaggle/input/predict/DataSet_Participant_6.csv')
df1 = remove_missing_columns(df1, 0.5)
df1.drop('batch_date', axis=1, inplace=True)
df1= df1.applymap(lambda x: 1 if x==True else (0 if x==False else x))
df1 = pd.get_dummies(df1, columns=df1.select_dtypes(include=['object']).columns.difference(['covid19_test_results']))
# Find the columns that are in the training data but not in the test data
missing_columns = set(df.columns) - set(df1.columns)

# Add the missing columns to the test data and fill them with default value
for column in missing_columns:
    df1[column] = 0

# Find the columns that are in the test data but not in the training data
extra_columns = set(df1.columns) - set(df.columns)

# Remove extra columns from the test data
df1.drop(columns=extra_columns, inplace=True)
df1 = df1[df.columns]
def impute_age(df1):
    """
    Impute missing values in the 'age' column of a pandas DataFrame
    """
    mean_age = df1["age"].mean()
    df1["age"] = df1["age"].fillna(mean_age)
    return df1
df1 = impute_age(df1)
def min_max_normalize(df1):
    """
    Normalizes the values in the 'age' column of a DataFrame using MinMaxScaler
    """
    scaler = MinMaxScaler()
    df1[['age']] = scaler.fit_transform(df1[['age']])
    return df1
df1 = min_max_normalize(df1)
df1.drop(columns=['covid19_test_results'], inplace=True)
missing_value_count = df1.shape[0] - df1.count()
print(missing_value_count)
df1.to_csv("modified_test_data1.csv", index=False)
test_predictions = knn.predict(df1)
df1['id'] = range(1, len(df1)+1)
df1['covid19_test_results'] = test_predictions
df1 = df1[['id', 'covid19_test_results']]
df1['covid19_test_results'] = df1['covid19_test_results'].replace({1: 'Positive', 0: 'Negative'})
df1.to_csv('submission_to_submit.csv', index=False)
