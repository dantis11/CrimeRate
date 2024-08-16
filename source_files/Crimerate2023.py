# Step 1: Loading the dataset
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import datetime

# Streamlit setup
st.title("Crime Classification and Prediction")

# Load the dataset (assuming it is in a CSV file)
df = pd.read_csv(r"data2023.csv")

# Step 2: Initial Exploration
# Get basic information about the dataset
print(df.info())

# Get summary statistics for numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check for outliers
df[['Weapon Used Cd', 'Premis Cd', 'Crm Cd 1']].boxplot(figsize=(12, 5))
# plt.show()
print(df['Premis Cd'].median())

# Step 3: Handling Missing Values
# Filling missing values

df['Mocodes'].fillna('Unknown', inplace=True)

vict_sex_mode = df['Vict Sex'].mode()[0]
vict_descent_mode = df['Vict Descent'].mode()[0]
df['Vict Sex'].fillna(vict_sex_mode, inplace=True)
df['Vict Descent'].fillna(vict_descent_mode, inplace=True)

premis_cd_median = df['Premis Cd'].median()
df['Premis Cd'].fillna(premis_cd_median, inplace=True)

premis_desc_mode = df['Premis Desc'].mode()[0]
df['Premis Desc'].fillna(premis_desc_mode, inplace=True)

df['Weapon Used Cd'].fillna(0, inplace=True)  # Replacing with 0 since it has enormous missing values: 455719
df['Weapon Desc'].fillna('Unknown', inplace=True)

crm_cd_1_median = df['Crm Cd 1'].median()
df['Crm Cd 1'].fillna(crm_cd_1_median, inplace=True)

# Drop 'Crm Cd 2', 'Crm Cd 3', and 'Crm Cd 4' due to a high proportion of missing values
df.drop(columns=['Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4'], inplace=True)

df['Cross Street'].fillna('Unknown', inplace=True)

print(df.isnull().sum())

df.isnull().sum().sort_values(ascending=True) # display count of null values per column

# drop all the null values from the dataset
df.dropna(inplace=True)

#df["Part 1-2"] = pd.to_numeric(df["Part 1-2"])
df["Part 1-2"] = df["Part 1-2"].astype('int64') #convert to whole number

df.isnull().sum().sort_values(ascending=True) # display count of null values per column

# Convert data types
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)  # Ensure time is in HHMM format

#vict age can't be 0
vict_age = df['Vict Age'].mean()
df['Vict Age'].replace(0,vict_age, inplace=True)
df['Vict Age'].replace(-1,vict_age, inplace=True)
df['Vict Age'].replace(-2,vict_age, inplace=True)
df['Vict Age'].replace(-3,vict_age, inplace=True)
df['Vict Age'].replace(-4,vict_age, inplace=True)

#Replacing all the -ve and 0 value with mean

#count plot
plt.figure(figsize=(12, 6))
sns.countplot(x=df['AREA'],data=df)  #plotting
plt.xticks(rotation=45)
plt.title('Crime Distribution by Area') #adding title
# plt.show()  #display


plt.figure(figsize=(10, 6))
sns.histplot(df['Vict Age'], bins=20)  #plotting
plt.title('Victim Age Distribution')#adding title
plt.xlabel('Age')#adding xlabel
plt.ylabel('Count')#adding ylabel
# plt.show()  #display


plt.figure(figsize=(10, 6))
crime_counts = df['AREA NAME'].value_counts()
plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)  #plotting
plt.title('Area with crime count')  #adding title
plt.tight_layout()
# plt.show()   #display


# Count occurrences of each Part 1-2 category for each age
age_counts = df.groupby(['Vict Age', 'Part 1-2']).size().unstack(fill_value=0)

# Plotting
plt.figure(figsize=(10, 6))

# Line plot
age_counts.plot(kind='line', marker='o')

# Adding labels and title
plt.title('Count of Part 1-2 Crime Classification by Victim Age')
plt.xlabel('Victim Age')
plt.ylabel('Count')

# Display legend
plt.legend(title='Part 1-2')

plt.grid(True)  # Add grid lines

plt.tight_layout()
# plt.show()  #display


status_counts = df.groupby(['Vict Sex', 'Part 1-2']).size().unstack(fill_value=0)

# Plotting
plt.figure(figsize=(10, 6))

# Bar plot for each Vict Sex
status_counts.plot(kind='bar', stacked=False, color=['blue', 'orange'], alpha=0.7)

# Adding labels and title
plt.title('Count of part 1-2 by Victim Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

# Display legend with appropriate labels
plt.legend(title='Part', labels=['Part 1', 'Part 2'])

plt.xticks(rotation=45)  # Rotate xticks for better readability
plt.grid(axis='y')  # Add horizontal grid lines

plt.tight_layout()
# plt.show()  #display

# Step 5: Feature Engineering
# Create new features if necessary. For example, we can extract year, month, and day from the "DATE OCC" and "Date Rptd" column.
# Convert "DATE OCC" and "Date Rptd" to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], errors='coerce')
# When this parameter is specified, it forces invalid parsing to be set as NaN (Not a Number).
# This is especially useful for handling errors that arise from converting strings to numeric values, dates, or other types.

# Extract year, month, and day
df['Year Rptd'] = df['Date Rptd'].dt.year
df['Month Rptd'] = df['Date Rptd'].dt.month
df['Day Rptd'] = df['Date Rptd'].dt.day
df['Day of Week Rptd'] = df['Date Rptd'].dt.dayofweek

df['Year Occ'] = df['DATE OCC'].dt.year
df['Month Occ'] = df['DATE OCC'].dt.month
df['Day Occ'] = df['DATE OCC'].dt.day
df['Day of Week Occ'] = df['DATE OCC'].dt.dayofweek

# Extract hour and minute from TIME OCC
df['Hour Occ'] = df['TIME OCC'].str[:2].astype(int)
df['Minute Occ'] = df['TIME OCC'].str[2:].astype(float)
# Drop columns 'DATE OCC', 'Date Rptd','TIME OCC'
df = df.drop(['DATE OCC', 'Date Rptd','TIME OCC'], axis=1)

# Reset index
df = df.reset_index(drop=True)


# Step 4: Encoding Categorical Variables
# List of categorical columns to be encoded (assuming these are required for your model)
categorical_columns = [
     'AREA','AREA NAME', 'Crm Cd','Crm Cd Desc',
    'Vict Sex', 'Vict Descent', 'Premis Cd','Premis Desc',
    'Weapon Used Cd','Weapon Desc', 'Status', 'Status Desc'
]
# columns left out from encoding assuming these are not required features: Date Rptd ,DATE OCC ,Mocodes,LOCATION ,Cross Street

# # Initialize the label encoder
# label_encoder = LabelEncoder()

# # # Apply label encoding to each necessary categorical column
# for column in categorical_columns:
#     df[column] = label_encoder.fit_transform(df[column])

# Initialize a dictionary to store a LabelEncoder for each column
label_encoder = {}

# Apply label encoding to each necessary categorical column
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoder[column] = le  # Store the encoderr

# Display the first few rows of the encoded dataframe
print(df.head())

print(df['AREA'].unique())

# Finding the number of unique values in each column
unique_counts = df.nunique()
print("Number of unique values in each column:")
print(unique_counts)

df = df.select_dtypes(include=[np.number])
# Generate correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
# plt.show()

#Drop the highly correlated columns
# df.drop('Status Desc', axis=1, inplace=True)
df.drop('Status', axis=1, inplace=True)
df.drop('Rpt Dist No', axis=1, inplace=True)
df.drop('Year Occ', axis=1, inplace=True)
df.drop('AREA', axis=1, inplace=True)
# df.drop('Premis Desc', axis=1, inplace=True)
# df.drop('Weapon Desc', axis=1, inplace=True)
df.drop('Premis Cd', axis=1, inplace=True)
df.drop('Weapon Used Cd', axis=1, inplace=True)
df.drop('Crm Cd Desc', axis=1, inplace=True)
df.drop('Crm Cd', axis=1, inplace=True)
df.drop('Crm Cd 1', axis=1, inplace=True)
df.drop('LAT', axis=1, inplace=True)
df.drop('LON', axis=1, inplace=True)
df.drop('Day Occ', axis=1, inplace=True)
df.drop('DR_NO', axis=1, inplace=True)
#df.drop('Month Occ', axis=1, inplace=True)

# df=df[['AREA', 'Vict Age','Vict Sex','Vict Descent',  'Premis Desc', 'Weapon Desc','Status', 'Year Rptd','Month Rptd','Day Rptd','Day of Week Rptd','Month Occ','Day of Week Occ','Hour Occ', 'Minute Occ']]
# Step 7: Splitting Training and Test Data
# Define target variable
target = 'Part 1-2'

# Column Analysis:
# Crm Cd (Criminal Code): This could be a potential target variable if you're interested in predicting or classifying types of crimes based on their codes.
# Part 1-2: This could be relevant if you're interested in classifying crimes as Part 1 (more serious) or Part 2 (less serious).
# Status: Depending on its meaning (e.g., case status), this could be relevant for predicting outcomes related to crime resolution.
# Vict Age, Vict Sex, Vict Descent: These could be used for predicting victim-related characteristics or demographics.

# Split the dataset into features (X) and target (y)
X = df.drop(columns=[target])
y = df[target]

# Split into training and testing sets 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verify the shapes
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


# Initialize and train the model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# # Make predictions and evaluate
# y_pred = log_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# # Calculate confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# # plt.show()

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# # Make predictions and evaluate
# y_pred = rf_model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")


#EXTRA
# Evaluate the models
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log))

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
###########################################################


# User Input Section in Streamlit
st.header("Predict Crime Type")

# Interactive inputs from the user
area_input = st.selectbox("Select Area", label_encoder['AREA NAME'].classes_)
vict_sex_input = st.selectbox("Select Victim Sex", label_encoder['Vict Sex'].classes_)
vict_descent_input = st.selectbox("Select Victim Descent", label_encoder['Vict Descent'].classes_)
premis_cd_input = st.selectbox("Enter Premise Code", label_encoder['Premis Desc'].classes_)
weapon_used_input = st.selectbox("Enter Weapon Used Code", label_encoder['Weapon Desc'].classes_)
status_input = st.selectbox("Enter Status", label_encoder['Status Desc'].classes_)
date_str = st.text_input("Enter the date and time of occurrence (YYYY-MM-DD HH:MM):", "2024-08-15 14:30")

# Parse the date and time
try:
    occ_datetime = pd.to_datetime(date_str, format="%Y-%m-%d %H:%M", errors='raise')
    year_occ = occ_datetime.year
    month_occ = occ_datetime.month
    day_occ = occ_datetime.day
    day_of_week_occ = occ_datetime.weekday()
    hour_occ = occ_datetime.hour
    minute_occ = occ_datetime.minute
except ValueError:
    st.error("Invalid date format. Please use YYYY-MM-DD HH:MM.")
    st.stop()

# Prepare input data for prediction
input_data = pd.DataFrame({
    'AREA NAME': [label_encoder['AREA NAME'].transform([area_input])[0]],
    'Vict Age': [X_train['Vict Age'].mean()],  # Assuming using the mean age for simplicity
    'Vict Sex': [label_encoder['Vict Sex'].transform([vict_sex_input])[0]],
    'Vict Descent': [label_encoder['Vict Descent'].transform([vict_descent_input])[0]],
    'Premis Desc': [label_encoder['Premis Desc'].transform([premis_cd_input])[0]],
    'Weapon Desc': [label_encoder['Weapon Desc'].transform([weapon_used_input])[0]],
    'Status Desc': [label_encoder['Status Desc'].transform([status_input])[0]],
    'Year Rptd': [year_occ],
    'Month Rptd': [month_occ],
    'Day Rptd': [day_occ],
    'Day of Week Rptd': [day_of_week_occ],
    'Month Occ': [month_occ],
    'Day of Week Occ': [day_of_week_occ],
    'Hour Occ': [hour_occ],
    'Minute Occ': [minute_occ]
})

print(input_data)
# Convert to DataFrame
input_df = pd.DataFrame(input_data)
print(input_df.dtypes)
# Scale input data
# Scale the 'LON' column (assuming 'scaler' was fitted on 'LON' earlier)
# input_df['LON'] = scaler.transform(input_df[['LON']])

# Ensure that input_df has the correct columns in the correct order
input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

# Predict with Logistic Regression
if st.button("Predict with Logistic Regression"):
    prediction = log_model.predict(input_data)
    st.write(f"Logistic Regression Prediction: {'Part 1 (Serious)' if prediction[0] == 1 else 'Part 2 (Less Serious)'}")

# Predict with Random Forest
if st.button("Predict with Random Forest"):
    prediction = rf_model.predict(input_data)
    st.write(f"Random Forest Prediction: {'Part 1 (Serious)' if prediction[0] == 1 else 'Part 2 (Less Serious)'}")

# # Displaying the confusion matrix for Logistic Regression
# if st.checkbox("Show Confusion Matrix for Logistic Regression"):
#     y_pred_log = log_model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred_log)
#     st.write("Confusion Matrix:")
#     st.write(cm)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     st.pyplot()

# # Displaying the confusion matrix for Random Forest
# if st.checkbox("Show Confusion Matrix for Random Forest"):
#     y_pred_rf = rf_model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred_rf)
#     st.write("Confusion Matrix:")
#     st.write(cm)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     st.pyplot()


