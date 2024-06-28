import pandas as pd

# List of your csv files
files = ['sleepDay_merged.csv', 'dailyActivity_merged.csv', 'dailyCalories_merged.csv', 'dailySteps_merged.csv', 'weightLogInfo_merged.csv']

# Read the first file, convert 'Date' to datetime and set the 'id' and 'Date' columns as index
df = pd.read_csv(files[0])
df['Date'] = pd.to_datetime(df['Date']).dt.date
df = df.set_index(['Id', 'Date'])

# Loop over the rest of the csv files
for i, file in enumerate(files[1:], 1):
    # Read each csv file, convert 'Date' to datetime and set the 'id' and 'Date' columns as index
    temp_df = pd.read_csv(file)
    temp_df['Date'] = pd.to_datetime(temp_df['Date']).dt.date
    temp_df = temp_df.set_index(['Id', 'Date'])

    # Join with the main dataframe (df) based on the index ('id' and 'Date')
    # Add suffixes to distinguish between overlapping columns
    # Use 'outer' join to include all rows
    df = df.join(temp_df, how='outer', lsuffix='_file'+str(i), rsuffix='_file'+str(i+1))

# Remove rows where 'TotalMinutesAsleep' is blank
df = df.dropna(subset=['TotalMinutesAsleep'])

# List of columns to remove
columns_to_remove = ['TotalSleepRecords', 'Loggedactivitiesdistance', 'StepTotal', 'TrackerDistance', 'Calories_file3', 'SedentaryActiveDistance', 'Fat', 'WeightKg', 'BMI', 'IsManualReport', 'WeightPounds', 'LoggedActivitiesDistance', 'LogId']

# Remove specified columns if they exist
for column in columns_to_remove:
    if column in df.columns:
        df = df.drop(columns=[column])

# Rename 'Calories_file2' column to 'Calories'
if 'Calories_file2' in df.columns:
    df = df.rename(columns={'Calories_file2': 'Calories'})

# Save the merged data to a new csv file
df.to_csv('merged.csv')
