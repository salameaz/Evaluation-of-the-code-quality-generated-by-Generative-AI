from scipy import stats
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
import matplotlib.pyplot as plt
from pandas.plotting import table

# pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.set_option('display.width', 400)

df_original = pd.read_excel('code_complexity_data.xlsx', sheet_name='sheet1')
df = df_original.copy()

print(df['ChatGPT'].head(10), '\n')
print(df.shape, '\n')

df[['ChatGPT_time', 'ChatGPT_space']] = df['ChatGPT'].str.split(' S', expand=True)
df['ChatGPT_space'] = 'S' + df['ChatGPT_space']

df[['Copilot_time', 'Copilot_space']] = df['Copilot'].str.split(' S', expand=True)
df['Copilot_space'] = 'S' + df['Copilot_space']

print(df['ChatGPT_time'].unique(), '\n')
print(df['ChatGPT_space'].unique(), '\n')
print(df['Copilot_time'].unique(), '\n')
print(df['Copilot_space'].unique(), '\n')

ordinal_mapping_chatGPT_T = {
    'col': 'ChatGPT_time',
    'mapping': {
        'T:O(n^3)': 1,
        'T:O(n^2*log n)': 2,
        'T:O(n^2)': 3,
        'T:O(n*log n)': 4,
        'T:O(n)': 5,
        'T:O(log n)': 6,
        'T:O(1)': 7
    }
}

ordinal_mapping_chatGPT_S = {
    'col': 'ChatGPT_space',
    'mapping': {
        'S:O(n^2)': 1,
        'S:O(n)': 2,
        'S:O(log n)': 3,
        'S:O(1)': 4
    }
}

ordinal_mapping_copilot_T = {
    'col': 'Copilot_time',
    'mapping': {
        'T:O(n^3)': 1,
        'T:O(n^2*log n)': 2,
        'T:O(n^2)': 3,
        'T:O(n*log n)': 4,
        'T:O(n)': 5,
        'T:O(log n)': 6,
        'T:O(1)': 7
    }
}

ordinal_mapping_copilot_S = {
    'col': 'Copilot_space',
    'mapping': {
        'S:O(n^2)': 1,
        'S:O(n)': 2,
        'S:O(log n)': 3,
        'S:O(1)': 4
    }
}

oe_chatGPT_T = OrdinalEncoder(mapping=[ordinal_mapping_chatGPT_T])
df['ChatGPT_time_encoded'] = oe_chatGPT_T.fit_transform(df[['ChatGPT_time']])

oe_chatGPT_S = OrdinalEncoder(mapping=[ordinal_mapping_chatGPT_S])
df['ChatGPT_space_encoded'] = oe_chatGPT_S.fit_transform(df[['ChatGPT_space']])

oe_copilot_T = OrdinalEncoder(mapping=[ordinal_mapping_copilot_T])
df['Copilot_time_encoded'] = oe_copilot_T.fit_transform(df[['Copilot_time']])

oe_copilot_S = OrdinalEncoder(mapping=[ordinal_mapping_copilot_S])
df['Copilot_space_encoded'] = oe_copilot_S.fit_transform(df[['Copilot_space']])

print(df[['ChatGPT_time', 'ChatGPT_time_encoded']].head(10), '\n')
print(df[['ChatGPT_space', 'ChatGPT_space_encoded']].head(10), '\n')
print(df[['Copilot_time', 'Copilot_time_encoded']].head(10), '\n')
print(df[['Copilot_space', 'Copilot_space_encoded']].head(10), '\n')

u_statistic, p_value = stats.mannwhitneyu(df['ChatGPT_time_encoded'], df['Copilot_time_encoded'],
                                          alternative='two-sided')

z_score = (u_statistic - (len(df['ChatGPT_time_encoded']) * len(df['Copilot_time_encoded']) / 2)) / \
          ((len(df['ChatGPT_time_encoded']) * len(df['Copilot_time_encoded']) * (
                  len(df['ChatGPT_time_encoded']) + len(df['Copilot_time_encoded']) + 1) / 12) ** 0.5)
effect_size = z_score / (len(df['ChatGPT_time_encoded']) ** 0.5)

# Prepare the results
results = {
    'Mann-Whitney U Test': ['U', 'z-score', 'p', 'Effect size r'],
    '': [u_statistic, z_score, p_value, effect_size]
}

# Create a DataFrame for the results
results_df = pd.DataFrame(results)
print(results_df, '\n')

u_statistic1, p_value1 = stats.mannwhitneyu(df['ChatGPT_space_encoded'], df['Copilot_space_encoded'],
                                            alternative='two-sided')

z_score1 = (u_statistic1 - (len(df['ChatGPT_space_encoded']) * len(df['Copilot_space_encoded']) / 2)) / \
           ((len(df['ChatGPT_space_encoded']) * len(df['Copilot_space_encoded']) * (
                   len(df['ChatGPT_space_encoded']) + len(df['Copilot_space_encoded']) + 1) / 12) ** 0.5)

effect_size1 = z_score1 / (len(df['ChatGPT_space_encoded']) ** 0.5)

# Prepare the results
results1 = {
    'Mann-Whitney U Test': ['U', 'z-score', 'p', 'Effect size r'],
    '': [u_statistic1, z_score1, p_value1, effect_size1]
}

# Create a DataFrame for the results
results_df1 = pd.DataFrame(results1)
print(results_df1, '\n')

categories_chatgpt_time = ['T:O(n^3)', 'T:O(n^2*log n)', 'T:O(n^2)', 'T:O(n*log n)', 'T:O(n)', 'T:O(log n)', 'T:O(1)']
categories_chatgpt_time = categories_chatgpt_time[::-1]
categories_chatgpt_space = ['S:O(n^2)', 'S:O(n)', 'S:O(log n)', 'S:O(1)']
categories_chatgpt_space = categories_chatgpt_space[::-1]
categories_copilot_time = ['T:O(n^3)', 'T:O(n^2)', 'T:O(n*log n)', 'T:O(n)', 'T:O(log n)', 'T:O(1)']
categories_copilot_time = categories_copilot_time[::-1]
categories_copilot_space = ['S:O(n^2)', 'S:O(n)', 'S:O(1)']
categories_copilot_space = categories_copilot_space[::-1]

df['ChatGPT_time'] = pd.Categorical(df['ChatGPT_time'], categories=categories_chatgpt_time, ordered=True)
df['ChatGPT_space'] = pd.Categorical(df['ChatGPT_space'], categories=categories_chatgpt_space, ordered=True)
df['Copilot_time'] = pd.Categorical(df['Copilot_time'], categories=categories_copilot_time, ordered=True)
df['Copilot_space'] = pd.Categorical(df['Copilot_space'], categories=categories_copilot_space, ordered=True)

# ChatGPT descriptive statistics
print("ChatGPT Time Descriptive Statistics\n")
# Frequency Distribution
frequency_chatgpt_time = df['ChatGPT_time'].value_counts().sort_index()
print("Frequency Distribution:\n", frequency_chatgpt_time, '\n')

# Mode
mode_chatgpt_time = df['ChatGPT_time'].mode()[0]
print("Mode:", mode_chatgpt_time, '\n')

# Median
median_chatgpt_time = df['ChatGPT_time'].cat.codes.median()
print("Median:", median_chatgpt_time, '\n')

# Percentiles
# Percentiles
percentiles_chatgpt_time = np.percentile(df['ChatGPT_time'].cat.codes, [25, 50, 75])
percentiles_chatgpt_time_values = [categories_chatgpt_time[int(percentile)] for percentile in percentiles_chatgpt_time]
print("25th Percentile:", percentiles_chatgpt_time[0], '(', percentiles_chatgpt_time_values[0], ')')
print("50th Percentile:", percentiles_chatgpt_time[1], '(', percentiles_chatgpt_time_values[1], ')')
print("75th Percentile:", percentiles_chatgpt_time[2], '(', percentiles_chatgpt_time_values[2], ')', '\n')

# Range
range_chatgpt_time = (df['ChatGPT_time'].min(), df['ChatGPT_time'].max())
print("Range:", range_chatgpt_time, '\n')

# Inter quartile Range
iqr_chatgpt_time = (percentiles_chatgpt_time_values[0], percentiles_chatgpt_time_values[2])
print("Inter quartile Range:", iqr_chatgpt_time, '\n')

# Copilot time descriptive statistics
print("Copilot Time Descriptive Statistics\n")
# Frequency Distribution
frequency_copilot_time = df['Copilot_time'].value_counts().sort_index()
print("Frequency Distribution:\n", frequency_copilot_time, '\n')

# Mode
mode_copilot_time = df['Copilot_time'].mode()[0]
print("Mode:", mode_copilot_time, '\n')

# Median
median_copilot_time = df['Copilot_time'].cat.codes.median()
print("Median:", median_copilot_time, '\n')

# Percentiles
percentiles_copilot_time = np.percentile(df['Copilot_time'].cat.codes, [25, 50, 75])
percentiles_copilot_time_values = [categories_copilot_time[int(percentile)] for percentile in percentiles_copilot_time]
print("25th Percentile:", percentiles_copilot_time[0], '(', percentiles_copilot_time_values[0], ')')
print("50th Percentile:", percentiles_copilot_time[1], '(', percentiles_copilot_time_values[1], ')')
print("75th Percentile:", percentiles_copilot_time[2], '(', percentiles_copilot_time_values[2], ')', '\n')

# Range
range_copilot_time = (df['Copilot_time'].min(), df['Copilot_time'].max())
print("Range:", range_copilot_time, '\n')

# Inter quartile Range
iqr_copilot_time = (percentiles_copilot_time_values[0], percentiles_copilot_time_values[2])
print("Inter quartile Range:", iqr_copilot_time, '\n')

# ChatGPT space descriptive statistics
print("ChatGPT Space Descriptive Statistics\n")
# Frequency Distribution
frequency_chatgpt_space = df['ChatGPT_space'].value_counts().sort_index()
print("Frequency Distribution:\n", frequency_chatgpt_space, '\n')

# Mode
mode_chatgpt_space = df['ChatGPT_space'].mode()[0]
print("Mode:", mode_chatgpt_space, '\n')

# Median
median_chatgpt_space = df['ChatGPT_space'].cat.codes.median()
print("Median:", median_chatgpt_space, '\n')

# Percentiles
percentiles_chatgpt_space = np.percentile(df['ChatGPT_space'].cat.codes, [25, 50, 75])
percentiles_chatgpt_space_values = [categories_chatgpt_space[int(percentile)] for percentile in
                                    percentiles_chatgpt_space]
print("25th Percentile:", percentiles_chatgpt_space[0], '(', percentiles_chatgpt_space_values[0], ')')
print("50th Percentile:", percentiles_chatgpt_space[1], '(', percentiles_chatgpt_space_values[1], ')')
print("75th Percentile:", percentiles_chatgpt_space[2], '(', percentiles_chatgpt_space_values[2], ')', '\n')

# Range
range_chatgpt_space = (df['ChatGPT_space'].min(), df['ChatGPT_space'].max())
print("Range:", range_chatgpt_space, '\n')

# Inter quartile Range
iqr_chatgpt_space = (percentiles_chatgpt_space_values[0], percentiles_chatgpt_space_values[2])
print("Inter quartile Range:", iqr_chatgpt_space, '\n')

# Copilot space descriptive statistics
print("Copilot Space Descriptive Statistics\n")

# Frequency Distribution
frequency_copilot_space = df['Copilot_space'].value_counts().sort_index()
print("Frequency Distribution:\n", frequency_copilot_space, '\n')

# Mode
mode_copilot_space = df['Copilot_space'].mode()[0]
print("Mode:", mode_copilot_space, '\n')

# Median
median_copilot_space = df['Copilot_space'].cat.codes.median()
print("Median:", median_copilot_space, '\n')

# Percentiles
percentiles_copilot_space = np.percentile(df['Copilot_space'].cat.codes, [25, 50, 75])
percentiles_copilot_space_values = [categories_copilot_space[int(percentile)] for percentile in
                                    percentiles_copilot_space]
print("25th Percentile:", percentiles_copilot_space[0], '(', percentiles_copilot_space_values[0], ')')
print("50th Percentile:", percentiles_copilot_space[1], '(', percentiles_copilot_space_values[1], ')')
print("75th Percentile:", percentiles_copilot_space[2], '(', percentiles_copilot_space_values[2], ')', '\n')

# Range
range_copilot_space = (df['Copilot_space'].min(), df['Copilot_space'].max())
print("Range:", range_copilot_space, '\n')

# Inter quartile Range
iqr_copilot_space = (percentiles_copilot_space_values[0], percentiles_copilot_space_values[2])
print("Inter quartile Range:", iqr_copilot_space, '\n')

# # Plotting the descriptive statistics
# fig, ax = plt.subplots(2, 2, figsize=(14, 10))
#
# # ChatGPT Time
# df['ChatGPT_time'].value_counts().sort_index().plot(kind='bar', ax=ax[0, 0])
# ax[0, 0].set_title('ChatGPT Time')
# ax[0, 0].set_xlabel('Time Complexity')
# ax[0, 0].set_ylabel('Frequency')
# plt.show()
