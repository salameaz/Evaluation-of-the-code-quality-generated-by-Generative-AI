import pandas as pd
from pandas.plotting import table
from scipy.stats import mannwhitneyu
from category_encoders import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.set_option('display.width', 400)

# Load data from Excel

data = pd.read_excel('code_correctness_data.xlsx', sheet_name='sheet1')
data = data.dropna(axis=1)

print(data["ChatGPT"].unique())
print(data.head())

# Define the ordinal mapping
ordinal_mapping = [{
    'col': 'ChatGPT',
    'mapping': {
        'נכון ': 5,  # Correct
        'נכון חלקית': 4,  # Partially Correct
        'לא נכון': 3,  # Incorrect
        'שגיאת קומפילציה': 2,
        'שגיאת זמן ריצה': 1
    }}]
ordinal_mapping1 = [{
    'col': 'Copilot',
    'mapping': {
        'נכון ': 5,  # Correct
        'נכון חלקית': 4,  # Partially Correct
        'לא נכון': 3,  # Incorrect
        'שגיאת קומפילציה': 2,
        'שגיאת זמן ריצה': 1
    }}]


oe = OrdinalEncoder(mapping=ordinal_mapping)
data['ChatGPT_encoded'] = oe.fit_transform(data["ChatGPT"])
oe1 = OrdinalEncoder(mapping=ordinal_mapping1)
data['Copilot_encoded'] = oe1.fit_transform(data["Copilot"])

print(data['ChatGPT_encoded'])
print(data['Copilot_encoded'])

np.random.seed(10)

plt.figure(figsize=(8, 6))

# Create boxplot without normalization
boxplot = data.boxplot(column=['Copilot_encoded', 'ChatGPT_encoded'], patch_artist=True)

# Customizing colors


# Adding details to the plot
plt.ylabel('Encoded Values')
plt.title('Box Plot of Copilot vs ChatGPT')
plt.xticks([1, 2], ['Copilot', 'ChatGPT'])
plt.grid(axis='y')

plt.show()

# Perform the Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(data['ChatGPT_encoded'], data['Copilot_encoded'])
z_score = abs((u_statistic - (len(data['ChatGPT_encoded']) * len(data['Copilot']) / 2)) / \
              ((len(data['ChatGPT_encoded']) * len(data['Copilot_encoded']) * (len(data['ChatGPT_encoded']) + len(data['Copilot_encoded']) + 1) / 12) ** 0.5))
effect_size = abs(z_score / (len(data['ChatGPT_encoded']) ** 0.5))

# Prepare the results
results = {
    'Mann-Whitney U Test': ['U', 'z-score', 'p', 'Effect size r'],
    '': [u_statistic, z_score, p_value, effect_size]
}

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Plot the table
fig, ax = plt.subplots(figsize=(4, 2))  # Set the size of the figure
ax.axis('tight')
ax.axis('off')
tbl = table(ax, results_df, loc='center', cellLoc='center', colWidths=[0.2] * len(results_df.columns))

# Adjust the font size of the table
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
plt.show()

#data.to_excel("test2.xlsx",sheet_name="Sheet1",index=False)

# Sample data
import pandas as pd
import numpy as np

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'ChatGPT': ['נכון', 'נכון חלקית', 'לא נכון', 'נכון', 'שגיאת זמן ריצה', 'לא נכון', 'נכון חלקית', 'נכון חלקית']
})

# Define the ordered categories
categories = ['נכון', 'נכון חלקית', 'לא נכון', 'שגיאת קומפילציה', 'שגיאת זמן ריצה']

# Convert 'ChatGPT' column to categorical
data['ChatGPT'] = pd.Categorical(data['ChatGPT'], categories=categories, ordered=True)

# Frequency Distribution
frequency_distribution = data['ChatGPT'].value_counts().sort_index()
print("Frequency Distribution:\n", frequency_distribution)

# Mode
mode = data['ChatGPT'].mode()[0]
print("\nMode:chatgpt", mode)

# Median
median = data['ChatGPT'].cat.codes.median()
print("Median:", median)

# Percentiles
percentiles = np.percentile(data['ChatGPT'].cat.codes, [25, 50, 75])
percentile_values = [categories[int(p)] for p in percentiles]
print("25th Percentile:", percentile_values[0])
print("50th Percentile (Median):", percentile_values[1])
print("75th Percentile:", percentile_values[2])

# Range
range_values = (data['ChatGPT'].min(), data['ChatGPT'].max())
print("Range:", range_values)

# Interquartile Range (IQR)
iqr = (percentile_values[0], percentile_values[2])
print("IQR:", iqr)

##########
data = pd.DataFrame({
    'Copilot': ['נכון', 'נכון חלקית', 'לא נכון', 'נכון', 'שגיאת זמן ריצה', 'לא נכון', 'נכון חלקית', 'נכון חלקית']
})
categories = ['נכון', 'נכון חלקית', 'לא נכון', 'שגיאת קומפילציה', 'שגיאת זמן ריצה']

# Convert 'ChatGPT' column to categorical
data['Copilot'] = pd.Categorical(data['Copilot'], categories=categories, ordered=True)

# Frequency Distribution
frequency_distribution = data['Copilot'].value_counts().sort_index()
print("Frequency Distribution:\n", frequency_distribution)

# Mode
mode = data['Copilot'].mode()[0]
print("\nMode:copilot", mode)

# Median
median = data['Copilot'].cat.codes.median()
print("Median:", median)

# Percentiles
percentiles = np.percentile(data['Copilot'].cat.codes, [25, 50, 75])
percentile_values = [categories[int(p)] for p in percentiles]
print("25th Percentile:", percentile_values[0])
print("50th Percentile (Median):", percentile_values[1])
print("75th Percentile:", percentile_values[2])

# Range
range_values = (data['Copilot'].min(), data['Copilot'].max())
print("Range:", range_values)

# Interquartile Range (IQR)
iqr = (percentile_values[0], percentile_values[2])
print("IQR:", iqr)
