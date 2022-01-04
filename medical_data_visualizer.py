import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
pd.set_option('display.max_columns', None)

# Add 'overweight' column
df['bmi'] = df['weight'] / ((df['height']/100) ** 2)

df['overweight'] = df['bmi'].apply(lambda x: 1 if x > 25 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)

df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# Draw Categorical Plot
def draw_cat_plot():
    cardio_df = df[['cardio','active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']]
    
    df_cat = pd.melt(cardio_df, id_vars=['cardio'], var_name='variable', value_name='value')

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind='count')
    g.set_axis_labels("variable", "total")
    fig = g.fig  #thanks to user Jagaya for helping resolve some testing errors with this line of code!
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    df_heat = df_heat.drop(columns='bmi')

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, vmin=-0.16, vmax=0.30, mask=mask, cmap='icefire', center=0.0, cbar_kws={'ticks': [-0.08, 0.0, 0.08, 0.16, 0.24], 'shrink': 0.5}, linewidth=0.7, annot=True, fmt='.1f')

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
