import pandas as pd
import plotly.express as px

# Load features.csv file generated from dsmri.py
# Create a new column after the first column in the generated features.csv file to provide the labels of your domain. 
# In our sample features.csv file, its "Manufacturer" column with 'GE', 'Siemens' and 'Philips' labels.
df = pd.read_csv('features.csv')

# For 3D tSNE
#fig = px.scatter_3d(df, x = 'TSNEX', y = 'TSNEY', z = 'TSNEZ',
fig = px.scatter(df, x = 'TSNEX', y = 'TSNEY', 
color=df.Manufacturer, title='ADNI2 MRI data distribution with 2D tSNE plot', symbol=df.Manufacturer)
fig.update_traces(marker_size=5)
fig.show()
