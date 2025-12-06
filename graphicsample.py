import numpy as np 
import pandas as pd
import seaborn as seab
import numpy as np
import matplotlib.pyplot as plt

# you need this dataset https://www.kaggle.com/code/dhwanimodi239/demo-dementia-classification/data?select=oasis_longitudinal.csv
df = pd.read_csv('dementia/oasis_longitudinal.csv')

#print first five rows of the dataset
df.head(5)

df.describe()

#determine number of subjects with dementia
seab.set_style("whitegrid")
ex_df = df.loc[df['Visit'] == 1]
palette=seab.color_palette("terrain")
seab.countplot(x='Group', data=ex_df,palette=palette)
print(palette[2])

#There are three groups so convert 
ex_df['Group'] = ex_df['Group'].replace(['Converted'], ['With Dimentia'])
df['Group'] = df['Group'].replace(['Converted'], ['With Dimentia'])
seab.countplot(x='Group', data=ex_df,palette=palette)

# bar drawing function
def bar_chart(feature):
    Dementia = ex_df[ex_df['Group']=='With Dementia'][feature].value_counts()
    NoDementia = ex_df[ex_df['Group']=='Without Dementia'][feature].value_counts()
    df_bar = pd.DataFrame([Dementia,NoDementia])
    df_bar.index = ['Dementia','NoDementia']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))
    print(df_bar)
          
#plot out data by age
plt.figure(figsize=(10,5))
seab.violinplot(x='CDR', y='Age', data=df)
plt.title('Violin plot of Age by CDR',fontsize=14)
plt.xlabel('Clinical Dementia Rating (CDR)',fontsize=13)
plt.ylabel('Age',fontsize=13)
plt.show()
