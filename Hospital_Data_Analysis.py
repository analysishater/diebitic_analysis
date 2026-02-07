import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\PC\Desktop\sara\JN\train.csv")
df
plt.figure(figsize=(20,10))
plt.hist(df['Age'],bins=10,color="Purple",edgecolor='Black')#drawing a histo(i like purple)
plt.title("age appearance")#we add title to our histo
plt.show()
