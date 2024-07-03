import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 

directory = './output/'
def barplot(xcol, ycol, data):
    sns.barplot(x = xcol, y=ycol, data =data)
    file_path = os.path.join(directory, f'{xcol}_{ycol}.png')
    plt.savefig(file_path)
    plt.show()
        
    
    
    