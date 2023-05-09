import matplotlib.pyplot as plt

from joblib import load

import pandas as pd

import numpy as np


model = load("assets/trained_model.joblib")

datatest=pd.read_csv('HotelsReviews.csv', sep=',',index_col=0, encoding = 'utf-8')

results = model.predict(datatest["review_text"])

count = datatest.count()

count0 = results.value_counts()[0]/count

count1 = results.value_counts()[1]/count

count3 = results.value_counts()[3]/count

plt.pie([count0,count1,count3])
plt.show()