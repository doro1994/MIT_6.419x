import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

with open('gamma-ray.csv') as csvfile:
    reader = csv.reader(csvfile)
    X = list()
    Y = list()
    emission_rate = list()
    line_count = 1
    for row in reader:
        if (line_count >= 2):
            X.append(float(row[0]))
            Y.append(float(row[1]))
            emission_rate.append(float(row[1])/float(row[0]))
            
        line_count += 1
print(X)
print(Y)
print(np.array(emission_rate).mean())

print(chi2.interval(0.95, 99))
print(chi2.ppf(0.95, 99))
lambda_0 = np.sum(X) / np.sum(np.array(Y))
X = np.array(X)
emission_rate = np.array(emission_rate)
test_statistic = -2*np.log(np.max(lambda_0**emission_rate*np.exp(-lambda_0))
                           /np.max(X**emission_rate*np.exp(-X)))
print((1-chi2.cdf(test_statistic, 99)))


