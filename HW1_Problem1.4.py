# Problem 1.4
import numpy as np
import zipfile
import csv
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
with zipfile.ZipFile("release_statsreview_release.zip") as zip_file:   
    golub_data, golub_classnames = ( np.genfromtxt(zip_file.open('data_and materials/golub_data/{}'.format(fname)), 
                                                   delimiter=',', names=True, 
                                                   converters={0: lambda s: int(s.strip(b'"'))}) for fname in ['golub.csv', 'golub_cl.csv'] )

#with open("./unzipped/golub_data/golub.csv")


golub_data_converted = np.array(golub_data.tolist())
X_ALL = golub_data_converted[:, 1:28]
X_AML = golub_data_converted[:, 28:]
N_ALL = X_ALL.shape[1]
N_AML = X_AML.shape[1]
X_ALL_bar = X_ALL.mean(axis = 1)
X_AML_bar = X_AML.mean(axis = 1)
s2_ALL = X_ALL.var(axis = 1, ddof = 1)
s2_AML = X_AML.var(axis = 1, ddof = 1)

t_Welch = (X_ALL_bar - X_AML_bar) / np.sqrt(s2_ALL/N_ALL + s2_AML/N_AML)

nu = (s2_ALL/N_ALL + s2_AML/N_AML)**2 / (
    (1/(N_ALL-1)*(s2_ALL/N_ALL)**2) + (1/(N_AML-1)*(s2_AML/N_AML)**2))

m = 3051

p_values = 2*(1 - t.cdf(np.abs(t_Welch), nu))
p_values.sort()
correction = np.array([m - i + 1 for i in range(1, m + 1)])
print("Initial:", p_values[p_values <= 0.05].shape)
print("Holm-Bonferroni:", p_values[p_values <= 0.05/m].shape)
print("Benjamini-Hochberg:", p_values[np.multiply(p_values, correction) <= 0.05].shape)

fdr_bh = multipletests(p_values,method='fdr_bh')
print("Bonferroni 2nd:", sum(fdr_bh[1] < 0.05))