import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import zipfile

    
with open('unzipped/syn_X.csv','r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ",")
    data = [data for data in data_iter]
syn_X_data = np.array(data, dtype = float)

with open('unzipped/syn_Y.csv','r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ",")
    data = [data for data in data_iter]
syn_Y_data = np.array(data, dtype = float)


ones = np.ones(syn_X_data.shape[0]).reshape([100, 1])
X = np.concatenate([ones, syn_X_data], axis = 1)
y = syn_Y_data
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(syn_Y_data)
beta_0 = np.zeros_like(beta)

#def gradient_descent(X, y, step_size, precision):
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#X[1], X[2] = np.meshgrid(X[1], X[2])
#surf = ax.plot_surface(X[1], X[2], syn_Y_data, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)


precision = 10e-6

def gradient_descent(X, y, step_size, precision = 10e-6):
    Loss_after = np.array([[10e6]])
    Loss_before = np.array([[10e7]])
    beta = np.zeros(X.shape[1]).reshape([X.shape[1],1])
    counter = 0
    while np.abs(Loss_after[0][0] - Loss_before[0][0]) > precision:
        Loss_before = Loss_after
        grad = -2*y.T.dot(X) + 2*beta.T.dot(X.T).dot(X)
        beta = beta - step_size*grad.T
        Loss_after = ((y - X.dot(beta)).T) @ (y - X.dot(beta))
        counter += 1
        if Loss_after > Loss_before:
            return (np.zeros_like(beta), 10e6)      
       
    return (beta, counter)

print(gradient_descent(X, y, 0.01, precision)[0])


step_sizes = [i/1000 for i in range(1,100)]

count_min = 100000
for step_size in step_sizes:
    beta, count = gradient_descent(X, y, step_size, precision)
    if count < count_min:
        count_min = count
        step_size_optimal = step_size
        beta_optimal = beta


# returns a 3-tuple of (list of city names, list of variable names, numpy record array with each variable as a field)
def read_mortality_csv(zip_file):
    import io
    import csv
    fields, cities, values = None, [], []
    with io.TextIOWrapper(zip_file.open('data_and materials/mortality.csv')) as wrap:
        csv_reader = csv.reader(wrap, delimiter=',', quotechar='"')
        fields = next(csv_reader)[1:]
        for row in csv_reader:
            cities.append(row[0])
            values.append(tuple(map(float, row[1:])))
    dtype = np.dtype([(name, float) for name in fields])
    return cities, fields, np.array(values, dtype=dtype).view(np.recarray)

with zipfile.ZipFile("release_statsreview_release.zip") as zip_file:
    m_cities, m_fields, m_values = read_mortality_csv(zip_file)
    
m_values_array = np.array(m_values.tolist())
ones = np.ones_like(m_values_array[:, 1].reshape([59, 1]))
X = np.concatenate([ones, m_values_array[:, 1:]], axis = 1)
y = m_values_array[:, 0].reshape([59, 1]) 

X = (X - X.mean(axis = 0, keepdims = True)) / X.std(axis = 0, keepdims = True)
X = np.concatenate([ones, X[:, 1:]], axis = 1)
y = (y - y.mean()) / y.std()
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

step_sizes = [i/1000 for i in range(1,100)]

print(gradient_descent(X, y, 0.0001, 10e-6))

Base_Loss = ((y - X.dot(beta)).T) @ (y - X.dot(beta))
m_values_array = np.array(m_values.tolist())
ones = np.ones_like(m_values_array[:, 1].reshape([59, 1]))
X = np.concatenate([ones, m_values_array[:, 1:]], axis = 1)
y = m_values_array[:, 0].reshape([59, 1]) 

column_number = 1
X = np.concatenate([ones, m_values_array[:, 1:]], axis = 1)
for parameter in m_fields[1:]:    
    y = m_values_array[:, 0].reshape([59, 1]) 
    X[:, column_number] = np.log(X[:, column_number])
    X = (X - X.mean(axis = 0, keepdims = True)) / X.std(axis = 0, keepdims = True)
    X = np.concatenate([ones, X[:, 1:]], axis = 1)
    y = (y - y.mean()) / y.std()
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    Loss = ((y - X.dot(beta)).T) @ (y - X.dot(beta))
    if Loss <= Base_Loss:
        print(parameter)
    else:
        X = np.concatenate([ones, m_values_array[:, 1:]], axis = 1)
    column_number += 1

X = np.concatenate([ones, m_values_array[:, 1:]], axis = 1)
y = m_values_array[:, 0].reshape([59, 1]) 
y = np.log(y)
X = (X - X.mean(axis = 0, keepdims = True)) / X.std(axis = 0, keepdims = True)
X = np.concatenate([ones, X[:, 1:]], axis = 1)
y = (y - y.mean()) / y.std()
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
Loss = ((y - X.dot(beta)).T) @ (y - X.dot(beta))
print(Loss < Base_Loss)
    
    