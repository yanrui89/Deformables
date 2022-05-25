import csv
import numpy as np

a = np.load('/home/clearlab/pybullet_robot/assets/bags/test22.npy')
print(np.max(a[:,0]))
b = a[:,0]
c = np.abs(b - 0.26)
d = c < 0.01
e = np.where(d == 1)
print(e)

'''
file = open('/home/clearlab/pybullet_robot/assets/bags/test.csv')
csvreader = csv.reader(file)
rows = []
rows_x = []
rows_z = []
for row in csvreader:
    print(float(row[2]))
    #row = row.split(',')
    rows.append(float(row[0]))
    rows_z.append(float(row[1]))
    rows_x.append(float(row[2]))


rows_array = np.array(rows)
print(rows_array)

min_idx= np.argmin(rows_array)
a = rows_z[min_idx]
b = rows_x[min_idx]

print(min_idx)
print(a)
print(b)
'''
