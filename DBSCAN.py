import matplotlib.pyplot as plt
import numpy as np
import xlwt
from xlwt import Workbook
import random

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

N = 500

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1]]
X, labels_true = make_blobs(n_samples=N, centers=centers, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

################################################################################
# write to excel
Y = db.fit_predict(X)
P = []

for i in range (0, N):
	if Y[i] == -1:
		P.append(-1)
	if Y[i] == 1:
		P.append(1)
	if Y[i] == 0:
		P.append(0)

# labels
sheet1.write(0, 0, 'X')
sheet1.write(0, 1, 'Y')
sheet1.write(0, 2, 'Actual')
sheet1.write(0, 3, 'Prediction')
sheet1.write(0, 4, 'Recall')
sheet1.write(0, 5, 'Precision')


TP = 0
FP = 0
FN = 0
actual = 0
prediction = 0

for i in range(1, N):
	sheet1.write(i, 0, X[i][0])
	sheet1.write(i, 1, X[i][1])

	if random.randint(5, 10) == 10:
		sheet1.write(i, 2, -1)
		actual = -1
	elif X[i][0] < 0 and X[i][1] < 0:
		sheet1.write(i, 2, 0)
		actual = 0
	else:
		sheet1.write(i, 2, 1)
		actual = 1

	sheet1.write(i, 3, P[i - 1])
	prediction = P[i - 1]

	# ----------------------------------

	if actual != -1 and prediction != -1:
		TP += 1
	elif actual == -1 and prediction != -1:
		FP += 1
	elif actual != -1 and prediction == -1:
		FN += 1

sheet1.write(1, 4, TP / (TP + FN))
sheet1.write(1, 5, TP / (TP + FP))

wb.save('dataset.xls')
################################################################################


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# #############################################################################
# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
