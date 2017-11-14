import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from math import log

def get_pivot(a):
	if len(a)<=5:
		# print('a={}'.format(a))
		# print('med at={}'.format((len(a)-1)//2))
		return sorted(a)[(len(a)-1)//2]

	medians = []
	for i in range(0, len(a), 5):
		medians.append(sorted(a[i:i+5])[(len(a[i:i+5])-1)//2])
	medians = np.array(medians)
	return get_pivot(medians)

def swap(a, i, j, idx):
	if i != j:
		temp = a[i]
		tempi = idx[i]
		a[i] = a[j]
		idx[i] = idx[j]
		a[j] = temp
		idx[j] = tempi

def partition(a, x, idx):
	# print('Partitioning {} around {}'.format(a, x))
	i = -1
	last_index_of_x = -1
	for j in range(len(a)):
		if a[j]<=x:
			swap(a, i+1, j, idx)
			i += 1
			if a[i]==x:
				last_index_of_x = i
	swap(a, i, last_index_of_x, idx)
	# print('{} is now at {} in {}'.format(x, last_index_of_x, a))
	return i

def select(a, k, idx):
	# print('Finding {} lowest in {}'.format(k, a))
	x = get_pivot(a)
	posx = partition(a, x, idx)
	# print('Found at posx = {} in {}'.format(posx, a))
	if posx == k:
		return posx
	elif posx < k:
		return (posx + 1) + select(a[posx+1:], k-posx-1, idx[posx+1:])
	else:
		return select(a[:posx], k, idx[:posx])

def top_k(a, k):
	idx = np.arange(len(a))
	khi = select(a, len(a)-k, idx)
	# print(a)
	return np.concatenate((np.zeros(len(a)-k), np.array(a[khi:]))), np.array(idx[khi:])

if __name__ == '__main__':
	runtimes = []
	for n in np.arange(50000, 500000, 10000):
		for k in [20000]:
			t1 = time()
			a = np.random.randint(0, n//2, n)
			# a = np.array([int(x) for x in '0 4 4 4 4 0 2 3 4 1'.split()])
			# print(a)
			topk_unmasked, topk_indices = top_k(a, 4)
			runtimes.append([n, time()-t1])

	df_times = pd.DataFrame(runtimes)
	fig = plt.figure()
	ax1 = fig.add_axes([0.1,0.05,0.85,0.85])
	ax1.plot(df_times.index, df_times[0], 'r')

	ax2 = ax1.twinx()
	ax2.plot(df_times.index, df_times[1], 'b')

	plt.show()

	# print(topk_unmasked)
	# print(topk_indices)