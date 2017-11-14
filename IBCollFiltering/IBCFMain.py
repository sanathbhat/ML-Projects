#Reference: http://files.grouplens.org/papers/www10_sarwar.pdf for collaborative filtering
import numpy as np
from numpy.linalg import norm
from time import time
np.set_printoptions(threshold=np.nan)

m = 943     #n_users
n = 1682    #n_items
k = 50

def readRatings(file):
    ratings_mat = np.zeros((m, n))
    with open(file, 'r') as f:
        for line in f:
            u_r = line.split()
            ratings_mat[int(u_r[0])-1, int(u_r[1])-1] = u_r[2]
    return ratings_mat

if __name__=='__main__':
    # np.random.seed(10)
    # R = np.random.randint(6, size=(5,3))
    # R = np.array([[1,1,0],[2,2,3],[2,2,5],[3,4,0],[5,5,3]])
    train_data = 'data/ml-100k/u1.base'
    test_data = 'data/ml-100k/u1.test'

    R = readRatings(train_data)
    # print(R)
    np.savetxt('ratings.txt', R, fmt='%d', delimiter=',',)

    t1 = time()
    #compute mean user rating for items rated by the user (zero ratings represent unrated items)
    R_mean = np.array([np.mean(R[i,:][R[i,:]!=0]) for i in range(R.shape[0])])
    # print(R_mean)

    #mean centering ratings(subtract average user ratings from user's ratings of rated items)
    R_m_c = np.where(R>0 ,R - np.repeat(R_mean, R.shape[1]).reshape(R.shape[0], R.shape[1]), R)
    # print(R_m_c)

    #COMPUTING PAIRWISE ADJUSTED COSINE SIMILARITIES:
    cos_sim = np.zeros((R.shape[1],R.shape[1]))

    #create a matrix from the ratings matrix with ones where user u has rated an item i and zeros otherwise
    R_exist = np.where(R>0, np.ones((R.shape[0], R.shape[1])), np.zeros((R.shape[0], R.shape[1])))
    #print(R_exist)

    for i in range(R.shape[1]):
        for j in range(R.shape[1]):
            if i==j:    #diagonal elements represent similarity of item to self which is irrelevant
                cos_sim[i, j] = 0
            elif j>i:
                #compute norms of item rating vectors from only mutually rated item ratings
                #for example for rating vectors of item i and j, disregard the elements of i where j is zero
                #and vice versa when computing norms of both vectors
                normi = norm(R_m_c[:, i] * R_exist[:, j])
                normj = norm(R_m_c[:, j] * R_exist[:, i])
                #zeroing out in numerator explicitly is not required as dot product automatically zeroes out
                #non-mutually rated terms
                cos_sim[i, j] = np.dot(R_m_c[:, i], R_m_c[:, j])/(normi*normj)
            elif i>j:   #symmetric other half copy
                cos_sim[i, j] = cos_sim[j, i]

    print('Finished computing similarities in {} secs'.format(time()-t1))

    #calculating ratings from test set
    t2 = time()
    #array which has 1 where ratings exist and -99 otherwise, will be used for ignoring unrated items in top-k later
    R_rem = np.where(R > 0, np.zeros((R.shape[0], R.shape[1])), np.full((R.shape[0], R.shape[1]), -99))

    ae = []
    # nancount = 0

    with open(test_data, 'r') as f:
        for line in f:
            u_r = line.split()
            u, t, actual_r = int(u_r[0])-1, int(u_r[1])-1, int(u_r[2])

            #remove cosine similarities for items which user u hasn't rated. Removal=make extremely negative
            #so that those are not included in top-k for significantly large k as well
            csr = R_rem[u, :] + cos_sim[t, :]
            #remove nans from cosine similarities for item t(can this be done by removing nans from cos_sim??
            #tried removing nans from cos_sim matrix, output incorrect for some reason
            csr[np.isnan(csr)] = 0
            #print(csr)
            #keep only top k ratings
            # print('user={}, item={} : topk indices={}'.format(u, t, np.argpartition(csr, len(csr) - k)[len(csr) - k:]))
            np.put(csr, np.argpartition(csr, len(csr) - k)[:len(csr) - k], 0)

            pred_r = np.sum(csr * R[u, :])/np.sum(np.abs(csr))

            #in case of failed prediction, use mean of mean(user u ratings) and mean(item t ratings)
            if np.isnan(pred_r):
                mean_u_t = np.array([np.mean(R[u, :][R[u, :]>0]), np.mean(R[:, t][R[:, t]>0])])
                pred_r = np.nan_to_num(np.nanmean(mean_u_t))
            ae.append(round(abs(actual_r-pred_r)))

    # print('total nan predictions = {}'.format(nancount))
    print('MAE = {}'.format(sum(ae)/len(ae)))
    print('Finished predictions in {} secs'.format(time() - t2))
    #print(time()-t1)
    np.savetxt('pairwise_acs.txt', cos_sim, fmt='%6.5f')
