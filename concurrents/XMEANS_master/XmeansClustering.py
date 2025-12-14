import numpy as np
import math as mt
import sys
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

'''
class XMeans:
    def loglikelihood(self, r, rn, var, m, k):
        l2=l4=l5=0
        l1 = - rn / 2.0 * mt.log(2 * mt.pi)
        if var>0: l2 = - rn * m / 2.0 * mt.log(var)
        l3 = - (rn - k) / 2.0
        if rn>0: l4 = rn * mt.log(rn)
        if r> 0: l5 = - rn * mt.log(r)

        return l1 + l2 + l3 + l4 + l5

    def __init__(self, X, kmax = 20):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(X, axis=1)
        self.KMax = kmax

    def fit(self):
        k = 1
        X = self.X
        M = self.dim
        num = self.num

        while(1):
            ok = k

            #Improve Params
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            m = kmeans.cluster_centers_

            #Improve Structure
            #Calculate BIC
            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                var = np.sum((X[labels == i] - m[i])**2)/float(rn - 1)
                obic[i] = self.loglikelihood(rn, rn, var, M, 1) - p/2.0*mt.log(rn)

            #Split each cluster into two subclusters and calculate BIC of each splitted cluster
            sk = 2 #The number of subclusters
            nbic = np.zeros(k)
            addk = 0

            for i in range(k):
                ci = X[labels == i]
                r = np.size(np.where(labels == i))
                if (r > sk): #correction frederic
                    kmeans = KMeans(n_clusters=sk).fit(ci)
                    ci_labels = kmeans.labels_
                    sm = kmeans.cluster_centers_

                    for l in range(sk):
                        rn = np.size(np.where(ci_labels == l))
                        var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
                        nbic[i] += self.loglikelihood(r, rn, var, M, sk)

                    p = sk * (M + 1)
                    nbic[i] -= p/2.0*mt.log(r)

                    if obic[i] < nbic[i]:
                        addk += 1

            k += addk

            if ok == k or k >= self.KMax:
                break


        #Calculate labels and centroids
        kmeans = KMeans(n_clusters=k).fit(X)
        self.labels = kmeans.labels_
        self.k = k
        self.m = kmeans.cluster_centers_
'''
import numpy as np
import math as mt
from sklearn.cluster import KMeans

import numpy as np
import math
from sklearn.cluster import KMeans


import numpy as np
from sklearn.cluster import KMeans

import numpy as np


import numpy as np
from sklearn.cluster import KMeans

import numpy as np
from sklearn.cluster import KMeans

class XMeans:
    def __init__(self, X, kmax=20, min_cluster=20, random_state=0):
        """
        X : array (n_samples, n_features)
        kmax : nombre maximal de clusters
        min_cluster : taille minimale d’un cluster pour autoriser un split
        """
        self.X = X
        self.n, self.dim = X.shape
        self.kmax = kmax
        self.min_cluster = min_cluster
        self.random_state = random_state

    # ----------------------------
    # Log-likelihood (variance globale)
    # ----------------------------
    def _loglikelihood(self, Xc, center, var):
        n = Xc.shape[0]
        if n <= 1:
            return -np.inf
        return (
            - n * self.dim / 2 * np.log(2 * np.pi * var)
            - np.sum((Xc - center) ** 2) / (2 * var)
        )

    # ----------------------------
    # Fit X-means
    # ----------------------------
    def fit(self):
        # Variance globale (clé du papier)
        global_var = np.mean(
            np.sum((self.X - np.mean(self.X, axis=0)) ** 2, axis=1)
        )
        global_var = max(global_var, 1e-6)

        # Initialisation : 1 cluster
        clusters = [self.X]

        while True:
            new_clusters = []
            split_done = False

            for Xc in clusters:
                n_c = Xc.shape[0]

                # Trop petit ou plafond atteint → pas de split
                if n_c < self.min_cluster or len(new_clusters) >= self.kmax:
                    new_clusters.append(Xc)
                    continue

                # Modèle parent
                center_p = np.mean(Xc, axis=0)
                ll_parent = self._loglikelihood(Xc, center_p, global_var)
                bic_parent = ll_parent - 0.5 * (self.dim + 1) * np.log(n_c)

                # Tentative de split (K=2)
                km = KMeans(n_clusters=2, n_init=10,
                            random_state=self.random_state).fit(Xc)
                labels = km.labels_
                X1 = Xc[labels == 0]
                X2 = Xc[labels == 1]

                if len(X1) < self.min_cluster or len(X2) < self.min_cluster:
                    new_clusters.append(Xc)
                    continue

                # Modèle enfants
                ll_1 = self._loglikelihood(X1, np.mean(X1, axis=0), global_var)
                ll_2 = self._loglikelihood(X2, np.mean(X2, axis=0), global_var)

                bic_children = (
                    ll_1 + ll_2
                    - 0.5 * 2 * (self.dim + 1) * np.log(n_c)
                )

                # Split seulement si amélioration BIC
                if bic_children > bic_parent and len(new_clusters) + 2 <= self.kmax:
                    new_clusters.extend([X1, X2])
                    split_done = True
                else:
                    new_clusters.append(Xc)

            clusters = new_clusters

            # Arrêt naturel (papier)
            if not split_done or len(clusters) >= self.kmax:
                break

        # Clustering final par KMeans
        self.k = len(clusters)

        if self.k <= 1:
            self.k = 1
            self.labels = np.zeros(self.n, dtype=int)
            self.centers = np.mean(self.X, axis=0, keepdims=True)
            return self

        kmeans = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state
        ).fit(self.X)

        self.labels = kmeans.labels_
        self.centers = kmeans.cluster_centers_

        return self

class XMeansTarget:
    def __init__(self, X, target_k=4, kmax=20, random_state=0):
        """
        X : array-like, shape (n_samples, n_features)
        target_k : int, nombre de clusters maximal visé
        kmax : int, plafond strict
        random_state : int pour reproductibilité
        """
        self.X = X
        self.n, self.dim = X.shape
        self.target_k = target_k
        self.KMax = kmax
        self.random_state = random_state

    def _loglikelihood(self, Xc, center):
        n = Xc.shape[0]
        if n <= 1:
            return -np.inf
        var = np.mean(np.sum((Xc - center) ** 2, axis=1))
        var = max(var, 1e-6)
        return - n * self.dim / 2 * np.log(2 * np.pi) - n * self.dim / 2 * np.log(var) - (n - 1) / 2

    def fit(self):
        clusters = [self.X]

        while True:
            new_clusters = []
            split_occurred = False

            for Xc in clusters:
                nc = Xc.shape[0]

                # Cluster trop petit → pas de split
                if nc <= 3:
                    new_clusters.append(Xc)
                    continue

                # Cluster parent
                center_p = np.mean(Xc, axis=0)
                ll_parent = self._loglikelihood(Xc, center_p)
                bic_parent = ll_parent - 0.5 * (self.dim + 1) * np.log(nc)

                # Split en 2
                km = KMeans(n_clusters=2, n_init=10, random_state=self.random_state).fit(Xc)
                X1 = Xc[km.labels_ == 0]
                X2 = Xc[km.labels_ == 1]

                if X1.shape[0] <= 1 or X2.shape[0] <= 1:
                    new_clusters.append(Xc)
                    continue

                ll_c1 = self._loglikelihood(X1, np.mean(X1, axis=0))
                ll_c2 = self._loglikelihood(X2, np.mean(X2, axis=0))
                bic_children = ll_c1 + ll_c2 - 0.5 * 2 * (self.dim + 1) * np.log(nc)

                # Split si BIC s'améliore et on ne dépasse pas target_k ou kmax
                if bic_children > bic_parent and (len(new_clusters) + 2 <= self.target_k) and (len(new_clusters) + 2 <= self.KMax):
                    new_clusters.extend([X1, X2])
                    split_occurred = True
                else:
                    new_clusters.append(Xc)

                # Arrêt immédiat si target atteint
                if len(new_clusters) >= self.target_k:
                    split_occurred = False
                    break

            clusters = new_clusters

            # Arrêt naturel si pas de split ou target_k atteint
            if not split_occurred or len(clusters) >= self.target_k:
                break

        # Clustering final
        k = len(clusters)
        if k == 0:
            self.k = 1
            self.labels = np.zeros(self.n, dtype=int)
            self.centers = np.mean(self.X, axis=0, keepdims=True)
            return self

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(self.X)
        self.labels = kmeans.labels_
        self.centers = kmeans.cluster_centers_
        self.k = k

        return self


'''
if __name__ == '__main__':

    #Blobs (Isotropic Gaussian distributions)
    X, TrueLabels = datasets.make_blobs(n_samples=1500, centers=3, n_features=3)

    xm = XMeans(X)
    xm.fit()

    purity = metrics.adjusted_rand_score(TrueLabels, xm.labels)
    nmi = metrics.normalized_mutual_info_score(TrueLabels, xm.labels)
    ari = metrics.adjusted_rand_score(TrueLabels, xm.labels)

    print("Blobs")
    print("True k = 3, Estimated k = " + str(xm.k) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
    
    #Iris dataset
    dataset = datasets.load_iris()
    X = dataset.data
    TrueLabels = dataset.target

    xm = XMeans(X)
    xm.fit()

    purity = metrics.adjusted_rand_score(TrueLabels, xm.labels)
    nmi = metrics.normalized_mutual_info_score(TrueLabels, xm.labels)
    ari = metrics.adjusted_rand_score(TrueLabels, xm.labels)

    print("Iris dataset")
    print("True k = 3, Estimated k = " + str(xm.k) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
'''