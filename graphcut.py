import numpy as np
from math import exp, pow
from sklearn.mixture import GaussianMixture
from collections import deque


class Segmentation:
    def __init__(self, img: np.ndarray, objLabel, bkgLabel,alpha):
        self.img = img
        self.bkg = bkgLabel
        self.obj = objLabel
        self.alpha = alpha

    def B(self, p, q):
        return 100 * exp(-pow(int(p) - int(q), 2) / (2 * pow(30, 2)))

    def get_gmm(self):
        bkg_values = [self.img[i[0]][i[1]] for i in self.bkg]
        obj_values = [self.img[i[0]][i[1]] for i in self.obj]
        hkg_hist, _ = np.histogram(bkg_values, 256, [0, 256])
        obj_hist, _ = np.histogram(obj_values, 256, [0, 256])

        obj_gmm = GaussianMixture(n_components=1)
        obj_gmm = obj_gmm.fit(X=np.expand_dims(obj_values, 1))
        bkg_gmm = GaussianMixture(n_components=1)
        bkg_gmm = bkg_gmm.fit(X=np.expand_dims(bkg_values, 1))
        return obj_gmm, bkg_gmm

    def R(self, p, obj_gmm, bkg_gmm):
        p_obj = np.exp(obj_gmm.score_samples([[p]]))
        p_bkg = np.exp(bkg_gmm.score_samples([[p]]))
        p_p_obj = p_obj / (p_obj + p_bkg)
        p_p_bkg = p_bkg / (p_obj + p_bkg)
        return -np.log(p_p_obj), -np.log(p_p_bkg)

    # Image to Graph
    def make_graph(self):
        r, c = self.img.shape
        V = r * c + 2
        G = np.zeros((V,V))
        K = 1e9

        # Regional Cost
        obj_gmm, bkg_gmm = self.get_gmm()
        for i in range(r):
            for j in range(c):
                if (i, j) in self.obj:
                    G[0][1 + c * i + j] = K
                    G[1 + c * i + j][V - 1] = 0
                elif (i, j) in self.bkg:
                    G[0][1 + c * i + j] = 0
                    G[1 + c * i + j][V - 1] = K
                else:
                    R_bkg, R_obj = self.R(self.img[i][j], obj_gmm, bkg_gmm)
                    G[0][1 + c * i + j] = self.alpha * R_obj
                    G[1 + c * i + j][V - 1] = self.alpha * R_bkg

        # Boundary Cost
        for i in range(r):
            for j in range(c):
                if i - 1 >= 0 and G[c * i + j + 1][c * (i - 1) + j + 1] == 0:
                    G[c * i + j + 1][c * (i - 1) + j + 1] = self.B(self.img[i][j], self.img[i - 1][j])
                    G[c * (i - 1) + j + 1][c * i + j + 1] = self.B(self.img[i][j], self.img[i - 1][j])
                if i + 1 < r and G[c * i + j + 1][c * (i + 1) + j + 1] == 0:
                    G[c * i + j + 1][c * (i + 1) + j + 1] = self.B(self.img[i][j], self.img[i + 1][j])
                    G[c * (i + 1) + j + 1][c * i + j + 1] = self.B(self.img[i][j], self.img[i + 1][j])
                if j - 1 >= 0 and G[c * i + j + 1][c * i + (j - 1) + 1] == 0:
                    G[c * i + j + 1][c * i + (j - 1) + 1] = self.B(self.img[i][j], self.img[i][j - 1])
                    G[c * i + (j - 1) + 1][c * i + j + 1] = self.B(self.img[i][j], self.img[i][j - 1])
                if j + 1 < c and G[c * i + (j + 1) + 1][c * i + (j + 1) + 1] == 0:
                    G[c * i + j + 1][c * i + (j + 1) + 1] = self.B(self.img[i][j + 1], self.img[i][j])
                    G[c * i + (j + 1) + 1][c * i + j + 1] = self.B(self.img[i][j + 1], self.img[i][j])
        return G

    def networkFlow(self, source, sink, capacity, flow):
        N, totalflow = len(capacity), 0
        while True:
            parent = [-1] * N
            queue = deque()
            parent[source] = source
            queue.append(source)
            while queue and parent[sink] == -1:
                p = queue.popleft()
                for q in range(N):
                    if capacity[p][q] - flow[p][q] > 0 and parent[q] == -1:
                        queue.append(q)
                        parent[q] = p
            if parent[sink] == -1: break
            p, amount = sink, int(1e9)
            while (p != source):
                amount = min(amount, capacity[parent[p]][p] - flow[parent[p]][p])
                p = parent[p]
            p = sink
            while (p != source):
                flow[parent[p]][p] += amount
                flow[p][parent[p]] -= amount
                p = parent[p]
            totalflow += amount
        return totalflow

    def run(self):
        r, c = self.img.shape
        V = r * c + 2
        G = self.make_graph()
        F = np.zeros((V, V))
        self.networkFlow(0, V - 1, G, F)

        def dfs(graph, s, visited):
            visited[s] = True
            for i in range(len(graph)):
                if graph[s][i] > 0 and not visited[i]:
                    dfs(graph, i, visited)

        visited = [False] * V
        dfs(G, 0, visited)
        for i in range(V):
            for j in range(V):
                if visited[i] and G[i][j] == F[i][j] != 0:
                    G[i][j] = 0
        group = [False] * V
        dfs(G, 0, group)
        mask = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                if group[1 + i * c + j]:
                    mask[i][j] = 1
        return mask
