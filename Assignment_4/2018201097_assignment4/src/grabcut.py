import sys
import numpy as np
import cv2 as cv
import igraph
from matplotlib import pyplot as plt
from copy import deepcopy
import os
from gaussian import *

BG = 0
FG = 1
PR_BG = 2
PR_FG = 3

class GrabCut:
    def __init__(self, img, mask, rect=None, gmm_components=5):
        self.img = np.asarray(img, dtype=np.float64)
        self.out_image = deepcopy(img)
        self.rows, self.cols, _ = img.shape
        self.mask = mask

        if rect is not None:
            self.mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
                      rect[2]] = PR_FG

        self.classify_pixels()

        self.gmm_components = gmm_components
        self.gamma = 50
        self.calculate_beta_smoothness()

        self.fgd_gmm = GaussianMixture(self.img[self.fgd_indexes])
        self.bgd_gmm = GaussianMixture(self.img[self.bgd_indexes])
        self.label_gmm = np.empty((self.rows, self.cols), dtype=np.uint32)

        self.gc_source = self.cols * self.rows  # "object" terminal S
        self.gc_sink = self.gc_source + 1  # "background" terminal T

    def calculate_beta_smoothness(self):
        left_diff = self.img[:, 1:] - self.img[:, :-1]
        up_left_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        up_diff = self.img[1:, :] - self.img[:-1, :]
        up_right_diff = self.img[1:, :-1] - self.img[:-1, 1:]

        self.beta = np.linalg.norm(left_diff)**2 + np.linalg.norm(
            up_left_diff)**2 + np.linalg.norm(up_diff)**2 + np.linalg.norm(
                up_right_diff)**2
        #         self.beta = np.sum(np.square(left_diff)) + np.sum(np.square(up_left_diff)) + \
        #             np.sum(np.square(up_diff)) + \
        #             np.sum(np.square(up_right_diff))
        self.beta = 2.0 * self.beta / (4 * self.rows * self.cols - 3 *
                                       (self.rows + self.cols) + 2.0)
        self.beta = 1 / self.beta
        print(self.beta)

        def calculate_V(x):
            return self.gamma * np.exp(
                -self.beta * np.sum(np.square(x), axis=2))

        self.left_V = calculate_V(left_diff)
        self.up_left_V = calculate_V(up_left_diff)
        self.up_V = calculate_V(up_diff)
        self.up_right_V = calculate_V(up_right_diff)

    def classify_pixels(self):
        self.fgd_indexes = np.where((self.mask == FG) | (self.mask == PR_FG))
        self.bgd_indexes = np.where((self.mask == BG) | (self.mask == PR_BG))

    def assign_GMM(self):
        self.label_gmm[self.fgd_indexes] = self.fgd_gmm.which_component(
            self.img[self.fgd_indexes])
        self.label_gmm[self.bgd_indexes] = self.bgd_gmm.which_component(
            self.img[self.bgd_indexes])

    def learn_GMM(self):
        self.fgd_gmm.fit(self.img[self.fgd_indexes],
                         self.label_gmm[self.fgd_indexes])
        self.bgd_gmm.fit(self.img[self.bgd_indexes],
                         self.label_gmm[self.bgd_indexes])

    def make_n_links(self, mask1, mask2, V):
        mask1 = mask1.reshape(-1)
        mask2 = mask2.reshape(-1)
        self.gc_graph_capacity += V.reshape(-1).tolist()
        return list(zip(mask1, mask2))

    def construct_gc_graph(self):
        fgd_indexes = np.where(self.mask.reshape(-1) == FG)
        bgd_indexes = np.where(self.mask.reshape(-1) == BG)
        pr_indexes = np.where((self.mask.reshape(-1) == PR_FG)
                              | (self.mask.reshape(-1) == PR_BG))

        self.gc_graph_capacity = []
        edges = []

        def make_edges(source, sinks):
            return list(zip([source] * sinks[0].size, sinks[0]))

        edges += make_edges(self.gc_source, pr_indexes)
        self.gc_graph_capacity += list(-np.log(
            self.bgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes])))

        edges += make_edges(self.gc_sink, pr_indexes)
        self.gc_graph_capacity += list(-np.log(
            self.fgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes])))

        edges += make_edges(self.gc_source, fgd_indexes)
        self.gc_graph_capacity += [9 * self.gamma] * fgd_indexes[0].size

        edges += make_edges(self.gc_sink, fgd_indexes)
        self.gc_graph_capacity += [0] * fgd_indexes[0].size

        edges += make_edges(self.gc_source, bgd_indexes)
        self.gc_graph_capacity += [0] * bgd_indexes[0].size

        edges += make_edges(self.gc_sink, bgd_indexes)
        self.gc_graph_capacity += [9 * self.gamma] * bgd_indexes[0].size

        img_indexes = np.arange(
            self.rows * self.cols, dtype=np.uint32).reshape(
                self.rows, self.cols)

        edges += self.make_n_links(img_indexes[:, 1:], img_indexes[:, :-1],
                                   self.left_V)
        edges += self.make_n_links(img_indexes[1:, 1:], img_indexes[:-1, :-1],
                                   self.up_left_V)
        edges += self.make_n_links(img_indexes[1:, :], img_indexes[:-1, :],
                                   self.up_V)
        edges += self.make_n_links(img_indexes[1:, :-1], img_indexes[:-1, 1:],
                                   self.up_right_V)

        assert (len(edges) == len(self.gc_graph_capacity))

        self.gc_graph = igraph.Graph(self.rows * self.cols + 2)
        self.gc_graph.add_edges(edges)

    def estimate_segmentation(self):
        mincut = self.gc_graph.st_mincut(self.gc_source, self.gc_sink,
                                         self.gc_graph_capacity)
        print('foreground pixels: %d, background pixels: %d' % (len(
            mincut.partition[0]), len(mincut.partition[1])))
        pr_indexes = np.where((self.mask == PR_FG) | (self.mask == PR_BG))
        img_indexes = np.arange(
            self.rows * self.cols, dtype=np.uint32).reshape(
                self.rows, self.cols)
        condition = np.isin(img_indexes[pr_indexes], mincut.partition[0])
        self.mask[pr_indexes] = np.where(condition, PR_FG, PR_BG)
        self.classify_pixels()

    def calculate_energy(self):
        U = 0
        bg_indexes = np.where((self.mask == BG) | (self.mask == PR_BG))
        fg_indexes = np.where((self.mask == FG) | (self.mask == PR_FG))
        for component in range(self.gmm_components):
            indexes = np.where((self.label_gmm == component) & (fg_indexes))
            U += np.sum(-np.log(self.fgd_gmm.coefs[component] * self.fgd_gmm.
                                calc_score(self.img[indexes], component)))
            indexes = np.where((self.label_gmm == component) & (bg_indexes))
            U += np.sum(-np.log(self.bgd_gmm.coefs[component] * self.bgd_gmm.
                                calc_score(self.img[indexes], component)))

        V = 0
        mask = self.mask.copy()
        mask[bg_indexes] = BG
        mask[fg_indexes] = FG

        V += np.sum(self.left_V * (mask[:, 1:] == mask[:, :-1]))
        V += np.sum(self.up_left_V * (mask[1:, 1:] == mask[:-1, :-1]))
        V += np.sum(self.up_V * (mask[1:, :] == mask[:-1, :]))
        V += np.sum(self.up_right_V * (mask[1:, :-1] == mask[:-1, 1:]))
        return U, V, U + V

    def modified_image(self):
        img2 = deepcopy(self.out_image)
        mask = self.mask.copy()
        mask2 = np.where((self.mask == 1) + (self.mask == 3), 255,
                         0).astype('uint8')
        return cv.bitwise_and(img2, img2, mask=mask2)

    def run(self, num_iters=2):
        for _ in range(num_iters):
            self.assign_GMM()
            self.learn_GMM()
            self.construct_gc_graph()
            self.estimate_segmentation()