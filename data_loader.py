import json
import pickle 
import os
import numpy as np
from numpy import array
from numpy.random import seed, shuffle


class DataLoader:

    def __init__(self):
        self.train_path_pk = (os.path.abspath("dataset\\train\\Viral.pk"),
                              os.path.abspath("dataset\\train\\Health.pk"))

        self.test_path_pk = (os.path.abspath("dataset\\test\\Viral.pk"),
                             os.path.abspath("dataset\\test\\Health.pk"))

    def pk_load(self, filename):
        return pickle.load(open(filename, "rb"))

    def subload(self, healthy, viral, seed_np, count_img=(0, 1600)):
        combo_h_v = healthy + viral
        combo_targets = [0 for x in range(len(healthy))] + [1 for y in range(len(viral))]
        assert len(combo_h_v) == len(combo_targets), "Different length"

        test_join = [x for x in zip(combo_h_v, combo_targets)]
        np.random.seed(seed_np)
        np.random.shuffle(test_join)
        test_join = test_join[count_img[0]:count_img[1]]

        targets = array([x[1] for x in test_join], ndmin=2)        
        result = array([x[0] for x in test_join], ndmin=2)
        print(f"Targets shape: {targets.shape}\nData shape: {result.shape}")
        return result, targets

    def load_train(self, seed_np: int = 7231, count_img=(0, 1600)):
        healthy = self.pk_load(self.train_path_pk[1])
        viral = self.pk_load(self.train_path_pk[0])
        return self.subload(healthy, viral, seed_np, count_img)

    def load_test(self, seed_np: int = 7231):
        healthy = self.pk_load(self.test_path_pk[1])
        viral = self.pk_load(self.test_path_pk[0])
        return self.subload(healthy, viral, seed_np)
