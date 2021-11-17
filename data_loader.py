import json
import pickle 
import os
import numpy as np
from numpy import array
from numpy.random import seed, shuffle


class DataLoader:

    def __init__(self):
        self.train_path_pk = (os.path.abspath("dataset\\train\\Viral3dV2_images_data.pk"),
                              os.path.abspath("dataset\\train\\Healthy3dV2_images_data.pk"))

        self.test_path_pk = (os.path.abspath("dataset\\test\\Viral3dV2_images_data.pk"),
                             os.path.abspath("dataset\\test\\Healthy3dV2_images_data.pk"))

    def pk_json_load(self, filename):
        if filename.endswith(".json"):
            return [x for x in json.load(open(filename, "r")).values()]
        if filename.endswith(".pk"):
            return [x for x in pickle.load(open(filename, "rb")).values()]

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

    def load_train(self, type_: str = "train", seed_np: int = 7231, count_img=(0, 1600)):
        healthy = self.pk_json_load(self.train_path_pk[1])
        viral = self.pk_json_load(self.train_path_pk[0])
        return self.subload(healthy, viral, seed_np, count_img)

    def load_test(self, seed_np: int = 7231):
        healthy = self.pk_json_load(self.test_path_pk[1])
        viral = self.pk_json_load(self.test_path_pk[0])
        return self.subload(healthy, viral, seed_np)

    def get_train(self, what):
        if what == "healthy":
            return array(self.pk_json_load(self.train_path_pk[1]), ndmin=2)
        if what == "viral":
            return array(self.pk_json_load(self.train_path_pk[0]), ndmin=2)
