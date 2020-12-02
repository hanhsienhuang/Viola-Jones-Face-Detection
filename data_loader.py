import os.path
import glob
from PIL import Image
import pickle
import numpy as np

def normalize(x):
    mean = np.mean(x, axis = (1,2), keepdims=True)
    var = np.mean(x**2, axis = (1,2), keepdims=True) - mean**2
    x = x/np.sqrt(var)
    return x


def load_from_folder(folder = "VJ_dataset"):
    train_sets = ["trainset", "testset"]
    is_face = ["faces", "non-faces"]
    def load_dataset(train, face):
        images = []
        filenames = glob.glob(os.path.join(folder, train, face, "*.png"))
        for filename in filenames:
            image = Image.open(filename)
            images.append(np.array(image, dtype=float))
        y = (1 if face == "faces" else -1) * np.ones(len(images), dtype=int)
        return np.stack(images), y

    images = {}
    for train in train_sets:
        for face in is_face:
            images[(train, face)] = load_dataset(train, face)
    train_x = np.concatenate([images[("trainset", "faces")][0], images[("trainset", "non-faces")][0]])
    train_y = np.concatenate([images[("trainset", "faces")][1], images[("trainset", "non-faces")][1]])
    test_x = np.concatenate([images[("testset", "faces")][0], images[("testset", "non-faces")][0]])
    test_y = np.concatenate([images[("testset", "faces")][1], images[("testset", "non-faces")][1]])
    train_x = normalize(train_x)
    test_x = normalize(test_x)
    return train_x, train_y, test_x, test_y

def load(folder = "VJ_dataset"):
    file = "dataset.pkl"
    try:
        with open(file, "rb") as f:
            data = pickle.load(f)
    except:
        data = load_from_folder(folder)
        with open(file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data
