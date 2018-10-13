import numpy as np
from model.utils.data_io import *
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


class MNISTDataSet(DataSet):

  def __init__(self,dataset_path,
               img_width=28, img_height=28, num_images=None, train=True,
               shuffle=False, low=-1, high=1):
    DataSet.__init__(self, dataset_path, img_height, img_width)
    self.images, self.attributes = maybe_download_minst(dataset_path, train)
    self.images = self.images.astype(np.float32)
    self.images = np.multiply(self.images, (high - low) / 255.0) + low
    self.num_images = len(self.images)
    self.indices = np.arange(self.num_images, dtype=np.int32)
    if shuffle:
      np.random.shuffle(self.indices)
    if num_images:
      self.num_images = min(self.num_images, num_images)
      self.indices = self.indices[:self.num_images]
    self.attributes = self.attributes[self.indices]
    self.images = self.images[self.indices]
    print(self.images.shape)
    self.data_info = ['{' + '\n'
                          + '\t\'id\': ' + str(i + 1) + '\n'
                          + '\t\'attributes\': ' + str(attr)
                          + '\n}' for i, attr in enumerate(self.attributes)]


def maybe_download_minst(train_dir, train=True, one_hot=True):
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if train:
    return train_images, train_labels
  else:
    return test_images, test_labels

def labels_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == '__main__':
    db = MNISTDataSet('../../data/mnist', shuffle=True)
    x, y = db[:10]
    saveSampleImages(x, 'test.png', 2, 5)
    print(y)
    print(x.shape, y.shape)