from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
import heapq
import numpy as np
import pandas as pd

data = []

class KNNModel:
  def __init__(self):
    self.k = 2
    self.encoder = 'ResNet'
    self.distance_metric = 'euclidean'
    self.X_train = []
    self.Y_train = []
    
  def set_encoder(self, encoder_name):
    encoder = {}
    encoder['name'] = encoder_name
    encoder['X'] = []
    encoder['Y'] = []
    
    if encoder['name'] == 'ResNet':
      for row in data:
        encoder['X'].append(row[1])
        encoder['Y'].append(row[3])
      
    elif encoder['name'] == 'VIT':
      for row in data:
        encoder['X'].append(row[2])
        encoder['Y'].append(row[3])
      
    else:
      raise ValueError('Unknown encoder')
      
    self.encoder = encoder
  
  def set_distance_metric(self, distance_metric):
    self.distance_metric = distance_metric
    
  def set_k(self, k):
    self.k = k
    
  def calculate_distance(self, x1, x2):
    if self.distance_metric == 'euclidean':
      return np.sqrt(np.sum((x1 - x2) ** 2))
    elif self.distance_metric == 'manhattan':
      return np.sum(np.abs(x1 - x2))
    elif self.distance_metric == 'cosine':
      # cosine distance = 1 - cosine similarity
      return 1 - (np.dot(x1.flatten(), x2.flatten()) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
    else:
      raise ValueError('Unknown distance metric')
    
  def fit(self, X_train, Y_train):
    self.X_train = np.array(X_train)
    self.Y_train = np.array(Y_train)
  
  def predict(self, X):
    y_pred = []
    for x in X:
      distances = []
      k_nearest = []
      for i, x_train in enumerate(self.X_train):
        heapq.heappush(distances, (self.calculate_distance(x, x_train), self.Y_train[i]))
      while distances and len(k_nearest) < self.k:
        k_nearest.append(heapq.heappop(distances)[1])
      y_pred.append(max(set(k_nearest), key = k_nearest.count))
    return y_pred
  
  def evaluate(self, X_test, Y_test):
    # predict the test data
    y_pred = self.predict(X_test)
    # calculate the metrics
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(Y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(Y_test, y_pred, average='macro', zero_division=0)
    return [accuracy*100, precision*100, recall*100, f1*100]
  
  def get_accuracy(self, X_test, Y_test):
    return self.evaluate(X_test, Y_test)[0]
  
  
if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python3 eval.py <embeddings_file>")
    sys.exit(1)
  
  input_test_file = sys.argv[1]
  data = np.load('./data.npy', allow_pickle=True)
  test_data = np.load(input_test_file, allow_pickle=True)
  
  Y_test = test_data[:, 3]
  
  knn = KNNModel()
  knn.set_encoder('ResNet')
  knn.set_distance_metric('manhattan')
  knn.set_k(13)
  X_test = test_data[:, 1]
  knn.fit(knn.encoder['X'], knn.encoder['Y'])
  print('RESNET, accuracy precision recall f1: ', *knn.evaluate(X_test, Y_test))

  knn = KNNModel()
  knn.set_encoder('VIT')
  knn.set_distance_metric('manhattan')
  knn.set_k(1)
  X_test = test_data[:, 2]
  knn.fit(knn.encoder['X'], knn.encoder['Y'])
  print('VIT, accuracy precision recall f1: ', *knn.evaluate(X_test, Y_test))