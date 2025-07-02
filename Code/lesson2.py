"""
MNISTのファッション版 (Fashion MNIST，クラス数10) をソフトマックス回帰によって分類する
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.modules['tensorflow'] = None

def load_fashionmnist():
    # パス
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    path = os.path.join(script_dir, '..', 'Data', 'Lecture02_dataset') 

    # 学習データ
    x_train_path = os.path.join(path, 'x_train.npy')
    y_train_path = os.path.join(path, 'y_train.npy')
    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    # テストデータ
    x_test_path = os.path.join(path, 'x_test.npy')
    x_test = np.load(x_test_path)
    
    # データの前処理
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32')]
    x_test = x_test.reshape(-1, 784).astype('float32') / 255

    return x_train, y_train, x_test

x_train, y_train, x_test = load_fashionmnist()

def softmax(x):
    # WRITE ME
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 重み
W = np.random.randn(784, 10) * 0.01 # WRITE ME
b = np.zeros(10)                    # WRITE ME

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

def train(x, t, eps=1.0):
    # WRITE ME
    global W, b
    y = softmax(np.dot(x, W) + b)

    grad = y - t
    dW = np.dot(x.T, grad) / len(x)
    db = np.mean(grad, axis=0)
    
    W = W - eps * dW
    b = b - eps * db
    
    return W, b

def valid(x, t):
    # WRITE ME
    y = softmax(np.dot(x, W) + b)
    # 予測値と正解ラベルをワンホット表現からクラスインデックスに変換
    y_pred_labels = np.argmax(y, axis=1)
    t_labels = np.argmax(t, axis=1)
    
    return accuracy_score(t_labels, y_pred_labels)

for epoch in range(10):
    # WRITE ME
    train(x_train, y_train, eps=1.0)
    accuracy = valid(x_valid, y_valid)
    print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}')

y_pred_softmax = softmax(np.dot(x_test, W) + b)
y_pred = np.argmax(y_pred_softmax, axis=1) # WRITE ME

# 動作確認済み