"""
MNISTのファッション版 (Fashion MNIST，クラス数10) を多層パーセプトロンによって分類する
"""
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))

def create_batch(data, batch_size):
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches:])
    return batched_data

def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def deriv_softmax(x):
    y = softmax(x)
    n, c = y.shape
    jac = np.zeros((n, c, c), dtype=y.dtype)
    for i in range(n):
        yi = y[i][:, None]
        jac[i] = np.diagflat(yi) - yi @ yi.T
    return jac

def crossentropy_loss(t, y):
    return -np.mean(np.sum(t * np_log(y), axis=1))

class Dense:
    def __init__(self, in_dim, out_dim, rng, activation=None):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation
        self.x = None
        self.u = None
        self.dW = None
        self.db = None

    def __call__(self, x):
        self.x = x
        self.u = x @ self.W + self.b
        if self.activation == "relu":
            return relu(self.u)
        elif self.activation == "softmax":
            return softmax(self.u)
        else:
            return self.u

    def backward(self, delta):
        if self.activation == "relu":
            delta = delta * deriv_relu(self.u)
        N = self.x.shape[0]
        self.dW = self.x.T @ delta / N
        self.db = delta.mean(axis=0)
        return delta @ self.W.T

    def step(self, dW, db):
        self.W += dW
        self.b += db

class Model:
    def __init__(self, rng, lr=0.01, momentum=0.9):
        self.layers = [
            Dense(784, 512, rng, activation="relu"),
            Dense(512, 256, rng, activation="relu"),
            Dense(256, 10, rng, activation="softmax"),
        ]
        self.lr = lr
        self.momentum = momentum
        self.vW = [np.zeros_like(layer.W) for layer in self.layers]
        self.vb = [np.zeros_like(layer.b) for layer in self.layers]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, t, y):
        delta = (y - t) / y.shape[0]
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def step(self):
        for i, layer in enumerate(self.layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * layer.dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * layer.db
            layer.W += self.vW[i]
            layer.b += self.vb[i]

def train_model(mlp, x_train, t_train, x_val, t_val, batch_size, n_epochs=10):
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        x_train, t_train = shuffle(x_train, t_train)
        x_train_batches = create_batch(x_train, batch_size)
        t_train_batches = create_batch(t_train, batch_size)

        x_val, t_val = shuffle(x_val, t_val)
        x_val_batches = create_batch(x_val, batch_size)
        t_val_batches = create_batch(t_val, batch_size)

        for x, t in zip(x_train_batches, t_train_batches):
            y = mlp(x)
            loss = crossentropy_loss(t, y)
            losses_train.append(loss)
            mlp.backward(t, y)
            mlp.step()
            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            train_num += x.shape[0]
            train_true_num += acc

        for x, t in zip(x_val_batches, t_val_batches):
            y = mlp(x)
            loss = crossentropy_loss(t, y)
            losses_valid.append(loss)
            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            valid_num += x.shape[0]
            valid_true_num += acc

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_true_num / train_num,
            np.mean(losses_valid),
            valid_true_num / valid_num
        ))



if __name__ == '__main__':
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, '..', 'Data', 'Lecture03_dataset')

    # 学習データの読み込み
    x_train_path = os.path.join(path, 'x_train.npy')
    t_train_path = os.path.join(path, 'y_train.npy')
    x_train = np.load(x_train_path)
    t_train = np.load(t_train_path)

    # テストデータの読み込み
    x_test_path = os.path.join(path, 'x_test.npy')
    x_test = np.load(x_test_path)

    # データの前処理
    x_train, x_test = x_train / 255., x_test / 255.
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    t_train = np.eye(N=10)[t_train.astype("int32").flatten()]

    # データの分割
    x_train, x_val, t_train, t_val = \
        train_test_split(x_train, t_train, test_size=10000, random_state=42)

    # ハイパーパラメータとモデルの初期化
    lr = 0.01
    momentum = 0.9
    n_epochs = 30
    batch_size = 128

    rng = np.random.RandomState(1234)
    mlp = Model(rng, lr=lr, momentum=momentum)

    # モデルの学習
    train_model(mlp, x_train, t_train, x_val, t_val, batch_size, n_epochs)

    # テストデータでの予測
    t_pred = []
    for x in x_test:
        x = x[np.newaxis, :]
        y = mlp(x)
        pred = y.argmax(1).tolist()
        t_pred.extend(pred)

# 動作確認済み