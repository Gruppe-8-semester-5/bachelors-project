import torch


X, y  = fashion_mnist_X_y()
(X_train, y_train), (X_test, y_test) = make_train_and_test_sets(X, y, 0.8)

X_train, y_train, X_test, y_test = map(
    torch.tensor, (X_train, y_train, X_test, y_test)
)