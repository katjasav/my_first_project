import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
!pip install torchmetrics
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from torchmetrics.classification import BinaryRecall

seed = 17
# деление выборки на test (20%) и train (80%)
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], 
                                                       train_size=0.8, 
                                                       random_state=seed)
# перевод DataFrame в numpy
X_train_n = X_train.to_numpy()
y_train_n = y_train.to_numpy()
X_test_n = X_test.to_numpy()
y_test_n = y_test.to_numpy()

# перевод в тензоры
X_train_t =  torch.tensor(X_train_n, dtype=torch.float32)
y_train_t =  torch.tensor(y_train_n, dtype=torch.long)
X_val_t =  torch.tensor(X_test_n, dtype=torch.float32)
y_val_t =  torch.tensor(y_test_n, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

batch_size = 200
train_dataloader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def test_activation_function(activation):
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1*10,500),
    nn.BatchNorm1d(500),
    activation(),
    nn.Linear(500,300),
    nn.BatchNorm1d(300),
    activation(),
    nn.Linear(300,200),
    nn.BatchNorm1d(200),
    activation(),
    nn.Linear(200,100),
    nn.BatchNorm1d(100),
    activation(),
    nn.Linear(100,2)
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
    recall = BinaryRecall()

    loaders = {"train": train_dataloader, "valid": valid_dataloader}
    max_epochs = 30
    recall_lst = {"train": [], "valid": []}
    for epoch in range(max_epochs):
        for k, dataloader in loaders.items():
            epoch_rec = 0
            epoch_all = 0
            for x_batch, y_batch in dataloader:
                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                      outp = model(x_batch)
                preds = outp.argmax(-1)
                rec =  recall(preds, y_batch).float()
                all = 1
                epoch_rec += rec
                epoch_all += all
                if k == "train":
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()
            if k == "train":
                print(f"Epoch: {epoch+1}")
            print(f"Loader: {k}. Recall: {epoch_rec/epoch_all}")
            recall_lst[k].append(epoch_rec/epoch_all)

            scheduler.step()
    return recall_lst["valid"]

# тестирование: Sigmoid, ReLu, LeakyReLu
sigmoid_recall = test_activation_function(nn.Sigmoid)
relu_recall = test_activation_function(nn.ReLU)
leaky_relu_recall = test_activation_function(nn.LeakyReLU)

# визуализация
plt.figure(figsize=(16, 10))
plt.title("Valid recall")
plt.plot(range(max_epochs), relu_recall, label="ReLU activation", linewidth=2)
plt.plot(range(max_epochs), leaky_relu_recall, label="LeakyReLU activation", linewidth=2)
plt.plot(range(max_epochs), sigmoid_recall, label="Sigmoid activation", linewidth=2)
plt.legend()
plt.xlabel("Epoch")
plt.show()
