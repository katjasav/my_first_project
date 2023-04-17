import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
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

activation = nn.ReLU
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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
recall = BinaryRecall()

loaders = {"train": train_dataloader, "valid": valid_dataloader}

max_epochs = 100
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

# оценка результатов
model.eval()
y_train_pred = model(X_train_t).max(1).indices.detach().numpy()
y_val_pred = model(X_val_t).max(1).indices.detach().numpy()
train_recall = recall_score(y_train_n, y_train_pred)
test_recall = recall_score(y_test_n, y_val_pred)
print(f'Recall на train: {round(train_recall, 4)}, Recall на test: {round(test_recall, 4)}')

y_train_predicted = model(X_train_t)[:, 1].detach().numpy()
y_test_predicted = model(X_val_t)[:, 1].detach().numpy()
train_auc = roc_auc_score(y_train_n, y_train_predicted)
test_auc = roc_auc_score(y_test_n, y_test_predicted)

plt.figure(figsize=(10,7))
plt.plot(*roc_curve(y_train, y_train_predicted)[:2], label='train AUC={:.4f}'.format(train_auc))
plt.plot(*roc_curve(y_test, y_test_predicted)[:2], label='test AUC={:.4f}'.format(test_auc))
legend_box = plt.legend(fontsize='large', framealpha=1).get_frame()
legend_box.set_facecolor("white")
legend_box.set_edgecolor("black")
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))
plt.show()
