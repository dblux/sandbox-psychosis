import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Data preprocessing ###

# Data versions
# v1: Before batch correction
file = 'data/tmp/sehwan/lyriks_265_395.csv'
# file = 'data/tmp/sehwan/lyriks_214_395.csv'
# file = 'data/tmp/sehwan/lyriks_50_395.csv'
lyriks = pd.read_csv(file, index_col=0)
lyriks = lyriks.T

file = 'data/astral/metadata/metadata-psy_602_16-v1.csv'
metadata = pd.read_csv(file, index_col=0)
metadata_lyriks = metadata.loc[lyriks.index]

# Split into batches according to extraction_date 
# Batch 1 := '4/9/24' - Classes: ['Control', 'UHR']
# Batch 2 := '5/9/24' - Classes: ['Control', 'UHR', 'FEP'] 
# Batch 3 := '28/8/24' - Classes: ['Control']

batch2 = lyriks[metadata_lyriks.extraction_date == '4/9/24']
# Remove 'FEP' state samples from batch 2
batch1 = lyriks[metadata_lyriks.extraction_date == '5/9/24']
# batch1 = lyriks[
#     (metadata_lyriks.extraction_date == '5/9/24') &
#     (metadata_lyriks.state != 'FEP')
# ]

# Split batch 1 into train and test
# Split stratified by class labels and partitioned by sn
metadata_b1 = metadata_lyriks.loc[batch1.index]
b1_sn = (
    metadata_b1.
        sort_values('timepoint').
        groupby('sn')[['timepoint', 'state']].
        last()
)
train_idx, test_idx = train_test_split(
    b1_sn.index,
    test_size=0.2,
    stratify=b1_sn.state,
    random_state=42
)

batch1_train = batch1.loc[metadata_b1.sn.isin(train_idx)]
batch1_test = batch1.loc[metadata_b1.sn.isin(test_idx)]

encoder = {
    'Control': 0,
    'UHR': 1,
    'FEP': 2
}
state_b1_train = metadata_lyriks.loc[batch1_train.index, 'state'].map(encoder)
state_b1_test = metadata_lyriks.loc[batch1_test.index, 'state'].map(encoder)
state_b2 = metadata_lyriks.loc[batch2.index, 'state'].map(encoder)

# Z-scaling
scaler = StandardScaler()
scaled_b1_train = scaler.fit_transform(batch1_train)
scaled_b1_test = scaler.fit_transform(batch1_test)
scaled_b2 = scaler.fit_transform(batch2)

X_b1_train = torch.tensor(scaled_b1_train, dtype=torch.float32)
X_b1_test = torch.tensor(scaled_b1_test, dtype=torch.float32)
X_b2 = torch.tensor(scaled_b2, dtype=torch.float32)

y_b1_train = torch.tensor(state_b1_train, dtype=torch.long)
y_b1_test = torch.tensor(state_b1_test, dtype=torch.long)
y_b1_train = torch.tensor(state_b1_train, dtype=torch.long)


### Model architecture ###

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = lyriks.shape[1]
n_classes = 3
model = MLP(input_dim=input_dim, hidden_dim=16, n_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)


epochs = 20
for epoch in range(epochs):
    # train
    _ = model.train()
    y_hat_train = model(X_b1_train)
    loss = criterion(y_hat_train, y_b1_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # validation
    if (epoch + 1) % 5 == 0 or epoch == 0:
        _ = model.eval()
        print('=' * 30)
        print(f'Epoch {epoch+1}/{epochs}')
        print('=' * 30)
        print(f'Loss: {loss.item():.4f}')
        with torch.no_grad():
            y_hat_val = model(X_b1_test)
            val_loss = criterion(y_hat_val, y_b1_test)
            print(f'Validation loss: {val_loss.item():.4f}')
            print()


### Evaluation ###

y_pred_val = torch.argmax(y_hat_val, dim=1)
acc_macro = balanced_accuracy_score(y_b1_test, y_pred_val)
f1_macro = f1_score(y_b1_test, y_pred_val, average='macro')
acc_micro = accuracy_score(y_b1_test, y_pred_val)

print(f'Evaluation (n = {y_pred_val.shape[0]})')
print('-' * 30)
cm_raw = confusion_matrix(y_b1_test, y_pred_val)
# print(torch.unique(y_b1_test, return_counts=True))
# labels0 = list(encoder.keys())[:2]
# labels1 = ['Predicted: ' + l for l in labels0]
print(pd.DataFrame(cm_raw))
print()

print(f'Micro Accuracy: {acc_micro:.4f}')
print(f'Macro Accuracy (Balanced): {acc_macro:.4f}')
print(f'Macro F1 Score: {f1_macro:.4f}')
