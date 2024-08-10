import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class mnist_dataset(Dataset):
    # def __init__(self,dataframe):
    #     super().__init__()

    #     self.y = dataframe['label']
    #     X = dataframe.drop(['label'],axis=1)

    #     tensor_list = [torch.tensor(np.array(X.iloc[i]).reshape(1,1, 28, 28),dtype=torch.float32) for i in range(len(self.y))]
    #     self.X_tensor = torch.cat(tensor_list, dim=0)
    #     self.y_tensor = torch.tensor(np.array(self.y))

    
    # def __len__(self):
    #     return len(self.y)

    # def __getitem__(self,idx):
    #     return self.X_tensor[idx],self.y_tensor[idx].unsqueeze(0)

    def __init__(self, dataframe):
        super().__init__()

        self.y = dataframe['label'].values
        X = dataframe.drop(['label'], axis=1).values.reshape(-1, 1, 28, 28)  # Reshape directly in numpy

        # Convert entire arrays to tensors
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long)  # Ensure long type for classification

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]
    
class mnist_test(Dataset):
    def __init__(self, dataframe):
        super().__init__()

        # self.y = dataframe['label'].values
        X = dataframe.values.reshape(-1, 1, 28, 28)  # Reshape directly in numpy

        # Convert entire arrays to tensors
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        # self.y_tensor = torch.tensor(self.y, dtype=torch.long)  # Ensure long type for classification

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx]

if __name__ == "__main__":
    df = pd.read_csv('data/train.csv')
    daset = mnist_dataset(df)

    for x,y in daset:
        print(x.shape)
        print(y.shape)
        break

    te = pd.read_csv('data/test.csv')
    te_dat = mnist_test(te)
    for x in te_dat:
        print(x.shape)
        break