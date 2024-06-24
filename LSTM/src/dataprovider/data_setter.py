import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from base.base_data_setter import BaseDataSetter

class StockDataSetter(BaseDataSetter):
    def __init__(self, csv_file, index_column='date', value_column='volume', seq_length=60):
        self.index_column = index_column
        self.value_column = value_column
        self.seq_length = seq_length
        self.data = pd.read_csv(csv_file)
        self.data = self.preprocess()
        self._create_sequences()

    def preprocess(self):
        mscaler = MinMaxScaler(feature_range=(-1, 1))
        sscaler = StandardScaler()

        data = self.data
        data = data[data["Name"] == "AAPL"]
        if 'Name' in self.data.columns:
            self.data.drop('Name', axis=1, inplace=True)
            self.data.drop(self.index_column, axis=1, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        #data[self.index_column] = pd.to_datetime(self.data[self.index_column])
        #data.set_index(self.index_column, inplace=True)
        data[self.value_column] = data[self.value_column].astype(float)
        data.iloc[:, self.data.columns != self.value_column] = sscaler.fit_transform(data.iloc[:, self.data.columns != self.value_column].values)
        data[self.value_column] = mscaler.fit_transform(data[self.value_column].values.reshape(-1, 1))

        data = data[[self.value_column]]
        return data

    def _create_sequences(self):
        xs, ys = [], []
        for i in tqdm(range(len(self.data) - self.seq_length)):
            x = self.data.iloc[i:i+self.seq_length]
            y = self.data.iloc[i+self.seq_length]
            xs.append(x)
            ys.append(y)
        self.xs = np.array(xs)
        self.ys = np.array(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return np.reshape(x, (np.shape(x)[0], 1)), y