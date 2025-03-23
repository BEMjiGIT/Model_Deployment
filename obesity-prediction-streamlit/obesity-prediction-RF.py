import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import pickle as pkl

class ModelRF:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.fitted_model = None
        
    def data_split(self, target_column):
        self.X = self.data.drop([target_column], axis=1)
        self.y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
    
    def data_preprocessing(self, scaler, encoder):
        self.encoder = encoder
        self.scaler = scaler
        
        categories = sorted(self.y.unique())
        self.target_vals = {label: idx for idx, label in enumerate(categories)}
        self.y_train = self.y_train.map(self.target_vals)
        self.y_test = self.y_test.map(self.target_vals)
        
        categorical_cols = self.X_train.select_dtypes(exclude=np.number).columns
        numerical_cols = self.X_train.select_dtypes(include=np.number).columns
        X_train_encoded = pd.DataFrame(self.encoder.fit_transform(self.X_train[categorical_cols]), columns=encoder.get_feature_names_out())
        X_test_encoded = pd.DataFrame(self.encoder.transform(self.X_test[categorical_cols]), columns=encoder.get_feature_names_out())
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.X_train[numerical_cols]), columns=self.X_train.select_dtypes(include=np.number).columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(self.X_test[numerical_cols]), columns=self.X_test.select_dtypes(include=np.number).columns)
        
        self.X_train = pd.concat([X_train_encoded, X_train_scaled], axis=1)
        self.X_test = pd.concat([X_test_encoded, X_test_scaled], axis=1)
        
        return self.target_vals
        
    def train(self, model, param_grid):
        self.fitted_model = GridSearchCV(model, param_grid, cv=5)
        self.fitted_model.fit(self.X_train, self.y_train)

    def evaluate(self, pred):
        print(f'Classification Report :\n{classification_report(self.y_test, pred)}')
        
    def test(self):
        if self.fitted_model == None:
            raise ValueError("No Model Found!")
        pred = self.fitted_model.predict(self.X_test)
        self.evaluate(pred)
    
    def best_params(self):
        if self.fitted_model == None:
            raise ValueError("No Model Found!")
        return self.fitted_model.best_params_
    
    def export_model(self, filename):
        if self.fitted_model == None:
            raise ValueError("No Model Found!")
        with open(filename, 'wb') as file:
            pkl.dump(self.fitted_model, file)

    def export_scaler(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.scaler, file)
        
    def export_encoder(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.encoder, file)
    
    def export_target_vals(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.target_vals, file)
    
    @staticmethod
    def import_model(filename):
        with open(filename, 'rb') as file:
            loaded_data = pkl.load(file)
        return loaded_data

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

model = ModelRF(data='ObesityDataSet_raw_and_data_sinthetic.csv')
model.data_split(target_column='NObeyesdad')
data_targets = model.data_preprocessing(scaler, encoder)
print(f'Data Targets: {data_targets}')

param_grid = {'n_estimators': [100, 300], 'max_depth': [None, 10]}
rf_model = RandomForestClassifier()
model.train(model=rf_model, param_grid=param_grid)
model.test()
model.best_params()

model.export_model('rf_model.pkl')
model.export_scaler('scaler.pkl')
model.export_encoder('encoder.pkl')
model.export_target_vals('target_vals.pkl')