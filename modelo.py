# Importação das bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from joblib import dump
import pandas as pd
import warnings

# Ignorar avisos desnecessários
warnings.filterwarnings('ignore')

# Importação das bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from joblib import dump
import pandas as pd
import warnings

# Ignorar avisos desnecessários
warnings.filterwarnings('ignore')

# Leitura e cópia do DataFrame
df_train = pd.read_csv("Dados/train.csv")
df1 = df_train.copy()

    # Substituição de valores nulos com base nas características
df1['MasVnrType'] = df1['MasVnrType'].fillna('Nm')
df1['Fence'] = df1['Fence'].fillna('Na')
df1['Alley'] = df1['Alley'].fillna('Na')
df1['PoolQC'] = df1['PoolQC'].fillna('Np')
df1['FireplaceQu'] = df1['FireplaceQu'].fillna('Nl')
df1['BsmtCond'] = df1['BsmtCond'].fillna('Nb')
df1['BsmtFinType1'] = df1['BsmtFinType1'].fillna('Nb')
df1['BsmtCond'] = df1['BsmtCond'].fillna('Nb')
df1['BsmtQual'] = df1['BsmtQual'].fillna('Nb')
df1['BsmtExposure'] = df1['BsmtExposure'].fillna('Nb')
df1['BsmtFinType2'] = df1['BsmtFinType2'].fillna('Nb')
df1['GarageQual'] = df1['GarageQual'].fillna('Ng')
df1['GarageType'] = df1['GarageType'].fillna('Ng')
df1['GarageQual'] = df1['GarageQual'].fillna('Ng')
df1['GarageFinish'] = df1['GarageFinish'].fillna('Ng')
df1['GarageCond'] = df1['GarageCond'].fillna('Ng')

    # Excluir colunas com alto número de valores nulos, sem uma definição clara
df1 = df1.drop(['MiscFeature'],axis=1)

    #Como os valores ausentes são relativamente poucos, foram substituídos pela média
df1['LotFrontage'] = df1['LotFrontage'].fillna(df1['LotFrontage'].mean())
df1['MasVnrArea'] = df1['MasVnrArea'].fillna(df1['MasVnrArea'].mean())
df1['GarageYrBlt'] = df1['GarageYrBlt'].fillna(df1['GarageYrBlt'].mean())

    # A coluna Electrical é composta por strings, portanto, não pode ser substituída pela média
df1['Electrical']=df1['Electrical'].fillna('Nulo')
sns.barplot(x='Electrical', y= 'SalePrice',data=df1)
df1['Electrical'] = df1['Electrical'].replace('Nulo','SBrkr')

    # Divisão dos dados em categóricos e numéricos
df2 = df1.copy()
categoricos= df2.select_dtypes(exclude=['float64', 'int64'])
numericos= df2.select_dtypes(include=['float64', 'int64'])

    # Divisão dos dados em numéricos e categórico
df3 = df2.copy()
df3_num = df2.select_dtypes(include=['float64', 'int64'])
df3_num = df3_num.drop(['SalePrice'],axis=1)
df3_cat = df2.select_dtypes(exclude=['float64', 'int64'])
df3['SalePrice_log'] = np.log1p(df3['SalePrice'])
    # Criar o MinMaxScaler
scaler = MinMaxScaler()

    # Aplicar o scaler ao DataFrame
df_scaled = scaler.fit_transform(df3_num)
    # Converter o resultado de volta para um DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df3_num.columns)
df3_num = df_scaled

    # Tranformação dos dados - Encoding
df3_cat= pd.get_dummies(df3_cat, columns=df3_cat.columns).astype(int)

    # Resultado Final
df4 = pd.concat([df3_num,df3_cat],axis=1)

x = df4
y = df3['SalePrice_log']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Instanciação do modelo XGBRegressor
modelo_xgb = XGBRegressor(
    alpha=0.1, colsample_bytree=0.5, reg_lambda=0, learning_rate=0.05, 
    max_depth=3, n_estimators=300, subsample=0.5
)


# Treinamento do modelo
modelo_xgb.fit(x_train, y_train)


# Previsões e avaliação do modelo
y_pred = modelo_xgb.predict(x_test)

dump(modelo_xgb, 'modelo_xgb.joblib')

