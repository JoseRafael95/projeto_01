## biblioteca
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
#from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
import joblib
from joblib import load, dump
import time

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 

modelo_xgb = load('./modelo_xgb.joblib') # persistência do modelo
validacao_cruzada_resultado = load("validacao_cruzada_resultado.pkl") #persistência dos resultados


# Abas da pagina, criação e atribuindo nomes.
tab1, tab2, tab3,  = st.tabs(["Sobre os Dados", "Análise de Dados", "Modelo de Machine Learning"])



with tab1: # Aba 1 -  Sobre os Dados 
    #Título
    st.title("Previsão de Valores de Imóveis Usando Modelos de Regressão")
    st.write("Este projeto utiliza dados do Kaggle: **House Prices - Advanced Regression Techniques**.")
    st.write("O objetivo é analisar as características dos imóveis e prever seus valores de venda.")

    #Carregamento e Visualização dos Dados
    st.subheader("Visualização dos Dados de Treinamento")
    df_train = pd.read_csv("Dados/train.csv")
    st.write(df_train.head(5))
    st.caption("(Tabela 1: Exibição das primeiras 5 linhas dos dados brutos, sem pré-processamento)")

    df1 = df_train.copy()

    # Porcentagem de Linhas Nulas por Coluna
    st.subheader("Porcentagem de Valores Nulos por Coluna")
    nulos = 100*(df1.isnull().sum()/ len(df1))
    nulos = nulos.sort_values(ascending=False)
    nulos = nulos[nulos>0]
    nulos
    st.caption("(Tabela 2: Porcentagem de valores nulos por coluna)")

    st.write("Algumas colunas possuem valores nulos que representam a ausência de alguma característica na casa.\
    Nesses casos, os valores nulos foram substituídos por categorias como 'Sem item'.")

    # Substituição de valores nulos com base nas características
    df1['MasVnrType'] = df1['MasVnrType'].fillna('Nm')
    df1['Fence'] = df1['Fence'].fillna('Na')
    df1['Alley'] = df1['Alley'].fillna('Na')
    df1['PoolQC'] = df1['PoolQC'].fillna('Np')
    df1['FireplaceQu'] = df1['FireplaceQu'].fillna('Nl')

    df1['BsmtCond'] = df1['BsmtCond'].fillna('Nb')
    df1['BsmtFinType1'] = df1['BsmtFinType1'].fillna('Nb')
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

    # Exibir colunas restantes com valores nulos
    st.write("Revisando a porcentagem de valores nulos após o tratamento inicial:")
    nulos = (df1.isnull().sum()/ len(df1))
    nulos = nulos.sort_values(ascending=False)
    nulos = nulos[nulos>0]
    st.write(nulos)
    st.caption("(Tabela 3: Porcentagem de valores nulos por coluna)")
    # Análise de colunas específicas 
    st.write("As colunas com poucos valores nulos foram tratadas com a média ou o valor mais frequente, conforme a natureza da variável.")
    aux = df1[['LotFrontage','MasVnrArea','Electrical','GarageYrBlt']]
    aux.info()

    #Como os valores ausentes são relativamente poucos, foram substituídos pela média
    df1['LotFrontage'] = df1['LotFrontage'].fillna(df1['LotFrontage'].mean())
    df1['MasVnrArea'] = df1['MasVnrArea'].fillna(df1['MasVnrArea'].mean())
    df1['GarageYrBlt'] = df1['GarageYrBlt'].fillna(df1['GarageYrBlt'].mean())

    # A coluna Electrical é composta por strings, portanto, não pode ser substituída pela média
    df1['Electrical']=df1['Electrical'].fillna('Nulo')
    sns.barplot(x='Electrical', y= 'SalePrice',data=df1)
    df1['Electrical'] = df1['Electrical'].replace('Nulo','SBrkr')

    # Exibir histograma
    st.subheader("Distribuição das Vendas")
    fig, ax = plt.subplots(figsize=(15, 5)) 
    sns.histplot(df1['SalePrice'], kde=True, ax=ax) 
    plt.title("Histograma - Vendas")
    st.pyplot(fig)  
    st.caption("(Gráfico 1: Distribuição assimétrica positiva dos preços de venda)")

    # Exibir descrição estatística
    st.write(df1['SalePrice'].describe())
    st.caption("(Tabela 4: Estatísticas descritivas dos preços de venda)")

    # Divisão dos dados em categóricos e numéricos
    df2 = df1.copy()
    categoricos= df2.select_dtypes(exclude=['float64', 'int64'])
    numericos= df2.select_dtypes(include=['float64', 'int64'])

    # Exibir correlações com o preço de venda
    corr = numericos.corr()
    st.write("Correlação das features numéricas com o preço de venda:")
    st.write(corr['SalePrice'].abs().sort_values(ascending=False))
    st.caption("(Tabela : Correlação entre as variáveis e o Preço de venda)")

with tab2: # Aba 2 - Tela de Análise de Dados

    #Título
    st.title("Análise Exploratória dos Dados de Imóveis")

    # Subtítulo e explicação inicial
    st.subheader("Distribuição das Variáveis")
    st.write("Nesta seção, realizamos uma análise exploratória das variáveis do conjunto de dados, visando identificar padrões,\
         distribuições e possíveis outliers.")
    
    # Observação
    st.markdown("<span style='color:red'>Observação: Para simplificar e otimizar o processo dentro do Streamlit, utilizei o algoritmo Boruta,\
         que me permitiu identificar e focar apenas nas variáveis com maior relevância para a análise.</span>", unsafe_allow_html=True)

    # Variáveis mais relevantes segundo o Boruta ('LotFrontage',LotArea','OverallQual', 'YearBuilt', 
    # 'YearRemodAdd','MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','GrLivArea', 
    # 'FullBath', 'TotRmsAbvGrd', 'Neighborhood', 'BsmtQual', 'CentralAir', 'GarageType'

    ## Caracteristicas do Terreno 

    st.title("Características do Terreno")
    fig, ax = plt.subplots(figsize=(12, 5))  
    plt.subplot(1, 2, 1)  
    sns.boxplot(df2['LotFrontage'])
    plt.subplot(1, 2, 2)  
    sns.boxplot(df2['LotArea'])
    st.pyplot(fig)
    st.caption("(Gráfico 1: Boxplots mostrando a distribuição de 'LotFrontage' e 'LotArea')")

    descricoes = pd.DataFrame({
    'LotFrontage': df1['LotFrontage'].describe(),
    'LotArea': df1['LotArea'].describe()
    })
    st.write(descricoes)
    st.caption("(Tabela 1: Estatísticas descritivas para 'LotFrontage' e 'LotArea')")
    plt.figure(figsize=(15,9))

    st.write("Considerações:")
    st.write("Observa-se uma concentração significativa de frentes de lote entre 60 e 79 pés. Com base nos quartis, valores\
        abaixo de 31,5 pés e acima de 107,7 pés são considerados outliers, indicando propriedades com frentes menores ou maiores que o usual.\
         Em relação à área total dos lotes, há uma variação de 7.550 a 11.600 pés quadrados, demonstrando uma diversidade nos tamanhos das propriedades analisadas.")
    st.write("---")

    ## Características da Estrutura 
    st.title("Características da Estrutura")

    fig, ax = plt.subplots(figsize=(10, 10))
    lista = ['OverallQual', 'YearBuilt', 'YearRemodAdd',
             'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
             'GrLivArea']   

    # Criar boxplots para as variáveis numéricas
    for i, var in enumerate(lista):
        plt.subplot(3, 3, i + 1)  # i + 1 porque o índice começa em 0
        sns.boxplot(y=df2[var])

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    st.pyplot(fig)
    st.caption("(Gráfico 2: Boxplots das variáveis numéricas mais relevantes segundo o algoritmo Boruta)")

    descricoes = pd.DataFrame({
    'OverallQual': df1['OverallQual'].describe(),
    'YearBuilt': df1['YearBuilt'].describe(),
    'YearRemodAdd': df1['YearRemodAdd'].describe(),
    'MasVnrArea': df1['MasVnrArea'].describe(),
    'BsmtFinSF1': df1['BsmtFinSF1'].describe(),
    'TotalBsmtSF': df1['TotalBsmtSF'].describe(),
    '1stFlrSF': df1['1stFlrSF'].describe(),
    '2ndFlrSF': df1['2ndFlrSF'].describe(),
    'GrLivArea': df1['GrLivArea'].describe()
    
    })
    st.write(descricoes)
    st.caption("(Tabela 2: Estatísticas descritivas das variáveis estruturais mais relevantes)")
    plt.figure(figsize=(15,9))
    
    st.write("Considerações:")
    st.write("A qualidade das casas está concentrada entre 5 e 7, indicando uma estrutura de qualidade moderada a boa.\
         O ano de construção varia entre 1954 e 2000, sugerindo uma mistura de imóveis antigos e relativamente novos. A área útil,\
         que varia de 1.130 a 1.777 pés quadrados, reflete o tamanho típico das casas dessa amostra.")
    st.write("---")

    ## Características de Banheiros e Cômodos
    st.title("Características de Banheiros e Cômodos") 
    fig, ax = plt.subplots(figsize=(13, 8)) 
    plt.subplot(1,2,1)
    sns.boxplot(df2['FullBath'])
    plt.subplot(1,2,2)
    sns.boxplot(df2['TotRmsAbvGrd'])
    st.pyplot(fig)
    st.caption("(Gráfico 3: Boxplots mostrando a distribuição de 'FullBath' e 'TotRmsAbvGrd')")
    descricoes = pd.DataFrame({
    'FullBath': df1['FullBath'].describe(),
    'TotRmsAbvGrd': df1['TotRmsAbvGrd'].describe()
    })
    st.write(descricoes)
    st.caption("(Tabela 3: Estatísticas descritivas para 'FullBath' e 'TotRmsAbvGrd')")
    plt.figure(figsize=(15,9))
   
    st.write("Considerações:")
    st.write("Os imóveis geralmente possuem entre 1 e 2 banheiros, com algumas casas apresentando até 3.\
         A quantidade de cômodos varia de 5 a 7.")
    st.write("---")

    ## Características Categóricas
    st.title("Características Categóricas")
    fig, ax = plt.subplots(figsize=(13, 12))
    plt.subplot(2,2,1)
    sns.boxplot(df2['Neighborhood'])
    plt.subplot(2,2,2)
    sns.boxplot(df2['BsmtQual'])
    plt.subplot(2,2,3)
    sns.boxplot(df2['CentralAir'])
    plt.subplot(2,2,4)
    sns.boxplot(df2['GarageType'])
    st.pyplot(fig)
    st.caption("(Gráfico 4: Boxplots mostrando a distribuição de variáveis categóricas como 'Neighborhood', 'BsmtQual', 'CentralAir' e 'GarageType')")


    descricoes = pd.DataFrame({
    'Neighborhood': df1['Neighborhood'].describe(),
    'BsmtQual': df1['BsmtQual'].describe(),
    'CentralAir': df1['CentralAir'].describe(),
    'GarageType': df1['GarageType'].describe()
    })
    st.write(descricoes)
    st.caption("(Tabela 4: Estatísticas descritivas para variáveis categóricas selecionadas)")
    st.write("---")

    ### Análise Bivariada
    st.subheader("Análise Bivariada")
    st.write("O objetivo principal aqui é entender o impacto de cada variável no valor de venda dos imóveis.")

    plt.figure(figsize=(20,10))

    plt.subplot(2,3,1)
    sns.histplot(x=df1['LotFrontage'],y=df1['SalePrice'])

    plt.subplot(2,3,2)
    sns.histplot(x=df1['LotArea'],y=df1['SalePrice'])

    plt.subplot(2,3,3)
    sns.histplot(x=df1['OverallQual'],y=df1['SalePrice'])

    plt.subplot(2,3,4)
    sns.histplot(x=df1['YearBuilt'],y=df1['SalePrice'])

    plt.subplot(2,3,5)
    sns.histplot(x=df1['Neighborhood'],y=df1['SalePrice'])

    plt.subplot(2,3,6)
    sns.histplot(x=df1['GarageCars'],y=df1['SalePrice'])
    st.pyplot(plt)
    st.caption("(Gráfico 5: Análise bivariada das variáveis em relação ao preço de venda dos imóveis)")

    st.write("Considerações:")
    st.write("As variáveis, como número de vagas de garagem, qualidade do imóvel, área total e ano de construção,\
         mostram uma forte correlação positiva com o valor de venda. No entanto, a frente do lote apresenta uma relação mais fraca.")
    st.write("---")

    ### Análise Multivariada - Numérica
    st.subheader("Análise Multivariada - Numérica")

    plt.figure(figsize=(25,15))
    corr1= df2[['OverallQual', 'YearBuilt', 'YearRemodAdd','MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','GrLivArea']].corr()
    sns.heatmap(corr1)
    st.pyplot(plt)
    st.caption("(Gráfico 6: Mapa de calor mostrando a correlação entre variáveis numéricas relevantes)")


    st.write("Considerações:")
    st.write("Identificamos uma correlação considerável entre 1stFlrSF (Área do primeiro andar) e TotalBsmtSF (Área total do porão).\
         Também notamos que 2ndFlrSF (Área do segundo andar) e GrLivArea (Área útil acima do solo) apresentam uma correlação leve.\
         Outras relações são mais fracas.")
    st.write("---")


with tab3:#Aba 3 - Modelo de Machine Learning
    
    # Titulo 
    st.header("Preparação dos Dados e Apicação das Machine Learning ")

    # Divisão dos dados em numéricos e categóricos
    df3 = df2.copy()
    df3_num = df2.select_dtypes(include=['float64', 'int64'])
    df3_num = df3_num.drop(['SalePrice'],axis=1)
    df3_cat = df2.select_dtypes(exclude=['float64', 'int64'])


    # Tranformação dos dados - Transformação Logarítmica
    st.subheader("Preparação dos dados - Traformação")
    st.write("Foi aplicada a transformação logarítmica nos valores da variável 'SalePrice', com o objetivo de reduzir\
         a assimetria e aproximar a distribuição dos dados de uma normal, facilitando o ajuste dos modelos preditivos.")
   
    # Histograma do SalePrice apos a tranformação log
    plt.figure(figsize=(10,5))
    df3['SalePrice_log'] = np.log1p(df3['SalePrice'])
    sns.histplot(df3['SalePrice_log'],kde=True)
    st.pyplot(plt)
    st.caption("(Gráfico 1: Histograma da variável 'SalePrice' após transformação logarítmica)")

    # Tranformação dos dados - Normalização
    st.subheader("Preparação dos dados - Normalização")
    # Normalização dos dados
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df3_num)
    # Converter o resultado de volta para um DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=df3_num.columns)
    df3_num = df_scaled
    st.write("Foi aplicada a técnica de normalização MinMaxScaler aos dados numéricos, escalando-os para o intervalo [0, 1],\
    a fim de uniformizar as variáveis e otimizar o desempenho dos modelos de machine learning.")
    st.write(df3_num)
    st.caption("(Tabela 1: Dados numéricos normalizados usando MinMaxScaler)")

    # Tranformação dos dados - Encoding
    st.subheader("Preparação dos dados - Encoding")
    st.write("Foi aplicado o processo de codificação (encoding) nas variáveis categóricas, convertendo-as em representações\
        numéricas para que pudessem ser interpretadas pelos modelos de machine learning.")
    df3_cat= pd.get_dummies(df3_cat, columns=df3_cat.columns).astype(int)
    st.write(df3_cat)
    st.caption("(Tabela 2: Dados categóricos codificados usando One-Hot Encoding)")

    # Resultado Final
    st.subheader("Dados para o treinamento")
    st.write("Após a aplicação das transformações nos dados numéricos e categóricos, os datasets foram combinados em um único conjunto final,\
         preparado para o treinamento do modelo de machine learning.")
    df4 = pd.concat([df3_num,df3_cat],axis=1)
    st.write(df4)
    st.caption("(Tabela 3: Conjunto final de dados transformados)")

    ## Divisão em treino e teste

    # Sem Tratamento 
    a = df3.drop(['SalePrice'],axis=1)
    b = df3['SalePrice']
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)
    a_test = a_test.reset_index(drop=True)
    b_test = b_test.reset_index(drop=True)
    #Com Tratamento
    x = df4
    y = df3['SalePrice_log']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    st.write("---")


    #### Modelo --  Modelo de Regressão  
    st.subheader("Modelo XGB Regressor")
    

    st.markdown("<h4>Fine Tunning - GridSearchCV</h4>", unsafe_allow_html=True)
    st.write("Parametros:")
    st.write("**alpha=0.1,**\n\n""**colsample_bytree=0.5,**\n\n""**reg_lambda=0,**\n\n""**learning_rate=0.05,**\n\n"\
        "**max_depth=3,**\n\n""**n_estimators=300,**\n\n""**subsample=0.5**")

    
    st.write("---")
    with st.spinner('Carregando o Validação Cruzada...'):
        
        st.markdown("<h4>Cross-Validate</h4>", unsafe_allow_html=True)
    

    # Resultados validação cruzada
    st.write(f"Resultado da validação cruzada (R2): {validacao_cruzada_resultado['r2']}")
    st.write(f"Resultado da validação cruzada (MSE): {validacao_cruzada_resultado['mse']}")
    st.write(f"Resultado da validação cruzada (RMSE): {validacao_cruzada_resultado['rmse']}")

    # Validação
    st.write("---")
    st.markdown("<h4>Validation</h4>", unsafe_allow_html=True)
    
    y_pred = modelo_xgb.predict(x_test)


    r2 = r2_score(y_test, y_pred)
    st.write(f"R²:, {r2}")

    mse = mean_squared_error(y_test, y_pred)
    st.write(f"MSE: {mse}")

    rmse = np.sqrt(mse)
    st.write(f"RMSE:, {rmse}")

    # Grafico de previsão 
    st.markdown("<h4>Gráfico - Previsão x Real</h4>", unsafe_allow_html=True)
    plt.figure(figsize=(7,7))
    sns.scatterplot(x=b_test, y=np.exp(y_pred), color="blue", label= "Previsão")
    plt.plot([min(b_test),max(b_test)],[min(b_test),max(b_test)], color="red", label="Previsto = Real")
    plt.ylabel("Previsto")
    plt.xlabel("Real")
    plt.legend()
    st.pyplot(plt)
    st.caption("(Gráfico 2: Comparação entre valores previstos e valores reais)")

    ### Previsão 
    
    st.write("---")
    st.markdown("<h4>Previsões </h4>", unsafe_allow_html=True)

    def funcao_previsao(indice, titulo):    
        st.write(titulo)
        st.write("Nesta linha temos as seguintes variáveis:")
        st.write(
        "Tamanho Frontal do Lote = ", a_test.loc[indice, 'LotFrontage'], ", ",
        "Área Total Lote = ", a_test.loc[indice, 'LotArea'], ", ",
        "Qualidade Geral = ", a_test.loc[indice, 'OverallQual'], ", ",
        "Ano de Construção = ", a_test.loc[indice, 'YearBuilt'], " ",
        "Bairro = ", a_test.loc[indice, 'Neighborhood'])
        st.table(a_test.iloc[[indice]])
        st.write("O modelo previu um valor de ", int(np.exp(y_pred[indice]))," enquanto o valor real observado foi de ", int(b_test.iloc[indice]))
        desvio = (float(np.exp(y_pred[indice])) - float(b_test.iloc[indice]))
        st.write("indicando uma diferença de ",int(desvio)," entre a previsão e o valor real.")
    
    ### Aplicação
    funcao_previsao(20,"**Previsao 1 **")
    funcao_previsao(30,"**Previsao 2 **")
    
      
 

