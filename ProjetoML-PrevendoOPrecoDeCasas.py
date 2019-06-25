#Importando os modulos necessarios
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as matplot
import sklearn

#Importando dataset que ja esta disponivel no sckit-learn , Precisamos apenas carrega-lo
from sklearn.datasets import load_boston
boston = load_boston()
type(boston)

#Visualizando o shape do dataset
boston.data.shape

# Descriçao do dataset
print(boston.DESCR)
print(boston.feature_names)


#Convertendo o dataset para um DataFrame
df = pd.DataFrame(boston.data)
df.head()

#Alterando o nome das colunas
df.columns = boston.feature_names
df.head()

#boston.target é um array com o preço das casas
boston.target

#Incluindo a variavel target ao dataframe
df["PRICE"] = boston.target
df.head()
df.tail()


#Importando o módulo de regressao linear
from sklearn.linear_model import LinearRegression

#Nao queremos os preços das casas como variavel depedente
X = df.drop("PRICE", axis=1)

#Definindo o Y
Y = df.PRICE


#Plotando um grafico com os dados
matplot.scatter(df.RM, Y)
matplot.xlabel("Média de Numero de quartos por casa")
matplot.ylabel("Preço da Casa")
matplot.title("Relaçao entre número de quartos e Preço")
matplot.show()

#Criando o modelo
regression = LinearRegression()
type(regression)

#Treinando o modelo
regression.fit(X, Y)

#Coeficientes
print("Coeficiente: ", regression.intercept_)
print("Numero de coeficientes: ", len(regression.coef_))
print("Score: ", regression.score(X,Y))


#Prevendo o preço da casa
regression.predict(X)


#Comparando preços originais X preços previstos
matplot.scatter(df.PRICE, regression.predict(X))
matplot.xlabel("Preço Original")
matplot.ylabel("Preço Previsto")
matplot.title("Preço original VS Preço previsto")
matplot.show()


#Podemos ver que existem alguns erros na prediçao do preço das casas

#Vamos calcular o MSE (Mean Squared error)
mse1 = np.mean((df.PRICE - regression.predict(X)) ** 2)
print(mse1)


#Aplicando regressao Linear para uma variavel e calculando o MSE
regression.fit(X[['PTRATIO']], df.PRICE)
mse2 = np.mean((df.PRICE - regression.predict(X[['PTRATIO']]))**2)
print(mse2)

'''
 Neste caso nota-se que o MSE aumentou indicando que uma unica caracteristica
 nao é bom predictor para o preço das casas.
''' 

'''
  Na pratica, divide-se o dataset em datasets de treino e de teste.Assim 
  o modelo é treinado nos dados de treino e depois verifica como o modelo 
  se comporta nos seus dados de teste  
'''

#DIVISAO DE FORMA MANUAL
#Dividindo X em dados de treino e teste
X_treino = X[:-50]
X_teste = X[-50:]

#Dividindo Y em dados de treino e teste
Y_treino = df.PRICE[:-50]
Y_teste = df.PRICE[-50:]

#Imprimindo o shape dos datasets
print(X_treino.shape, X_teste.shape, Y_treino.shape, Y_teste.shape)


#DIVISAO RANDOMICA (IDEAL)

from sklearn.model_selection import train_test_split

#Dividindo X e Y em dados de treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, df.PRICE, 
                                     test_size = 0.30, random_state = 5)

#Imprimindo o shape dos datasets
print(X_treino.shape, X_teste.shape, Y_treino.shape, Y_teste.shape)


#Construindo o modelo de regressao Linear
regression = LinearRegression()

#Treinando o modelo
regression.fit(X_treino, Y_treino)

predict_treino = regression.predict(X_treino)
predict_teste = regression.predict(X_teste)


#Comparando preços originais X preços previstos
matplot.scatter(regression.predict(X_treino), regression.predict(X_treino) - Y_treino, c = "b", s = 40, alpha=0.5)
matplot.scatter(regression.predict(X_teste), regression.predict(X_teste) - Y_teste, c = "g", s = 40, alpha=0.5)
matplot.hlines(y=0, xmin=0, xmax=50)
matplot.ylabel("Resíduo")
matplot.title("Residual Plot - Treino(Azul), Teste(Verde)")
matplot.show()


