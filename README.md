<p class="text-justify">
  
# 1 Introdução

Este trabalho propõe o desenvolvimento de uma técnica de rede neural artificial do tipo Multilayer Perceptron para predição da subscrição de depósitos a prazo por clientes bancários. A técnica proposta utiliza um pipeline para pré-processamento dos dados e posterior treinamento da rede com ajuste de hiperparâmetros. O dataset utilizado é disponibilizado publicamente por meio do link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#, com dados provenientes de um programa de telemarketing de um banco português. Os resultados alcançados foram considerados satisfatórios quando comparados aos resultados de outros trabalhos da literatura. 

Para o desenvolvimento deste estudo foram utilizadas algumas tecnologias, como Marchin Learn, K-means, Decision Tree e Pipeline.

# 2 Variáveis de Entrada e Saída

A seguir serão apresentadas as informações dos atributos utilizados no projeto, detalhando, inclusive, a categorização de cada variável.

## 2.1 Dados do cliente do banco:

* age: idade (numérico); 
* job: tipo de emprego (administrador, colarinho azul, empreendedor, empregada doméstica, gerenciamento, aposentado, autônomo, serviços, estudante, técnico, desempregado, desconhecido); 
* marital: estado civil (divorciado, casado, solteiro, desconhecido, nota: divorciado significa divorciado ou viúvo); 
* education: escolaridade (básico.4anos, básico.6 anos, básico.9 anos, ensino médio, analfabetos, curso profissional, grau universitário, desconhecido); 
* default: tem crédito inadimplente? (não, sim, desconhecido); 
* housing: tem crédito de habitação? (não, sim, desconhecido); 
* loan: tem empréstimo pessoal? (não, sim, desconhecido);

## 2.2 Dados relacionados com o último contato da campanha atual:

* contact: tipo de comunicação do contato (celular, telefone);
* month: último mês de contato do ano (jan, fev, mar, ..., nov, dec); 
* day of week: último dia de contato da semana (seg, ter, qua, qui, sex); 
* duration: duração do último contato, em segundos (numérico). Observação importante: esse atributo afeta muito o destino de saída (por exemplo, se duração = 0, então y = não). No entanto, a duração não é conhecida antes de uma chamada ser realizada. Além disso, após o término da chamada, y é obviamente conhecido.

## 2.3 Outros atributos:

* campaign: número de contatos realizados durante esta campanha e para este cliente (numérico, inclui último contato); 
* pdays: número de dias que se passaram após o último contato com o cliente de uma campanha anterior (numérico; 999 significa que o cliente não foi contatado anteriormente); 
* previous: número de contatos realizados antes desta campanha e para este cliente (numérico); 
* Poutcome: resultado da campanha de marketing anterior (fracasso, inexistente, sucesso);

##  2.4 Atributos do contexto social e econômico
* emp.var.rate: taxa de variação do emprego - indicador trimestral (numérico); 
* cons.price.idx: índice de preços ao consumidor - indicador mensal (numérico); 
* cons.conf.idx: índice de confiança do consumidor - indicador mensal (numérico); 
* euribor3m: taxa de 3 meses euribor - indicador diário (numérico); 
* nr.employed: número de funcionários - indicador trimestral (numérico).

## 2.5 Saída
* y - o cliente realizou um depósito a prazo? (sim, não).

# 3 <i> Workflow </i>
  
Todo o processo foi realizado por meio do workflow:
  
  ![Workflow](https://github.com/ivanovitchm/ppgeecmachinelearning/blob/main/images/workflow.png)

# 4 ETAPAS DO PROCESSO

## 4.1 Instalando bibliotecas

~~~
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras import layers
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
~~~

## 4.2 Importando o dataset
  
Esta etapa é opcionall, o usuário por importar o arquivo direto do local do dataset. 
~~~
from google.colab import drive
drive.mount('/content/drive')
~~~
  
## 4.2 Pré-processamento do dataset
  
### 4.2.1 Organização do dataset

Para carregar o dataset editado
~~~
df_train = pd.read_csv("/content/drive/MyDrive/dataset_bank.csv",sep=",")
df_test = pd.read_csv("/content/drive/MyDrive/dataset_bank_test.csv",sep=",")
~~~
Apagar coluna indesejada:
~~~
df_train = df_train.drop(["Unnamed: 0"],axis=1)
df_test = df_test.drop(["Unnamed: 0"],axis=1)
~~~
Concatena e preprocessa os dados categóricos. 
~~~
frames = [df_train,df_test]
df = pd.concat(frames)
col_categoric = df.select_dtypes("object")
col = df.select_dtypes("object")

for i in col:
  label_encoder = preprocessing.LabelEncoder()
  df[i] = label_encoder.fit_transform(df[i])
~~~
  
### 4.2.2 Normalização dos dados

O método utilizado para normalização foi o StandardScaler. 
~~~
X = df.drop(['y'], axis=1)
y = df['y']
X_scaled = StandardScaler().fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
~~~

## 4.3 Machine Learning
  
O modelo utilizado foi o <i>Multi-layer Perceptron classifier</i>. Este modelo otimiza a função log-loss usando LBFGS ou gradiente descendente estocástico. A documentação e os parâmetros podem ser vistos em: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
  
### 4.3.1 Teste Modelo
Dados dos parâmetros:
~~~
h = (20,50)
f = "relu"
s = "adam"
a = 0.1
m = 500  
model1 = MLPClassifier(hidden_layer_sizes=h, activation=f, solver=s, alpha=a, max_iter=m, random_state=42)  
model1.fit(train_x, train_y)
~~~
Visualizando as métricas e a Matriz de Confusão
~~~
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

pred_y = model1.predict(test_x)
report = metrics.classification_report(test_y, pred_y, 
                                       target_names=['0','1'])
print(report)

ConfusionMatrixDisplay.from_predictions(test_y, pred_y)
plt.show()
~~~
  
### 4.3.2 Teste Modelo com downsample

#### 4.3.2.1 Balanceamento dos dados com downsample
~~~
label0 = df[df['y'] == 0]
label1 = df[df['y'] == 1]
~~~
~~~
from sklearn.utils import resample
df_downsampled = resample(label0,
                          replace=False,
                          n_samples=5289,
                          random_state=42)
~~~
  
~~~
df_resample = pd.concat([label1, df_downsampled])
~~~
#### 4.3.2.2 Pré-processamento dos dados
~~~
X = df_resample.drop(['y'], axis=1)
y = df_resample['y']
X_scaled = StandardScaler().fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
~~~
#### 4.3.2.3 Modelo
Dados dos parâmetros:
~~~
h = (20,50)
f = "relu"
s = "adam"
a = 0.1
m = 1000
model2 = MLPClassifier(hidden_layer_sizes=h, activation=f, solver=s, alpha=a, max_iter=m, random_state=42, verbose=True)
model2.fit(train_x, train_y)
~~~
Visualizando as métricas e a Matriz de Confusão
~~~
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

pred_y = model2.predict(test_x)

report = metrics.classification_report(test_y, pred_y, 
                                       target_names=['0','1'])
print(report)

ConfusionMatrixDisplay.from_predictions(test_y, pred_y)
plt.show()
~~~
  
### 4.3.3 Teste Modelo com upsample

#### 4.3.3.1 Divisão dos dados em treino e teste
~~~
X = df.drop(['y'], axis=1)
y = df['y']
X_scaled = StandardScaler().fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
~~~

##### 4.3.3.1.1 Upsample just train data
~~~
df_train = pd.DataFrame(np.column_stack([train_x, train_y]))
label0 = df_train[df_train[16] == 0]
label1 = df_train[df_train[16] == 1]

df_upsampled = resample(label1,
                          replace=True,
                          n_samples=31906,
                          random_state=42)
df_train_resample = pd.concat([df_upsampled, label0])
train_x = df_train_resample.drop([16], axis=1).to_numpy()
train_y = df_train_resample[[16]].to_numpy()  
~~~
  
#### 4.3.3.3 Modelo
Dados dos parâmetros:
~~~
h = (20,50)
f = "relu"
s = "adam"
a = 0.1
m = 500
model3 = MLPClassifier(hidden_layer_sizes=h, activation=f, solver=s, alpha=a, max_iter=m, random_state=42)
model3.fit(train_x, train_y)
~~~
Visualizando as métricas e a Matriz de Confusão
~~~
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

pred_y = model3.predict(test_x)

report = metrics.classification_report(test_y, pred_y, 
                                       target_names=['0','1'])
print(report)

ConfusionMatrixDisplay.from_predictions(test_y, pred_y)
plt.show()
~~~
  
### 4.3.4 Teste Modelo com downsample e upsample
  
#### 4.3.4.1 Balanceamento dos dados com downsample

~~~
label0 = df[df['y'] == 0]
label1 = df[df['y'] == 1]
from sklearn.utils import resample
df_downsampled = resample(label0,
                          replace=False,
                          n_samples=10000,
                          random_state=42)
df_resample = pd.concat([label1, df_downsampled])
~~~

#### 4.3.4.2 Division of data into train and test

~~~
X = df_resample.drop(['y'], axis=1)
y = df_resample['y']
X_scaled = StandardScaler().fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
~~~

#### 4.3.4.3 Upsample just train data

~~~
df_train = pd.DataFrame(np.column_stack([train_x, train_y]))
label0 = df_train[df_train[16] == 0]
label1 = df_train[df_train[16] == 1]

df_upsampled = resample(label1,
                          replace=True,
                          n_samples=8012,
                          random_state=42)
df_train_resample = pd.concat([df_upsampled, label0])
train_x = df_train_resample.drop([16], axis=1).to_numpy()
train_y = df_train_resample[[16]].to_numpy()
~~~

#### 4.3.4.4 ### Modelo
Dados dos parâmetros:
~~~
h = (20,50)
f = "relu"
s = "adam"
a = 0.1
m = 500
model4 = MLPClassifier(hidden_layer_sizes=h, activation=f, solver=s, alpha=a, max_iter=m, random_state=42)
model4.fit(train_x, train_y)
~~~
Visualizando as métricas e a Matriz de Confusão
~~~
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

pred_y = model4.predict(test_x)

report = metrics.classification_report(test_y, pred_y, 
                                       target_names=['0','1'])
print(report)

ConfusionMatrixDisplay.from_predictions(test_y, pred_y)
plt.show()
~~~
  
### 4.3.5 Teste Modelo com downsample e tuning of parameters

#### 4.3.5.1 Balanceamento dos dados com downsample

~~~
label0 = df[df['y'] == 0]
label1 = df[df['y'] == 1]
from sklearn.utils import resample
df_downsampled = resample(label0,
                          replace=False,
                          n_samples=5289,
                          random_state=42)
df_resample = pd.concat([label1, df_downsampled])
~~~

#### 4.3.5.2 Pré-processamento dos dados

~~~
X = df_resample.drop(['y'], axis=1)
y = df_resample['y']
X_scaled = StandardScaler().fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
~~~

#### 4.3.5.3 Modelo

~~~
from sklearn.model_selection import GridSearchCV

clf = MLPClassifier(hidden_layer_sizes=(20,50), activation='relu', solver='adam', alpha=0.1, max_iter=1000, random_state=42)

# Every combination you want to try
params = {
    'hidden_layer_sizes' : [(20,50), (50, 60), (20, 30, 40)], 
    'activation' : ['relu', 'tanh'], 
    'alpha' : [0.1, 0.01]
}

gscv = GridSearchCV(clf, params, verbose=1)

gscv.fit(np.array(train_x), np.array(train_y))
print(gscv.best_params_) 

pred_y = gscv.predict(test_x)

report = metrics.classification_report(test_y, pred_y, target_names=['0','1'])
print(report)

ConfusionMatrixDisplay.from_predictions(test_y, pred_y)
plt.show()
~~~
  
### 4.3.6 Teste Modelo com downsample e tuning of parameters
  
</p>
  
