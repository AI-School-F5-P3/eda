# Análisis Exploratorio de Datos (EDA) en el Titanic Dataset

Este proyecto realiza un **Análisis Exploratorio de Datos (EDA)** del dataset del Titanic, utilizando herramientas avanzadas de Python como pandas, seaborn y matplotlib. Además, se exploran técnicas de clustering como **K-Means** y **Análisis Jerárquico**.

## Estructura del Proyecto

1. **Carga y Preparación de Datos**
   - Importación de datasets desde archivos CSV.
   - Selección de variables relevantes.
   - Limpieza de valores faltantes y codificación de variables categóricas.

2. **Visualización y Análisis Exploratorio**
   - Distribución de variables importantes como `Age` y `Fare`.
   - Análisis de relaciones entre variables clave como `Survived` y `Pclass`.

3. **Análisis de Correlación**
   - Generación de una matriz de correlación para entender las relaciones entre variables numéricas.

4. **Técnicas de Clustering**
   - Aplicación de K-Means para segmentar pasajeros.
   - Análisis jerárquico para identificar patrones en los datos.

---

## Instalación y Configuración

1. **Clonar el Repositorio**
   ```bash
   git clone https://github.com/tu-repositorio/eda-titanic.git
   cd eda-titanic

## Crear un Entorno Virtual
virtualenv -p python3.10 venv
venv\Scripts\activate

# Scripts Principales
### Carga y Preparación de Datos

import pandas as pd

# Cargar los datos
train_data = pd.read_csv("data_train.csv")

# Limpiar valores faltantes
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Codificar variables categóricas
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)


### Visualización de Datos

Distribución de Edad

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train_data['Age'], kde=True, bins=30)
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

Relación entre Clase y Supervivencia

sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Supervivencia por Clase')
plt.show()


###  Análisis de Correlación
Generar la Matriz de Correlación

correlation_matrix = train_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()

### Técnicas de Clustering
Aplicar K-Means

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Seleccionar características
features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
data_for_clustering = train_data[features].dropna()

# Escalar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_clustering)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data_for_clustering['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualización
sns.scatterplot(x=data_for_clustering['Age'], y=data_for_clustering['Fare'], hue=data_for_clustering['Cluster'], palette='Set1', alpha=0.7)
plt.title('Clustering K-Means')
plt.xlabel('Edad')
plt.ylabel('Tarifa')
plt.show()

Análisis Jerárquico

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(data_scaled, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Dendrograma')
plt.show()



# Requisitos
Python 3.8 o superior.
Librerías necesarias:
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy


Instálalas con:
pip install -r requirements.txt


# 
---

### **Instrucciones para Usarlo**
1. Guarda el contenido en un archivo llamado `README.md` en el directorio principal del proyecto.
2. Crea un archivo `requirements.txt` para especificar las dependencias:
   ```plaintext
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   scipy
