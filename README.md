import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Datos de ejemplo
data = {
    'nombre_clase': ['Matemáticas', 'Programación', 'Historia', 'Inglés', 'Ciencias'],
    'dificultad': [3, 5, 2, 4, 3],
    'duración': [60, 90, 45, 75, 60],
    'cantidad_estudiantes': [100, 150, 80, 120, 90]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Seleccionar características relevantes
X = df[['dificultad', 'duración', 'cantidad_estudiantes']]

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar clusters
plt.figure(figsize=(8, 6))

colors = ['red', 'green', 'blue']
for cluster, color in zip(range(3), colors):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['dificultad'], cluster_data['duración'], c=color, label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='black', label='Centroides')
plt.xlabel('Dificultad')
plt.ylabel('Duración')
plt.title('Clustering de Clases en Línea')
plt.legend()
plt.grid(True)
plt.show()
