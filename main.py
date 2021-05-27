# Importación de la libreria para la creación de la matriz de cada una de las componentes del grafo
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Area Relativa de cada zona,Vulnerabilidad de Cada Especie, Indice de dispersión, Cercania a cuerpos hidricos, Número de especies presente en cada zona
cp1 = np.array([[0.26, 0.23, 0.3, 0.47, 0.45],  # 1
                [0.34, 0.63, 0.4, 0.88, 0.34],  # 2
                [0.07, 0.05, 0.12, 0.19, 0.19],  # 3
                [0.05, 0.1, 0.06, 0.32, 0.13],  # 4
                [0.29, 0.17, 0.63, 0.68, 0.4],  # 5
                [0.12, 0.8, 0.20, 0.79, 0.36],  # 7
                [0.30, 0.45, 0.52, 0.75, 0.84],  # 8
                [0.20, 0.32, 0.39, 0.75, 0.52],  # 9
                [0.11, 0.03, 0.06, 0.41, 0.25],  # 10
                [0.05, 0.34, 0.10, 0.49, 0.21],  # 11
                [0.20, 0.13, 0.5, 0.90, 0.15],  # 12
                [0.05, 0.21, 0.02, 0.69, 0.07],  # 13
                [0.08, 0.13, 0.4, 0.71, 0.15],  # 14
                [0.62, 0.35, 0.9, 0.57, 0.66],  # 15
                [0.18, 0.85, 0.8, 0.79, 0.23],  # 16
                [0.47, 0.23, 0.7, 0.37, 0.92],  # 17
                [0.06, 0.40, 0.43, 0, 0.35],  # 18
                [0.27, 0.65, 0.05, 0.74, 0.91],  # 19
                [0.10, 0.43, 0.3, 0.37, 0.3],  # 30
                [0.12, 0.60, 0.1, 0.51, 0.14],  # 31
                [0.05, 0.14, 0.3, 0.42, 0.25],  # 32
                [0.12, 0.93, 0.01, 0.10, 0.11],  # 34
                ]
               )
cp2 = np.array([[0.17, 0.7, 0.5, 0.75, 0.55],  # 20
                [0.16, 0.67, 0.2, 0.54, 0.28],  # 21
                [0.06, 0.08, 0.16, 0.98, 0.14],  # 22
                [0.49, 0.74, 0.34, 0.84, 0.58],  # 23
                [0.38, 0.3, 0.25, 0.45, 0.22],  # 24
                [0.17, 0.33, 0.68, 0.86, 0.73],  # 25
                [1, 0.68, 0.78, 0, 0.79],  # 26
                [0.27, 0.44, 0.17, 0.18, 0.57],  # 27
                [0.60, 0.31, 0.56, 0.54, 0.63],  # 28
                [0.13, 0.05, 0.07, 0.8, 0.33],  # 29
                [0.06, 0.56, 0.1, 0.1, 0.34],  # 33
                ]
               )
# Cercania promedio a cascos urbanos, Probabilidad de explotación minera,Distancia a Carretera
cn1 = np.array([[0.45, 0.3, 0.65],  # 1
                [0.34, 0.78, 0.97],  # 2
                [0.19, 0.4, 0.43],  # 3
                [0.13, 0.56, 0.5],  # 4
                [0.4, 0.18, 0.79],  # 5
                [0.36, 0.5, 0.87],  # 7
                [0.84, 0.69, 0.96],  # 8
                [0.52, 0.38, 0.87],  # 9
                [0.25, 0.2, 0.55],  # 10
                [0.21, 0.48, 0.56],  # 11
                [0.15, 0.1, 0.98],  # 12
                [0.07, 0.2, 0.76],  # 13
                [0.15, 0.38, 0.78],  # 14
                [0.66, 0.25, 0.69],  # 15
                [0.23, 0.6, 0.89],  # 16
                [0.92, 0.85, 0.35],  # 17
                [0.35, 0.7, 0],  # 18
                [0.91, 0.4, 0.81],  # 19
                [0.30, 0.35, 0.78],  # 30
                [0.14, 0.5, 0.64],  # 31
                [0.25, 0.42, 0.65],  # 32
                [0.11, 0.1, 0.22],  # 34
                ]
               )
cn2 = np.array([[0.55, 0.75, 0],  # 20
                [0.28, 0.68, 0],  # 21
                [0.14, 0.78, 0],  # 22
                [0.58, 0.54, 0],  # 23
                [0.22, 0.49, 0],  # 24
                [0.73, 0.20, 0],  # 25
                [0.79, 0.96, 0],  # 26
                [0.57, 0.85, 0],  # 27
                [0.63, 0.56, 0],  # 28
                [0.37, 0.3, 0],  # 29
                [0.34, 0.78, 0],  # 33
                ]
               )
# Distancias: Por componentes- Distancias con repecto a cada uno de los parches/ Distancia Maxima por componente
# -----------------------1---2---3----4----5----7----8---9----10---11----12---13---14---15----16----17----18---19---30---31---32---34
D1 = np.array([[0, 5.42, 5.16, 2.40, 2.61, 8.02, 4.42, 3.38, 2.48, 4.67, 10.37, 2.31, 7.75, 7.42, 10.40, 6.52, 9.06,
                9.75, 4.63, 7.09, 2.78, 7.46],  # 1
               [0, 0, 9.51, 9.27, 8.19, 10.72, 5.68, 4.45, 8.75, 10.43, 13.36, 3.80, 3.64, 5.75, 5.00, 8.85, 13.83,
                13.97, 2.77, 12.82, 4.57, 12.48],  # 2
               [0, 0, 0, 4.02, 9.77, 13.80, 9.83, 7.73, 5.35, 9.30, 15.33, 6.49, 8.25, 6.14, 10.77, 2.08, 4.14, 14.44,
                6.72, 10.79, 4.68, 4.35],  # 3
               [0, 0, 0, 0, 2.97, 10.11, 7.13, 5.49, 1.78, 5.44, 11.52, 5.47, 9.28, 8.17, 11.78, 5.59, 6.37, 10.46,
                6.95, 7.46, 4.67, 4.40],  # 4
               [0, 0, 0, 0, 0, 4.17, 3.04, 3.33, 4.70, 3.07, 6.54, 5.53, 10.93, 11.33, 12.94, 11.41, 12.31, 6.37, 8.11,
                5.08, 7.36, 11.22],  # 5
               [0, 0, 0, 0, 0, 0, 5.06, 6.85, 8.66, 5.89, 2.52, 9.18, 14.08, 14.55, 15.89, 15.26, 16.33, 4.36, 11.61,
                6.46, 10.83, 14.24],  # 7
               [0, 0, 0, 0, 0, 0, 0, 1.91, 6.99, 6.30, 8.04, 3.94, 9.00, 10.18, 10.93, 11.34, 13.98, 8.29, 6.60, 8.20,
                6.88, 12.00],  # 8
               [0, 0, 0, 0, 0, 0, 0, 0, 5.42, 6.42, 9.63, 1.62, 7.62, 8.19, 9.41, 8.75, 12.27, 10.15, 4.67, 8.17, 4.45,
                10.29],  # 9
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.75, 10.97, 5.19, 10.22, 8.91, 12.35, 8.13, 7.50, 8.85, 7.20, 5.92, 5.33,
                5.53],  # 10
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.20, 7.44, 13.40, 12.50, 15.97, 11.55, 11.04, 4.94, 9.82, 2.04, 8.04,
                8.65],  # 11
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11.26, 16.92, 18.45, 18.90, 18.47, 17.82, 2.90, 14.66, 6.74, 13.73,
                14.94],  # 12
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.32, 6.47, 8.11, 6.82, 10.94, 11.91, 2.92, 9.94, 2.46, 9.58],  # 13
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.39, 2.71, 6.54, 12.78, 17.31, 2.89, 15.13, 4.93, 12.70],  # 14
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.73, 4.39, 10.21, 17.76, 3.50, 14.55, 4.04, 10.08],  # 15
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8.73, 16.02, 18.84, 5.67, 17.53, 7.18, 15.37],  # 16
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.09, 16.23, 5.74, 13.31, 4.42, 6.79],  # 17
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.76, 11.60, 11.68, 9.04, 2.53],  # 18
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13.96, 3.93, 13.42, 12.85],  # 19
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.51, 2.18, 10.10],  # 30
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.57, 8.74],  # 31
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8.38],  # 32
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 34
               ]
              )
# ----------------------20---21--22--23---24----25--26---27----28----29----33
D2 = np.array([[0, 2.87, 2.47, 2.13, 4.00, 9.01, 8.31, 10.11, 9.32, 12.89, 4.10],  # 20
               [0, 0, 5.37, 3.12, 4.74, 11.63, 10.08, 11.92, 13.03, 16.24, 2.34],  # 21
               [0, 0, 0, 3.57, 6.74, 6.97, 10.39, 9.13, 9.29, 11.52, 6.82],  # 22
               [0, 0, 0, 0, 5.70, 10.99, 10.45, 12.22, 11.84, 14.58, 4.91],  # 23
               [0, 0, 0, 0, 0, 9.33, 4.31, 6.59, 8.38, 12.93, 2.86],  # 24
               [0, 0, 0, 0, 0, 0, 8.58, 5.20, 3.53, 4.40, 11.92],  # 25
               [0, 0, 0, 0, 0, 0, 0, 3.44, 6.43, 11.80, 7.83],  # 26
               [0, 0, 0, 0, 0, 0, 0, 0, 6.65, 8.11, 10.63],  # 27
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 4.16, 11.66],  # 28
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15.67],  # 29
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 33
               ]
              )
# Hallamos las distancias maximas de cada una de las matrices y se las dividimos a cada una de ellas para normalizarlas
max_distance_c1 = np.max(D1)
max_distance_c2 = np.max(D2)
ND1 = (1 / max_distance_c1) * D1
ND2 = (1 / max_distance_c2) * D2


# funciones ###################################
def ponderacion(I1, I2, D):
    # calcula los pesos de las aristas de acuerdo al metodo establecido en el documento
    # inputs:
    # I1,I2->vectores I+ e I- de un componente
    # D-> matriz de distancia de un componente
    # output:
    # W-> matriz de pesos (como es simetrica solo se calculan los valores por encima de la diagonal entonces es triangular superior)
    W = np.zeros((I1.shape[0], I1.shape[0]))
    for i in range(0, I1.shape[0]):
        for j in range(i + 1, I1.shape[0]):
            # computa el peso de la arista ij
            p = 0.5 * (0.7 * (I1[i] + I1[j]) - (0.3) * (I2[i] + I2[j])) - 0.5 * D[i, j]
            if p > 0:
                W[i, j] = p

    return W


#################

# Creacion del vector I+ y el vector I-
# Definición de los valores de las coeficientes de I_+, I_-
fac_p = np.array([0.4, 0.25, 0.05, 0.2, 0.1])
fac_n = np.array([0.25, 0.4, 0.35])

# Separando los vectores I+ e I- para cada componente
Ip_c1 = np.dot(cp1, fac_p)
In_c1 = np.dot(cn1, fac_n)
Ip_c2 = np.dot(cp2, fac_p)
In_c2 = np.dot(cn2, fac_n)

# se obtienen las matrices con los pesos de todas las aristas
W1 = ponderacion(Ip_c1, In_c1, ND1)
W2 = ponderacion(Ip_c2, In_c2, ND2)

# Indices de los parches X componente
ind_c1 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 31, 32, 34]
ind_c2 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33]
# Creación del grafo Inicial
Grafo = nx.Graph()
# Creación de los vertices del grafo
vertices = []
for i in range(1, 36):
    vertices.append("v" + str(i))
Grafo.add_nodes_from(vertices)
# Creación de las aristas del grafo X componente
aristas = []

# aristas de componente 1
for i in range(len(ind_c1)):
    for j in range(i + 1, len(ind_c1)):
        if (i + 1) != len(ind_c1):
            aristas.append(("v" + str(ind_c1[i]), "v" + str(ind_c1[j]), W1[i, j]))

# aristas de componente 2
for i in range(len(ind_c2)):
    for j in range(i + 1, len(ind_c2)):
        if (i + 1) != len(ind_c2):
            aristas.append(("v" + str(ind_c2[i]), "v" + str(ind_c2[j]), W2[i, j]))

# aristas de componente 3
aristas.append(("v" + str(6), "v" + str(35), 0.7))
# Agregamos las aristas con sus respectivos pesos
Grafo.add_weighted_edges_from(aristas)
# Ejecución del algoritmo de Kruskal
mst = nx.tree.maximum_spanning_edges(Grafo, algorithm="kruskal", data=False)
# Edgelist contiene las aristas que se seleccionaron por el algoritmo anterior
edgelist = list(mst)
sorted(sorted(e) for e in edgelist)
# Construcción del bosque de expansión
Trees = nx.Graph()
Trees.add_nodes_from(vertices)
Trees.add_edges_from(edgelist)
nx.draw(Trees, with_labels=True)
plt.show()
