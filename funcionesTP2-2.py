# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:45:32 2024

@author: nacha

"""

import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import solve_triangular
dir ='C:/Users/nacha/Downloads/'
data = pd.read_excel(dir+ 'matrizlatina2011_compressed_0.xlsx',sheet_name=1)
#%%
#ejercicio 2
'''
elevarMatriz
esta función calcula la potencia i de la matriz A
'''
def elevarMatriz(A,i):
    matrizElevada = np.eye(A.shape[0])
    for j in range(i):
        matrizElevada = np.dot(matrizElevada,A)
    return matrizElevada
'''
crearVector 
esta función crea el vector que cada uno de sus elementos representa la norma 2
de la matriz elevada a la potencia que corresponde a su índice
'''
def crearVector(A):
    vector = np.zeros(250)
    for i in range(0, 250, 1):
        vector[i] = (sc.norm(elevarMatriz(A,i), ord=2))
    return vector 

A1 = np.array([[0.186, 0.521, 0.014, 0.32, 0.134],
               [0.24, 0.073, 0.219, 0.013, 0.327],
               [0.098, 0.12, 0.311, 0.302, 0.208],
               [0.173, 0.03, 0.133, 0.14, 0.074],
               [0.303, 0.256, 0.323, 0.225, 0.257]])

vector1 = crearVector(A1)

A2 = np.array([[0.186, 0.521, 0.014, 0.32, 0.134],
               [0.24, 0.073, 0.219, 0.013, 0.327],
               [0.098, 0.12, 0.311, 0.302, 0.208],
               [0.173, 0.03, 0.133, 0.14, 0.074],
               [0.003, 0.256, 0.323, 0.225, 0.257]])

vector2 = crearVector(A2)


# Gráfico de la norma de A1
plt.figure(figsize=(10, 5))  
plt.plot(vector1, label='Norma de A1', color='darkgreen')
plt.title('Norma de las potencias de A1')
plt.xlabel('n')
plt.ylabel('Norma 2')
plt.xlim(0, 50)  
plt.ylim(1, 1.05)  
plt.grid()
plt.legend()
plt.show()

# Gráfico de la norma de A2
plt.figure(figsize=(10, 5))  
plt.plot(vector2, label='Norma de A2', color='darkblue')
plt.title('Norma de las potencias de A2')
plt.xlabel('n')
plt.ylabel('Norma 2')
plt.xlim(0, len(vector2))
plt.ylim(0, 1)  
plt.grid()
plt.legend()
plt.show()
#%%
#ejercicio 3
'''
metodoDeLaPotencia 
esta función calcula el autovector aproximado de una matriz a través del método de
la potencia 
'''
def metodoDeLaPotencia(A, k):
    v = np.random.rand(A.shape[0])
    for i in range(k):
        v_siguiente = A @ v
        v = v_siguiente / np.linalg.norm(v_siguiente,2) 
    return(v) 

'''
obtenerAutovalorAproximado 
esta función calcula el autovalor aproximado de una matriz con el vector obtenido en 
el método de la potencia 
'''
def obtenerAutovalorAproximado(A,v):
    autovalor = np.dot(v, np.dot(A,v)) / np.dot(v,v)
    return autovalor 

'''
monteCarlo
esta función realiza el cálculo del autovector estadísticamente con el método de Monte Carlo
'''
def monteCarlo(A):
    lista_de_autovalores = []
    k = 50
    for i in range(251):
        v = metodoDeLaPotencia(A, k)
        autovalor_aproximado = obtenerAutovalorAproximado(A, v)
        lista_de_autovalores.append(autovalor_aproximado)
    promedio = np.mean(lista_de_autovalores)
    desvio_estandar = np.std(lista_de_autovalores)
    return promedio, desvio_estandar
        
        
promedio1, desvio_estandar1 = monteCarlo(A1)

promedio2, desvio_estandar2 = monteCarlo(A2)

promedioydesvio = {
    'Matriz': ['A1', 'A2'],
    'Promedio Autovalor': [promedio1, promedio2],
    'Desviación Estándar': [desvio_estandar1, desvio_estandar2]
}

tabla_resultados = pd.DataFrame(promedioydesvio)
#%%
#ejercico 4
'''
serieDePotencia
esta función calcula la serie de potencia para una matriz hasta un cierto n
'''
def serieDePotencias(A, n):
    suma = np.zeros(A.shape[0])
    for i in range(n + 1):
        suma = suma + elevarMatriz(A, i); 
    return suma

serieDePotenciasA1_10 = serieDePotencias(A1, 10)
serieDePotenciasA1_100 = serieDePotencias(A1, 100)
serieDePotenciasA2_10 = serieDePotencias(A2, 10)
serieDePotenciasA12_100 = serieDePotencias(A2, 100)
serieDePotenciasA12_1000 = serieDePotencias(A2, 1000)

'''
intercambiarfilas
Esta función toma una matriz A y cambia de lugar dos filas elegidas 
'''
def intercambiarfilas(A, fila1, fila2):
    A[[fila1, fila2]] = A[[fila2, fila1]]
    return A

'''
Facotirzación LU
Esta función toma a una matriz A  y le calcula su factorización LU
con permutaciones si es necesario
'''

def calcularLU(A):
    m, n = A.shape
    '''Si no es matriz cuadrada, no es invertible, 
    entonces no podemos calcular la factorización LU'''
    if m != n:
        print('Matriz no cuadrada')
        return
    '''
    Iniciamos el vector de permutaciones
    '''
    P = np.eye(n)
    Ac = A.copy()
    '''
    Recorremos las filas de la matriz A y si el pivote es cero, intercambiamos
    la fila con la siguiente
    '''  
    for fila in range(m):
        if Ac[fila, fila] == 0:
            '''
            Nos aseguramos de no estar en la última fila
            '''
            if fila + 1 < m: 
                intercambiarfilas(Ac, fila, fila + 1)
                intercambiarfilas(P, fila, fila + 1)
            else:
                print("La matriz no tiene factorización LU.")
                return
        '''Recorremos la matriz Ac. En cada paso, se calcula un factor 
        y se utiliza para restar las filas y obtener la eliminación gaussiana'''
        for i in range(fila + 1, m):
            factor = Ac[i, fila] / Ac[fila, fila]
            Ac[i, fila] = factor  
            Ac[i, fila + 1:] -= factor * Ac[fila, fila + 1:]
        '''Calculamos las matrices L y U que componen la factorización LU de la matriz original.
        L toma la parte triangular inferior estricta de la matriz Ac y le añadimos una matriz identidad''' 
        L = np.tril(Ac, -1) + np.eye(m) 
        U = np.triu(Ac) 
    return L, U, P

'''
inversaLU
Esta función toma la descomposición LU de una matriz y
calcula la inversa de la misma
'''
def inversaLU(L, U, P):
    n = L.shape[0]
    Inv = np.zeros((n, n))  # Inicializa una matriz de ceros
    id = np.eye(n)  # Crea una matriz identidad

    for i in range(n):
        y = solve_triangular(L, np.dot(P, id[:, i]), lower=True)  # Resuelve L * y = P * e_i
        x = solve_triangular(U, y)  # Resuelve U * x = y
        Inv[:, i] = x  # Almacena la columna en Inv

    return Inv

'''
crearVectorError
esta función obtiene el error en la serie de potencia como un vector
'''
def crearVectorError(A, inversa, n):
    e = np.zeros(n + 1)
    for i in range(n + 1):
        e[i] = sc.norm((serieDePotencias(A, i) - inversa),2)
    return e

Id_menos_A2 = np.array([[0.832, 0.521, 0.014, 0.32, 0.134],
               [0.24, 0.927, 0.219, 0.013, 0.327],
               [0.098, 0.12, 0.689, 0.302, 0.208],
               [0.173, 0.03, 0.133, 0.86, 0.074],
               [0.003, 0.256, 0.323, 0.225, 0.743]])

L2, U2, P2 = calcularLU(Id_menos_A2)
inversa_Id_menos_A2 = inversaLU(L2, U2, P2)

e2_10 = crearVectorError(A2, inversa_Id_menos_A2, 10)
e2_100 = crearVectorError(A2, inversa_Id_menos_A2, 100)

plt.figure(figsize=(10, 5))  
plt.plot(e2_100, label='error inversa de I-A2', color='orange')
plt.title('error en la inversa de I-A2')
plt.xlabel('n')
plt.ylabel('error')
plt.xlim(0, len(e2_100))
plt.ylim(0, 15)  
plt.grid()
plt.legend()
plt.show()
#%%
#ejercicio 5
'''
generadorMatrizZ
Esta función toma como parámetros el excel con la información de los paises
y dos paises seleccionados para armar las matrices de flujo de capitales intrarregionales
e interregionales
'''

def generadorMatrizZ(data,PAIS1,PAIS2):
    '''
    PAIS 1 corresponde a filas y PAIS 2 corresponde a columnas
    '''
    Columnas = data.drop([col for col in data.columns if not col.startswith(PAIS2) and not col.startswith("Country_iso3")], axis = 1)
    FilasYColumnas = Columnas[(Columnas["Country_iso3"]== PAIS1 )]
    Matriz = FilasYColumnas.reset_index(drop=True).drop([col for col in FilasYColumnas.columns if col.startswith("Country_iso3")],axis = 1)

    return Matriz

'''
produccionesPais
Esta función toma como parámetros el excel y un país para calcular el total de producción del mismo
'''

def produccionesPais(data,PAIS):
    total = data.drop([col for col in data.columns if not col.startswith("Output") and not col.startswith("Country_iso3")], axis = 1)
    totalPAIS = total[(total["Country_iso3"]==PAIS)].reset_index(drop=True)
    totalPAIS = totalPAIS.drop([col for col in totalPAIS.columns if col.startswith("Country")],axis = 1)
    totalPAIS = totalPAIS.to_numpy()
    return totalPAIS

'''
IdxP
Esta función toma como parámetro un país y devuleve la matriz diagonal con el total de producción
'''

def IdxP(pPAIS):
  n = pPAIS.shape[0]
  Id = np.eye(n)
  for i in range(len(Id)):
    for j in range(len(Id[i])):
      if i == j :
        Id[i][j] = Id[i][j] * pPAIS[j]
        if Id [i][j] == 0:
            Id[i][j] = 1

  return(Id)


'''
AInsumoProducto
Esta función devuelve la matriz de coeficientes técnicos intrarregional
'''
def AInsumoProducto(ZP1P2,Id_InvP2):
    AP1P2 = ZP1P2 @ Id_InvP2
    AP1P2 = AP1P2.to_numpy()

    return AP1P2

ZCriCri = generadorMatrizZ(data,"CRI","CRI")
pCri = produccionesPais(data,"CRI")
IdPCri = IdxP(pCri)
L_Id_Cri, U_Id_Cri, P_Id_Cri =calcularLU(IdPCri)
IdCri_inv = inversaLU(L_Id_Cri, U_Id_Cri, P_Id_Cri)
ACriCri = AInsumoProducto(ZCriCri,IdCri_inv)
AvecACriCri = metodoDeLaPotencia(ACriCri, 50)
AvalCriCri = obtenerAutovalorAproximado(ACriCri, AvecACriCri)

ZNicNic = generadorMatrizZ(data,"NIC","NIC")
pNic = produccionesPais(data,"NIC")
IdPNic = IdxP(pNic)
L_Id_Nic, U_Id_Nic, P_Id_Nic =calcularLU(IdPNic)
IdNic_inv = inversaLU(L_Id_Nic, U_Id_Nic, P_Id_Nic)
ANicNic = AInsumoProducto(ZNicNic,IdNic_inv)
avecNic = metodoDeLaPotencia(ANicNic,150)
autovaloresNicP = obtenerAutovalorAproximado(ANicNic,avecNic)
#%%
#consigna 7
I40 = np.eye(40)
v40 = np.ones(40)
E_40 = I40 - 1/40*(np.outer(v40,v40))

ACriCri_normalizada = np.dot(E_40, ACriCri)

C = np.dot(ACriCri_normalizada.T, ACriCri_normalizada) / 39

'''
metodo_de_la_potencia
esta función ejecuta nuevamente el método de la potencia pero con un criterio de parada
'''
def metodo_de_la_potencia(C, epsilon=1e-6, max_iter=1000):
    v = np.random.rand(C.shape[0])
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        v_siguiente = C @ v
        v_siguiente = v_siguiente / np.linalg.norm(v_siguiente,2)

        if np.linalg.norm(v_siguiente - v,2) < epsilon:
            break
        v = v_siguiente

    autovalor = np.dot(v.T, C @ v)

    return v, autovalor

v1, lambda1 = metodo_de_la_potencia(C)
print("Primer autovalor:", lambda1)
print("Primer autovector:", v1)

C_prima = C - lambda1 * np.outer(v1, v1)
v2, lambda2 = metodo_de_la_potencia(C_prima)
print("Segundo autovalor:", lambda2)
print("Segundo autovector:", v2)
#%%
#ejercicio 8
'''
proyectar_datos
esta función proyecta las filas de la matriz en el subespacio definido por 
los dos autovectores principales.
'''
def proyectar_datos(Arr, autovector_1, autovector_2):
    # Combina los autovectores en una matriz de proyección
    matriz_proyeccion = np.vstack((autovector_1, autovector_2)).T  # Tamaño (m, 2)
    
    # Proyecta las filas de Arr en el espacio de dos dimensiones
    Arr_proj = Arr @ matriz_proyeccion  # Tamaño (n, 2)
    
    return Arr_proj

ACriCri_proyectada = proyectar_datos(ACriCri, v1, v2)