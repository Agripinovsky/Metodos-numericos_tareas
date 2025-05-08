#Tarea 2 
#1.- Interseccion de trayectorias
# Definimos las columnas a y b para hacer la eliminación de Gauss y obtener las intersecciones

import numpy as np
a = np.array([[2.0, -1.0, 3.0],
              [0, 2.0, -1.0],
              [7.0, -5, 0]])

b = np.array([[24.0], [14.0], [6.0]])

#Usamos el algoritmo para la eliminación de gauss
def gaussElimin(a,b):
  n = len(b)
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b

print(f'Las intersecciones son : \n {gaussElimin(a,b)}')

#2.-Carga de los quarks
#Definimos las dos matrices a y b
a = np.array([[2.0, 1.0],
              [1.0, 2.0],
            ])

b = np.array([[1.0], [0]])
#Utilizamosla eliminación gausseana para obtener la carga de los quarks

print(f"Las cargas de los quarks u y d son: \n {gaussElimin(a,b)}")

#3.- Meteoros
#Definimos las dos matrices a y b

a = np.array([[1.0, 5.0,10.0,20.0],
              [0.0, 1.0,-4.0,0.0],
               [-1.0, 2.0,0.0,0.0],
                [1.0, 1.0,1.0,1.0]
            ])


b = np.array([[95.0], [0.0],[1.0],[26.0]])

print(f"la cantidad de meteoros es: \n {gaussElimin(a,b)}")
