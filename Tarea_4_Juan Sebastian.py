#EJERCICIO 1
import sys
import math
from math import *
from numpy import sign
from math import sin, cos

## Modulo Newton-Raphson
## raiz = newtonRaphson(f,df,a,b,tol=1.0e-9).
## Encuentra la raiz de f(x) = 0 combinando Newton-Raphson
## con biseccion. La raiz debe estar en el intervalo (a,b).
## Los usuarios definen f(x) y su derivada df(x).
def err(string):
  print(string)
  input('Press return to exit')
  sys.exit()

def newtonRaphson(f,df,a,b,tol=1.0e-9):
  from numpy import sign
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb): err('La raiz no esta en el intervalo')
  x = 0.5*(a + b)
  for i in range(30):
    ###print(i)###
    fx = f(x)
    if fx == 0.0: return x 
    if sign(fa) != sign(fx): b = x # Haz el intervalo mas pequeño
    else: a = x
    dfx = df(x)  
    try: dx = -fx/dfx # Trata un paso con la expresion de Delta x
    except ZeroDivisionError: dx = b - a # Si division diverge, intervalo afuera
    x = x + dx # avanza en x
    if (b - x)*(x - a) < 0.0: # Si el resultado esta fuera, usa biseccion
      dx = 0.5*(b - a)
      x = a + dx 
    if abs(dx) < tol*max(abs(b),1.0): return x # Checa la convergencia y sal
  print('Too many iterations in Newton-Raphson')
def f(x): return x*sin(x) + 3*cos(x) - x
def df(x): return x*cos(x) - 2*sin(x) - 1
a = -6
b = 6
root = newtonRaphson(f, df, a, b)
print('Raíz =', root)

#EJERCICIO 2
def newtonRaphson(f,df,a,b,tol=1.0e-9):
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb): err('La raiz no esta en el intervalo')
  x = 0.5*(a + b)
  for i in range(30):
    fx = f(x)
    if fx == 0.0: return x 
    if sign(fa) != sign(fx): b = x 
    else: a = x
    dfx = df(x)  
    try: dx = -fx/dfx 
    except ZeroDivisionError: dx = b - a 
    x = x + dx # avanza en x
    if (b - x)*(x - a) < 0.0:
      dx = 0.5*(b - a)
      x = a + dx 
    if abs(dx) < tol*max(abs(b),1.0): return x 
  print('Too many iterations in Newton-Raphson')
  
def f(x): return x*sin(x) + 3*cos(x) - x
def df(x): return x*cos(x) - 2*sin(x) - 1
a = -6
b = 6
root = newtonRaphson(f, df, a, b)
print('Raíz =', root)

#EJERCICIO 3
x = [0.0, 0.1, 0.2, 0.3, 0.4]
fx = [0.000000, 0.078348, 0.138910, 0.192916, 0.244981]
h = 0.1
dfx = (fx[3] - fx[1]) / (2 * h)

def derivada(x_vals, f_vals, x):
    i = x_vals.index(x)
    h = x_vals[i+1] - x_vals[i]
    return (f_vals[i+1] - f_vals[i-1]) / (2*h)
x_val = [0.0, 0.1, 0.2, 0.3, 0.4]
f_val= [0.000000, 0.078348, 0.138910, 0.192916, 0.244981]
x = 0.2
res = derivada(x_val, f_val ,x)
print(f"Aproximación centrada:       {res:.3f}")

#EJERCICIO 3 
def df_forward(x, h, f, n):
    return (-3*f(x,n) + 4*f(x+h,n) - f(x+2*h,n)) / (2*h)

def df_central(x, h, f): 
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x): 
    return sin(x)
x = 0.8
h = 0.01
print("Aproximación forward:", df_forward(x, h, f))
print("Aproximación central:", df_central(x, h, f))

#para ver cual es mas exacta
print("Error forward:", abs(df_forward(x, h, f) - cos(x)))
print("Error central:", abs(df_central(x, h, f) - cos(x)))

#EJERCICIO 4 
def trapecio_recursiva(f,a,b,Iold,k):
  if k == 1: Inew = (f(a) + f(b))*(b - a)/2.0
  else:
    n = 2**(k -2 ) # numero de nuevos puntos
    h = (b - a)/n # espaciamiento de nuevos puntos
    x = a + h/2.0
    sum = 0.0
    for i in range(n):
      sum = sum + f(x)
      x = x + h
      Inew = (Iold + h*sum)/2.0
  return Inew

def f(x): 
    return math.log(1 + math.tan(x))
Iold = 0.0
for k in range(1, 21):
    Inew = trapecio_recursiva(f, 0.0, math.pi / 4, Iold, k)
    if (k > 1) and (abs(Inew - Iold) < 1.0e-6):
        break
    Iold = Inew

print('Integral aproximada =', Inew)
print('Número de paneles =', 2**(k - 1))

