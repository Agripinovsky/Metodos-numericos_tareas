#EJERCICIO 1


f=float(input("Ingresa una temperatura en Fahrenheit: "))
c=(f-32)*5/9
print("La temperatura en Celsius es: ",c, "\n")

#EJERCICIO 2

import math 

x=float(input("Ingresvalor: "))
r=math.sinh(x)
print('El seno hiperbolico es : ' , r)


sh=(math.e**x-math.e**-x)/2
print(f'El resultado usnado la formula es {sh} ')

p=(math.exp(x)-math.exp(-x))/2
print(f'Usando la funcion exp: {p}')

#EJERCICIO 3
import cmath

x=float(input("Ingrese un valor: "))
q=math.sinh(x)
print("sinh (x)=",q)
i=1j*q
print("i*sinh(x)=",i,)
s=cmath.sin(1j*x)
print("sin(ix)=", s, "\n")

#relacion euler
d=math.cos(x)
e=1j*math.sin(x)
print("Cos x + iSen x=",d+e)
f=math.e**(1j*x)
print("e^ix=",f,)

#EJERCICIO 4

import numpy.lib.scimath as nls

a=float(input("Ingresa un valor para a "))
b=float(input("Ingresa un valor para b: "))
c=float(input("Ingresa un valor para c: "))
print("f(x)=",a,"x^2 +",b,"x+",c)
z1=(-b+nls.sqrt(b**2-4*a*c))/(2*a)
z2=(-b-nls.sqrt(b**2-4*a*b))/(2*a)
print("x1=",z1,"y x2=",z2)

