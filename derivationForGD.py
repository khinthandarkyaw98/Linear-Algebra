from sympy import *
import numpy as np 

# for derivation
w = Symbol('w')
y = w**4 - 3*(w**3) + 2
yprime = y.diff(w)

'''convert a SymPy expression to an expression that 
can be numerically evaluated'''

f = lambdify(w, yprime, 'numpy')

weight = -1.5
step = 0.001

# Gradient descent
for i in range(100):
	weight_grad = f(weight)
	weight = weight - (step*weight_grad)

print(weight)


