# gradient_descent is finding the local minimum

import numpy as np
import matplotlib.pyplot as plt

x_old = 3.5
alpha = 0.01
precision = 0.0001

# Define function

x_input = np.arange(-1, 3.5, 0.01) # for list range with float step

# f = @(x) x.^4 - 3*x.^3 + 2;

def function(x):
	return x**4 - 3*(x**3) + 2

def df(x):
	return 4*(x**3) - 9*(x**2) 

y_output =function(x_input)

plt.plot(x_input, y_output)
plt.title('Derivation')
plt.xlabel('x_input')
plt.ylabel('y_output')
plt.show()

while 1:
	tmpDelta = x_old - alpha*(df(x_old))
	diffOldTmp = abs(tmpDelta - x_old)
	x_old = tmpDelta
	if diffOldTmp < precision:
		break

print('The local minimum is ', x_old)

""" %% Gradient descent algorithm
% Keep repeating until convergence
while 1
    % Evalulate gradients
    tmpDelta = x_old - alpha*(df(x_old));    
    % Check Convergence
    diffOldTmp = abs(tmpDelta - x_old);
    if diffOldTmp < precision
        break;
    end
    % Update parameters
    x_old = tmpDelta;     
end
"""