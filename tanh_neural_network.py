import numpy as np
np.random.seed(42)

x = np.array([0.6, 0.1])  
w_input_hidden = np.random.uniform(-0.5, 0.5, (2, 2))   
w_hidden_output = np.random.uniform(-0.5, 0.5, 2)      

b1 = 0.5   
b2 = 0.7  

def tanh(x):
    return np.tanh(x)
z1 = np.dot(w_input_hidden, x) + b1
a1 = tanh(z1)

z2 = np.dot(w_hidden_output, a1) + b2
output = tanh(z2)

print("Weights Input-Hidden:\n", w_input_hidden)
print("Weights Hidden-Output:\n", w_hidden_output)
print("Hidden Layer Output:\n", a1)
print("Final Network Output:\n", output)
