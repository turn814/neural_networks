import dense


b = 2
x = [b] * 3
w = [x + [b]] * 5

print(b)
print(x)
print(w, '\n')

layer = dense.Dense(w, b)
print(layer.outputs)
print(layer.l, '\n')

output = layer.forward(x, layer.outputs)
print(output)
