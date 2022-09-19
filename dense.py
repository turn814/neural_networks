## Just need to create a layer of neurons that take the weighted sum of an input, run it through an activation function, and then spit out an output

class Dense:
    def __init__(self, weights, biases):
        self.weights = weights   # tensor (dim:dim_input + 1)
        self.biases = biases     # tensor (dim:dim_input - 1)
        self.i = 0               # "global" counter variable
        self.outputs = self.create_output(weights, biases)   # tensor (dim:dim_input)
        self.l = [None] * self.i                             # list of number of elements in each dimension of output
        n = self.outputs
        for i in range(self.i):
            self.l[i] = len(n)
            n = n[0]
        self.i = 0
        self.n = 0               # another counter
        self.w = weights         # temp weights
        self.b = biases          # temp biases

    def create_output(self, w, b):
        self.i += 1
        if isinstance(b, int):
            return [0] * len(w)
        o = [self.create_output(w[0], b[0])] * len(w)
        return o
    
## Assumption: there are weights included for a bias in the input
## Trying: multiple base cases
    def forward(self, inputs, outputs):
        if isinstance(inputs[0], int):
            o = [0] * self.l[-1]
            for n in range(self.l[-1]):
                for i in range(len(inputs)):
                    o[n] += self.w[self.i][i] * inputs[i]
                o[n] += self.w[self.i][-1] * self.b
            return o                               # just a single element of output, not the entire output
        self.w = self.w[self.i]
        self.b = self.b[self.n]
        for i in range(len(outputs)):
            outputs[i] = self.forward(inputs[self.n], outputs[i])
            self.i = i
        self.n += 1
        return outputs
