import { getRandom, Matrix } from "./utils.js";

// Softmax function for normalizing output probabilities
function softMax(x){
    let max = Math.max(...x[0]); // Prevents large exponent values for numerical stability
    let expX = x.map(e => e.map(u => Math.exp(u - max))); // Compute exponentials
    let sumExp = 0;
    expX[0].forEach(e => sumExp += e); // Sum all exponentials
    return expX.map(e => e.map(u => u / sumExp)); // Normalize to obtain probabilities
}

// Dictionary of activation functions and their derivatives
let fns = {
    'Tanh': {'fn': (x) => Math.tanh(x), 'fn_prime': (x) => (1 - Math.tanh(x) ** 2)},
    'ReLu': {'fn': (x) => Math.max(0, x), 'fn_prime': (x) => x > 0 ? 1 : 0},
    'Sigmoide': {
        'fn': (x) => (1 / (1 + Math.exp(-x))), 
        'fn_prime': (x) => {
            let sig = 1 / (1 + Math.exp(-x));
            return sig * (1 - sig);
        }
    }
};

// Dictionary of loss functions and their derivatives
let loss_fns = {
    'Squared': {
        'fn': (x, y, n) => (y - x) ** 2 / n, // Mean Squared Error (MSE)
        'fn_prime': (x, y, n) => (2 / n) * (y - x) // Derivative of MSE
    }
};

// Fully connected layer (Dense layer)
class Layer_Dense {
    constructor(n_inputs, n_neurons){
        // Initialize weights and biases with random values
        this.weights = Matrix.randn(n_inputs, n_neurons);
        this.bias = Matrix.randn(1, n_neurons, [-1, 1]);
    }
    
    forward(inputs){
        // Compute the layer output: output = (inputs * weights) + bias
        this.inputs = inputs;
        this.output = Matrix.add(this.bias, Matrix.dot(inputs, this.weights));
    }

    backward(out_gradient, learning_rate){
        // Compute gradients and update weights and biases
        let grad_weights = Matrix.dot(Matrix.T(this.inputs), out_gradient);
        this.weights = Matrix.add(this.weights, Matrix.scale(grad_weights, -learning_rate));
        this.bias = Matrix.add(this.bias, Matrix.scale(out_gradient, -learning_rate));

        // Return gradient for previous layer
        return Matrix.dot(out_gradient, Matrix.T(this.weights));
    }
}

// Activation layer (applies a non-linearity)
class Activation {
    constructor(fn, fn_prime){
        this.fn = fn;
        this.fn_prime = fn_prime;
    }

    forward(inputs){
        // Apply activation function element-wise
        this.inputs = inputs;
        this.output = inputs.map(e => e.map(i => this.fn(i)));
    }

    backward(out_gradient, learning_rate){
        // Compute gradient of activation function
        return Matrix.hdmr(out_gradient, this.inputs.map(e => e.map(i => this.fn_prime(i))));
    }
}

// Neural Network class
class NeuralNetwork {
    constructor(X, Y, n_layers, n_neurons, fn = 'ReLu', classification = false){
        this.X = X;
        this.Y = Y;
        this.fn_name = fn;
        this.n_layers = n_layers - 1;
        this.n_neurons = n_neurons;
        this.classification = classification;
        this.layers = this.#createLayers();
    }

    #createLayers(){
        let layers = [];
        // Input layer
        layers.push(new Layer_Dense(this.X[0].length, this.n_neurons[0]));
        layers.push(new Activation(fns[this.fn_name].fn, fns[this.fn_name].fn_prime));

        // Hidden layers
        for (let l = 0; l < this.n_layers; l++){
            layers.push(new Layer_Dense(this.n_neurons[l], this.n_neurons[l + 1]));
            layers.push(new Activation(fns[this.fn_name].fn, fns[this.fn_name].fn_prime));
        }

        // Output layer
        layers.push(new Layer_Dense(this.n_neurons[this.n_layers], 1));
        if (this.classification) {
            layers.push(new Activation(fns['Sigmoide'].fn, fns['Sigmoide'].fn_prime)); // Sigmoid for classification
        } else {
            layers.push(new Activation(fns['Tanh'].fn, fns['Tanh'].fn_prime)); // Tanh for regression
        }

        return layers;
    }

    fit(learning_rate = 0.01, epochs = 1000, loss_function = 'Squared'){
        let errs = [];

        for (let epoch = 0; epoch < epochs; epoch++){
            let total_loss = 0;
            for (let j = 0; j < this.X.length; j++){
                let input = [this.X[j]];
                let target = [this.Y[j]];

                // Forward propagation
                let out = input;
                this.layers.forEach(layer => {
                    layer.forward(out);
                    out = layer.output;
                });

                // Compute loss
                total_loss += loss_fns[loss_function].fn(target[0], out[0], target.length);

                // Backpropagation
                let out_gradient = [[loss_fns[loss_function].fn_prime(target[0], out[0], target.length)]];
                for (let k = this.layers.length - 1; k > 0; k--){
                    out_gradient = this.layers[k].backward(out_gradient, learning_rate);
                }
            }

            if (epoch % 100 === 0) {
                errs.push([epoch, total_loss / this.X.length]);
            }
        }
        
        return errs;
    }

    predict(test_input){
        // Perform forward propagation for inference
        let out = test_input;
        this.layers.forEach(layer => {
            layer.forward(out);
            out = layer.output;
        });
        return out;
    }
}

// Genetic Algorithm for Neural Network optimization
class NN_GA {
    constructor(X, Y, n_neurons, n_layers = 2, fn = 'ReLu', classification = false, n_population = 10, generations = 100){
        this.n_population = n_population;
        this.generations = generations;
        this.X = X;
        this.Y = Y;
        this.fn_name = fn;
        this.n_layers = n_layers;
        this.n_neurons = n_neurons;
        this.classification = classification;
        this.population = this.#createPopulation(n_population);
    }
    
    #createPopulation(n_population){
        // Initialize population with random neural networks
        let population = new Array(n_population);
        for (let i = 0; i < n_population; i++){
            population[i] = new NeuralNetwork(this.X, this.Y, this.n_layers, this.n_neurons, this.fn_name, this.classification);
        }
        return population;
    }

    #error(brain){
        // Compute error for a given neural network
        let error = 0;
        for (let i = 0; i < this.X.length; i++){
            let predict = brain.predict([this.X[i]]);
            error += loss_fns.Squared.fn(predict[0][0], this.Y[i][0], this.X.length);
        }
        return 1 / error * 100;
    }

    #fitness(){
        // Rank networks based on their error (lower is better)
        let ranked = this.population.map(brain => ({
            fitness: this.#error(brain),
            brain
        }));
        ranked.sort((a, b) => b.fitness - a.fitness);
        return ranked;
    }

    #mutation(brain){
        // Mutate network by slightly modifying weights and biases
        let newBrain = brain.copy();
        newBrain.layers.forEach(layer => {
            if (layer instanceof Layer_Dense){
                let mut_weight = Matrix.randn(...Matrix.shape(layer.weights), [-0.5, 0.5]);
                let mut_bias = Matrix.randn(...Matrix.shape(layer.bias), [-0.5, 0.5]);
                layer.weights = Matrix.add(layer.weights, mut_weight);
                layer.bias = Matrix.add(layer.bias, mut_bias);
            }
        });
        return newBrain;
    }

    train(){
        for (let g = 0; g < this.generations; g++){
            let ranked = this.#fitness();
            this.population = ranked.slice(0, this.n_population).map(r => r.brain);
        }
        return this.population[0];
    }
}

export {
    NeuralNetwork,
    NN_GA
}
