import { getRandom, Matrix } from "./utils.js";


function softMax(x){
    let max = Math.max(...x[0]);
    let expX = x.map(e=>e.map(u=>Math.exp(u-max)));
    let sumExp = 0;
    expX[0].forEach(e=>sumExp+=e);
    return expX.map(e=>e.map(u=>u/sumExp));
}

let fns = {
    'Tanh':{'fn':(x)=>Math.tanh(x), 'fn_prime':(x)=>(1 - Math.tanh(x)**2)},
    'ReLu':{'fn':(x)=>Math.max(0,x), 'fn_prime':(x)=>x>0?1:0},
    'Sigmoide':{'fn':(x)=>(1 / (1 + Math.exp(-x))), 'fn_prime':(x)=>((1 / (1 + Math.exp(-x))) * (1 - (1 / (1 + Math.exp(-x)))))},
}

let loss_fns = {
    'Squared':{'fn':(x,y,n)=>(y - x)**2 / n, 'fn_prime':(x,y,n)=>(2/n)*(y-x)}
}

class Layer_Dense{
    constructor(n_inputs, n_neurons){
        this.weights = Matrix.randn(n_inputs, n_neurons);
        this.bias = Matrix.randn(1,n_neurons,[-1,1]);
    }
    
    forward(inputs){
        this.inputs = inputs;
        this.output = Matrix.add(this.bias ,Matrix.dot(inputs, this.weights));
    }

    backward(out_gradient, learning_rate){        
        let grad_weights = Matrix.dot(Matrix.T(this.inputs),out_gradient);
        this.weights = Matrix.add(this.weights, Matrix.scale(grad_weights, -1 * learning_rate));
        this.bias = Matrix.add(this.bias, Matrix.scale(out_gradient, -1 * learning_rate));
        
        return Matrix.dot(out_gradient, Matrix.T(this.weights));
    }

}

class Activation{
    constructor(fn, fn_prime){
        this.fn = fn;
        this.fn_prime = fn_prime;
    }

    forward(inputs){
        this.inputs = inputs;
        this.output = inputs.map(e=>e.map(i=>this.fn(i)));
    }

    backward(out_gradient, learning_rate){
        return Matrix.hdmr(out_gradient, this.inputs.map(e => e.map(i => this.fn_prime(i))));
    }
}


class NeuralNetwork{
    constructor(X,Y,n_layers, n_neurons, fn='ReLu', classification=false){
        this.X = X;
        this.Y = Y;
        this.fn_name = fn;
        this.n_layers = n_layers-1;
        this.n_neurons = n_neurons;
        this.classification = classification;
        this.layers = this.#createLayers();
        
    }

    #createLayers(){
        let layers = [];
        layers.push(new Layer_Dense(this.X[0].length,this.n_neurons[0]));
        layers.push(new Activation(fns[this.fn_name].fn, fns[this.fn_name].fn_prime));
        for(let l=0; l<this.n_layers; l++){
            layers.push(new Layer_Dense(this.n_neurons[l],this.n_neurons[l+1]));
            layers.push(new Activation(fns[this.fn_name].fn, fns[this.fn_name].fn_prime));
        }
        layers.push(new Layer_Dense(this.n_neurons[this.n_layers],1));
        if(this.classification){
            layers.push(new Activation(fns['Sigmoide'].fn, fns['Sigmoide'].fn_prime));
        }else{
            layers.push(new Activation(fns['Tanh'].fn, fns['Tanh'].fn_prime));
        }

        return layers;
    }

    fit(learning_rate=0.01, epochs=1000, loss_function='Squared'){

        let errs = [];

        for(let epoch=0; epoch<epochs; epoch++){

            let [X, Y] = [this.X, this.Y];
        
            let total_loss = 0;
            for(let j=0; j<X.length; j++){
                let input = [X[j]];
                let target = [Y[j]];
                
        
                let out = input;
                this.layers.forEach(e=>{
                    e.forward(out);
                    out = e.output;
                })
                
                
                total_loss += loss_fns[loss_function].fn(target[0], out[0], target.length);
                
        
                let out_gradient = [[loss_fns[loss_function].fn_prime(target[0], out[0], target.length)]];
        
                for(let k=this.layers.length - 1; k>0; k--){
                    out_gradient = this.layers[k].backward(out_gradient, learning_rate);
                }
        
            }
        
            if (epoch % 100 === 0) {
                errs.push([epoch,total_loss / X.length]);
            }
        }
        
        return errs;
    }

    predict(test_input){
        let out = test_input;
        this.layers.forEach(layer => {
            layer.forward(out);
            out = layer.output;
        });
        return out;
    }

    test(test_X, test_Y) {
        if (this.classification) {
            let correct = 0;

            for (let i = 0; i < test_X.length; i++) {
                let prediction = this.predict([test_X[i]]);
                
                let predicted_label = prediction[0][0] >= 0.5 ? 1 : 0;

                if (predicted_label === test_Y[i][0]) {
                    correct++;
                }
            }

            let accuracy = correct / test_X.length;
            
            return accuracy*100;
        } else {
            let total_error = 0;

            for (let i = 0; i < test_X.length; i++) {
                let prediction = this.predict([test_X[i]]);
                let error = (prediction[0][0] - test_Y[i]) ** 2;
                total_error += error;
            }

            let mse = total_error / test_X.length;
            return mse*100;
        }
    }
    copy() {
        let newNN = new NeuralNetwork(
            this.X, 
            this.Y, 
            this.n_layers, 
            this.n_neurons, 
            this.fn_name, 
            this.classification
        );

        newNN.layers = this.layers.map(layer => layer.copy ? layer.copy() : Object.assign(Object.create(Object.getPrototypeOf(layer)), layer));
        return newNN;
    }
}


class NN_GA{
    constructor(X, Y, n_neurons, n_layers=2, fn='ReLu', classification=false, n_population=10, generations=100){
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
        let population = new Array(n_population);

        for(let brain = 0; brain<n_population; brain++) 
            population[brain] = new NeuralNetwork(this.X, this.Y, this.n_layers, this.n_neurons, this.fn_name, this.classification);

        return population;
    }

    #error(brain){
        let error = 0;
        for(let i=0; i<this.X.length; i++){
            let predict = brain.predict([this.X[i]])
            
            error += loss_fns.Squared.fn(predict[0][0],this.Y[i][0], this.X.length);
        }
        return 1/error*100;
    }

    #fitness(){
        let ranked = new Array(this.n_population);
        this.population.forEach((brain,i)=>{
            ranked[i] = {'fitness': (this.#error(brain)), 'brain':brain}
        });

        ranked.sort((a,b)=>b.fitness - a.fitness);

        return ranked;
    }

    #crossingOver(A,B){
        let child1 = new NeuralNetwork(this.X, this.Y, this.n_layers, this.n_neurons, this.fn_name, this.classification);
        let child2 = new NeuralNetwork(this.X, this.Y, this.n_layers, this.n_neurons, this.fn_name, this.classification);

        let j=0;
        for(let i=0; i<this.n_layers; i++){
            if(Math.random()>0.5){
                child1.layers[j] = A.layers[j];
                child2.layers[j] = B.layers[j];
                j++;
                child1.layers[j] = A.layers[j];
                child2.layers[j] = B.layers[j];
                j++
            }else{
                child1.layers[j] = B.layers[j];
                child2.layers[j] = A.layers[j];
                j++;
                child1.layers[j] = B.layers[j];
                child2.layers[j] = A.layers[j];
                j++
            }
        }
        
        return [child1, child2];
    }

    #mutation(brain){
        let newBrain = brain.copy();
        newBrain.layers.forEach(layer=>{
            if(layer instanceof Layer_Dense){
                let w_size = Matrix.shape(layer.weights);
                let b_size = Matrix.shape(layer.bias);
                let mut_weight = Matrix.randn(w_size[0],w_size[1],[-0.5, 0.5]);
                let mut_bias = Matrix.randn(b_size[0],b_size[1],[-0.5, 0.5]);
                layer.weights = Matrix.add(layer.weights, mut_weight);
                layer.bias = Matrix.add(layer.bias,mut_bias);
            }
        })

        return newBrain;
    }


    train(){
        for(let g=0; g<this.generations; g++){
            let ranked = this.#fitness(this.population);
            
            
            
            let newPopulation = [];
            let n_selects = Math.floor(this.n_population/2);
            for(let i=0; i<3; i++){
                newPopulation.push(ranked[i].brain);
            }
            for(let i=0; i<n_selects; i+=2){
                let match = Math.floor(getRandom(n_selects,this.n_population-1) + 0.1);
                let childs = this.#crossingOver(ranked[i].brain, ranked[match].brain);
                newPopulation.push(childs[0]);
                newPopulation.push(childs[1]);
            }
            

            let void_size = ranked.length - newPopulation.length;
            
            for(let i=0; i<void_size; i++){
                if(i<void_size - Math.floor(void_size/4)){
                    let probability = Math.random();
                    newPopulation.push((probability>0.5)?this.#mutation(ranked[i].brain):ranked[i].brain);
                }else{
                    newPopulation.push(new NeuralNetwork(this.X, this.Y, this.n_layers, this.n_neurons, this.fn_name, this.classification));
                }
            }

            this.population = [...newPopulation];
            
        }

        return this.population[0];
    }
    
}


export{
    NeuralNetwork,
    NN_GA
}