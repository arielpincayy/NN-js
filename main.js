import { NeuralNetwork, NN_GA } from "./NN.js";
import { createCircles, createSin, drawPoint, split } from "./data.js";

// Get the canvas element and its context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set canvas dimensions dynamically based on window size
canvas.width = window.innerWidth - (window.innerWidth / 30);
canvas.height = window.innerHeight - (window.innerHeight / 4);
let width = canvas.width;
let height = canvas.height;

// Translate the coordinate system to center the canvas
ctx.translate(width / 2, height / 2);

// Generate a dataset of circles for classification
let [X, Y] = createCircles(800, 2, [0, 0], 1, [-0.5, 0.5], ctx, 100);

// Split dataset into training and testing sets
let [X_train, Y_train, X_test, Y_test] = split(X, Y, 30);

console.log('Creating layers...');

// Define the neural network structure with hidden layers
const n_neurons = [4, 8];
const NN = new NeuralNetwork(X_train, Y_train, n_neurons.length, n_neurons, 'Tanh', true);

// Genetic algorithm-based neural network (currently commented out)
// let brains = new NN_GA(X_train, Y_train, 4, 5, 'Tanh', true, 20, 500);

console.log('Layers created');

// Add an event listener to train the neural network when the button is clicked
document.getElementById('train').addEventListener('click', () => {
    console.log('Training NN...');
    
    // Train the neural network with a learning rate of 0.01 and 1000 epochs
    NN.fit(0.01, 1000);
    
    // Train the genetic algorithm-based neural network (currently commented out)
    // NN = brains.train();

    console.log(`Accuracy: ${NN.test(X_test, Y_test)}%`);
    console.log('NN trained');
    console.log(NN);
});

let x_pred = [];

// Event listener to capture user clicks on the canvas
canvas.addEventListener('click', function(event) {
    // Get the bounding rectangle of the canvas
    const rect = canvas.getBoundingClientRect();

    // Convert mouse click position to canvas coordinates
    const x = -(width / 2 - event.clientX - rect.left) - 20;
    const y = -(height - event.clientY - rect.top) + 70;

    console.log(`New data: (${x / 100}, ${y / 100})`);
    
    // Draw the clicked point on the canvas in black
    drawPoint(x, y, ctx, 2, 'black');

    // Store the new point for prediction
    x_pred = [x / 100, y / 100];
});

// Event listener for making predictions with the trained model
document.getElementById('predict').addEventListener('click', () => {
    let x = x_pred;
    console.log('Predicting new data...');
    
    // Get prediction from the neural network
    const predict = NN.predict([x]);
    
    console.log(`Prediction: ${predict[0][0]}`);  
    
    console.log('End');
});
