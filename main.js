import { NeuralNetwork, NN_GA } from "./NN.js";
import { createCircles, createSin, drawPoint, split } from "./data.js";


const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth - (window.innerWidth/30);
canvas.height = window.innerHeight - (window.innerHeight/4);
let width = canvas.width;
let height = canvas.height;

ctx.translate(width / 2, height / 2);

let [X,Y] = createCircles(800,2,[0,0],1,[-0.5,0.5],ctx,100);
let [X_train, Y_train ,X_test, Y_test] = split(X,Y,30);

console.log('Creating layers...');
const n_neurons = [4,8];
const NN = new NeuralNetwork(X_train,Y_train,n_neurons.length,n_neurons,'Tanh',true);
//let brains = new NN_GA(X_train,Y_train,4,5,'Tanh',true,20,500);
console.log('Layers created');

//let NN;
document.getElementById('train').addEventListener('click',()=>{
    console.log('Trainning NN...');
    NN.fit(0.01,1000);
    //NN = brains.train();
    console.log(`accuracy: ${NN.test(X_test,Y_test)}%`);
    console.log('NN trained');
    console.log(NN);
});

let x_pred = [];

canvas.addEventListener('click', function(event) {
    const rect = canvas.getBoundingClientRect();
    const x = -(width/2 - event.clientX - rect.left) - 20;
    const y = -(height - event.clientY -  rect.top) + 70;

    console.log(`New data: (${x/100}, ${y/100})`);
    drawPoint(x, y, ctx, 2, 'black');
    x_pred = [x/100,y/100];
});

document.getElementById('predict').addEventListener('click',()=>{
    let x = x_pred;
    console.log('Predicting new data...');
    
    const predict = NN.predict([x]);
    
    console.log(`Prediction: ${predict[0][0]}`);  
    
    console.log('End');
    
});








