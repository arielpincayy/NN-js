# Neural Network with Gradient Descent and Genetic Algorithm Training

This project implements a neural network with two different training modes:
1. **Traditional Gradient Descent** - Using backpropagation and optimization with gradient descent.
2. **Genetic Algorithm (GA) Optimization** - Using evolutionary techniques to optimize the network.

## Features
- Custom dataset generation with circular patterns.
- Interactive visualization using an HTML5 Canvas.
- User interaction for testing the network by clicking on the canvas.
- Training with either gradient descent or a genetic algorithm.

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/arielpincayy/NN-js.git
   cd yourrepository
   ```
2. Install dependencies (if required):
   ```sh
   npm install  # Only if using external dependencies
   ```
3. Open `index.html` in your browser.

## Usage
- Click the **Train** button to train the neural network using gradient descent.
- Click anywhere on the canvas to add a new point.
- Click the **Predict** button to classify the new point.
- (Optional) Uncomment the GA code to train the network with a genetic algorithm.

## File Structure
```
.
├── data.js           # Functions to generate training data
├── NN.js             # Neural Network implementation
├── main.js           # Main script for visualization and interaction
├── index.html        # Web interface
├── styles.css        # Styling for the interface
└── README.md         # Project documentation
```

## Contributing
Feel free to fork this repository and submit pull requests for improvements!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
