import { getRandom } from "./utils.js";

function createCircles(n_samples, radius, center, diff, noise = [-1,1], ctx = null, scaleFigure = 1) {
    let x = [];
    let target = [];

    let theta;
    for (let i = 0; i < Math.floor(n_samples / 2); i++) {
        // Generate points for the outer circle with noise
        theta = getRandom(0, 2 * Math.PI);
        x.push([
            getRandom(...noise) + (center[0] + radius * Math.cos(theta)),
            getRandom(...noise) + (center[1] + radius * Math.sin(theta))
        ]);
        target.push([0]);

        // Generate points for the inner circle with noise
        x.push([
            getRandom(...noise) + (center[0] + (radius - diff) * Math.cos(theta)),
            getRandom(...noise) + (center[1] + (radius - diff) * Math.sin(theta))
        ]);
        target.push([1]);
    }

    // If a drawing context is provided, visualize the generated points
    if (ctx) {
        for (let i = 0; i < x.length; i++) {
            if (target[i] == '0') {
                drawPoint(x[i][0] * scaleFigure, x[i][1] * scaleFigure, ctx, 2, 'red');
            } else {
                drawPoint(x[i][0] * scaleFigure, x[i][1] * scaleFigure, ctx, 2, 'blue');
            }
        }
    }

    return [x, target];
}

function createSin(n_samples, ctx = null, noise = [-0.1, 0.1]) {
    let x = [];
    let y = [];

    // Generate noisy sine wave points
    for (let k = 0; k < Math.floor(n_samples / 60); k++) {
        for (let i = -3; i < 3; i += 0.1) {
            x.push([i + getRandom(...noise)]);
            y.push(Math.sin(i) + getRandom(...noise));
        }
    }

    // If a drawing context is provided, visualize the sine wave
    if (ctx) {
        for (let i = 0; i < x.length; i++) {
            drawPoint(x[i][0] * 100, y[i] * 100, ctx, 2, 'red');
        }
    }

    return [x, y];
}

function drawPoint(x, y, ctx, radius = 2, color = 'black') {
    // Draws a point on the given canvas context
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function split(X, Y, k) {
    // Splits dataset into two parts at index k
    return [X.slice(k, X.length), Y.slice(k, Y.length), X.slice(0, k), Y.slice(0, k)];
}

export {
    createCircles,
    createSin,
    drawPoint,
    split
}
