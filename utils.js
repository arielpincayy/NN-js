function getRandom(min, max) {
    // Generates a random number in the range [min, max)
    let random = Math.random() * (max - min) + min;
    return random;
}

function shuffleData(dataX, dataY) {
    // Ensures both datasets have the same length
    if (dataX.length !== dataY.length) {
        throw new Error("Data must have the same shape");
    }

    // Creates an array of indices for shuffling
    let indices = Array.from({ length: dataX.length }, (v, k) => k);
    
    // Fisher-Yates shuffle algorithm to randomize indices
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // Creates new arrays to store shuffled data
    let shuffledDataX = new Array(dataX.length);
    let shuffledDataY = new Array(dataX.length);

    // Rearranges data according to shuffled indices
    for (let i = 0; i < indices.length; i++) {
        shuffledDataX[i] = dataX[indices[i]];
        shuffledDataY[i] = dataY[indices[i]];
    }

    return [shuffledDataX, shuffledDataY];
}

class Matrix {

    // Gets the shape (dimensions) of a matrix
    static shape(m) {
        return [m.length, m[0].length];
    }
    
    // Matrix multiplication (dot product)
    static dot(m1, m2) {
        let size1 = this.shape(m1);
        let size2 = this.shape(m2);
        
        // Ensures matrices are compatible for multiplication
        if (size1[1] !== size2[0]) throw new Error('Mismatch error. Unable to multiply');

        let m3 = [];

        // Computes matrix multiplication using standard rules
        for (let i = 0; i < size1[0]; i++) {
            let row = [];
            for (let j = 0; j < size2[1]; j++) {
                let d = 0;
                for (let k = 0; k < size1[1]; k++) {
                    d += m1[i][k] * m2[k][j];
                }
                row.push(d);
            }
            m3.push(row);
        }

        return m3;
    }

    // Matrix addition
    static add(m1, m2) {
        let size1 = this.shape(m1);
        let size2 = this.shape(m2);

        // Ensures matrices have the same shape
        if (size1[0] !== size2[0] || size1[1] !== size2[1]) throw new Error('Mismatch error. Unable to add');

        return m1.map((e, i) => e.map((n, j) => n + m2[i][j]));
    }

    // Scales a matrix by a scalar k
    static scale(matrix, k) {
        return matrix.map(e => e.map(m => m * k));
    }

    // Transposes a matrix
    static T(matrix) {
        let [m, n] = this.shape(matrix);
        let trps = [];

        for (let i = 0; i < n; i++) {
            let row = [];
            for (let j = 0; j < m; j++) {
                row.push(matrix[j][i]);
            }
            trps.push(row);
        }

        return trps;
    }

    // Element-wise matrix multiplication (Hadamard product)
    static hdmr(m1, m2) {
        let size1 = this.shape(m1);
        let size2 = this.shape(m2);

        // Ensures matrices have the same shape
        if (size1[0] !== size2[0] || size1[1] !== size2[1]) throw new Error('Mismatch error. Unable to perform Hadamard product');

        return m1.map((e, i) => e.map((n, j) => n * m2[i][j]));
    }

    // Generates a (m x n) matrix with random values in a given range
    static randn(m, n, range = [-1, 1]) {
        let newArr = [];
        let [min, max] = range;

        for (let i = 0; i < m; i++) {
            let row = [];
            for (let j = 0; j < n; j++) {
                row.push(getRandom(min, max));
            }
            newArr.push(row);
        }

        return newArr;
    }
}

export {
    Matrix,
    shuffleData,
    getRandom
}
