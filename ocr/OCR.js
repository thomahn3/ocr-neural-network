const fs = require('fs');
const csv = require('csv-parser');
const { Matrix } = require('ml-matrix');
const readline = require('readline')
const { create, all } = require('mathjs');

const math = create(all);

// Utility functions for mathematical operations
const sigmoid = (Z) => {
    const A = Z.map(val => 1 / (1 + Math.exp(-val)));
    return { A, Z };
};

const tanh = (Z) => {
    const A = Z.map(val => Math.tanh(val));
    return { A, Z };
};

const sigmoidBackward = (dA, Z) => {
    const s = Z.map(val => 1 / (1 + Math.exp(-val)));
    return math.dotMultiply(dA, math.dotMultiply(s, math.subtract(1, s)));
};

const tanhBackward = (dA, Z) => {
    const s = Z.map(val => Math.tanh(val));
    return math.dotMultiply(dA, math.subtract(1, math.square(s)));
};

// Initialize parameters
const initParams = (layerDims) => {
    math.config({ randomSeed: 3 });
    const params = {};
    const L = layerDims.length;
    for (let l = 1; l < L; l++) {
        params['W' + l] = math.multiply(math.random([layerDims[l], layerDims[l - 1]], -0.01, 0.01), 0.01);
        params['b' + l] = math.zeros([layerDims[l], 1]);
    }
    return params;
};

// Forward propagation
const forwardProp = (X, params) => {
    let A = X;
    const caches = [];
    const L = Object.keys(params).length / 2;

    for (let l = 1; l <= L; l++) {
        const A_prev = A;
        const W = params['W' + l];
        const b = params['b' + l];
        const Z = math.add(math.multiply(W, A_prev), b);
        const { A: A_tanh, Z: Z_tanh } = tanh(Z);
        A = A_tanh;
        caches.push({ A_prev, W, b, Z: Z_tanh });
    }

    return { A, caches };
};

// Cost function
const costFunction = (A, Y) => {
    const m = Y.size()[1];
    const cost = (-1 / m) * math.sum(math.add(math.dotMultiply(math.log(A), Y), math.dotMultiply(math.log(math.subtract(1, A)), math.subtract(1, Y))));
    return cost;
};

// Backward propagation
const oneLayerBackward = (dA, cache) => {
    const { A_prev, W, b, Z } = cache;
    const dZ = tanhBackward(dA, Z);
    const m = A_prev.size()[1];
    const dW = math.multiply(1 / m, math.multiply(dZ, math.transpose(A_prev)));
    const db = math.multiply(1 / m, math.sum(dZ, 1));
    const dA_prev = math.multiply(math.transpose(W), dZ);
    return { dA_prev, dW, db };
};

const backprop = (AL, Y, caches) => {
    const grads = {};
    const L = caches.length;
    const m = AL.size()[1];
    const dAL = math.dotDivide(Y, AL).map(val => -val).add(math.dotDivide(math.subtract(1, Y), math.subtract(1, AL)));
    const currentCache = caches[L - 1];
    let { dA_prev, dW, db } = oneLayerBackward(dAL, currentCache);

    grads['dA' + L] = dA_prev;
    grads['dW' + L] = dW;
    grads['db' + L] = db;

    for (let l = L - 2; l >= 0; l--) {
        const currentCache = caches[l];
        ({ dA_prev, dW, db } = oneLayerBackward(grads['dA' + (l + 2)], currentCache));
        grads['dA' + (l + 1)] = dA_prev;
        grads['dW' + (l + 1)] = dW;
        grads['db' + (l + 1)] = db;
    }

    return grads;
};

// Update parameters
const updateParameters = (parameters, grads, learningRate) => {
    const L = Object.keys(parameters).length / 2;
    for (let l = 0; l < L; l++) {
        parameters['W' + (l + 1)] = math.subtract(parameters['W' + (l + 1)], math.multiply(learningRate, grads['dW' + (l + 1)]));
        parameters['b' + (l + 1)] = math.subtract(parameters['b' + (l + 1)], math.multiply(learningRate, grads['db' + (l + 1)]));
    }
    return parameters;
};

// Training function
const train = async (X, Y, layerDims, epochs, learningRate, batchSize) => {
    let params = initParams(layerDims);
    const costHistory = [];
    const m = X.size()[1];

    for (let i = 0; i < epochs; i++) {
        const permutation = math.randomInt(0, m, m);
        const X_shuffled = X.subset(math.index(math.range(0, X.size()[0]), permutation));
        const Y_shuffled = Y.subset(math.index(math.range(0, Y.size()[0]), permutation));

        for (let j = 0; j < m; j += batchSize) {
            const X_batch = X_shuffled.subset(math.index(math.range(0, X_shuffled.size()[0]), math.range(j, j + batchSize)));
            const Y_batch = Y_shuffled.subset(math.index(math.range(0, Y_shuffled.size()[0]), math.range(j, j + batchSize)));

            const { A: Y_hat, caches } = forwardProp(X_batch, params);
            const cost = costFunction(Y_hat, Y_batch);
            const grads = backprop(Y_hat, Y_batch, caches);
            params = updateParameters(params, grads, learningRate);
        }

        costHistory.push(cost);
        if (i % 100 === 0) {
            console.log(`Cost after epoch ${i}: ${cost}`);
        }
    }

    return { params, costHistory };
};

// Prediction function
const predict = (X, params) => {
    const { A: Y_hat } = forwardProp(X, params);
    return Y_hat.map(val => (val > 0.5 ? 1 : 0));
};

class OneHotEncoder {
    constructor() {
        this.categories = null;
    }

    fit(y) {
        this.categories = Array.from(new Set(y));
        return this;
    }

    transform(y) {
        if (!this.categories) {
            throw new Error('OneHotEncoder not fitted yet.');
        }
        const categoryMap = new Map(this.categories.map((cat, idx) => [cat, idx]));
        return y.map(value => {
            const row = Array(this.categories.length).fill(0);
            row[categoryMap.get(value)] = 1;
            return row;
        });
    }

    fitTransform(y) {
        return this.fit(y).transform(y);
    }
}


// Load and preprocess data
const loadAndPreprocessData = async (csvFile) => {
    const data = [];

    return new Promise((resolve, reject) => {
        fs.createReadStream(csvFile)
            .pipe(csv())
            .on('data', (row) => {
                const rowData = Object.values(row).map(Number);
                if (rowData.length > 0) {
                    data.push(rowData);
                }
            })
            .on('end', () => {
                if (data.length === 0) {
                    return reject(new Error("CSV data is empty or invalid"));
                }

                const X = data.map(row => row.slice(1).map(val => val / 255));
                const y = data.map(row => row[0]);

                if (X.length === 0 || y.length === 0) {
                    return reject(new Error("Parsed data arrays are empty"));
                }

                try {
                    const X_train = math.matrix(X); // Convert to Math.js matrix
                    const y_train = y; // Keep y as a 1D array

                    // Example: Implement your own OneHotEncoder function
                    const OneHotEncoder = (y_train) => {
                        // Implement your one-hot encoding logic here
                        throw new Error('OneHotEncoder is not implemented');
                    };

                    const Y_train_encoded = OneHotEncoder(y_train);
                    const Y_train = math.matrix(Y_train_encoded).transpose(); // Convert to Math.js matrix

                    resolve({ X_train, Y_train });
                } catch (err) {
                    reject(new Error(`Error creating matrices: ${err.message}`));
                }
            })
            .on('error', reject);
    });
};

// Save parameters to a file
const saveParams = (params, filename) => {
    fs.writeFileSync(filename, JSON.stringify(params));
};

// Load parameters from a file
const loadParams = (filename) => {
    const data = fs.readFileSync(filename, 'utf8');
    return JSON.parse(data);
};

// Main function
const main = async () => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const question = (query) => new Promise((resolve) => rl.question(query, resolve));

    const startTime = Date.now();
    const csvFile = 'Data/mnist_train.csv';
    const { X_train, Y_train } = await loadAndPreprocessData(csvFile);

    const layerDims = [784, 16, 16, 10];
    const learningRate = 0.01;
    const batchSize = parseInt(await question('Data Processing Size?: '), 10);

    const mode = (await question('Enter mode (train/retrain/predict): ')).trim().toLowerCase();
    const paramsFile = 'Data/model_params.json';

    if (mode === 'train') {
        const epochs = parseInt(await question('How many iterations?: '), 10);
        const { params, costHistory } = await train(X_train, Y_train, layerDims, epochs, learningRate, batchSize);
        saveParams(params, paramsFile);
        console.log('Training completed and parameters saved.');
    } else if (mode === 'retrain') {
        const epochs = parseInt(await question('How many iterations?: '), 10);
        const params = loadParams(paramsFile);
        const { params: updatedParams, costHistory } = await train(X_train, Y_train, layerDims, epochs, learningRate, batchSize);
        saveParams(updatedParams, paramsFile);
        console.log('Retraining completed and parameters updated.');
    } else if (mode === 'predict') {
        const { X_test, Y_test } = await loadAndPreprocessData('Data/mnist_test.csv'); // Add test data processing
        const params = loadParams(paramsFile);
        const predictions = predict(X_test, params);
        const accuracy = math.mean(math.equal(math.argmax(predictions, 1), math.argmax(Y_test, 1))) * 100;
        console.log(`Accuracy on test set: ${accuracy}%`);
    } else {
        console.log("Invalid mode. Please choose from 'train', 'retrain', or 'predict'.");
    }

    rl.close();
    console.log(`Time taken: ${((Date.now() - startTime) / 60000).toFixed(2)} mins`);
};

main();