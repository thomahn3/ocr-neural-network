# ocr-neural-network
- An Optical Character Recgonition neural network in python
- Uses MNIST Datasets in .csv form
- No Tensorflow, or pytorch

## Downsides
- No gpu functionality therefore slow as built from scratch

## Info
- I realised python's limited ML functionality as it doesn't utilise the GPU
- Tried to make the equivalent in JS but as to no prevail
- Benchmarks with an Macbook Air m2 16gb in `ocr/data.txt`

## Python Usage
1. Clone repository
2. Install requirements
```
pip install -r requirements.txt
```
3. Downlaod [datasets](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and put into `ocr/data`
4. Run ocr.py and answer prompts in the terminal;
  - Procesing Size: The number of rows in the csv to keep in ram at a given time
  - Iterations: How many backpropigations

## JS Usage (Not working)
1. Clone repository
2. Install requirements
```
npm i
```
3. Downlaod [datasets](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and put into `ocr/data`
4. Run JS file with:
```
node ocr.js
```
