class NeuralNetwork {
  static ActivationFunctions = {
    Sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    ReLU: (x) => Math.max(0, x),
    Step: (x, limit) => (x > limit ? 1 : 0),
  };

  constructor(neuronCounts) {
    this.levels = [];
    for (let i = 0; i < neuronCounts.length - 1; i++) {
      this.levels.push(new Level(neuronCounts[i], neuronCounts[i + 1]));
    }
  }

  toJSON() {
    return {
      levels: this.levels.map((level) => level.toJSON()),
    };
  }

  static fromJSON(object) {
    const network = new NeuralNetwork([]);
    network.levels = object.levels.map((level) => Level.fromJSON(level));
    return network;
  }

  static fetch(uri) {
    return new Promise((resolve, reject) => {
      fetch(uri)
        .then((response) => {
          if (!response.ok) throw new Error(`Modelo não encontrado em ${uri}`);
          return response.json();
        })
        .then((data) => {
          const network = NeuralNetwork.fromJSON(data);
          resolve(network);
        })
        .catch((err) => reject(err));
    });
  }

  static train(network, inputs, outputs, learningRate = 0.1) {
    if (inputs.length !== outputs.length) {
      throw new Error(
        "O número de entradas deve corresponder ao número de saídas."
      );
    }

    for (let i = 0; i < inputs.length; i++) {
      let predictedOutputs = this.feedForward(inputs[i], network);

      let errors = [];
      if (Array.isArray(outputs[i])) {
        for (let j = 0; j < outputs[i].length; j++) {
          if (isNaN(predictedOutputs[j])) {
            throw new Error(
              `Saída prevista é NaN para input[${i}]: ${inputs[i]}`
            );
          }
          errors.push(outputs[i][j] - predictedOutputs[j]);
        }
      } else {
        if (isNaN(predictedOutputs[0])) {
          throw new Error(
            `Saída prevista é NaN para input[${i}]: ${inputs[i]}`
          );
        }
        errors.push(outputs[i] - predictedOutputs[0]);
      }

      for (let k = network.levels.length - 1; k >= 0; k--) {
        let level = network.levels[k];
        let nextErrors = new Array(level.inputs.length).fill(0);

        for (let o = 0; o < level.outputs.length; o++) {
          let outputError = errors[o];
          if (isNaN(outputError)) {
            throw new Error(
              `Erro de saída é NaN na camada ${k}, neurônio ${o}`
            );
          }

          level.biases[o] += outputError * learningRate;

          for (let i = 0; i < level.inputs.length; i++) {
            nextErrors[i] += outputError * level.weights[i][o];
            level.weights[i][o] += level.inputs[i] * outputError * learningRate;
          }
        }

        errors = nextErrors;
      }
    }
  }

  static feedForward(inputs, network) {
    let outputs = inputs;
    for (let i = 0; i < network.levels.length; i++) {
      outputs = Level.feedForward(outputs, network.levels[i]);
    }
    return outputs;
  }

  static mutate(network, amount = 1) {
    network.levels.forEach((level) => {
      for (let i = 0; i < level.biases.length; i++) {
        level.biases[i] = NeuralNetwork.lerp(
          level.biases[i],
          Math.random() * 2 - 1,
          amount
        );
        if (isNaN(level.biases[i])) {
          throw new Error(`Bias inválido após mutação em nível ${i}`);
        }
      }
      for (let i = 0; i < level.weights.length; i++) {
        for (let j = 0; j < level.weights[i].length; j++) {
          level.weights[i][j] = NeuralNetwork.lerp(
            level.weights[i][j],
            Math.random() * 2 - 1,
            amount
          );
          if (isNaN(level.weights[i][j])) {
            throw new Error(`Peso inválido após mutação em [${i}][${j}]`);
          }
        }
      }
    });
  }

  static meanAbsolutePercentageError(yTrue, yPred) {
    const n = yTrue.length;
    let error = 0;

    for (let i = 0; i < n; i++) {
      if (yTrue[i] !== 0) error += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
    }

    return error / n; // Retorna erro percentual
  }

  static normalizeFeature(values) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    return values.map((value) => (value - min) / (max - min));
  }

  static unnormalizeFeature(inputs, value) {
    const min = Math.min(...inputs);
    const max = Math.max(...inputs);
    return value * (max - min) + min;
  }

  static lerp(A, B, t) {
    return A + (B - A) * t;
  }

  static applyDropout(inputs, dropoutRate = 0.2) {
    return inputs
      .map((value) => (Math.random() < dropoutRate ? null : value))
      .filter((value) => value !== null);
  }
}

class Level {
  constructor(inputCount, outputCount) {
    this.inputs = new Array(inputCount).fill(0);
    this.outputs = new Array(outputCount).fill(0);
    this.biases = new Array(outputCount).fill(0);

    this.weights = [];
    for (let i = 0; i < inputCount; i++) {
      this.weights[i] = new Array(outputCount).fill(0);
    }

    Level.randomize(this);
  }

  static randomize(level) {
    for (let i = 0; i < level.weights.length; i++) {
      for (let j = 0; j < level.weights[i].length; j++) {
        level.weights[i][j] = Math.random() * 2 - 1; // Gera valores entre -1 e 1
      }
    }

    for (let i = 0; i < level.biases.length; i++) {
      level.biases[i] = Math.random() * 2 - 1; // Inicializa os vieses entre -1 e 1
    }
  }

  static feedForward(inputs, level) {
    level.inputs = inputs.slice(); // Garante que os inputs sejam copiados corretamente

    for (let i = 0; i < level.outputs.length; i++) {
      let sum = 0;
      for (let j = 0; j < level.inputs.length; j++) {
        if (isNaN(level.inputs[j]) || isNaN(level.weights[j][i])) {
          throw new Error(
            `Valor inválido detectado: input[${j}] ou weight[${j}][${i}] é NaN`
          );
        }
        sum += level.inputs[j] * level.weights[j][i];
      }
      sum += level.biases[i];

      // Aplica a função de ativação
      if (isNaN(sum)) {
        throw new Error(`Soma inválida para saída[${i}]: ${sum}`);
      }

      level.outputs[i] = NeuralNetwork.ActivationFunctions.ReLU(sum);
    }

    return level.outputs;
  }

  toJSON() {
    return {
      inputs: this.inputs,
      outputs: this.outputs,
      biases: this.biases,
      weights: this.weights,
    };
  }

  static fromJSON(object) {
    let level = new Level(object.inputs.length, object.outputs.length);
    level.inputs = object.inputs;
    level.outputs = object.outputs;
    level.biases = JSON.parse(JSON.stringify(object.biases));
    level.weights = JSON.parse(JSON.stringify(object.weights));
    return level;
  }
}

class Attention {
  constructor(inputSize) {
    this.queryWeights = Attention.initializeMatrix(inputSize, inputSize); // Pesos para Queries
    this.keyWeights = Attention.initializeMatrix(inputSize, inputSize);   // Pesos para Keys
    this.valueWeights = Attention.initializeMatrix(inputSize, inputSize); // Pesos para Values
  }

  computeAttention(inputs) {
    // 1. Calcula Queries (Q), Keys (K), e Values (V)
    const queries = Attention.multiply(inputs, this.queryWeights); // [n, d]
    const keys = Attention.multiply(inputs, this.keyWeights);     // [n, d]
    const values = Attention.multiply(inputs, this.valueWeights); // [n, d]

    // 2. Calcula a Similaridade entre Queries e Keys
    const scores = Attention.multiply(queries, Attention.transpose(keys)); // [n, n]

    // 3. Escala os Scores pelo Raiz Quadrada da Dimensão
    const dK = Math.sqrt(keys[0].length); // d = dimensão de cada vetor
    const scaledScores = scores.map(row => row.map(value => value / dK)); // [n, n]

    // 4. Normaliza os Scores usando Softmax
    const attentionScores = scaledScores.map(row => Attention.softmax(row)); // [n, n]

    // 5. Calcula o Context Vector (Pesos x Values)
    const contextVectors = attentionScores.map(row => {
      return row.reduce((sum, score, index) => {
        const value = values[index];
        return sum.map((val, i) => val + score * value[i]);
      }, new Array(values[0].length).fill(0));
    }); // [n, d]

    return {
      scores: attentionScores,      // Pontuações de atenção [n, n]
      weightedInputs: contextVectors, // Entradas ponderadas [n, d]
      contextVectors,               // Vetores de contexto finais [n, d]
    };
  }

  static initializeMatrix(rows, cols) {
    const limit = 1///Math.sqrt(6 / (rows + cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() * 2 * limit - limit)
    );
  }

  
  static transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
  }

  
  static softmax(scores) {
    const maxScore = Math.max(...scores); // Evita overflow
    const exps = scores.map(score => Math.exp(score - maxScore));
    const sumExps = exps.reduce((sum, value) => sum + value, 0);
    return exps.map(value => value / sumExps);
  }

  
  static multiply(matrixA, matrixB) {
    if (matrixA[0].length !== matrixB.length) {
      throw new Error("Dimensões incompatíveis para multiplicação.");
    }
    const result = Array.from({ length: matrixA.length }, () =>
      new Array(matrixB[0].length).fill(0)
    );
    for (let i = 0; i < matrixA.length; i++) {
      for (let j = 0; j < matrixB[0].length; j++) {
        for (let k = 0; k < matrixA[0].length; k++) {
          result[i][j] += matrixA[i][k] * matrixB[k][j];
        }
      }
    }
    return result;
  }
}

function calculateSHAP(model, inputs, featureIndex) {
  // Clona os inputs originais
  const inputsCopy = JSON.parse(JSON.stringify(inputs));

  // Remove a feature (substitui pelo valor neutro, como média ou mediana)
  const originalFeature = inputsCopy[0][featureIndex];
  inputsCopy[0][featureIndex] = 0.5; // Exemplo: substituindo pela média

  // Faz as previsões com e sem a feature
  const predictionWithFeature = NeuralNetwork.feedForward(inputs[0], model);
  const predictionWithoutFeature = NeuralNetwork.feedForward(inputsCopy[0], model);

  // Calcula a diferença
  return predictionWithFeature - predictionWithoutFeature;
}