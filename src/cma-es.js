const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const identity = (size) => Array.from({ length: size }, (_, row) =>
  Array.from({ length: size }, (_, column) => Number(row === column)));

const zeros = (size) => Array.from({ length: size }, () => 0);

const norm = (vector) => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));

const gaussian = (random) => {
  const a = Math.max(Number.EPSILON, random());
  const b = random();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
};

function symmetricEigen(matrix) {
  const size = matrix.length;
  const values = matrix.map((row) => [...row]);
  const vectors = identity(size);
  const maxIterations = size * size * 30;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let p = 0;
    let q = 1;
    let largest = 0;
    for (let row = 0; row < size; row += 1) {
      for (let column = row + 1; column < size; column += 1) {
        const magnitude = Math.abs(values[row][column]);
        if (magnitude > largest) {
          largest = magnitude;
          p = row;
          q = column;
        }
      }
    }
    if (largest < 1e-12) break;

    const angle = 0.5 * Math.atan2(2 * values[p][q], values[q][q] - values[p][p]);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const app = values[p][p];
    const aqq = values[q][q];
    const apq = values[p][q];
    values[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
    values[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
    values[p][q] = values[q][p] = 0;

    for (let index = 0; index < size; index += 1) {
      if (index !== p && index !== q) {
        const aip = values[index][p];
        const aiq = values[index][q];
        values[index][p] = values[p][index] = cosine * aip - sine * aiq;
        values[index][q] = values[q][index] = sine * aip + cosine * aiq;
      }
      const vip = vectors[index][p];
      const viq = vectors[index][q];
      vectors[index][p] = cosine * vip - sine * viq;
      vectors[index][q] = sine * vip + cosine * viq;
    }
  }

  const order = Array.from({ length: size }, (_, index) => index)
    .sort((left, right) => values[right][right] - values[left][left]);
  return {
    values: order.map((index) => Math.max(1e-12, values[index][index])),
    vectors: vectors.map((row) => order.map((index) => row[index])),
  };
}

function transform(vectors, scales, input) {
  return vectors.map((row) => row.reduce(
    (sum, value, column) => sum + value * scales[column] * input[column],
    0,
  ));
}

function inverseTransform(vectors, scales, input) {
  const projected = scales.map((scale, column) => vectors.reduce(
    (sum, row, rowIndex) => sum + row[column] * input[rowIndex],
    0,
  ) / scale);
  return vectors.map((row) => row.reduce(
    (sum, value, column) => sum + value * projected[column],
    0,
  ));
}

export class CmaEsOptimizer {
  #random;
  #covariance;
  #pathCovariance;
  #pathSigma;
  #weights;
  #muEffective;
  #cc;
  #cs;
  #c1;
  #cMu;
  #damping;
  #chiN;
  #sequence = 0;

  constructor({ mean, sigma = 0.18, populationSize, random = Math.random } = {}) {
    if (!Array.isArray(mean) || mean.length < 2 || !mean.every(Number.isFinite)) {
      throw new TypeError("CMA-ES requires a finite mean vector with at least two dimensions.");
    }
    if (typeof random !== "function") throw new TypeError("random must be a function.");
    this.dimension = mean.length;
    this.mean = mean.map((value) => clamp(value, 0, 1));
    this.sigma = clamp(Number(sigma) || 0.18, 0.005, 1);
    this.populationSize = Math.max(4, Math.round(populationSize || 4 + 3 * Math.log(this.dimension)));
    this.parentCount = Math.floor(this.populationSize / 2);
    this.generation = 0;
    this.#random = random;
    this.#covariance = identity(this.dimension);
    this.#pathCovariance = zeros(this.dimension);
    this.#pathSigma = zeros(this.dimension);

    const rawWeights = Array.from({ length: this.parentCount }, (_, index) =>
      Math.log(this.parentCount + 0.5) - Math.log(index + 1));
    const weightSum = rawWeights.reduce((sum, value) => sum + value, 0);
    this.#weights = rawWeights.map((value) => value / weightSum);
    this.#muEffective = 1 / this.#weights.reduce((sum, value) => sum + value * value, 0);
    const n = this.dimension;
    this.#cc = (4 + this.#muEffective / n) / (n + 4 + 2 * this.#muEffective / n);
    this.#cs = (this.#muEffective + 2) / (n + this.#muEffective + 5);
    this.#c1 = 2 / ((n + 1.3) ** 2 + this.#muEffective);
    this.#cMu = Math.min(
      1 - this.#c1,
      2 * (this.#muEffective - 2 + 1 / this.#muEffective) / ((n + 2) ** 2 + this.#muEffective),
    );
    this.#damping = 1 + 2 * Math.max(0, Math.sqrt((this.#muEffective - 1) / (n + 1)) - 1) + this.#cs;
    this.#chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  }

  ask() {
    const eigen = symmetricEigen(this.#covariance);
    const scales = eigen.values.map(Math.sqrt);
    return Array.from({ length: this.populationSize }, () => {
      const z = Array.from({ length: this.dimension }, () => gaussian(this.#random));
      const y = transform(eigen.vectors, scales, z);
      const values = this.mean.map((value, index) => clamp(value + this.sigma * y[index], 0, 1));
      return Object.freeze({
        id: `${this.generation}-${++this.#sequence}`,
        generation: this.generation,
        values: Object.freeze(values),
      });
    });
  }

  tell(evaluations) {
    if (!Array.isArray(evaluations) || evaluations.length !== this.populationSize) {
      throw new RangeError(`CMA-ES expected ${this.populationSize} evaluations.`);
    }
    const ranked = evaluations.map(({ candidate, score }) => {
      if (candidate?.generation !== this.generation || candidate.values?.length !== this.dimension) {
        throw new TypeError("CMA-ES received a candidate from the wrong generation.");
      }
      if (!Number.isFinite(score)) throw new TypeError("CMA-ES scores must be finite.");
      return { candidate, score };
    }).sort((left, right) => right.score - left.score);

    const oldMean = [...this.mean];
    const parents = ranked.slice(0, this.parentCount);
    this.mean = oldMean.map((_, dimension) => parents.reduce(
      (sum, evaluation, index) => sum + this.#weights[index] * evaluation.candidate.values[dimension],
      0,
    ));
    const weightedStep = this.mean.map((value, index) => (value - oldMean[index]) / this.sigma);
    const eigen = symmetricEigen(this.#covariance);
    const inverseStep = inverseTransform(eigen.vectors, eigen.values.map(Math.sqrt), weightedStep);
    const sigmaFactor = Math.sqrt(this.#cs * (2 - this.#cs) * this.#muEffective);
    this.#pathSigma = this.#pathSigma.map((value, index) =>
      (1 - this.#cs) * value + sigmaFactor * inverseStep[index]);

    const pathNorm = norm(this.#pathSigma);
    const normalization = Math.sqrt(1 - (1 - this.#cs) ** (2 * (this.generation + 1)));
    const hSigma = pathNorm / Math.max(1e-12, normalization) / this.#chiN < 1.4 + 2 / (this.dimension + 1);
    const covarianceFactor = Math.sqrt(this.#cc * (2 - this.#cc) * this.#muEffective);
    this.#pathCovariance = this.#pathCovariance.map((value, index) =>
      (1 - this.#cc) * value + Number(hSigma) * covarianceFactor * weightedStep[index]);

    const oldCovariance = this.#covariance.map((row) => [...row]);
    const decay = 1 - this.#c1 - this.#cMu
      + this.#c1 * (1 - Number(hSigma)) * this.#cc * (2 - this.#cc);
    for (let row = 0; row < this.dimension; row += 1) {
      for (let column = 0; column < this.dimension; column += 1) {
        let rankMu = 0;
        parents.forEach((evaluation, index) => {
          const stepRow = (evaluation.candidate.values[row] - oldMean[row]) / this.sigma;
          const stepColumn = (evaluation.candidate.values[column] - oldMean[column]) / this.sigma;
          rankMu += this.#weights[index] * stepRow * stepColumn;
        });
        this.#covariance[row][column] = decay * oldCovariance[row][column]
          + this.#c1 * this.#pathCovariance[row] * this.#pathCovariance[column]
          + this.#cMu * rankMu;
      }
    }
    for (let row = 0; row < this.dimension; row += 1) {
      this.#covariance[row][row] = Math.max(1e-12, this.#covariance[row][row]);
      for (let column = row + 1; column < this.dimension; column += 1) {
        const symmetric = (this.#covariance[row][column] + this.#covariance[column][row]) / 2;
        this.#covariance[row][column] = this.#covariance[column][row] = symmetric;
      }
    }
    this.sigma = clamp(
      this.sigma * Math.exp(this.#cs / this.#damping * (pathNorm / this.#chiN - 1)),
      0.001,
      1.5,
    );
    this.generation += 1;
    return ranked;
  }
}
