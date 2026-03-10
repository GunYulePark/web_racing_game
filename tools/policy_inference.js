export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function relu(value) {
  return value > 0 ? value : 0;
}

export function runDenseLayer(input, layer) {
  const rows = Array.isArray(layer.weights) ? layer.weights.length : 0;
  const out = new Array(rows).fill(0);
  for (let row = 0; row < rows; row += 1) {
    const weights = layer.weights[row] || [];
    let acc = Array.isArray(layer.bias) ? (layer.bias[row] || 0) : 0;
    for (let col = 0; col < weights.length; col += 1) {
      acc += (input[col] || 0) * weights[col];
    }
    out[row] = layer.activation === 'relu' ? relu(acc) : acc;
  }
  return out;
}

export function runDenseNetwork(input, policy) {
  if (!policy || !Array.isArray(policy.layers)) {
    throw new Error('policy must define layers');
  }
  let activations = Array.from(input);
  for (const layer of policy.layers) {
    activations = runDenseLayer(activations, layer);
  }
  return activations;
}

export function actionFromPolicy(input, policy) {
  const raw = runDenseNetwork(input, policy);
  if (raw.length < 4) {
    throw new Error('policy output must contain at least 4 values');
  }
  return [
    clamp(raw[0], -1, 1),
    clamp(raw[1], 0, 1),
    clamp(raw[2], 0, 1),
    clamp(raw[3], 0, 1),
  ];
}

export function createHeuristicPolicyStub(name = 'heuristic-stub') {
  return {
    format: 'dense-mlp-v1',
    name,
    note: 'Placeholder structure. Replace with weights exported from rl/export_policy.py.',
    observationDim: 20,
    actionDim: 4,
    layers: [],
  };
}
