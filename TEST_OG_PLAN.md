# Keras Test Suite Porting & 1-to-1 Compatibility Plan

This plan tracks the porting of the official Keras test suite into `zero-keras`, ensuring 1-to-1 API compatibility and numerical equivalence. Each component of `zero-keras` will be validated side-by-side with the official `keras` package using identical inputs.

## Phase 1: Environment & Tooling Setup
- [x] Create `requirements-test.txt` with `pytest`, `pytest-cov`, `numpy`, and `keras` (for reference outputs).
- [ ] Set up a testing utility/harness (e.g., `tests/utils.py`) to easily compare outputs of `zero_keras` vs `keras` using `numpy.testing.assert_allclose`.
- [ ] Configure `pytest` to automatically discover and run the parity tests.
- [ ] Integrate parity tests into the GitHub Actions CI workflow.

## Phase 2: Activations 1-to-1 Parity
- [ ] `test_relu`: Port and assert `allclose`.
- [ ] `test_sigmoid`: Port and assert `allclose`.
- [ ] `test_softmax`: Port and assert `allclose`.
- [ ] `test_softplus`: Port and assert `allclose`.
- [ ] `test_softsign`: Port and assert `allclose`.
- [ ] `test_tanh`: Port and assert `allclose`.
- [ ] `test_selu`: Port and assert `allclose`.
- [ ] `test_elu`: Port and assert `allclose`.
- [ ] `test_exponential`: Port and assert `allclose`.
- [ ] `test_leaky_relu`: Port and assert `allclose`.
- [ ] `test_silu` (swish): Port and assert `allclose`.
- [ ] `test_gelu`: Port and assert `allclose`.
- [ ] `test_mish`: Port and assert `allclose`.
- [ ] `test_linear`: Port and assert `allclose`.

## Phase 3: Initializers 1-to-1 Parity
- [ ] `test_Zeros`: Port and assert shape/value match.
- [ ] `test_Ones`: Port and assert shape/value match.
- [ ] `test_Constant`: Port and assert shape/value match.
- [ ] `test_RandomNormal`: Verify statistical distribution or exact seed equivalence.
- [ ] `test_RandomUniform`: Verify statistical distribution or exact seed equivalence.
- [ ] `test_TruncatedNormal`: Verify statistical distribution.
- [ ] `test_VarianceScaling`: Verify statistical properties.
- [ ] `test_GlorotNormal` / `test_GlorotUniform`: Verify statistical properties.
- [ ] `test_HeNormal` / `test_HeUniform`: Verify statistical properties.
- [ ] `test_LecunNormal` / `test_LecunUniform`: Verify statistical properties.
- [ ] `test_Orthogonal`: Verify statistical properties and orthogonal matrix characteristics.
- [ ] `test_Identity`: Verify shape/value match.

## Phase 4: Losses 1-to-1 Parity
- [ ] `test_MeanSquaredError` (MSE): Port and assert `allclose`.
- [ ] `test_MeanAbsoluteError` (MAE): Port and assert `allclose`.
- [ ] `test_MeanAbsolutePercentageError` (MAPE): Port and assert `allclose`.
- [ ] `test_MeanSquaredLogarithmicError` (MSLE): Port and assert `allclose`.
- [ ] `test_CosineSimilarity`: Port and assert `allclose`.
- [ ] `test_Huber`: Port and assert `allclose`.
- [ ] `test_LogCosh`: Port and assert `allclose`.
- [ ] `test_CategoricalCrossentropy`: Port and assert `allclose`.
- [ ] `test_SparseCategoricalCrossentropy`: Port and assert `allclose`.
- [ ] `test_BinaryCrossentropy`: Port and assert `allclose`.
- [ ] `test_Hinge`: Port and assert `allclose`.
- [ ] `test_SquaredHinge`: Port and assert `allclose`.
- [ ] `test_CategoricalHinge`: Port and assert `allclose`.
- [ ] `test_Poisson`: Port and assert `allclose`.
- [ ] `test_KLDivergence`: Port and assert `allclose`.

## Phase 5: Metrics 1-to-1 Parity
- [ ] `test_Accuracy`: Port and assert identical state updates and results.
- [ ] `test_BinaryAccuracy`: Port and assert identical state updates and results.
- [ ] `test_CategoricalAccuracy`: Port and assert identical state updates and results.
- [ ] `test_SparseCategoricalAccuracy`: Port and assert identical state updates and results.
- [ ] `test_TopKCategoricalAccuracy`: Port and assert identical state updates and results.
- [ ] `test_SparseTopKCategoricalAccuracy`: Port and assert identical state updates and results.
- [ ] `test_Precision`: Port and assert identical state updates and results.
- [ ] `test_Recall`: Port and assert identical state updates and results.
- [ ] `test_TruePositives` / `test_TrueNegatives` / `test_FalsePositives` / `test_FalseNegatives`: Port and assert identical state updates.
- [ ] `test_AUC`: Port and assert identical state updates and results.
- [ ] `test_Mean`: Port and assert identical state updates and results.
- [ ] `test_Sum`: Port and assert identical state updates and results.

## Phase 6: Core Layers 1-to-1 Parity
- [ ] `test_Dense`: Verify parameter shapes, initialization, forward pass, and `get_weights()`/`set_weights()`.
- [ ] `test_Activation`: Verify forward pass.
- [ ] `test_Embedding`: Verify parameter shapes, initialization, forward pass.
- [ ] `test_InputLayer`: Verify behavior and metadata.
- [ ] `test_Dropout`: Verify behavior during `training=True` vs `training=False`.
- [ ] `test_SpatialDropout1D`, `2D`, `3D`: Verify behavior.
- [ ] `test_Flatten`: Verify shape transformations.
- [ ] `test_Reshape`: Verify shape transformations.
- [ ] `test_Permute`: Verify dimension reordering.
- [ ] `test_RepeatVector`: Verify shape transformations.
- [ ] `test_Lambda`: Verify custom function execution.
- [ ] `test_Masking`: Verify masking propagation and behavior.
- [ ] `test_Concatenate` / `test_Average` / `test_Maximum` / `test_Minimum` / `test_Add` / `test_Subtract` / `test_Multiply` / `test_Dot`: Verify forward passes and shape handling.
- [ ] `test_LayerNormalization`: Verify standardization forward pass and parameters.

## Phase 7: Optimizers 1-to-1 Parity
- [ ] `test_SGD`: Verify state updates and step logic for parameters.
- [ ] `test_Adam`: Verify state updates, moments, and step logic.
- [ ] `test_AdamW`: Verify state updates, moments, and step logic.
- [ ] `test_RMSprop`: Verify state updates and step logic.
- [ ] `test_Adadelta`: Verify state updates and step logic.
- [ ] `test_Adagrad`: Verify state updates and step logic.
- [ ] `test_Adamax`: Verify state updates and step logic.
- [ ] `test_Nadam`: Verify state updates and step logic.
- [ ] `test_Ftrl`: Verify state updates and step logic.
- [ ] `test_Lion`: Verify state updates and step logic.
- [ ] `test_LossScaleOptimizer`: Verify scaling logic.

## Phase 8: Integration and Full Model Parity
- [ ] `test_Sequential_Model`: Construct identical Sequential models. Feed random inputs and initialized weights, verify output `allclose`.
- [ ] `test_Functional_Model`: Construct identical Functional API models. Feed random inputs and initialized weights, verify output `allclose`.
- [ ] `test_Model_Compile`: Verify compile arguments (`loss`, `optimizer`, `metrics`) match configuration behavior.
- [ ] `test_Model_Fit`: Verify identical loss trajectories over a small number of steps given identical seeds.
- [ ] `test_Model_Evaluate`: Verify identical evaluation metrics.
- [ ] `test_Model_Predict`: Verify `allclose` predictions.

## Phase 9: Serialization and Config Parity
- [ ] `test_get_config` / `from_config`: Verify identical JSON configurations for all layers/models.
- [ ] `test_save_weights` / `load_weights`: Verify weight formats are compatible.
- [ ] `test_model_saving`: Verify `.keras` archive format compatibility (if supported).
