1. Potential biases & mitigation

MNIST biases

Dataset skew: MNIST digits are centered, high-contrast; a model may fail on differently written, rotated, or faded digits.

Mitigation: augment training (rotation, scale, noise); evaluate on diverse handwriting sets; use fairness/robustness metrics.

Amazon Reviews biases

Reviewers self-selection bias, demographic skew, and brand mention frequency differences can bias entity recognition and sentiment.

Mitigation: sample from varied sources; use model calibration; rule-based checks for low-confidence extractions; apply fairness tools (e.g., audit per subgroup).

Tools

TensorFlow Fairness Indicators can show per-slice performance differences (e.g., different product categories).

spaCy rule-based systems allow manual constraints and lower-risk extraction for high-stakes decisions — but still need human review & continuous monitoring.

2. Troubleshooting — common TF bugs and fixes

Dimension mismatch in conv/dense layers

Symptom: errors like "Shapes (None, X) and (None, Y) are incompatible".

Fix: print shapes (model.summary() or tf.shape(tensor)) and ensure flattening or global pooling before dense layers. Use layers.Flatten() or GlobalAveragePooling2D().

Wrong loss for labels

Symptom: ValueError: Shapes (batch, n) and (batch, ) or poor training.

Fix: for multiclass integer labels use sparse_categorical_crossentropy; for one-hot labels use categorical_crossentropy.

Learning stuck / loss NaN

Fix: reduce LR, add BatchNormalization, check data scaling (0-1), clip gradients or use different optimizer.
