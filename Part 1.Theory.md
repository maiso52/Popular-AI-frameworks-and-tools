Q1 — Primary differences: TensorFlow vs PyTorch (straight, strategic)

Computation model

PyTorch: dynamic (eager) execution by default → code reads & debugs like regular Python. Favoured in research and fast prototyping.

TensorFlow: historically static graph (TF1) but modern TF2 uses eager by default and tf.function to compile graphs. Better options for production/serving historically.

Ecosystem & production

TensorFlow: strong production tools (TensorFlow Serving, TF Lite, TF.js), good for deploying models to mobile, edge, and production pipelines.

PyTorch: strong research-first ecosystem, now closing the gap with TorchServe, TorchScript, ONNX export.

APIs & ergonomics

PyTorch: more “pythonic”, simpler debugging, transparent tensors/nn modules.

TensorFlow: higher-level Keras API is simple for many users; lower-level TF gives optimization leverage.

When to choose

Choose PyTorch for research, fast iteration, custom models, and if you prefer readable Python-first code.

Choose TensorFlow if you need mature production/serving tools, mobile/edge integration, or specific TF ecosystem features.

Q2 — Two use cases for Jupyter Notebooks in AI

Exploratory data analysis & visualization — iterate quickly over data transforms, plots, and inspect samples inline.

Research / prototyping — build and test model ideas, train small experiments, document hyperparameters and results alongside code (reproducible record).

Q3 — How spaCy improves NLP vs basic Python string ops

Robust tokenization that understands punctuation, contractions, hyphens, multi-word expressions.

Syntactic parsing & POS tagging giving structure (subject/object) rather than ad-hoc regex.

Named Entity Recognition (NER): pretrained models extract proper nouns (PRODUCT, ORG) with probabilities.

Pipelines (tokenizer → tagger → parser → NER) are optimized and fast; string ops are brittle and fail on edge cases, languages, or messy text.

Comparative analysis — Scikit-learn vs TensorFlow
Aspect	Scikit-learn	TensorFlow
Target applications	Classical ML: regression, SVMs, trees, clustering, pipelines	Deep learning / neural networks (CNNs, RNNs, transformers), large-scale differentiable models
Ease for beginners	Very approachable API, quick to get models working on tabular data	Keras API in TF2 simplifies NN building; deeper concepts required for advanced customization
Community support	Longstanding, lots of examples for standard ML tasks	Huge, active community with many deep-learning tutorials & production tooling
