# Syllabus

<style>
/* Style only the syllabus table */
h1 + table,
h1 ~ table:first-of-type {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.95rem;
}

h1 + table th,
h1 ~ table:first-of-type th {
  background: var(--md-default-fg-color--lightest);
  font-weight: 600;
  padding: 0.75rem 0.8rem;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
  text-align: left;
}

h1 + table td,
h1 ~ table:first-of-type td {
  padding: 0.7rem 0.8rem;
  vertical-align: top;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}

h1 + table tr:hover td,
h1 ~ table:first-of-type tr:hover td {
  background: color-mix(
    in srgb,
    var(--md-default-bg-color) 92%,
    var(--md-accent-fg-color) 8%
  );
}

h1 + table td:first-child,
h1 ~ table:first-of-type td:first-child {
  width: 4.5rem;
  white-space: nowrap;
  color: var(--md-default-fg-color--light);
}

h1 + table td:last-child,
h1 ~ table:first-of-type td:last-child {
  width: 16rem;
}

table th:first-child,
table td:first-child {
  width: 2.8rem;
  white-space: nowrap;
  text-align: right;
}

</style>

<div class="admonition warning">
  <p class="admonition-title">Important</p>
  <p>
    The content of this syllabus is subject to change. Please consistently check the course page on Blackboard and the <a href="https://www.ada.edu.az/en/academics/academic-calendar">ADA University Academic Calendar</a> for modifications. The last day of the add/drop period, holidays, and other academic deadlines are noted in the calendar.
  </p>
</div>

<div class="admonition info">
  <p class="admonition-title">Info</p>
  <p>
   Square brackets in the <em>Assessment / Notes</em> column indicate the range of classes whose material is covered by the assessment. For example, <em>Quiz 1 [1–3]</em> means that the quiz assesses material covered in classes 1 through 3.
  </p>
</div>

| Class | Topic | Learning Outcomes | Assessment / Notes |
|---|---|---|---|
| 1 | **Deep Learning (DL) Overview / Course Structure** | Describe the scope of DL and the course syllabus. Fulfill technological requirements. | |
| 2 | **Mathematics of DL: Linear Algebra / Calculus** | Work with vectors, matrices, and tensors; apply norms and inner products. Compute partial derivatives and apply the chain rule. Optional: intuition behind eigenvectors and SVD. | |
| 3 | **Gradient Descent / Backpropagation I** | Compute gradients on computational graphs. Perform forward and backward passes. Understand gradient descent updates and automatic differentiation (PyTorch autograd, micrograd). | |
| 4 | **Gradient Descent / Backpropagation II** | Implement full backpropagation. | **Feb 3:** Quiz 1 \[1–3\]<br>Last day to submit team member details |
| 5 | **Activation Functions / Neuron** | Implement activation functions and understand non-linearity. Backpropagate over an N-dimensional neuron. | |
| 6 | **Multilayer Perceptron (MLP)** | Construct an MLP from stacked neurons. Train a simple MLP classifier on a small dataset. | **Feb 10:** Project proposal deadline |
| 7 | **Images as Tensors / MLP on MNIST / Batching & Cross-Entropy** | Understand image representations, tensor shapes, and batching. Use torchvision datasets and dataloaders. Train an MLP on MNIST with SGD + cross-entropy. | **Feb 12:** Quiz 2 \[5–6\] |
| 8 | **Convolutional Neural Networks (CNN)** | Define and implement 2D cross-correlation (convolution) and pooling with kernels, including padding and stride. Train a LeNet-style CNN on MNIST. Compare MLP with CNN. | |
| 9 | **Mathematics of DL: Probability Theory** | Describe random variables; distinguish discrete and continuous distributions; work with PMF/PDF. Compute expectation, variance, and covariance. Use conditional probability, independence, and Bayes’ rule. Recognize common distributions. | **Feb 19:** Quiz 3 \[7–8\] |
| 10 | **Regularization** | Apply weight decay and dropout. Handle exploding and vanishing gradients. Use Xavier and He initialization. Distinguish local minima from saddle points in training dynamics. | |
| 11 | **Optimization** | Adjust learning rate and apply schedules. Use SGD with momentum. Apply RMSProp and Adam. Compare optimizers based on convergence behavior and practical performance. | |
| 12 | **Regularization / Optimization** | Train a regularized CNN on CIFAR-10 using optimizers. Apply hyperparameter tuning. | **Mar 3:** Quiz 4 \[10–11\] |
| 13 | **Paper: AlexNet** | Discuss AlexNet, its key ideas, what is outdated, and the paper structure. | |
| 14 | **Bigram Model / Negative Log-Likelihood / Softmax** | Build a character-level bigram model and sample from it. Distinguish probability vs likelihood. Compute average negative log-likelihood as a loss. Explain the purpose of softmax. | |
| 15 | **Neural Network N-gram Model / Mini-Batch Training** | Construct a neural N-gram model. Train the model with mini-batch updates. | **Mar 12:** Quiz 5 \[14\]<br>Project milestone 1 deadline |
| 16 | **Midterm Exam** | — | **Tuesday, Mar 17:** Midterm Exam \[1–15\] |
| 17 | **Midterm Exam Review** | Half-semester overview. | |
| — | **Holidays** | — | **Mar 20–30** |
| 18 | **Batch Normalization / Layer Normalization** | Explain why normalization helps training deep networks. Implement batch normalization and understand training vs evaluation behavior. Understand batch-size effects and when to prefer layer normalization. | |
| 19 | **Residual Blocks / Residual Network for NLP** | Understand residual (skip) connections. Add a residual block to a feed-forward N-gram model with correct dimensions. Connect residuals to vanishing gradients and regularization. | |
| 20 | **Sequence Modeling: Autoregressive Models and RNN/LSTM** | Explain autoregressive modeling beyond fixed context windows. Describe how RNNs maintain state. Identify limitations of RNN/LSTM/GRU. | **Apr 7:** Quiz 6 \[18–19\] |
| 21 | **Attention Mechanism** | Understand attention as weighted information selection. Derive queries, keys, and values at the tensor level. Implement attention with matrix operations and verify shapes and normalization. | |
| 22 | **Transformer Architecture / Self-Attention** | Explain self-attention and Transformer blocks. Explain how Transformers scale. | **Apr 14:** Quiz 7 \[20–21\] |
| 23 | **Transformer Blocks** | Assemble a Transformer block from self-attention and feed-forward sublayers. Trace signal flow. Analyze training stability and sensitivity to initialization and learning rate. | |
| 24 | **Paper Reading: Transformer, Vision Transformer, Swin Transformer** | Extract core architectural ideas and compare attention for sequences vs images. Discuss scalability and efficiency constraints. | **Apr 21:** Quiz 8 \[22–23\] |
| 25 | **Mathematics of DL: Information Theory and Probabilistic Modeling** | Compute entropy, cross-entropy, and KL divergence. Derive cross-entropy loss from maximum likelihood. Interpret common losses as probabilistic objectives. | |
| 26 | **Variational Autoencoders I** | Introduce latent-variable generative models. Explain latent representations and probabilistic encoders/decoders. Explain approximate inference and why variational methods are needed. | **Apr 28:** Project milestone 2 deadline |
| 27 | **Variational Autoencoders II** | Understand the VAE objective (ELBO). Implement a VAE. Interpret reconstruction and regularization terms and their trade-off. | |
| 28 | **Generative Adversarial Networks (GAN) / Diffusion Models / Score Matching** | Explain adversarial training between generator and discriminator. Describe common failure modes (mode collapse, instability) and stabilization techniques. Formulate diffusion via forward noising and learned reverse denoising. Interpret the training objective as denoising score matching. Explain sampling as iterative probabilistic inference. | **May 5:** Quiz 9 \[25–27\] |
| 29 | **Foundation Models and Modern Trends** | Explain large-scale pretraining and transfer learning. Examine GPT, BERT, CLIP, and latent diffusion models (LDMs). Discuss scaling behavior and limitations. | |
| — | **Final Exam** | — | **Tuesday, May 12:** Final Exam \[1–29\] |
