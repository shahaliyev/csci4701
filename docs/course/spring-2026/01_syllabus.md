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

| Class | Topic | Learning Outcomes | Assessment | Materials |
|---|---|---|---|---|
| 1 | **Deep Learning Overview / Course Structure** | Describe the scope of deep learning and the course syllabus. Fulfill technological requirements. | | [introduction/01_deep_learning](../../../introduction/01_deep_learning) |
| 2 | **Mathematics of Deep Learning: Calculus / Linear Algebra** |  Compute partial derivatives and apply the chain rule. Work with vectors, matrices, and tensors; apply norms and inner products. Understand intuition behind eigenvectors and SVD. | | [mathematics/01_calculus](../../../mathematics/01_calculus) <br> [mathematics/02_linear_algebra](../../../mathematics/02_linear_algebra) <br> [supplementary/svd](../../../supplementary/svd) |
| 3 | **Gradient Descent / Backpropagation I** | Compute gradients on computational graphs. Perform forward and backward passes. Understand gradient descent updates and automatic differentiation (PyTorch autograd, micrograd). | | [notebooks/01_backprop](../../../notebooks/01_backprop) |
| 4 | **Gradient Descent / Backpropagation II** | Implement full backpropagation. | **Feb 3:** Quiz 1 \[1–3\]<br>Last day to submit team member details | [notebooks/01_backprop](../../../notebooks/01_backprop) | 
| 5 | **Activation Functions / Neuron** | Implement activation functions and understand non-linearity. Backpropagate over an N-dimensional neuron. | |  [notebooks/02_neural_network](../../../notebooks/02_neural_network)  |
| 6 | **Multilayer Perceptron (MLP) / Cross-Entropy** | Construct an MLP from stacked neurons. Train a simple MLP classifier on a small dataset. Understand cross-entropy loss. | **Feb 10:** Project proposal deadline |  [notebooks/02_neural_network](../../../notebooks/02_neural_network) |
| 7 | **Images as Tensors / Training MLP on MNIST Dataset** | Understand image representations, tensor shapes, and batching. Use torchvision datasets and dataloaders. Train an MLP on MNIST. | **Feb 12:** Quiz 2 \[5–6\] | [notebooks/03_cnn_torch](../../../notebooks/03_cnn_torch) |
| 8 | **Convolutional Neural Networks (CNN)** | Define and implement 2D cross-correlation (convolution) and pooling with kernels, including padding and stride. Train a LeNet-style CNN on MNIST. Compare MLP with CNN. | | [notebooks/03_cnn_torch](../../../notebooks/03_cnn_torch) |
| 9 | **Mathematics of Deep Learning: Probability Theory** | Describe random variables; distinguish discrete and continuous distributions; work with PMF/PDF. Compute expectation, variance, and covariance. Use conditional probability, independence, and Bayes’ rule. Recognize common distributions. | **Feb 19:** Quiz 3 \[7–8\] | [mathematics/03_probability](../../../mathematics/03_probability) |
| 10 | **Regularization / Initialization** | Recall overfitting and understand how regularization helps with it. Apply data augmentation. Apply weight decay and dropout. Handle exploding and vanishing gradients. Use Xavier and He initialization.| | [notebooks/04_regul_optim](../../../notebooks/04_regul_optim) |
| 11 | **Optimization** | Distinguish local minima from saddle points in training dynamics. Adjust learning rate and apply schedules. Use stochastic gradient decent (SGD) and explain its purpose. Apply momentum, RMSProp, and Adam to optimize learning. Compare optimizers based on convergence behavior and practical performance. | | [notebooks/04_regul_optim](../../../notebooks/04_regul_optim) |
| 12 | **Training CNN on CIFAR-10 Dataset / Hyperparameter Tuning** | Train a regularized and optimized CNN on CIFAR-10. Apply hyperparameter tuning. | **Mar 3:** Quiz 4 \[10–11\] | [notebooks/04_regul_optim](../../../notebooks/04_regul_optim) |
| 13 | **CNN Architectures / Batch Normalization** | Compare classical and modern CNN architectures (e.g. AlexNet, VGG, Inception, EfficientNet) in terms of depth, parameter count, and training purpose and stability. Understand how receptive field influence predictions. Explain why normalization helps training deep networks. Implement batch normalization and understand training vs evaluation behavior. Understand batch-size effects and when to prefer layer normalization. | |  [notebooks/05_batchnorm_resnet](../../../notebooks/05_batchnorm_resnet) <br> [ [tensorspace.js](https://tensorspace.org/html/playground/index.html) ] |
| 14 | **Residual Network / Transfer Learning / Fine-tuning** | Understand residual (skip) connections. Explain the deep reasoning why residual blocks help training big models. Describe how pretrained CNNs enable transfer learning, and how fine-tuning adapts them to new datasets. Demonstrate fine-tuning of an ImageNet-pretrained CNN on CIFAR-10. | | [notebooks/05_batchnorm_resnet](../../../notebooks/05_batchnorm_resnet) |
| 15 | **Paper Reading: AlexNet** | Discuss AlexNet paper, its key ideas, and determine what is outdated. Describe the paper structure. | **Mar 12:** Quiz 5 \[13-14\]<br>Project milestone 1 deadline  | [[ alexnet ](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)]|
| 16 | **Midterm Exam** | — | **Mar 17:** Midterm Exam \[1–15\] | |
| 17 | **Midterm Exam Review** | Half-semester overview. | | |
| — | **Holidays** | — | **Mar 20–30** | | 
| 18 | **Mathematics of deep learning: Information Theory and Probabilistic Modeling** | Compute entropy, cross-entropy, and KL divergence. Derive cross-entropy loss from maximum likelihood. Interpret common losses as probabilistic objectives. | | [mathematics/04_information](../../../mathematics/04_information) <br> [mathematics/05_prob_modeling](../../../mathematics/05_prob_modeling) <br>  [ [probabalistic models](https://www.cs.toronto.edu/~rgrosse/courses/csc311_f20/readings/L07%20Probabilistic%20Models.pdf) ]  |
| 19 | **Sequence Modeling: Tokenization / Bigram Model / Perplexity** | Understand the aims of sequence modeling. Tokenize and build a character-level bigram model and sample from it. Implement average negative log-likelihood loss and perplexity. | | [notebooks/06_nn_ngram](../../../notebooks/06_nn_ngram) |
| 20 | **Neural N-gram Language Model** | Construct a neural N-gram model and train it with mini-batch updates. | **Apr 7:** Quiz 6 \[18–19\] | [notebooks/06_nn_ngram](../../../notebooks/06_nn_ngram) | 
| 21 | **Autoregressive Modeling: RNN / LSTM** | Explain autoregressive modeling. Describe how RNNs maintain state. Implement RNN and LSTM and identify their limitations. | | |
| 22 | **Attention Mechanism** | Understand attention as weighted information selection. Derive queries, keys, and values at the tensor level. Implement attention with matrix operations and verify shapes and normalization. | **Apr 14:** Quiz 7 \[20–21\] | |
| 23 | **Transformer Architecture / Self-Attention** | Explain self-attention and the motivation for Transformer models. Describe token embeddings and positional encoding. Explain multi-head attention and how attention heads capture different relationships. Assemble a Transformer block from self-attention, normalization, residual connections, and feed-forward layers. Trace tensor shapes through the model and discuss scaling behavior and training stability. |  | |
| 24 | **Paper Reading: Transformer, Vision Transformer, Swin Transformer** | Extract core architectural ideas and compare attention for sequences vs images. Discuss scalability and efficiency constraints. | **Apr 21:** Quiz 8 \[22–23\] | |
| 25 | **Variational Autoencoders I** | Introduce latent-variable generative models. Explain latent representations and probabilistic encoders/decoders. Explain approximate inference and why variational methods are needed. | | [notebooks/notebooks/07_vae](../../../notebooks/07_vae) |
| 26 | **Variational Autoencoders II** | Understand the VAE objective (ELBO). Implement a VAE. Interpret reconstruction and regularization terms and their trade-off. |  **Apr 28:** Project milestone 2 deadline | [notebooks/notebooks/07_vae](../../../notebooks/07_vae) |
| 27 | **Generative Adversarial Networks** | Explain adversarial training between a generator and discriminator. Formulate the GAN objective as a minimax game. Implement a basic GAN and examine training dynamics. Analyze common failure modes such as mode collapse and instability, and discuss stabilization techniques (e.g. normalization) | | |
| 28 | **Diffusion Models** | Introduce diffusion as generative modeling via gradual noise corruption and learned denoising. Describe the forward noising process and the reverse denoising model. Derive the training objective and connect it to score-based learning. Explain sampling as iterative probabilistic inference and discuss computational trade-offs and scalability. | **May 5:** Quiz 9 \[25–27\] | |
| 29 | **Foundation Models and Modern Trends** | Explain large-scale pretraining and transfer learning. Examine GPT, BERT, CLIP, and latent diffusion models (LDMs). Discuss scaling behavior and limitations. | | |
| — | **Final Exam** | — | **Tuesday, May 12:** Final Exam \[1–29\] |
