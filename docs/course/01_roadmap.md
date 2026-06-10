# Roadmap

The course roadmap for **CSCI 4701: Deep Learning**. Every node is a link — hover to see related topics highlighted. See the navigation bar for Syllabus and other related information.


<style>
.course-roadmap {
  --rm-gap: 0.55rem;
  --rm-radius: 0.65rem;
  --rm-node-h: 3.85rem;
  --rm-node-w: 8.4rem;
  --rm-phase-w: 7rem;
  --rm-cluster-pad: 0.55rem;
  --rm-border: 1.25px solid color-mix(in srgb, var(--md-default-fg-color--light) 62%, transparent);
  --rm-dash: 1.5px dashed color-mix(
    in srgb,
    var(--md-accent-fg-color) 58%,
    var(--md-default-fg-color--light) 42%
  );
  --rm-orange: #c45c3e;
  --rm-green: #4f9d69;
  --rm-blue: #3b82f6;
  --rm-purple: #8b5cf6;
  font-size: 0.58rem;
  line-height: 1.12;
  margin: 1.2rem 0 2rem;
  max-width: 100%;
}

.course-roadmap * {
  box-sizing: border-box;
}

.course-roadmap .rm-block {
  margin-bottom: 0.75rem;
}

.course-roadmap .rm-cluster {
  border: var(--rm-dash);
  border-radius: 0.85rem;
  padding: var(--rm-cluster-pad);
  background: color-mix(in srgb, var(--md-default-bg-color) 95%, var(--md-accent-fg-color) 5%);
  min-width: 0;
  overflow: hidden;
}

.course-roadmap .rm-label {
  display: flex;
  align-items: center;
  min-height: 1.35rem;
  font-size: 0.53rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  opacity: 0.72;
  margin-bottom: 0.45rem;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: normal;
  line-height: 1.08;
}

.course-roadmap a.rm-node {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: var(--rm-node-h);
  width: 100%;
  height: 100%;
  padding: 0.28rem 0.38rem;
  border: var(--rm-border);
  border-radius: var(--rm-radius);
  background: var(--md-code-bg-color);
  color: var(--md-default-fg-color);
  text-decoration: none;
  text-align: center;
  overflow: hidden;
  overflow-wrap: anywhere;
  word-break: normal;
  hyphens: none;
  font-weight: inherit;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
}

.course-roadmap a.rm-node:hover {
  border-color: var(--md-accent-fg-color);
  background: color-mix(in srgb, var(--md-accent-fg-color) 15%, var(--md-code-bg-color) 85%);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--md-accent-fg-color) 34%, transparent);
}

.course-roadmap a.rm-node.rm-lit {
  border-color: color-mix(in srgb, var(--rm-purple) 58%, var(--md-default-fg-color--light) 42%);
  background: color-mix(in srgb, var(--rm-purple) 10%, var(--md-code-bg-color) 90%);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--rm-purple) 24%, transparent);
}

.course-roadmap a.rm-ml {
  border-color: var(--rm-orange);
  background: color-mix(in srgb, var(--rm-orange) 10%, var(--md-code-bg-color) 90%);
  font-weight: inherit;
}

.course-roadmap a.rm-ml:hover {
  border-color: var(--rm-orange);
  background: color-mix(in srgb, var(--rm-orange) 17%, var(--md-code-bg-color) 83%);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--rm-orange) 38%, transparent);
}

.course-roadmap a.rm-python {
  border-color: var(--rm-green);
  background: color-mix(in srgb, var(--rm-green) 10%, var(--md-code-bg-color) 90%);
  font-weight: inherit;
}

.course-roadmap a.rm-python:hover {
  border-color: var(--rm-green);
  background: color-mix(in srgb, var(--rm-green) 17%, var(--md-code-bg-color) 83%);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--rm-green) 38%, transparent);
}

.course-roadmap a.rm-phase {
  border-color: color-mix(in srgb, var(--rm-blue) 72%, var(--md-default-fg-color--light) 28%);
  background: color-mix(in srgb, var(--rm-blue) 13%, var(--md-code-bg-color) 87%);
  font-weight: 700;
}

.course-roadmap .rm-num {
  display: block;
  font-weight: 700;
  margin-bottom: 0.18rem;
}

/* Shared 4-column grid: phase | narrow | wide | narrow — aligns all rows */
.course-roadmap .rm-grid-row {
  display: grid;
  grid-template-columns:
    var(--rm-phase-w)
    calc(var(--rm-node-w) + var(--rm-cluster-pad) * 2)
    calc(var(--rm-node-w) * 2 + var(--rm-gap) + var(--rm-cluster-pad) * 2)
    calc(var(--rm-node-w) + var(--rm-cluster-pad) * 2);
  gap: var(--rm-gap);
  align-items: stretch;
  width: fit-content;
  max-width: 100%;
}


.course-roadmap .rm-grid-row > * {
  min-width: 0;
  max-width: 100%;
}

.course-roadmap .rm-col-phase { grid-column: 1; z-index: 1; }
.course-roadmap .rm-col-narrow { grid-column: 2; z-index: 1; }
.course-roadmap .rm-col-wide { grid-column: 3; z-index: 1; }
.course-roadmap .rm-col-fm { grid-column: 4; z-index: 2; }
.course-roadmap .rm-span-prereq { grid-column: 1 / 3; z-index: 1; }
.course-roadmap .rm-span-wide { grid-column: 3; z-index: 1; }
.course-roadmap .rm-span-fm { grid-column: 4; z-index: 2; }

.course-roadmap .rm-col-fm .rm-node,
.course-roadmap .rm-span-fm .rm-node {
  position: relative;
  z-index: 1;
}

.course-roadmap .rm-phase-card {
  min-height: calc(var(--rm-node-h) * 2 + var(--rm-gap) + 1.35rem);
}

.course-roadmap .rm-two-plus-centered {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(2, minmax(var(--rm-node-h), 1fr));
  gap: var(--rm-gap);
  width: 100%;
}

.course-roadmap .rm-two-plus-centered .rm-centered {
  grid-column: 1 / -1;
  width: 100%;
  max-width: calc(50% - var(--rm-gap) / 2);
  justify-self: center;
}

.course-roadmap .rm-one-col-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: repeat(2, minmax(var(--rm-node-h), 1fr));
  gap: var(--rm-gap);
  width: 100%;
}

.course-roadmap .rm-math-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: repeat(2, minmax(var(--rm-node-h), 1fr));
  gap: var(--rm-gap);
  width: 100%;
}

.course-roadmap .rm-foundations-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(2, minmax(var(--rm-node-h), 1fr));
  gap: var(--rm-gap);
  width: 100%;
}

.course-roadmap .rm-foundations-grid .rm-centered {
  grid-column: 1 / -1;
  width: 100%;
  max-width: calc(50% - var(--rm-gap) / 2);
  justify-self: center;
}

.course-roadmap .rm-three-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(2, minmax(var(--rm-node-h), 1fr));
  gap: var(--rm-gap);
  width: 100%;
}

.course-roadmap .rm-three-grid .rm-centered {
  grid-column: 1 / -1;
  width: 100%;
  max-width: calc(50% - var(--rm-gap) / 2);
  justify-self: center;
}

.course-roadmap .rm-supp-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(2, minmax(var(--rm-node-h), 1fr));
  gap: var(--rm-gap);
  width: 100%;
}

.course-roadmap .rm-supp-grid .rm-centered {
  grid-column: 1 / -1;
  width: 100%;
  max-width: calc(50% - var(--rm-gap) / 2);
  justify-self: center;
}

@media (max-width: 1100px) {
  .course-roadmap {
    font-size: 0.64rem;
    line-height: 1.15;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  .course-roadmap .rm-label {
    font-size: 0.58rem;
    white-space: nowrap;
    overflow-wrap: normal;
  }

  .course-roadmap a.rm-node {
    white-space: nowrap;
    overflow: visible;
    overflow-wrap: normal;
  }

  .course-roadmap .rm-grid-row {
    min-width: max-content;
  }
}
</style>

<div class="course-roadmap" data-roadmap="csci4701">

  <div class="rm-block">
    <div class="rm-grid-row">

      <div class="rm-cluster rm-span-prereq">
        <div class="rm-label">Prerequisites</div>
        <div class="rm-two-plus-centered">
          <a class="rm-node rm-ml" data-id="ml" href="../introduction/02_machine_learning/">CSCI 4734:<br>Machine Learning</a>
          <a class="rm-node" data-id="dl-intro" href="../introduction/01_deep_learning">Introduction:<br>Deep Learning</a>
          <a class="rm-node rm-python rm-centered" data-id="python" href="https://cs231n.github.io/python-numpy-tutorial/" target="_blank" rel="noopener">Python &amp;<br>NumPy</a>
        </div>
      </div>

      <div class="rm-cluster rm-span-wide">
        <div class="rm-label">Prerequisite Mathematics</div>
        <div class="rm-two-plus-centered">
          <a class="rm-node" data-id="calc" href="../mathematics/01_calculus">Calculus</a>
          <a class="rm-node" data-id="linalg" href="../mathematics/02_linear_algebra">Linear<br>Algebra</a>
          <a class="rm-node rm-centered" data-id="prob-theory" href="../mathematics/03_probability">Probability<br>Theory</a>
        </div>
      </div>

      <div class="rm-cluster rm-span-fm">
        <div class="rm-label">Mathematics</div>
        <div class="rm-math-grid">
          <a class="rm-node" data-id="info" href="../mathematics/04_information">Information<br>Theory</a>
          <a class="rm-node" data-id="prob-model" href="../mathematics/05_prob_modeling">Probabilistic<br>Modeling</a>
        </div>
      </div>

    </div>
  </div>

  <div class="rm-block">
    <div class="rm-grid-row">

      <a class="rm-node rm-phase rm-phase-card rm-col-phase" data-id="pre-midterm" href="spring-2026/01_syllabus">Pre-midterm</a>

      <div class="rm-cluster rm-col-wide">
        <div class="rm-label">Foundations</div>
        <div class="rm-foundations-grid">
          <a class="rm-node" data-id="backprop" href="../notebooks/01_backprop"><span><span class="rm-num">01</span>Backpropagation</span></a>
          <a class="rm-node" data-id="nn" href="../notebooks/02_neural_network"><span><span class="rm-num">02</span>Neural<br>Network</span></a>
          <a class="rm-node rm-centered" data-id="regul" href="../notebooks/04_regul_optim"><span><span class="rm-num">04</span>Regularization<br>&amp; Optimization</span></a>
        </div>
      </div>

      <div class="rm-cluster rm-col-fm">
        <div class="rm-label">Computer Vision</div>
        <div class="rm-one-col-grid">
          <a class="rm-node" data-id="cnn" href="../notebooks/03_cnn_torch"><span><span class="rm-num">03</span>Convolutional<br>Neural Network</span></a>
          <a class="rm-node" data-id="cnn-arch" href="../notebooks/05_cnn_architectures"><span><span class="rm-num">05</span>Convolutional<br>Architectures</span></a>
        </div>
      </div>

    </div>
  </div>

  <div class="rm-block">
    <div class="rm-grid-row">

      <a class="rm-node rm-phase rm-phase-card rm-col-phase" data-id="post-midterm" href="spring-2026/01_syllabus">Post-midterm</a>

      <div class="rm-cluster rm-col-narrow">
        <div class="rm-label">Neural Language Processing</div>
        <div class="rm-one-col-grid">
          <a class="rm-node" data-id="rnn" href="../notebooks/06_rnn_sequential"><span><span class="rm-num">06</span>Recurrent Neural<br>Networks &amp;<br>Sequential Data</span></a>
          <a class="rm-node" data-id="trf" href="../notebooks/07_transformer"><span><span class="rm-num">07</span>Attention &amp;<br>Transformers</span></a>
        </div>
      </div>

      <div class="rm-cluster rm-col-wide">
        <div class="rm-label">Generative Modeling</div>
        <div class="rm-three-grid">
          <a class="rm-node" data-id="gan" href="../notebooks/08_gan"><span><span class="rm-num">08</span>Generative<br>Adversarial<br>Networks</span></a>
          <a class="rm-node" data-id="vae" href="../notebooks/09_vae"><span><span class="rm-num">09</span>Variational<br>Autoencoders</span></a>
          <a class="rm-node rm-centered" data-id="diff" href="../notebooks/10_diffusion"><span><span class="rm-num">10</span>Diffusion<br>Models</span></a>
        </div>
      </div>

      <div class="rm-cluster rm-col-fm">
        <div class="rm-label">Modern Ecosystem</div>
        <div class="rm-one-col-grid">
          <a class="rm-node" data-id="frontier" href="">Foundation Models<br>&amp; Modern Trends</a>
          <a class="rm-node" data-id="hardware" href="">Hardware &amp;<br>Industry Standards</a>
        </div>
      </div>

    </div>
  </div>

  <div class="rm-block">
    <div class="rm-grid-row">
      <a class="rm-node rm-phase rm-phase-card rm-col-phase" data-id="optional" href="../advanced/">Optional</a>

      <div class="rm-cluster rm-col-wide">
        <div class="rm-label">Supplementary</div>
        <div class="rm-supp-grid">
          <a class="rm-node" data-id="svd" href="../supplementary/svd">Singular Value<br>Decomposition</a>
          <a class="rm-node" data-id="pca" href="../supplementary/pca">Principal<br>Component<br>Analysis</a>
          <a class="rm-node rm-centered" data-id="tsne" href="../supplementary/tsne">t-distributed<br>Stochastic<br>Neighbor Embedding</a>
        </div>
      </div>

      <div class="rm-cluster rm-col-fm">
        <div class="rm-label">Advanced</div>
        <div class="rm-one-col-grid">
          <a class="rm-node" data-id="inp-loss" href="../advanced/inpainting_losses">Inpainting<br>Losses</a>
          <a class="rm-node" data-id="inp-met" href="../advanced/inpainting_metrics">Inpainting<br>Metrics</a>
        </div>
      </div>

    </div>
  </div>

</div>














