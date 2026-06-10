# Exam Sample

<style>
.exam-rendered {
  margin: 0.8rem 0;
}

.exam-question-card {
  margin: 0.8rem 0;
  padding: 0.72rem 0.82rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 10px;
  background: var(--md-default-bg-color);
  box-shadow: 0 0.08rem 0.24rem rgba(0, 0, 0, 0.035);
}

.exam-card-head {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0 0 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}

.exam-number {
  color: var(--md-default-fg-color--light);
  font-size: 0.82rem;
  font-weight: 650;
  line-height: 1;
}

.exam-task {
  font-size: 0.8rem;
  line-height: 1.25;
  font-weight: 650;
  color: var(--md-default-fg-color--light);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.exam-question-card p,
.exam-question-card ul,
.exam-question-card ol,
.exam-question-card table {
  font-size: 0.82rem;
  line-height: 1.46;
}

.exam-question-card p {
  margin: 0.38rem 0;
}

.exam-question-card table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}

.exam-question-card th,
.exam-question-card td {
  padding: 0.38rem 0.48rem;
  vertical-align: top;
}

.exam-question-card pre {
  white-space: pre-wrap !important;
  overflow-x: hidden !important;
  word-break: break-word;
  overflow-wrap: anywhere;
  font-size: 0.7rem;
  line-height: 1.38;
  border-radius: 8px;
}

.exam-question-card code {
  white-space: pre-wrap !important;
  word-break: break-word;
  overflow-wrap: anywhere;
  font-size: 0.7rem;
}

.exam-question-card .highlight,
.exam-question-card .highlight pre {
  overflow-x: hidden !important;
}

.exam-question-card .highlight {
  background: var(--md-code-bg-color);
  border-radius: 8px;
}

.exam-question-card .k,
.exam-question-card .kn {
  color: #7c4dff;
  font-weight: 650;
}

.exam-question-card .s,
.exam-question-card .sd {
  color: #0b7a75;
}

.exam-question-card .c,
.exam-question-card .c1 {
  color: var(--md-default-fg-color--light);
  font-style: italic;
}

.exam-question-card .n {
  color: var(--md-default-fg-color);
}

.exam-question-card .nb,
.exam-question-card .nf {
  color: #0b69a3;
}

.exam-question-card .mi,
.exam-question-card .mf {
  color: #9a6700;
}

.exam-code-main {
  margin: 0.48rem 0;
}

.exam-code-tests {
  margin: 0.55rem 0;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 10px;
  background: var(--md-code-bg-color);
  overflow: hidden;
}

.exam-code-tests summary {
  list-style: none;
  padding: 0.46rem 0.62rem;
  cursor: pointer;
  font-size: 1em;
  font-weight: 650;
  color: var(--md-default-fg-color);
  background: color-mix(in srgb, var(--md-code-bg-color) 82%, var(--md-default-fg-color) 6%);
  user-select: none;
}

.exam-code-tests summary::-webkit-details-marker {
  display: none;
}

.exam-code-tests summary::before {
  content: "▸";
  display: inline-block;
  margin-right: 0.42rem;
  color: var(--md-accent-fg-color);
  transition: transform 0.15s ease;
}

.exam-code-tests[open] summary {
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}

.exam-code-tests[open] summary::before {
  transform: rotate(90deg);
}

.exam-code-tests pre {
  margin: 0;
  border-radius: 0;
}

.exam-options {
  display: grid;
  gap: 0.36rem;
  margin-top: 0.55rem;
}

.exam-option {
  display: block;
  width: 100%;
  padding: 0.46rem 0.58rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  background: var(--md-code-bg-color);
  color: var(--md-default-fg-color);
  text-align: left;
  cursor: pointer;
  font: inherit;
  font-size: 0.8rem;
  line-height: 1.4;
}

.exam-option:hover {
  border-color: var(--md-accent-fg-color);
}

.exam-option.is-selected {
  border-color: var(--md-accent-fg-color);
  background: color-mix(in srgb, var(--md-accent-fg-color) 7%, var(--md-default-bg-color));
}

.exam-option.is-correct {
  border-color: #2e7d48;
  background: color-mix(in srgb, #2e7d48 9%, var(--md-default-bg-color));
}

.exam-option.is-incorrect {
  border-color: #a6423b;
  background: color-mix(in srgb, #a6423b 9%, var(--md-default-bg-color));
}

.exam-submit-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.6rem;
  margin: 1rem 0 0.35rem;
  padding-top: 0.65rem;
  border-top: 1px solid var(--md-default-fg-color--lightest);
}

.exam-submit {
  border: 1px solid var(--md-accent-fg-color);
  border-radius: 999px;
  padding: 0.36rem 0.78rem;
  background: transparent;
  color: var(--md-accent-fg-color);
  cursor: pointer;
  font: inherit;
  font-size: 0.8rem;
  line-height: 1.2;
}

.exam-submit:hover {
  background: color-mix(in srgb, var(--md-accent-fg-color) 7%, var(--md-default-bg-color));
}

.exam-result {
  display: none;
  padding: 0;
  border: 0;
  background: transparent;
  align-items: center;
  gap: 0.35rem;
}

.exam-result.is-visible {
  display: inline-flex;
}

.exam-score {
  font-size: 0.8rem;
  font-weight: 650;
  color: #2e7d48;
  white-space: nowrap;
}

.exam-comment {
  font-size: 0.8rem;
  color: var(--md-default-fg-color--light);
  white-space: nowrap;
}

.exam-hidden-source {
  display: none !important;
}
</style>

<script>
(function () {
  const answers = {
    6: "C",
    7: "C",
    8: "C",
    9: "A",
    10: "B",
    11: "D",
    12: "C",
    13: "A",
    14: "B",
    15: "D",
    16: "A",
    17: "A",
    18: "B",
    19: "C",
    20: "D"
  };

  function initExamFormatting() {
    const root = document.querySelector(".md-content__inner") || document.querySelector("article") || document.body;
    if (!root || root.dataset.examFormatted === "1") return;

    const questionHeadings = Array.from(root.querySelectorAll("h2")).filter((node) => {
      return /^Question\s+\d+$/i.test(node.textContent.trim());
    });

    if (!questionHeadings.length) return;

    root.dataset.examFormatted = "1";

    const oldRendered = root.querySelector(".exam-rendered");
    if (oldRendered) oldRendered.remove();

    const rendered = document.createElement("div");
    rendered.className = "exam-rendered";

    const selected = new Map();

    questionHeadings.forEach((qHeading) => {
      const number = Number(qHeading.textContent.match(/\d+/)[0]);
      const nodes = [];
      let node = qHeading.nextElementSibling;

      while (node && node.tagName !== "H2" && node.tagName !== "SCRIPT") {
        nodes.push(node);
        node = node.nextElementSibling;
      }

      qHeading.classList.add("exam-hidden-source");
      nodes.forEach((item) => item.classList.add("exam-hidden-source"));

      const card = document.createElement("section");
      card.className = "exam-question-card";
      card.id = qHeading.id || `question-${number}`;

      const hasCode = nodes.some((item) => item.tagName === "PRE" || item.querySelector("pre"));
      const isChoice = Boolean(answers[number]) && !hasCode;
      card.appendChild(makeQuestionHeader(number, hasCode ? "Code" : "Multiple Choice"));

      if (hasCode) {
        renderCodeQuestion(card, nodes);
      } else if (isChoice) {
        renderChoiceQuestion(card, nodes, number, selected);
      } else {
        nodes.forEach((item) => card.appendChild(cloneClean(item)));
      }

      rendered.appendChild(card);
    });

    const submitRow = document.createElement("div");
    submitRow.className = "exam-submit-row";

    const result = document.createElement("div");
    result.className = "exam-result";

    const score = document.createElement("span");
    score.className = "exam-score";
    result.appendChild(score);

    const comment = document.createElement("span");
    comment.className = "exam-comment";
    result.appendChild(comment);

    const submit = document.createElement("button");
    submit.type = "button";
    submit.className = "exam-submit";
    submit.textContent = "Submit";
    submit.addEventListener("click", () => {
      let correct = 0;
      let total = 0;

      Object.keys(answers).forEach((key) => {
        const number = Number(key);
        total += 1;
        if (selected.get(number) === answers[number]) correct += 1;
      });

      rendered.querySelectorAll(".exam-option").forEach((button) => {
        const number = Number(button.dataset.question);
        const label = button.dataset.label;

        button.classList.remove("is-correct", "is-incorrect");

        if (answers[number] === label) {
          button.classList.add("is-correct");
        } else if (selected.get(number) === label) {
          button.classList.add("is-incorrect");
        }
      });

      score.textContent = `Score: ${correct}/${total}`;
      comment.textContent = "Fingers crossed 🤞";
      result.classList.add("is-visible");
    });

    submitRow.appendChild(result);
    submitRow.appendChild(submit);
    rendered.appendChild(submitRow);

    questionHeadings[0].parentNode.insertBefore(rendered, questionHeadings[0]);

  }

  function makeQuestionHeader(number, label) {
    const header = document.createElement("div");
    header.className = "exam-card-head";

    const badge = document.createElement("span");
    badge.className = "exam-number";
    badge.textContent = `${number}.`;
    header.appendChild(badge);

    const task = document.createElement("span");
    task.className = "exam-task";
    task.textContent = label;
    header.appendChild(task);

    return header;
  }

  function renderCodeQuestion(card, nodes) {
    nodes.forEach((item) => {
      const pre = item.tagName === "PRE" ? item : item.querySelector("pre");

      if (!pre) {
        card.appendChild(cloneClean(item));
        return;
      }

      const code = pre.querySelector("code") || pre;
      const text = code.textContent;
      const marker = "# Tests do not modify";

      if (!text.includes(marker)) {
        card.appendChild(cloneClean(item));
        return;
      }

      const index = text.indexOf(marker);
      const mainCode = text.slice(0, index).trimEnd();
      const testCode = text.slice(index).trimStart();

      const main = document.createElement("div");
      main.className = "exam-code-main";
      main.appendChild(makeCodeBlock(mainCode));
      card.appendChild(main);

      const details = document.createElement("details");
      details.className = "exam-code-tests";

      const summary = document.createElement("summary");
      summary.textContent = "Show test cases";
      details.appendChild(summary);

      details.appendChild(makeCodeBlock(testCode));
      card.appendChild(details);
    });
  }

  function renderChoiceQuestion(card, nodes, number, selected) {
    const options = [];
    const stem = [];

    nodes.forEach((item) => {
      const text = item.textContent.trim();

      if (/^[A-D]\.\s/.test(text)) {
        options.push(text);
      } else {
        stem.push(item);
      }
    });

    stem.forEach((item) => card.appendChild(cloneClean(item)));

    const box = document.createElement("div");
    box.className = "exam-options";

    options.forEach((text) => {
      const label = text.slice(0, 1);
      const button = document.createElement("button");

      button.type = "button";
      button.className = "exam-option";
      button.dataset.question = String(number);
      button.dataset.label = label;
      button.textContent = text;

      button.addEventListener("click", () => {
        selected.set(number, label);

        box.querySelectorAll(".exam-option").forEach((option) => {
          option.classList.remove("is-selected", "is-correct", "is-incorrect");
        });

        button.classList.add("is-selected");
      });

      box.appendChild(button);
    });

    card.appendChild(box);
  }

  function makeCodeBlock(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "highlight";

    const pre = document.createElement("pre");
    const code = document.createElement("code");

    code.className = "language-python python";
    code.innerHTML = highlightPython(text);

    pre.appendChild(code);
    wrapper.appendChild(pre);

    return wrapper;
  }

  function highlightPython(text) {
    const pattern = /(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#.*|\b(?:import|from|def|return|pass|if|else|elif|for|while|in|as|None|True|False|assert)\b|\b(?:np|print|len|range|abs|int|float|str|list|dict|set|tuple)\b|\b\d+(?:\.\d+)?\b)/g;
    let html = "";
    let last = 0;

    text.replace(pattern, (match, _unused, offset) => {
      html += escapeHtml(text.slice(last, offset));
      html += wrapPythonToken(match);
      last = offset + match.length;
      return match;
    });

    html += escapeHtml(text.slice(last));
    return html;
  }

  function wrapPythonToken(token) {
    const safe = escapeHtml(token);
    if (token.startsWith("#")) return `<span class="c1">${safe}</span>`;
    if (token.startsWith('"""') || token.startsWith("'''")) return `<span class="sd">${safe}</span>`;
    if (token.startsWith('"') || token.startsWith("'")) return `<span class="s">${safe}</span>`;
    if (/^\d/.test(token)) return `<span class="mi">${safe}</span>`;
    if (/^(np|print|len|range|abs|int|float|str|list|dict|set|tuple)$/.test(token)) return `<span class="nb">${safe}</span>`;
    return `<span class="k">${safe}</span>`;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function cloneClean(node) {
    const clone = node.cloneNode(true);
    clone.classList.remove("exam-hidden-source");

    clone.querySelectorAll(".exam-hidden-source").forEach((child) => {
      child.classList.remove("exam-hidden-source");
    });

    return clone;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initExamFormatting);
  } else {
    initExamFormatting();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(function () {
      setTimeout(initExamFormatting, 0);
    });
  }

  window.addEventListener("hashchange", function () {
    setTimeout(initExamFormatting, 0);
  });
})();
</script>

!!! info
    These questions are from [Spring 2026](../01_syllabus) Final Exam. The original exam was computer-based. The exam took place for 60 minutes with 20 questions:

    - 4 questions (1 code, 3 mc) covered pre-midterm topics (20%)
    - 5 questions (1 code, 4 mc) covered [10_diffusion](../../../notebooks/10_diffusion) (25%).
    - 11 questions (3 code, 8 mc) covered other topics (post-midterm, before DDPM).

## Question 1

Complete and test the full function.

```python
import numpy as np

def forward_sample(x0, betas, t):
    """
    x0: NumPy array containing clean data
    betas: NumPy array containing the DDPM noise schedule
    t: integer timestep

    Return:
    noisy sample x_t at timestep t

    Hints:
    np.random.randn(*A.shape) creates standard Gaussian noise with the same shape as A.
    np.cumprod(a) returns cumulative products: [a0, a0*a1, a0*a1*a2, ...].

    Recall:
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
    where alpha_t = 1 - beta_t and alpha_bar_t is the cumulative product of alpha values up to timestep t.
    """
    pass

# Tests do not modify
np.random.seed(0)
x0 = np.ones((2, 2))
betas = np.array([0.1, 0.2, 0.3])
x_t = forward_sample(x0, betas, 0)
expected = np.array([
    [1.50652563, 1.07522412],
    [1.25818742, 1.65731595]
])
assert x_t.shape == x0.shape
assert np.allclose(x_t, expected)

np.random.seed(1)
x0 = np.ones((2, 2))
betas = np.array([0.1, 0.2, 0.3])
x_t = forward_sample(x0, betas, 1)
expected = np.array([
    [1.70805091, 0.52481707],
    [0.56904592, 0.28076651]
])
assert x_t.shape == x0.shape
assert np.allclose(x_t, expected)

np.random.seed(2)
x0 = np.array([[1.0, -1.0]])
betas = np.array([0.1, 0.2, 0.3])
x_t = forward_sample(x0, betas, 2)
expected = np.array([[0.41641841, -0.74955676]])
assert x_t.shape == x0.shape
assert np.allclose(x_t, expected)

np.random.seed(3)
x0 = np.array([[0.5, -0.5], [1.0, -1.0]])
betas = np.array([0.05, 0.10, 0.20, 0.30])
x_t = forward_sample(x0, betas, 3)
expected = np.array([
    [1.63726258, -0.03084216],
    [0.76161931, -2.03728708]
])
assert x_t.shape == x0.shape
assert np.allclose(x_t, expected)
```

## Question 2

Complete and test the full function.

```python
import numpy as np

def attention_score(q, k):
    """
    q: 1D query vector
    k: 1D key vector

    Return:
    scaled dot-product attention score
    """
    pass

# Tests do not modify
q = np.array([1.0, 2.0])
k = np.array([3.0, 4.0])

score = attention_score(q, k)
assert np.allclose(score, 7.77817459)

q = np.array([2.0, 0.0, -1.0])
k = np.array([1.0, 3.0, 2.0])

score = attention_score(q, k)
assert np.allclose(score, 0.0)

q = np.array([1.0, 1.0, 1.0, 1.0])
k = np.array([2.0, 2.0, 2.0, 2.0])

score = attention_score(q, k)
assert np.allclose(score, 4.0)
```

## Question 3

Complete and test the full function.

```python
import numpy as np

def kl_divergence(P, Q):
    """
    P : numpy array of shape (N, C), true distributions
    Q : numpy array of shape (N, C), model distributions

    TODO: Compute the average KL divergence D_KL(P || Q) across all N examples.

    Hint: you can use axis=1 for summing over classes.
    """
    pass

# Tests do not modify
P = np.array([[0.5, 0.5]])
Q = np.array([[0.5, 0.5]])
assert np.isclose(kl_divergence(P, Q), 0.0)

P = np.array([[0.9, 0.1]])
Q = np.array([[0.5, 0.5]])
expected = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
assert np.isclose(kl_divergence(P, Q), expected)

P = np.array([
    [0.8, 0.2],
    [0.6, 0.4]
])
Q = np.array([
    [0.7, 0.3],
    [0.5, 0.5]
])
expected = (
    (0.8 * np.log(0.8 / 0.7) + 0.2 * np.log(0.2 / 0.3)) +
    (0.6 * np.log(0.6 / 0.5) + 0.4 * np.log(0.4 / 0.5))
) / 2
assert np.isclose(kl_divergence(P, Q), expected)

print("Tests passed.")
```

## Question 4

Complete and test the full function.

```python
import numpy as np

def reparameterize(mean, logvar):
    """
    mean: NumPy array containing encoder means
    logvar: NumPy array containing encoder log-variances

    Return:
    latent sample z using the VAE reparameterization trick

    Hint:
    np.random.randn(N, M) generates an array of shape (N, M)
    whose entries are sampled from the standard normal distribution N(0, 1).
    """
    pass

# Tests do not modify
np.random.seed(0)
mean = np.zeros((2, 3))
logvar = np.zeros((2, 3))
z = reparameterize(mean, logvar)
expected = np.array([
    [1.76405235, 0.40015721, 0.97873798],
    [2.24089320, 1.86755799, -0.97727788]
])
assert z.shape == (2, 3)
assert np.allclose(z, expected)

np.random.seed(1)
mean = np.ones((2, 2))
logvar = np.zeros((2, 2))
z = reparameterize(mean, logvar)
expected = np.array([
    [2.62434536, 0.38824359],
    [0.47182825, -0.07296862]
])
assert z.shape == (2, 2)
assert np.allclose(z, expected)

np.random.seed(2)
mean = np.array([[1.0, 2.0]])
logvar = np.array([[np.log(4.0), np.log(9.0)]])
z = reparameterize(mean, logvar)
expected = np.array([[0.16648431, 1.83119952]])
assert z.shape == mean.shape
assert np.allclose(z, expected)
```

## Question 5

Complete and test the full function.

```python
import numpy as np

def momentum_step(param, grad, v=1.0, lr=0.1, beta=0.9):
    """
    param: scalar current parameter value
    grad: scalar gradient at the current parameter
    v: scalar velocity state
    lr: learning rate
    beta: momentum coefficient

    Return:
    tuple (v_next, param_next) of updated velocity and parameter

    Hint:
    velocity is an exponential moving average of gradients.
    """
    pass

# Tests do not modify
param = 5.0
grad = 4.0

v_next, param_next = momentum_step(param, grad)

assert abs(param_next - 4.87) < 1e-9
assert abs(v_next - 1.3) < 1e-9

print("Tests passed.")
```

## Question 6

A convolutional architecture uses DenseNet-style connections, where each layer receives the concatenation of all previous feature maps rather than only the output of the previous layer. What best explains the practical advantage of this design?

A. Concatenation guarantees that spatial resolution remains unchanged throughout the entire network.

B. Concatenation replaces nonlinear activation functions by preserving the raw outputs of all earlier convolution layers.

C. Each layer can directly access earlier representations, encouraging feature reuse and improving gradient flow through the network.

D. Concatenating feature maps ensures that the number of channels remains constant across layers, simplifying the architecture design.

## Question 7

A neural network is trained using standard gradient descent with a fixed learning rate. The loss surface contains a long, narrow valley: gradients are large in one direction but small along the direction that leads toward the minimum. During training the parameters oscillate and the progress toward the minimum is slow. A researcher switches to an optimizer that keeps a running average of past gradients and uses it to update the parameters. Which explanation best describes why this change improves convergence?

A. The optimizer introduces noise into the gradient updates, allowing the parameters to escape poor local minima more easily.

B. The introduced optimizer reduces the learning rate whenever gradients become large, preventing the parameters from overshooting the minimum.

C. Averaging recent gradients reduces oscillations across steep directions and encourages updates that follow consistent directions along the valley.

D. The optimizer estimates second derivatives of the loss function and rescales the gradient based on curvature of the loss surface.

## Question 8

"Multi-head attention is helpful mainly because it increases total parameter count. If we gave a single head the same total width, it would recover the same essential behavior." Which explanation is correct?

A. A single wider head is effectively equivalent, because any interaction pattern that several heads can represent can be merged into one larger attention map without loss.

B. Multi-head attention mainly helps optimization by forcing each head to specialize on different sequence lengths, which one full-width head cannot express explicitly.

C. Several heads matter because each head learns its own projections, so the model can compare the same sequence through different representational views before recombining them.

D. The main benefit comes from reducing head dimension, because smaller heads automatically preserve sharper positional distinctions than one wider head can maintain.

## Question 9

A student says: "The forward process destroys the image, so the final noisy image should still contain a weak hidden version of the original image. Otherwise, the reverse process would have nothing to recover." Which explanation is correct?

A. The final noisy image should be close to standard Gaussian noise, because generation starts from this simple distribution.

B. The final noisy image should keep visible traces of the data, because the sampler needs them as a reconstruction guide.

C. The final noisy image must preserve the original image, because generation reconstructs hidden structure from the starting noise.

D. The final noisy image should be partly corrupted, because full Gaussian noise would make the reverse chain impossible to learn.

## Question 10

A student says: "During generation, a VAE should pass a real input image through the encoder first, sample $z$ from approximate posterior, and then decode it. If we discard the encoder and sample $z$ only from the prior, the model is no longer using what it learned during training." Which explanation is correct?

A. The encoder and decoder are both discarded during generation because the component needed to sample from the learned data distribution is ELBO.

B. The encoder is discarded for unconditional generation because new samples are produced by drawing $z$ from the prior and passing it through the learned decoder.

C. The encoder replaces the prior after training, because the approximate posterior becomes the true source distribution from which all new latent vectors should be drawn.

D. The encoder must be used during generation because the decoder was trained only on latent vectors produced from real input images, not on samples from the prior.

## Question 11

A student is asked to explain why sequence modeling is needed at all, instead of using a separate set of parameters for each token position in a sequence. They propose several explanations. Which explanation is correct?

A. Sequence models mainly exist to reduce vocabulary size, since using the same parameters across time forces rare tokens to share the same embedding statistics.

B. Sequence models are preferred because each time step should have exactly the same hidden state, and repeated parameters guarantee that identical representation throughout the sequence.

C. Sequence models are necessary because probabilities at different time steps must be independent, and parameter sharing is the simplest way to enforce that independence.

D. Sequence models reuse the same update rule across time, which allows them to handle variable-length inputs and learn a general temporal pattern instead of separate position-specific rules.

## Question 12

You are building a next-character predictor. You compare two setups:

- Model A uses a fixed window of size 1 (bigram).
- Model B uses a fixed window of size 5.

Both are simple feedforward models trained with the same loss. During evaluation, you notice that Model B performs better on validation data. But when you shuffle characters inside each window during inference, Model B's performance drops sharply, while Model A is unaffected. Which explanation is correct?

A. Model A is invariant to input permutations because bigram models implicitly learn position-independent token embeddings.

B. Model B fails because larger windows require normalization across positions, and shuffling breaks this normalization assumption.

C. Model B relies on ordered context within the window, so shuffling destroys meaningful dependencies, while Model A only depends on the last character and is unaffected.

D. Model B overfits to the exact training sequences, so shuffling introduces unseen tokens that were not present in the training vocabulary.

## Question 13

A student says: "Since DDPM starts generation from random noise, the model is basically learning to turn any random tensor directly into an image. The intermediate reverse steps are just repeated refinements." Which explanation is correct?

A. The model learns one denoising step at a time, and generation works by repeatedly applying those learned local steps.

B. The model first predicts a clean image from noise, and later reverse steps gradually remove small remaining artifacts.

C. The model learns a separate generator for each step, and sampling chooses the best generated image among them.

D. The model directly maps random noise to an image, and the reverse steps only improve sharpness and color quality.

## Question 14

"Causal masking already gives the decoder a notion of order, since each position can see a different prefix. Once that triangular structure is present, positional encoding is no longer necessary." Which explanation is correct?

A. The mask already encodes relative order fully, so positional encoding mainly helps the feed-forward sublayer distinguish tokens that share similar embeddings.

B. The mask restricts which positions may be used, but it does not itself encode where visible tokens lie, so positional information is still needed inside the attended representations.

C. Positional encoding becomes unnecessary once masking is applied, because the decoder can reconstruct absolute order from the length of each visible prefix during training.

D. The triangular mask is already enough, because the model can infer token order directly from which positions are visible and which positions are hidden at each row.

## Question 15

In a convolutional layer, two important hyperparameters are padding and stride. These choices directly affect the output feature map size, how much boundary information is preserved, and how quickly spatial resolution is reduced. Which option best explains the practical purpose of choosing specific padding and stride values in a CNN?

A. Padding is mainly used to prevent gradients from vanishing near the image boundaries, while stride is used to stabilize optimization by reducing the number of kernel multiplications. Larger padding is typically chosen only in deeper layers because early layers rarely need boundary information.

B. Padding is mainly used to increase the number of trainable parameters so the CNN can learn more complex patterns, while stride is used to reduce overfitting by skipping pixels during convolution. Larger padding and larger stride are usually chosen together because they maximize feature extraction efficiency.

C. Padding is used to ensure that convolution always produces a smaller output than the input, which is necessary for feature extraction, while stride is used to guarantee translation invariance. Larger stride often improves accuracy because it forces the network to focus only on global patterns.

D. Padding is used to preserve spatial size and avoid losing too much border information, especially in early layers, while stride is used to control downsampling and computational cost. Larger stride reduces output resolution faster, which speeds up computation but may discard fine spatial details.

## Question 16

You train a basic encoder-decoder model for machine translation without attention. It performs acceptably on short sentences, but quality drops sharply on longer sentences with several important details. What is the most plausible explanation?

A. The encoder compresses the whole source sentence into a fixed-size final state, which can become a bottleneck when the input is long or information-dense.

B. The model fails because recurrent decoders require bidirectional encoders whenever source and target sentences differ in word order across the two languages.

C. The translation quality drops because longer sentences always require beam search during training, even if greedy decoding is used later during inference.

D. The decoder cannot generate long outputs unless the source and target vocabularies are approximately the same size and contain comparable punctuation patterns.

## Question 17

Before training on the full translation dataset, you repeatedly train the seq2seq model on a single fixed batch. Even after many steps, the loss barely moves. What is the most justified conclusion?

A. This strongly suggests a bug or optimization problem, because a correctly implemented model should usually be able to overfit one fixed batch.

B. This is normal when teacher forcing is used, because feeding correct previous tokens prevents the model from learning the batch-specific mapping efficiently.

C. This mainly shows that the vocabulary is too large, because overfitting a single batch is possible only when the target vocabulary has fewer than one hundred tokens.

D. This is expected, because recurrent encoder-decoder models are not designed to memorize a single batch and usually need large-scale data before loss can decrease.

## Question 18

A student says: "If the VAE decoder receives $z$ as input, then $z$ can be any random vector with the correct dimension. The prior $p(z)$ is only a convenient source of noise, so sampling from a uniform distribution or from a much wider Gaussian should work about the same after training." Which explanation is correct?

A. The decoder can use any latent distribution after training, as long as the sampled vectors have the same dimensionality as the encoder output.

B. The decoder is trained with latent codes regularized toward the chosen prior, so generation should sample from that same prior distribution.

C. The prior only affects the KL-divergence term during training, while the reconstruction term determines all latent vectors the decoder can use later.

D. The exact sampling distribution does not matter because the decoder learns a general mapping from latent vectors to images, not a probability model tied to one prior.

## Question 19

A student says: "The timestep embedding is just an extra label telling the U-Net which step it is on. The noisy image already contains the real information, so the embedding should not affect denoising much." Which explanation is correct?

A. The embedding replaces image features at high noise levels, so the U-Net can generate images from time alone.

B. The embedding is mainly used to order samples in the batch, so the U-Net can process timesteps consistently.

C. The embedding conditions the U-Net on the noise level, so the same image features can be interpreted differently at different stages.

D. The embedding is mostly decorative because the visible noise level already tells the U-Net how much denoising is needed.

## Question 20

A student says: "In DDPM training, we should generate a noisy image by repeatedly applying the forward process step by step until the chosen timestep. Otherwise, the model will not learn the real diffusion chain." Which explanation is correct?

A. Direct noising is enough during training because the model only needs highly corrupted images close to pure Gaussian noise.

B. Step-by-step noising is required during training because each intermediate image teaches the model a separate reverse transition.

C. Step-by-step noising is required during training because direct noising removes the Markov structure from the diffusion process.

D. Direct noising is enough during training because the chosen noisy image has the same distribution as if all earlier steps had been simulated.


