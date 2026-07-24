(() => {
  const KATEX_CSS_ATTR = "data-katex-css"

  function katexCssHref() {
    const scripts = document.getElementsByTagName("script")
    for (const s of scripts) {
      if (s.src && s.src.includes("/assets/app/katex.js")) {
        return s.src.replace(/app\/katex\.js(?:\?.*)?$/, "vendor/katex/katex.min.css")
      }
    }
    return "../../assets/vendor/katex/katex.min.css"
  }

  function ensureKatexCss() {
    if (document.querySelector(`link[${KATEX_CSS_ATTR}]`)) {
      return Promise.resolve()
    }
    return new Promise((resolve) => {
      const link = document.createElement("link")
      link.rel = "stylesheet"
      link.href = katexCssHref()
      link.setAttribute(KATEX_CSS_ATTR, "")
      link.onload = () => resolve()
      link.onerror = () => resolve()
      document.head.appendChild(link)
    })
  }

  document$.subscribe(({ body }) => {
    ensureKatexCss().then(() => {
      renderMathInElement(body, {
        delimiters: [
          { left: "$$",  right: "$$",  display: true },
          { left: "$",   right: "$",   display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ],
      })
    })
  })
})()
