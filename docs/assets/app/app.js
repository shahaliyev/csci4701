(() => {
    const rootSel = ".md-content__inner";
    const pad = 10;
    const closeDelayMs = 120;
  
    let tClose = null;
    let openWrap = null;
  
    const closeAll = (root) => {
      root.querySelectorAll(".fn-refwrap.fn-open")
        .forEach(x => x.classList.remove("fn-open"));
      openWrap = null;
    };
  
    const placeTooltip = (refwrap) => {
      const a = refwrap.querySelector("a.fn-ref");
      const tip = refwrap.querySelector(".fn-tooltip");
      if (!a || !tip) return;
  
      tip.style.left = "0px";
      tip.style.top = "0px";
  
      const ar = a.getBoundingClientRect();
  
      tip.style.display = "block";
      const tr = tip.getBoundingClientRect();
  
      let left = ar.left + (ar.width / 2) - (tr.width / 2);
      left = Math.max(pad, Math.min(left, window.innerWidth - tr.width - pad));
  
      let top = ar.top - tr.height - 10;
      if (top < pad) top = ar.bottom + 10;
      top = Math.max(pad, Math.min(top, window.innerHeight - tr.height - pad));
  
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
      tip.style.display = "";
    };
  
    const jumpTo = (id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.scrollIntoView({ block: "start", behavior: "auto" });
      history.replaceState(null, "", `#${id}`);
    };
  
    const build = () => {
  
      const root = document.querySelector(rootSel);
      if (!root) return;
  
      const existing = root.querySelector(".fn-footnotes");
      if (existing) existing.remove();
  
      closeAll(root);
  
      root.querySelectorAll("span.fn")
        .forEach(fn => fn.querySelectorAll(".fn-refwrap").forEach(x => x.remove()));
  
      const bodies = Array.from(root.querySelectorAll("span.fn > span.fn-body"))
        .filter(Boolean);
  
      if (!bodies.length) return;
  
      const wrap = document.createElement("div");
      wrap.className = "fn-footnotes";
  
      const ol = document.createElement("ol");
      wrap.appendChild(ol);
  
      bodies.forEach((body, i) => {
  
        const n = i + 1;
        const html = body.innerHTML.trim();
  
        const refId = `fnref-${n}`;
        const fnId = `fn-${n}`;
  
        const fn = body.closest(".fn");
  
        fn.insertAdjacentHTML(
          "afterbegin",
          `<span class="fn-refwrap">
            <sup><a class="fn-ref" href="#${fnId}" id="${refId}">${n}</a></sup>
            <span class="fn-tooltip">${html}</span>
          </span>`
        );
  
        const li = document.createElement("li");
        li.id = fnId;
        li.innerHTML = `${html} <a class="fn-backref" href="#${refId}">↩</a>`;
        ol.appendChild(li);
  
      });
  
      root.appendChild(wrap);
  
      root.addEventListener("mouseover", (e) => {
  
        const wrap = e.target.closest(".fn-refwrap");
        if (!wrap) return;
  
        if (tClose) {
          clearTimeout(tClose);
          tClose = null;
        }
  
        if (openWrap && openWrap !== wrap)
          openWrap.classList.remove("fn-open");
  
        openWrap = wrap;
        wrap.classList.add("fn-open");
        placeTooltip(wrap);
  
      });
  
      root.addEventListener("mouseout", (e) => {
  
        const fromWrap = e.target.closest(".fn-refwrap");
        const toWrap = e.relatedTarget?.closest?.(".fn-refwrap");
        const toTip = e.relatedTarget?.closest?.(".fn-tooltip");
  
        if (!fromWrap || toWrap || toTip) return;
  
        if (tClose) clearTimeout(tClose);
  
        tClose = setTimeout(() => {
          if (openWrap) openWrap.classList.remove("fn-open");
          openWrap = null;
        }, closeDelayMs);
  
      });
  
      root.addEventListener("click", (e) => {
  
        const ref = e.target.closest("a.fn-ref");
        const back = e.target.closest("a.fn-backref");
  
        if (ref) {
          e.preventDefault();
          closeAll(root);
          jumpTo(ref.getAttribute("href").slice(1));
          return;
        }
  
        if (back) {
          e.preventDefault();
          closeAll(root);
          jumpTo(back.getAttribute("href").slice(1));
          return;
        }
  
        if (!e.target.closest(".fn-tooltip"))
          closeAll(root);
  
      });
  
      window.addEventListener("scroll", () => closeAll(root), { passive: true });
      window.addEventListener("resize", () => closeAll(root), { passive: true });
  
    };
  
    document.addEventListener("DOMContentLoaded", build);
  
    if (window.document$ && typeof window.document$.subscribe === "function")
      window.document$.subscribe(build);
  
  })();