import { DISCLOSE } from "./copy.js";
import { esc } from "./text.js";

const PREFIX = "bookworm.expand.";
const controllers = new WeakMap();

export function readState(key) {
  try {
    return window.localStorage.getItem(PREFIX + key) === "1";
  } catch (error) {
    return false;
  }
}

export function writeState(key, open) {
  try {
    window.localStorage.setItem(PREFIX + key, open ? "1" : "0");
  } catch (error) {
    return;
  }
}

const CHEVRON = '<svg class="dz-chev" viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M3 6l5 5 5-5"/></svg>';

export function discloseButton(key, bodyId, place, labels) {
  const L = labels || DISCLOSE;
  return (
    '<button type="button" class="dz-btn' + (place === "end" ? " is-end" : "") + '" data-dz="' + esc(key) + '" aria-expanded="false" aria-controls="' + esc(bodyId) + '">' +
    '<span class="dz-l">' + esc(L.open) + "</span>" + CHEVRON + "</button>"
  );
}

export function discloseBar(key, bodyId, hint) {
  return '<div class="dz-bar">' + discloseButton(key, bodyId) + (hint ? '<p class="dz-hint">' + esc(hint) + "</p>" : "") + "</div>";
}

export function discloseEnd(key, bodyId) {
  return '<div class="dz-endbar">' + discloseButton(key, bodyId, "end") + "</div>";
}

export function paintButton(btn, open, labels) {
  const L = labels || DISCLOSE;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  btn.querySelector(".dz-l").textContent = open ? L.close : L.open;
}

export function bindDisclose(root, key, opts) {
  const o = opts || {};
  const buttons = Array.from(root.querySelectorAll('[data-dz="' + key + '"]'));
  if (!buttons.length) return null;
  const body = document.getElementById(buttons[0].getAttribute("aria-controls"));
  if (!body) return null;
  body.setAttribute("data-dz-body", key);
  let open = o.initial != null ? !!o.initial : readState(key);

  function paint() {
    body.hidden = !open;
    root.classList.toggle("is-dz-open", open);
    buttons.forEach((b) => {
      paintButton(b, open, o.labels);
      if (b.classList.contains("is-end")) b.hidden = !open;
    });
  }

  function set(next, how) {
    const was = open;
    open = !!next;
    paint();
    if (how !== "quiet") writeState(key, open);
    if (was !== open && o.onChange) o.onChange(open);
  }

  buttons.forEach((b) => {
    b.addEventListener("click", () => {
      const fromEnd = b.classList.contains("is-end");
      set(!open);
      if (!open && fromEnd) {
        const top = buttons[0];
        top.scrollIntoView({ block: "center", behavior: "auto" });
        try {
          top.focus({ preventScroll: true });
        } catch (error) {
          top.focus();
        }
      }
    });
  });

  const ctl = { set, isOpen: () => open, body, button: buttons[0] };
  controllers.set(body, ctl);
  paint();
  if (o.onChange) o.onChange(open);
  return ctl;
}

export function reveal(el) {
  let n = el;
  let changed = false;
  while (n && n.nodeType === 1) {
    if (n.hasAttribute("data-dz-body") && n.hidden) {
      const ctl = controllers.get(n);
      if (ctl) {
        ctl.set(true, "quiet");
        changed = true;
      }
    }
    n = n.parentElement;
  }
  return changed;
}
