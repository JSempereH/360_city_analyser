// Presentation mode: the viewer surveys the dataset by itself, the way an engineer would.
//
// It walks the panorama sequence, turns the camera to each building, clicks it,
// looks at it from every view the survey kept (showing the crop the model reads),
// streams the assistant's reasoning, fills in floors and the GEM structure, "saves"
// (the files are not touched), cross-checks the record with the cadastre when the
// building is matched to it, and moves on. Each walk ends with the map colored by
// what was found and the review queue an engineer would work through; then it
// starts again. The HUD names the phase (walk, look, reason, record, check).
// Space pauses, the right arrow skips to the next building, Escape ends it.
// Everything shown comes from the dataset's attribute files; answers from
// city-analyser-demo are marked as a demo dataset.

import { answerLine, framingOf, visibleReasoning } from "./assistant.mjs";
import { cadastreCheck, planTour, reasoningScript, surveySummary } from "./presentation-script.mjs";

const SIMULATED_PREFIX = "vlm:simulated";

// The phases of one visit, named in the HUD so an audience can follow what the assistant is doing.
const PHASES = [
  ["walk", "Walk", "Walks the street from capture to capture, as in Street View."],
  ["look", "Look", "Finds the building's facade in the panorama (the orange mask) and reads each view the model is given."],
  ["reason", "Reason", "Counts storeys and reads the structure across the views, explaining what it sees."],
  ["record", "Record", "Writes the GEM v3 record (material / lateral system / height) and colors the map."],
  ["check", "Check", "Compares the record with the cadastre and leaves doubtful answers to an engineer."],
];

class Stopped extends Error {}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function easeInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

// api: { host, datasetId(), imageIds(), currentImageId(), showImage(id), selectBuilding(id), fetchSummary(), fetchCard(id),
//        getCamera(), setCamera({ yawDeg, pitchDeg, fov }), bearingToBuilding(id), revealBuilding(id), enter(), exit(),
//        nextDataset(go): the dataset the presentation continues with (null to stay), loaded when `go`,
//        frame(view): the camera framing of a crop's view in the current panorama (assistant.framingOf),
//        showColors(mode): color the map by an attribute, showReviewQueue(on): open or close the review queue,
//        summaryView(on): enlarge the map and frame it on the saved buildings }
export function startPresentation(api, { speed = 1, minScore = 0.03 } = {}) {
  const host = api.host;
  let paused = false;
  let stopped = false;
  let skip = false;
  const root = element("div", "presentation");
  const hud = element("div", "presentation-hud");
  const hudTitle = element("strong", "", "AI structural survey assistant");
  const hudProgress = element("span", "presentation-progress");
  const hudBar = element("span", "presentation-bar");
  const hudFill = element("span");
  hudBar.append(hudFill);
  const hudBadge = element("span", "presentation-badge");
  const hudKeys = element("span", "presentation-keys", "Space pause · → next · Esc exit");
  const hudPhases = element("ol", "presentation-phases");
  const phaseItems = new Map(PHASES.map(([key, label, help]) => {
    const item = element("li", "", label);
    item.title = help;
    hudPhases.append(item);
    return [key, item];
  }));
  hud.append(hudTitle, hudProgress, hudPhases, hudBar, hudBadge, hudKeys);
  // The current phase is lit, the earlier ones of this visit marked done; null clears them.
  const setPhase = (key) => {
    let reached = false;
    for (const [name, item] of [...phaseItems.entries()].reverse()) {
      if (name === key) reached = true;
      item.classList.toggle("active", name === key);
      item.classList.toggle("done", reached && name !== key);
    }
  };
  const frame = (view) => (api.frame ? api.frame(view) : framingOf(view));
  const saved = new Set();
  const panel = element("section", "presentation-panel");
  panel.setAttribute("aria-live", "polite");
  const cursor = element("div", "presentation-cursor");
  cursor.setAttribute("aria-hidden", "true");
  root.append(hud, panel, cursor);
  host.append(root);

  const wait = async (ms) => {
    let left = ms / speed;
    while (left > 0) {
      if (stopped) throw new Stopped();
      if (skip) return;
      const step = Math.min(left, 50);
      await new Promise((resolve) => window.setTimeout(resolve, step));
      if (!paused) left -= step;
    }
  };

  const animate = async (ms, frame) => {
    const duration = ms / speed;
    let elapsed = 0;
    let last = performance.now();
    while (elapsed < duration) {
      if (stopped) throw new Stopped();
      if (skip) break;
      await new Promise((resolve) => window.requestAnimationFrame(resolve));
      const now = performance.now();
      if (!paused) elapsed += now - last;
      last = now;
      frame(easeInOut(Math.min(elapsed / duration, 1)));
    }
    frame(1);
  };

  // The camera moves like Street View: a horizontal pan with the horizon level, then a vertical
  // tilt, then the zoom; never two axes at once. Duration follows the angle, as a hand on a mouse.
  const turnTo = async (target) => {
    const start = api.getCamera();
    const delta = ((target.yawDeg - start.yawDeg + 540) % 360) - 180;
    if (Math.abs(start.pitchDeg) > 1) {
      const from = start.pitchDeg;
      await animate(250 + Math.abs(from) * 18, (t) => api.setCamera({ ...start, pitchDeg: from * (1 - t) }));
    }
    if (Math.abs(delta) > 0.5) {
      // About 60 degrees a second at full speed: slow enough for an audience to follow the turn.
      await animate(700 + Math.abs(delta) * 14, (t) => api.setCamera({ yawDeg: start.yawDeg + delta * t, pitchDeg: 0, fov: start.fov }));
    }
    const level = { yawDeg: target.yawDeg, pitchDeg: 0, fov: start.fov };
    if (Math.abs(target.pitchDeg) > 1) {
      await wait(150);
      await animate(300 + Math.abs(target.pitchDeg) * 25, (t) => api.setCamera({ ...level, pitchDeg: target.pitchDeg * t }));
    }
    if (Math.abs(target.fov - start.fov) > 1) {
      await animate(600, (t) => api.setCamera({ ...level, pitchDeg: target.pitchDeg, fov: start.fov + (target.fov - start.fov) * t }));
    }
  };

  // Walking to another capture, as one does in Street View: face it, step forward, arrive.
  const walkTo = async (imageId) => {
    if (!imageId || imageId === api.currentImageId()) return;
    const order = api.imageIds();
    const from = order.indexOf(api.currentImageId());
    const to = order.indexOf(imageId);
    // Captures in between are walked through one by one when they are few.
    const path = from >= 0 && to >= 0 && Math.abs(to - from) <= 4
      ? Array.from({ length: Math.abs(to - from) }, (_, step) => order[from + Math.sign(to - from) * (step + 1)])
      : [imageId];
    for (const next of path) {
      const heading = api.yawToImage(next);
      const camera = api.getCamera();
      if (heading !== null) await turnTo({ yawDeg: heading, pitchDeg: 0, fov: camera.fov });
      const start = api.getCamera();
      await animate(500, (t) => api.setCamera({ ...start, fov: start.fov - 12 * t }));
      await api.showImage(next);
      const arrived = api.getCamera();
      await animate(600, (t) => api.setCamera({ ...arrived, fov: arrived.fov + 12 * t }));
      await wait(next === imageId ? 300 : 500);
    }
  };

  const moveCursor = async (x, y, ms = 900) => {
    const from = { x: Number.parseFloat(cursor.style.left) || host.clientWidth * 0.62, y: Number.parseFloat(cursor.style.top) || host.clientHeight * 0.75 };
    cursor.classList.add("visible");
    // A slight arc, as a wrist moves.
    const bend = { x: (from.x + x) / 2 + (y - from.y) * 0.18, y: (from.y + y) / 2 - (x - from.x) * 0.18 };
    await animate(ms, (t) => {
      const a = (1 - t) ** 2, b = 2 * (1 - t) * t, c = t * t;
      cursor.style.left = `${a * from.x + b * bend.x + c * x}px`;
      cursor.style.top = `${a * from.y + b * bend.y + c * y}px`;
    });
  };

  const click = async () => {
    cursor.classList.remove("clicking");
    void cursor.offsetWidth;
    cursor.classList.add("clicking");
    await wait(350);
  };

  // Claude-style thinking block: a shimmering header while it streams, collapsed to "Thought for Ns" when done.
  let thinking = null;
  let clockTimer = null;
  const beginThinking = (title) => {
    panel.replaceChildren();
    const header = element("header", "presentation-thinking-header");
    const icon = element("span", "presentation-spark");
    icon.setAttribute("aria-hidden", "true");
    const label = element("span", "presentation-shimmer", "Thinking");
    const clock = element("span", "presentation-clock", "0s");
    const subject = element("span", "presentation-subject", title);
    header.append(icon, label, clock, subject);
    const body = element("div", "presentation-thinking-body");
    const status = element("p", "presentation-status");
    status.hidden = true;
    panel.append(header, body, status);
    thinking = { header, label, clock, body, status, elapsed: 0, last: performance.now() };
    // A live clock, as a model that is still working shows; it stops while paused.
    window.clearInterval(clockTimer);
    clockTimer = window.setInterval(() => {
      const now = performance.now();
      if (!paused) thinking.elapsed += now - thinking.last;
      thinking.last = now;
      thinking.clock.textContent = `${Math.floor(thinking.elapsed / 1000)}s`;
    }, 200);
  };

  // A tool step with a spinner and the latency a real model call has.
  const working = async (text, ms) => {
    thinking.status.hidden = false;
    thinking.status.textContent = text;
    await wait(ms * (0.75 + Math.random() * 0.5));
    thinking.status.hidden = true;
  };

  const stream = async (text, className = "", container = thinking.body) => {
    const paragraph = element("p", className);
    const caret = element("span", "presentation-caret");
    container.append(paragraph);
    paragraph.append(caret);
    // Text arrives in token-sized chunks at an uneven rate, slowing after punctuation as a model does
    // between clauses; about 330 words a minute on average, so an audience can read along.
    const tokens = text.match(/\s*\S{1,5}|\s+/g) || [];
    for (const token of tokens) {
      caret.before(document.createTextNode(token));
      container.scrollTop = container.scrollHeight;
      // On a phone the panel itself scrolls: keep the newest line in view.
      panel.scrollTop = panel.scrollHeight;
      const pause = /[.;:]$/.test(token) ? 260 + Math.random() * 380 : /,$/.test(token) ? 120 + Math.random() * 120 : 0;
      await wait(35 + token.length * 11 + Math.random() * 45 + pause);
      if (skip) break;
    }
    if (skip) paragraph.textContent = text;
    caret.remove();
    await wait(380);
  };

  const finishThinking = async () => {
    window.clearInterval(clockTimer);
    thinking.clock.remove();
    const seconds = Math.max(1, Math.round(thinking.elapsed / 1000));
    thinking.label.textContent = `Thought for ${seconds}s`;
    thinking.label.classList.remove("presentation-shimmer");
    thinking.header.classList.add("done");
    await wait(500);
    thinking.body.classList.add("collapsed");
  };

  // The crop the model was given for this view, so the audience sees what it reads.
  const showCrop = (url, index) => {
    if (!url) return;
    const figure = element("figure", "presentation-crop");
    const image = element("img");
    image.src = url;
    image.alt = `View ${index + 1} as given to the model`;
    image.addEventListener("error", () => figure.remove());
    figure.append(image, element("figcaption", "", `View ${index + 1}: what the model reads`));
    thinking.body.append(figure);
    thinking.body.scrollTop = thinking.body.scrollHeight;
  };

  // After saving, the record is compared with the cadastre, when the building is matched to it.
  const crossCheck = async (card) => {
    const check = cadastreCheck(card);
    if (!check) return;
    setPhase("check");
    const box = panel.querySelector(".presentation-decision") || panel;
    await wait(600);
    await stream(check.text, `presentation-check${check.differs ? " differs" : ""}`, box);
  };

  const decide = async (decision) => {
    const box = element("section", "presentation-decision");
    const heading = element("h3", "", "Building record");
    const list = element("dl");
    box.append(heading, list);
    panel.append(box);
    for (const field of decision.fields) {
      const dt = element("dt", "", field.label);
      const dd = element("dd", "typing");
      list.append(dt, dd);
      for (let index = 1; index <= field.value.length; index += 1) {
        dd.textContent = field.value.slice(0, index);
        await wait(22);
        if (skip) {
          dd.textContent = field.value;
          break;
        }
      }
      dd.classList.remove("typing");
      dd.classList.add("filled");
      panel.scrollTop = panel.scrollHeight;
      await wait(450);
    }
    const gem = element("p", "presentation-gem", decision.gem);
    gem.title = "GEM Building Taxonomy v3: material / lateral load-resisting system / height";
    box.append(gem);
    await wait(700);
    const save = element("p", "presentation-save saving", "Saving to the building record...");
    box.append(save);
    await wait(900 + Math.random() * 700);
    save.classList.remove("saving");
    save.classList.add(decision.needsReview ? "flagged" : "saved");
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    save.textContent = `${decision.needsReview ? "⚑" : "✓"} ${decision.saveText} ${time}`;
  };

  // Live: the model analyzes the building now. Events are played in order, so the camera reaches a view
  // before that view's notes stream in; text that arrives meanwhile waits in the queue.
  const liveVisit = async (building, savedCard) => {
    beginThinking(building.osmId);
    const queue = [];
    let ended = false;
    let failure = null;
    let wake = null;
    const push = (event) => {
      queue.push(event);
      wake?.();
    };
    const request = api.analyzeLive(building.osmId, push)
      .catch((error) => { failure = error; })
      .finally(() => { ended = true; wake?.(); });
    let paragraph = null;
    let raw = "";
    let card = null;
    for (;;) {
      if (stopped) throw new Stopped();
      if (skip) {
        api.cancelLive();
        break;
      }
      if (!queue.length) {
        if (ended) break;
        await new Promise((resolve) => { wake = resolve; window.setTimeout(resolve, 250); });
        wake = null;
        continue;
      }
      const event = queue.shift();
      if (event.type === "start") {
        setPhase("look");
        thinking.status.hidden = false;
        thinking.status.textContent = `${event.model} is reading ${event.views} view${event.views === 1 ? "" : "s"}...`;
      } else if (event.type === "view") {
        const reference = event.reference || {};
        setPhase("walk");
        await walkTo(reference.image_id ? String(reference.image_id) : null);
        setPhase("look");
        const target = frame(reference.view);
        if (target) await turnTo(target);
        await moveCursor(host.clientWidth / 2, host.clientHeight / 2 - 10);
        await click();
        api.selectBuilding(building.osmId);
        thinking.status.hidden = false;
        thinking.status.textContent = `Analyzing view ${event.index + 1}...`;
        paragraph = element("p", "presentation-step view");
        thinking.body.append(paragraph);
        raw = "";
      } else if (event.type === "text" && paragraph) {
        raw += event.text;
        const notes = visibleReasoning(raw);
        if (notes) thinking.status.hidden = true;
        paragraph.textContent = `View ${thinking.body.querySelectorAll("p.view").length}. ${notes}`;
        thinking.body.scrollTop = thinking.body.scrollHeight;
      } else if (event.type === "answer" && paragraph) {
        const line = event.error ? `no usable answer (${event.error})` : answerLine(event.answer);
        if (line) paragraph.textContent += `\n→ ${line}`;
        await wait(400);
      } else if (event.type === "done") {
        card = event.card;
      } else if (event.type === "error") {
        failure = new Error(event.error);
      }
    }
    await request;
    thinking.status.hidden = true;
    if (!card) {
      if (skip) return;
      await stream(`The model could not finish (${failure?.message || "no answer"}); showing the saved record instead.`, "presentation-step");
      card = savedCard;
    }
    if (!card) return;
    // The reflection across views comes from the fused answer, as in replay.
    const steps = reasoningScript(card);
    setPhase("reason");
    for (const step of steps.filter((item) => item.type === "floors" || item.type === "structure")) {
      if (skip) break;
      await working(step.type === "floors" ? "Comparing the storey counts across views..." : "Weighing the evidence across views...", 700);
      await stream(step.text, `presentation-step ${step.type}`);
    }
    const decision = steps.find((step) => step.type === "decision");
    setPhase("record");
    await finishThinking();
    await decide({ ...decision, saveText: steps.at(-1).text });
    api.revealBuilding(building.osmId);
    saved.add(building.osmId);
    if (!skip) await crossCheck(card);
  };

  const visit = async (building, number, total) => {
    hudProgress.textContent = `Building ${number} of ${total}`;
    hudFill.style.width = `${Math.round(((number - 1) / total) * 100)}%`;
    const card = await api.fetchCard(building.osmId);
    if (api.liveModel?.()) {
      hudBadge.textContent = `Live · ${api.liveModel()}`;
      hudBadge.hidden = false;
      await liveVisit(building, card);
      cursor.classList.remove("visible");
      hudFill.style.width = `${Math.round((number / total) * 100)}%`;
      await wait(4000);
      return;
    }
    if (!card) return;
    const simulated = String(card.model || "").startsWith(SIMULATED_PREFIX);
    const curated = String(card.model || "").startsWith("vlm:curated");
    // The badge stays honest about the source, without interrupting the show.
    hudBadge.textContent = simulated || curated ? "Demo dataset" : card.model ? `Model: ${card.model.replace(/^vlm:/, "")}` : "";
    hudBadge.hidden = !hudBadge.textContent;
    const steps = reasoningScript(card);
    beginThinking(building.osmId);
    setPhase("walk");
    await working("Loading the building's footprint and evidence...", 1100);
    // Walk to the first view: through the panoramas in between when they are few, as one would.
    const first = steps.find((step) => step.type === "view");
    const target = first?.imageId || building.image;
    await walkTo(target);
    for (const step of steps) {
      if (skip) break;
      if (step.type === "view") {
        if (step.imageId !== api.currentImageId()) {
          setPhase("walk");
          await working(`Moving to another capture for view ${step.index + 1}...`, 400);
          await walkTo(step.imageId);
        }
        setPhase("look");
        await turnTo(frame(step.view) || { yawDeg: api.bearingToBuilding(building.osmId) ?? api.getCamera().yawDeg, pitchDeg: -6, fov: 75 });
        await moveCursor(host.clientWidth / 2 + (Math.random() - 0.5) * 30, host.clientHeight / 2 - 10 + (Math.random() - 0.5) * 20);
        await click();
        api.selectBuilding(building.osmId);
        await wait(400);
        showCrop(step.cropUrl, step.index);
        await working(`Analyzing view ${step.index + 1}...`, 1600);
      }
      if (step.type === "floors" || step.type === "structure") setPhase("reason");
      if (step.type === "floors") await working("Counting window rows across the views...", 1200);
      if (step.type === "structure") await working("Comparing the views and weighting the evidence...", 1400);
      if (step.type === "decision") {
        setPhase("record");
        await working("Writing the building record...", 800);
      }
      if (step.type === "decision" || step.type === "save") {
        await finishThinking();
        await decide({ ...step, saveText: steps.at(-1).text });
        api.revealBuilding(building.osmId);
        saved.add(building.osmId);
        if (!skip) await crossCheck(card);
        break;
      }
      await stream(step.text ?? "", `presentation-step ${step.type}`);
    }
    cursor.classList.remove("visible");
    hudFill.style.width = `${Math.round((number / total) * 100)}%`;
    await wait(4000);
  };

  // What the audience is about to see, once, before the first building; → skips it.
  const introduce = async (total) => {
    const box = element("section", "presentation-intro");
    box.append(element("h3", "", "How the AI surveyor works"), element("p", "", `It will walk ${api.datasetId()} and describe ${total} buildings, one at a time:`));
    const list = element("ol");
    for (const [key, label, help] of PHASES) {
      const item = element("li");
      item.append(element("strong", "", label), document.createTextNode(` ${help}`));
      item.dataset.phase = key;
      list.append(item);
    }
    box.append(list, element("p", "presentation-intro-keys", "Space pauses, → skips ahead, Esc returns to the viewer."));
    panel.replaceChildren(box);
    await wait(9000);
    skip = false;
  };

  // The end of a walk: what was found, the map colored by each attribute, and the engineer's queue.
  const conclude = async (next) => {
    setPhase(null);
    hudProgress.textContent = "Survey complete";
    hudFill.style.width = "100%";
    beginThinking("Survey complete");
    const summary = await api.fetchSummary();
    const text = surveySummary(summary, saved) || `${saved.size} buildings described and saved.`;
    api.summaryView?.(true);
    await stream(text);
    for (const [mode, caption] of [["gem_material", "Map colored by GEM material"], ["gem_llrs", "by lateral load-resisting system"], ["floors", "by number of floors"]]) {
      if (skip) break;
      api.showColors?.(mode);
      thinking.status.hidden = false;
      thinking.status.textContent = caption;
      await wait(3500);
    }
    if (!skip && api.showReviewQueue) {
      const waiting = summary?.counts?.needs_review || 0;
      thinking.status.textContent = waiting
        ? `The review queue: ${waiting} answer${waiting === 1 ? "" : "s"} of ${api.datasetId()} for an engineer to check, weakest first`
        : "The review queue an engineer works through";
      api.showReviewQueue(true);
      await wait(6000);
      api.showReviewQueue(false);
    }
    api.showColors?.("gem_material");
    api.summaryView?.(false);
    thinking.status.hidden = true;
    skip = false;
    await stream(next ? `Moving on to ${next}.` : "Starting the walk again.");
    await wait(2000);
  };

  const run = async () => {
    api.enter();
    try {
      let tour = planTour(await api.fetchSummary(), api.imageIds(), { minScore });
      if (!tour.length) {
        panel.replaceChildren(element("p", "presentation-step", "No building of this dataset has a model answer yet. Run city-analyser-survey, or city-analyser-demo for a presentation copy."));
        return;
      }
      await introduce(tour.length);
      for (let lap = 0; !stopped; lap += 1) {
        for (let index = 0; index < tour.length && !stopped; index += 1) {
          skip = false;
          await visit(tour[index], index + 1, tour.length);
        }
        skip = false;
        const next = api.nextDataset ? await api.nextDataset(false) : null;
        await conclude(next);
        saved.clear();
        if (next) {
          await api.nextDataset(true);
          tour = planTour(await api.fetchSummary(), api.imageIds(), { minScore });
        }
      }
    } catch (error) {
      if (!(error instanceof Stopped)) {
        console.error(error);
        panel.replaceChildren(element("p", "presentation-step", `Presentation stopped: ${error.message}`));
      }
    }
  };

  const onKey = (event) => {
    if (event.key === " ") {
      event.preventDefault();
      event.stopImmediatePropagation();
      paused = !paused;
      root.classList.toggle("paused", paused);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      event.stopImmediatePropagation();
      skip = true;
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopImmediatePropagation();
      controller.stop();
    }
  };
  window.addEventListener("keydown", onKey, true);

  const controller = {
    stop() {
      if (stopped) return;
      stopped = true;
      window.clearInterval(clockTimer);
      window.removeEventListener("keydown", onKey, true);
      root.remove();
      api.exit();
    },
    pause() { paused = true; },
    resume() { paused = false; },
    get paused() { return paused; },
  };
  controller.done = run();
  return controller;
}
