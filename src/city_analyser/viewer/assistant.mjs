// Talking to the viewer's assistant: streamed events and what of the model's text to show.
// Pure helpers plus one network call, so they are tested without a browser.

/** Split a server-sent-events buffer into parsed events and the unfinished rest. */
export function parseEvents(buffer) {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop();
  const events = [];
  for (const block of blocks) {
    const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("\n");
    if (!data) continue;
    try {
      events.push(JSON.parse(data));
    } catch {
      // A malformed event is skipped; the stream goes on.
    }
  }
  return { events, rest };
}

/** POST a request and call `onEvent` for every event the server streams; resolves when the stream ends. */
export async function streamEvents(url, body, onEvent, { signal } = {}) {
  const response = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}), signal,
  });
  if (!response.ok || !response.body) {
    let message = `status ${response.status}`;
    try {
      message = (await response.json()).error || message;
    } catch {
      // Keep the status.
    }
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const parsed = parseEvents(buffer + decoder.decode(value, { stream: true }));
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  parseEvents(`${buffer}\n\n`).events.forEach(onEvent);
}

/** The field notes before the model's JSON answer, also while the answer is still arriving (mirrors vlm.visible_reasoning). */
export function visibleReasoning(text) {
  let cut = text.length;
  for (const marker of ["```", "\n{", "{\""]) {
    const index = text.indexOf(marker);
    if (index >= 0) cut = Math.min(cut, index);
  }
  if (text.trimStart().startsWith("{")) cut = 0;
  return text.slice(0, cut).replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trim();
}

/** One line for a view's parsed answer: "4 storeys · MUR · LWAL". */
export function answerLine(answer) {
  if (!answer) return "";
  return [answer.floors ? `${answer.floors} storeys` : null, answer.gem_material, answer.gem_llrs].filter(Boolean).join(" · ");
}

/** Pieces of a model's text with **bold** marked: [{ text, bold }]. Rendered as text nodes, never as HTML. */
export function boldSpans(text) {
  const spans = [];
  const pattern = /\*\*([^*\n]+)\*\*/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) spans.push({ text: text.slice(last, match.index), bold: false });
    spans.push({ text: match[1], bold: true });
    last = match.index + match[0].length;
  }
  if (last < text.length) spans.push({ text: text.slice(last), bold: false });
  return spans;
}

/**
 * Camera framing for a crop's view `{ yaw_deg, pitch_deg, fov_deg }`, as Street View users look: its direction,
 * a tilt of at most 22 degrees up, and a field of view wide enough to show the storeys. Null without a direction.
 * The renderer's pitch is positive down, hence the sign.
 */
export function framingOf(view) {
  if (!view || !Number.isFinite(view.yaw_deg)) return null;
  return {
    yawDeg: view.yaw_deg,
    pitchDeg: -Math.max(-5, Math.min(22, view.pitch_deg || 0)),
    fov: Math.max(60, Math.min(95, (view.fov_deg || 70) * 1.1)),
  };
}
