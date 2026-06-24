import * as ort from "onnxruntime-web/webgpu";

let forwardSession = null;
let residualSession = null;

let forwardBackend = "unknown";
let residualBackend = "unknown";

let pending = false;
let running = false;

const RE_MIN = 900.0;
const RE_MAX = 1100.0;

const D_MIN = 1.9;
const D_MAX = 3.3;

const TRI_MIN = 0.0;
const TRI_MAX = 0.01;

const BACKEND_OPTIONS = [
  {
    value: "webgpu-wasm",
    label: "forward: WebGPU, residual: WASM",
    forward: "webgpu",
    residual: "wasm",
  },
  {
    value: "webgpu-webgpu",
    label: "forward: WebGPU, residual: WebGPU",
    forward: "webgpu",
    residual: "webgpu",
  },
  {
    value: "wasm-webgpu",
    label: "forward: WASM, residual: WebGPU",
    forward: "wasm",
    residual: "webgpu",
  },
  {
    value: "wasm-wasm",
    label: "forward: WASM, residual: WASM",
    forward: "wasm",
    residual: "wasm",
  },
];

function byId(...ids) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) return el;
  }
  return null;
}

function getControls() {
  const ranges = Array.from(document.querySelectorAll('input[type="range"]'));
  const selects = Array.from(document.querySelectorAll("select"));

  const controls = {
    nx: byId("nx") ?? ranges[0],
    ny: byId("ny") ?? ranges[1],
    re: byId("re", "Re") ?? ranges[2],
    d: byId("d", "D", "depth", "Depth") ?? ranges[3],
    tri: byId("tri", "Tri") ?? ranges[4],

    nxValue: byId("nxValue") ?? null,
    nyValue: byId("nyValue") ?? null,
    reValue: byId("reValue", "ReValue") ?? null,
    dValue: byId("dValue", "DValue", "depthValue", "DepthValue") ?? null,
    triValue: byId("triValue", "TriValue") ?? null,

    solutionField: byId("solutionField") ?? selects[0],
    residualField: byId("residualField") ?? selects[1],
    backendMode: byId("backendMode") ?? null,

    status: byId("status"),
    residualStats: byId("residualStats"),

    solutionCanvas: byId("solutionCanvas"),
    residualCanvas: byId("residualCanvas"),
  };

  const optional = new Set([
    "nxValue",
    "nyValue",
    "reValue",
    "dValue",
    "triValue",
    "backendMode",
  ]);

  const missing = Object.entries(controls)
    .filter(([key, value]) => !optional.has(key) && !value)
    .map(([key]) => key);

  if (missing.length > 0) {
    throw new Error(`Missing required controls: ${missing.join(", ")}`);
  }

  return controls;
}

function setText(el, text) {
  if (el) el.textContent = text;
}

function ensureBackendSelector() {
  let selector = byId("backendMode");
  if (selector) return selector;

  const controls = getControls();
  const anchor = controls.residualField;

  const row = document.createElement("div");
  row.className = "backend-row";
  row.style.display = "contents";

  const label = document.createElement("label");
  label.htmlFor = "backendMode";
  label.textContent = "backend";

  selector = document.createElement("select");
  selector.id = "backendMode";

  for (const option of BACKEND_OPTIONS) {
    const opt = document.createElement("option");
    opt.value = option.value;
    opt.textContent = option.label;
    selector.appendChild(opt);
  }

  selector.value = "wasm-wasm";

  const controlsContainer =
    anchor.closest(".controls") ??
    anchor.parentElement?.parentElement ??
    anchor.parentElement ??
    document.body;

  if (controlsContainer.classList.contains("controls")) {
    controlsContainer.appendChild(label);
    controlsContainer.appendChild(selector);
    const spacer = document.createElement("span");
    spacer.textContent = "";
    controlsContainer.appendChild(spacer);
  } else {
    const wrapper = document.createElement("div");
    wrapper.style.margin = "8px 0";
    wrapper.appendChild(label);
    wrapper.appendChild(document.createTextNode(" "));
    wrapper.appendChild(selector);
    anchor.insertAdjacentElement("afterend", wrapper);
  }

  return selector;
}

function getBackendMode() {
  const selector = ensureBackendSelector();
  const selected = BACKEND_OPTIONS.find((x) => x.value === selector.value);

  if (!selected) {
    return BACKEND_OPTIONS[0];
  }

  return selected;
}

function providerLabel(provider) {
  if (provider === "webgpu") return "WebGPU";
  if (provider === "wasm") return "WASM";
  return provider;
}

async function createSession(modelPath, label, provider) {
  console.log(`Loading ${label} with ${providerLabel(provider)}: ${modelPath}`);

  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: [provider],
  });

  console.log(`${label} loaded with ${providerLabel(provider)}.`);

  return {
    session,
    backend: providerLabel(provider),
  };
}

async function loadModels() {
  const controls = getControls();
  ensureBackendSelector();

  const base = import.meta.env.BASE_URL;
  const version = Date.now();

  const mode = getBackendMode();

  const forwardModelPath = `${base}models/lid_cavity_forward.onnx?v=${version}`;
  const residualModelPath = `${base}models/lid_cavity_residual.onnx?v=${version}`;

  forwardSession = null;
  residualSession = null;

  controls.status.textContent =
    `Loading models: forward ${providerLabel(mode.forward)}, residual ${providerLabel(
      mode.residual,
    )}...`;

  try {
    const forwardLoaded = await createSession(
      forwardModelPath,
      "forward model",
      mode.forward,
    );

    forwardSession = forwardLoaded.session;
    forwardBackend = forwardLoaded.backend;

    const residualLoaded = await createSession(
      residualModelPath,
      "residual model",
      mode.residual,
    );

    residualSession = residualLoaded.session;
    residualBackend = residualLoaded.backend;

    console.log("Loaded forward model.");
    console.log("Forward backend:", forwardBackend);
    console.log("Forward input names:", forwardSession.inputNames);
    console.log("Forward output names:", forwardSession.outputNames);

    console.log("Loaded residual model.");
    console.log("Residual backend:", residualBackend);
    console.log("Residual input names:", residualSession.inputNames);
    console.log("Residual output names:", residualSession.outputNames);

    controls.status.textContent =
      `Models loaded. Forward: ${forwardBackend}, residual: ${residualBackend}.`;
  } catch (err) {
    forwardSession = null;
    residualSession = null;
    forwardBackend = "failed";
    residualBackend = "failed";

    console.error(err);

    controls.status.textContent =
      `Error loading selected backend mode: ${err.message}`;
  }
}

async function reloadModelsAndRun() {
  await loadModels();
  await runInference();
}

function normalizeRe(re) {
  return (re - RE_MIN) / (RE_MAX - RE_MIN);
}

function normalizeD(d) {
  return (d - D_MIN) / (D_MAX - D_MIN);
}

function normalizeTri(tri) {
  return (tri - TRI_MIN) / (TRI_MAX - TRI_MIN);
}

function makeGrid(nx, ny, re, d, tri) {
  const B = nx * ny;
  const data = new Float32Array(B * 5);

  const reNorm = normalizeRe(re);
  const dNorm = normalizeD(d);
  const triNorm = normalizeTri(tri);

  let k = 0;

  for (let j = 0; j < ny; j++) {
    const y = ny > 1 ? j / (ny - 1) : 0.0;

    for (let i = 0; i < nx; i++) {
      const x = nx > 1 ? i / (nx - 1) : 0.0;

      data[5 * k + 0] = x;
      data[5 * k + 1] = y;
      data[5 * k + 2] = reNorm;
      data[5 * k + 3] = dNorm;
      data[5 * k + 4] = triNorm;

      k++;
    }
  }

  return data;
}

async function runInference() {
  if (!forwardSession || !residualSession) return;
  if (running) return;

  running = true;

  try {
    const controls = getControls();

    const nx = Number(controls.nx.value);
    const ny = Number(controls.ny.value);
    const re = Number(controls.re.value);
    const d = Number(controls.d.value);
    const tri = Number(controls.tri.value);

    const solutionField = controls.solutionField.value;
    const residualField = controls.residualField.value;

    setText(controls.nxValue, String(nx));
    setText(controls.nyValue, String(ny));
    setText(controls.reValue, re.toFixed(0));
    setText(controls.dValue, d.toFixed(2));
    setText(controls.triValue, tri.toFixed(4));

    const inputData = makeGrid(nx, ny, re, d, tri);
    const inputTensor = new ort.Tensor("float32", inputData, [nx * ny, 5]);

    const start = performance.now();

    const forwardResults = await forwardSession.run({
      [forwardSession.inputNames[0]]: inputTensor,
    });

    const residualResults = await residualSession.run({
      [residualSession.inputNames[0]]: inputTensor,
    });

    const elapsedMs = performance.now() - start;

    const forward = forwardResults[forwardSession.outputNames[0]].data;
    const residual = residualResults[residualSession.outputNames[0]].data;

    const solutionValues = selectSolutionField(forward, solutionField);
    const residualValues = selectResidualField(residual, residualField);

    const solutionSymmetric =
      solutionField === "u" || solutionField === "v" || solutionField === "p";

    const residualSymmetric = residualField === "ru" || residualField === "rv";

    drawHeatmap("solutionCanvas", solutionValues, nx, ny, {
      symmetric: solutionSymmetric,
    });

    drawHeatmap("residualCanvas", residualValues, nx, ny, {
      symmetric: residualSymmetric,
    });

    const stats = computeResidualStats(residual);

    controls.status.textContent =
      `Inference complete: ${nx} x ${ny} grid, ${elapsedMs.toFixed(2)} ms ` +
      `(forward: ${forwardBackend}, residual: ${residualBackend})`;

    controls.residualStats.textContent =
      `Residual: mean |R| = ${stats.meanMag.toExponential(3)}, ` +
      `RMS |R| = ${stats.rmsMag.toExponential(3)}, ` +
      `max |R| = ${stats.maxMag.toExponential(3)}`;
  } catch (err) {
    console.error(err);

    const status = byId("status");
    if (status) status.textContent = `Error: ${err.message}`;
  } finally {
    running = false;
  }
}

function selectSolutionField(forward, field) {
  const n = forward.length / 4;
  const out = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const u = forward[4 * i + 0];
    const v = forward[4 * i + 1];
    const p = forward[4 * i + 2];
    const speed = forward[4 * i + 3];

    if (field === "u") out[i] = u;
    else if (field === "v") out[i] = v;
    else if (field === "p") out[i] = p;
    else out[i] = speed;
  }

  return out;
}

function selectResidualField(residual, field) {
  const n = residual.length / 3;
  const out = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const ru = residual[3 * i + 0];
    const rv = residual[3 * i + 1];
    const rmag = residual[3 * i + 2];

    if (field === "ru") out[i] = ru;
    else if (field === "rv") out[i] = rv;
    else out[i] = rmag;
  }

  return out;
}

function drawHeatmap(canvasId, values, nx, ny, options = {}) {
  const { symmetric = false } = options;

  const canvas = document.getElementById(canvasId);
  if (!canvas) throw new Error(`Missing canvas with id="${canvasId}"`);

  const ctx = canvas.getContext("2d");

  canvas.width = nx;
  canvas.height = ny;

  const image = ctx.createImageData(nx, ny);

  let min = Infinity;
  let max = -Infinity;

  if (symmetric) {
    let maxAbs = 0.0;

    for (const v of values) {
      const a = Math.abs(v);
      if (a > maxAbs) maxAbs = a;
    }

    min = -maxAbs;
    max = maxAbs;
  } else {
    for (const v of values) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  const scale = max > min ? 1.0 / (max - min) : 1.0;

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const normalized = clamp01((values[k] - min) * scale);
      const [r, g, b] = turboColormap(normalized);

      const p = 4 * k;
      image.data[p + 0] = r;
      image.data[p + 1] = g;
      image.data[p + 2] = b;
      image.data[p + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function computeResidualStats(residual) {
  const n = residual.length / 3;

  let sumMag = 0.0;
  let sumMagSq = 0.0;
  let maxMag = 0.0;

  for (let i = 0; i < n; i++) {
    const rmag = Math.abs(residual[3 * i + 2]);

    sumMag += rmag;
    sumMagSq += rmag * rmag;
    if (rmag > maxMag) maxMag = rmag;
  }

  return {
    meanMag: sumMag / n,
    rmsMag: Math.sqrt(sumMagSq / n),
    maxMag,
  };
}

function scheduleInference() {
  if (pending) return;

  pending = true;

  requestAnimationFrame(async () => {
    pending = false;
    await runInference();
  });
}

function clamp01(x) {
  return Math.max(0.0, Math.min(1.0, x));
}

function turboColormap(v) {
  const x = clamp01(v);

  const r =
    34.61 +
    x *
      (1172.33 +
        x *
          (-10793.56 +
            x * (33300.12 + x * (-38394.49 + x * 14825.05))));

  const g =
    23.31 +
    x *
      (557.33 +
        x *
          (1225.33 +
            x * (-3574.96 + x * (1073.77 + x * 707.56))));

  const b =
    27.2 +
    x *
      (3211.1 +
        x *
          (-15327.97 +
            x * (27814.0 + x * (-22569.18 + x * 6838.66))));

  return [
    Math.round(Math.max(0, Math.min(255, r))),
    Math.round(Math.max(0, Math.min(255, g))),
    Math.round(Math.max(0, Math.min(255, b))),
  ];
}

async function main() {
  ensureBackendSelector();

  await loadModels();
  await runInference();

  const controls = getControls();

  const interactiveElements = [
    controls.nx,
    controls.ny,
    controls.re,
    controls.d,
    controls.tri,
    controls.solutionField,
    controls.residualField,
  ];

  for (const element of interactiveElements) {
    element.addEventListener("input", scheduleInference);
    element.addEventListener("change", scheduleInference);
  }

  const backendSelector = ensureBackendSelector();
  backendSelector.addEventListener("change", reloadModelsAndRun);
}

main().catch((err) => {
  console.error(err);

  const status = byId("status");
  if (status) status.textContent = `Error: ${err.message}`;
});
