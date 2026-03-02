import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TAP_NODE_NAME = "FreeFuseMaskTap";
const REASSEMBLE_NODE_NAME = "FreeFuseMaskReassemble";
const SLOT_COUNT = 10;
const SLOT_REF_PREFIX = "mask_image_";
const LEGACY_MASK_SLOT_NAMES = [
  "mask_00",
  "mask_01",
  "mask_02",
  "mask_03",
  "mask_04",
  "mask_05",
  "mask_06",
  "mask_07",
  "mask_08",
  "mask_09",
];
const UI_SLOT_WIDGET = "Edit Slot";
const UI_OPEN_BUTTON = "Open Mask Editor";
const UI_CLEAR_BUTTON = "Clear Slot Edit";
const LEGACY_UI_SLOT_WIDGET = "ff_edit_slot";
const LEGACY_UI_OPEN_BUTTON = "ff_open_editor";
const LEGACY_UI_CLEAR_BUTTON = "ff_clear_slot";
const SIMPLE_MODE_CLASS = "ff-masktap-simple-mode";
const SIMPLE_MODE_STYLE_ID = "ff-masktap-simple-style";
const BOARD_EDITOR_STYLE_ID = "ff-masktap-board-editor-style";
const USER_EDIT_REF_PREFIX = "freefuse-masktap-edit-";
const USER_EDIT_SEED_RE = /^freefuse-masktap-edit-s(\d+)-/i;
const TOOL_INDEX_MASK_PEN = 0;
const TOOL_INDEX_ERASER = 2;
const SIMPLE_ALLOWED_TOOLS = new Set([TOOL_INDEX_MASK_PEN, TOOL_INDEX_ERASER]);
const BOARD_TOOL_BRUSH = "brush";
const BOARD_TOOL_ERASER = "eraser";

let activeBoardEditorCleanup = null;

function pad2(n) {
  return String(n).padStart(2, "0");
}

function slotRefName(index) {
  return `${SLOT_REF_PREFIX}${pad2(index)}`;
}

function ensureSimpleModeStyle() {
  if (document.getElementById(SIMPLE_MODE_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = SIMPLE_MODE_STYLE_ID;
  style.textContent = `
body.${SIMPLE_MODE_CLASS} .mask-editor-dialog #maskEditorCanvasContainer canvas {
  image-rendering: pixelated !important;
}
`;
  document.head.appendChild(style);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function dispatchValueChange(input, value) {
  if (!input) return;
  const text = String(value);
  if (input.value === text) return;
  input.value = text;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function dispatchSelectChange(select, value) {
  if (!select) return;
  const text = String(value);
  if (select.value === text) return;
  select.value = text;
  select.dispatchEvent(new Event("input", { bubbles: true }));
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function setInputToMax(input, fallback = 1) {
  if (!input) return;
  const max = Number.parseFloat(input.max);
  const value = Number.isFinite(max) ? max : fallback;
  dispatchValueChange(input, value);
}

function getToolButtons(root) {
  if (!root) return [];
  return Array.from(root.querySelectorAll(".maskEditor_toolPanelContainer"));
}

function getSelectedToolIndex(toolButtons) {
  if (!Array.isArray(toolButtons)) return -1;
  return toolButtons.findIndex((button) =>
    button.classList.contains("maskEditor_toolPanelContainerSelected")
  );
}

function setImportantStyle(element, prop, value) {
  if (!element || !element.style) return;
  element.style.setProperty(prop, value, "important");
}

function enforceSimpleToolVisibility(root) {
  const toolButtons = getToolButtons(root);
  if (!toolButtons.length) {
    return { toolButtons, maskPenBtn: null, eraserBtn: null };
  }
  for (let i = 0; i < toolButtons.length; i++) {
    const visible = SIMPLE_ALLOWED_TOOLS.has(i);
    if (visible) {
      toolButtons[i].style.removeProperty("display");
      toolButtons[i].style.removeProperty("pointer-events");
    } else {
      toolButtons[i].style.setProperty("display", "none", "important");
      toolButtons[i].style.setProperty("pointer-events", "none");
    }
  }
  return {
    toolButtons,
    maskPenBtn: toolButtons[TOOL_INDEX_MASK_PEN] || null,
    eraserBtn: toolButtons[TOOL_INDEX_ERASER] || null,
  };
}

function findMaskLayerActivateButton(root) {
  const layerChecks = root.querySelectorAll(".maskEditor_sidePanelLayerCheckbox");
  if (layerChecks.length >= 1) {
    const maskRow = layerChecks[0].closest("div");
    if (maskRow) {
      const buttonInRow = maskRow.querySelector("button");
      if (buttonInRow) {
        return buttonInRow;
      }
    }
  }

  const panel = root.querySelector(".maskEditor_sidePanelContainer");
  if (panel) {
    const buttons = panel.querySelectorAll("button");
    for (const button of buttons) {
      const text = (button.textContent || "").toLowerCase();
      if (text.includes("mask")) {
        return button;
      }
    }
  }

  const previews = root.querySelectorAll(".maskEditor_sidePanelLayerPreviewContainer");
  for (const preview of previews) {
    const next = preview.nextElementSibling;
    if (next && next.tagName === "BUTTON") {
      return next;
    }
  }
  return null;
}

function enforceMaskLayerEditing(root) {
  const activateMaskBtn = findMaskLayerActivateButton(root);
  if (!activateMaskBtn || activateMaskBtn.disabled) {
    return;
  }
  const text = (activateMaskBtn.textContent || "").toLowerCase();
  // Only click when this looks like an explicit "activate" action.
  if (text.includes("activate")) {
    activateMaskBtn.click();
  }
}

function enforceSimpleMaskCanvasView(root) {
  if (!root) return;
  const container = root.querySelector("#maskEditorCanvasContainer");
  if (!container) return;

  const canvases = container.querySelectorAll("canvas");
  for (const canvas of canvases) {
    canvas.style.removeProperty("opacity");
    canvas.style.removeProperty("filter");
    canvas.style.removeProperty("mix-blend-mode");
    setImportantStyle(canvas, "image-rendering", "pixelated");
  }

  const bg = container.querySelector("div");
  if (bg) {
    setImportantStyle(bg, "background-color", "#000000");
  }
}

function enforceSimpleMaskBlendAndLayers(root) {
  if (!root) return;
  dispatchSelectChange(root.querySelector(".maskEditor_sidePanelDropdown"), "white");
}

function enforceSimpleMaskBrushDefaults(root) {
  if (!root) return;
  const numberInputs = root.querySelectorAll(".maskEditor_sidePanel input[type='number']");
  // Expected common order: thickness, opacity, hardness, step.
  if (numberInputs.length >= 2) {
    setInputToMax(numberInputs[1], 1);
  }
  if (numberInputs.length >= 3) {
    setInputToMax(numberInputs[2], 1);
  }

  const rangeInputs = root.querySelectorAll(".maskEditor_sidePanel input[type='range']");
  if (rangeInputs.length >= 2) {
    setInputToMax(rangeInputs[1], 1);
  }
  if (rangeInputs.length >= 3) {
    setInputToMax(rangeInputs[2], 1);
  }
  dispatchValueChange(root.querySelector(".maskEditor_sidePanel input[type='color']"), "#ffffff");
}

function enforceSimpleMaskEditorState(root) {
  if (!root) return;
  const { toolButtons, maskPenBtn } = enforceSimpleToolVisibility(root);

  const selectedTool = getSelectedToolIndex(toolButtons);
  if (!SIMPLE_ALLOWED_TOOLS.has(selectedTool) && maskPenBtn) {
    maskPenBtn.click();
  }

  enforceMaskLayerEditing(root);
  enforceSimpleMaskBlendAndLayers(root);
  enforceSimpleMaskBrushDefaults(root);
  enforceSimpleMaskCanvasView(root);
}

function scheduleSimpleMaskEditorStabilization(root) {
  const delays = [0, 80, 180, 320, 480];
  for (const delay of delays) {
    setTimeout(() => {
      if (!document.body?.contains(root)) return;
      enforceSimpleMaskEditorState(root);
    }, delay);
  }
}

function initSimpleMaskEditor(root) {
  if (!root) {
    return;
  }

  if (root.dataset.ffMaskTapSimpleInit === "1") {
    return;
  }
  root.dataset.ffMaskTapSimpleInit = "1";
  scheduleSimpleMaskEditorStabilization(root);
}

async function setupSimpleMaskEditorMode() {
  ensureSimpleModeStyle();
  if (!document.body) {
    return;
  }
  document.body.classList.add(SIMPLE_MODE_CLASS);

  for (let i = 0; i < 80; i++) {
    const root = document.querySelector(".mask-editor-dialog");
    if (root) {
      initSimpleMaskEditor(root);
      return;
    }
    await sleep(50);
  }
}

function compactLegacyMaskOutputs(node) {
  if (!node || typeof node.removeOutput !== "function" || !Array.isArray(node.outputs)) {
    return;
  }
  const names = new Set(LEGACY_MASK_SLOT_NAMES);
  for (let i = node.outputs.length - 1; i >= 0; i--) {
    const slot = node.outputs[i];
    if (!slot || !names.has(slot.name)) continue;
    const hasLinks = Array.isArray(slot.links) ? slot.links.length > 0 : slot.links != null;
    if (hasLinks) continue;
    node.removeOutput(i);
  }
}

function compactLegacyMaskInputs(node) {
  if (!node || typeof node.removeInput !== "function" || !Array.isArray(node.inputs)) {
    return;
  }
  const names = new Set(LEGACY_MASK_SLOT_NAMES);
  for (let i = node.inputs.length - 1; i >= 0; i--) {
    const slot = node.inputs[i];
    if (!slot || !names.has(slot.name)) continue;
    if (slot.link != null) continue;
    node.removeInput(i);
  }
}

function compactNodeSize(node) {
  if (!node || typeof node.computeSize !== "function" || typeof node.setSize !== "function") {
    return;
  }
  try {
    node.setSize(node.computeSize());
  } catch {
    // no-op
  }
}

function getWidget(node, name) {
  return node.widgets?.find((w) => w?.name === name) || null;
}

function getAnyWidget(node, names) {
  for (const name of names) {
    const w = getWidget(node, name);
    if (w) return w;
  }
  return null;
}

function getSlotRefWidget(node, index) {
  return getWidget(node, slotRefName(index));
}

function hideWidget(widget) {
  if (!widget) return;
  widget.computeSize = () => [0, -4];
}

function parseWidgetRef(value) {
  if (!value) return null;

  if (typeof value === "object" && value.filename) {
    return {
      filename: String(value.filename),
      subfolder: value.subfolder ? String(value.subfolder) : "",
      type: value.type ? String(value.type) : "input",
    };
  }

  if (typeof value !== "string") {
    return null;
  }

  const text = value.trim();
  if (!text) return null;

  try {
    if (text.startsWith("http://") || text.startsWith("https://") || text.startsWith("/view?")) {
      const url = text.startsWith("/") ? new URL(text, window.location.origin) : new URL(text);
      const filename = url.searchParams.get("filename");
      if (filename) {
        return {
          filename,
          subfolder: url.searchParams.get("subfolder") || "",
          type: url.searchParams.get("type") || "input",
        };
      }
    }
  } catch (error) {
    console.warn("[FreeFuseMaskTapUI] Failed to parse URL ref:", error);
  }

  const match = text.match(/^(.*?)(?:\s*\[([^\]]+)\])?$/);
  if (!match) return null;

  const pathPart = (match[1] || "").trim();
  if (!pathPart) return null;
  const type = (match[2] || "input").trim();

  const slash = pathPart.lastIndexOf("/");
  if (slash >= 0) {
    return {
      filename: pathPart.slice(slash + 1),
      subfolder: pathPart.slice(0, slash),
      type,
    };
  }

  return {
    filename: pathPart,
    subfolder: "",
    type,
  };
}

function formatWidgetRef(ref) {
  if (!ref || !ref.filename) return "";
  const type = ref.type || "input";
  const path = ref.subfolder ? `${ref.subfolder}/${ref.filename}` : ref.filename;
  return `${path} [${type}]`;
}

function buildViewUrl(ref) {
  const params = new URLSearchParams();
  params.set("filename", ref.filename);
  params.set("type", ref.type || "input");
  if (ref.subfolder) {
    params.set("subfolder", ref.subfolder);
  }
  const tail = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
  return api.apiURL(`/view?${params.toString()}${tail}`);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = (error) => reject(error);
    img.src = url;
  });
}

async function uploadCanvasAsImageRef(canvas, prefix = "freefuse-masktap") {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) {
    throw new Error("Failed to create image blob.");
  }

  const filename = `${prefix}-${Date.now()}.png`;
  const formData = new FormData();
  formData.append("image", blob, filename);
  formData.append("type", "input");
  formData.append("subfolder", "clipspace");

  const response = await api.fetchApi("/upload/image", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Upload failed with status ${response.status}`);
  }

  const data = await response.json().catch(() => ({}));
  return {
    filename: data?.name || filename,
    subfolder: data?.subfolder || "clipspace",
    type: data?.type || "input",
  };
}

async function uploadBlankImageRef() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(0,0,0,1)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  return uploadCanvasAsImageRef(canvas, "freefuse-masktap-blank");
}

function ensureBoardEditorStyle() {
  if (document.getElementById(BOARD_EDITOR_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = BOARD_EDITOR_STYLE_ID;
  style.textContent = `
.ff-masktap-board-overlay {
  position: fixed;
  inset: 0;
  z-index: 99999;
  background: rgba(0, 0, 0, 0.72);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}
.ff-masktap-board-dialog {
  width: min(1180px, 100%);
  max-height: 95vh;
  border-radius: 14px;
  border: 1px solid #2a2d35;
  background: #151821;
  color: #e8ecf3;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.ff-masktap-board-toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  padding: 12px 14px;
  border-bottom: 1px solid #232734;
}
.ff-masktap-board-title {
  font-size: 17px;
  font-weight: 700;
  margin-right: 10px;
}
.ff-masktap-board-spacer {
  flex: 1 1 auto;
}
.ff-masktap-board-btn {
  border: 1px solid #2d3342;
  background: #1c2230;
  color: #edf1f8;
  border-radius: 9px;
  padding: 7px 12px;
  cursor: pointer;
}
.ff-masktap-board-btn:hover {
  background: #252c3d;
}
.ff-masktap-board-btn.active {
  background: #257be8;
  border-color: #257be8;
  color: #ffffff;
}
.ff-masktap-board-btn.ghost {
  background: transparent;
}
.ff-masktap-board-content {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14px;
  overflow: auto;
}
.ff-masktap-board-canvas-wrap {
  background: #000000;
  border: 1px solid #2e3442;
  border-radius: 12px;
  padding: 10px;
  box-shadow: inset 0 0 0 1px #111623;
}
.ff-masktap-board-canvas {
  display: block;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
  background: #000000;
  cursor: crosshair;
  touch-action: none;
}
.ff-masktap-board-size {
  width: 140px;
}
.ff-masktap-board-size-num {
  width: 60px;
  border-radius: 8px;
  border: 1px solid #2d3342;
  background: #111521;
  color: #edf1f8;
  padding: 5px 7px;
}
.ff-masktap-board-meta {
  opacity: 0.86;
  font-size: 12px;
}
`;
  document.head.appendChild(style);
}

function normalizeMaskCanvasFromImage(canvas, image, targetWidth = null, targetHeight = null) {
  const width = Math.max(
    1,
    Number(
      targetWidth ?? image?.naturalWidth ?? image?.width ?? 0
    )
  );
  const height = Math.max(
    1,
    Number(
      targetHeight ?? image?.naturalHeight ?? image?.height ?? 0
    )
  );
  canvas.width = width;
  canvas.height = height;

  const temp = document.createElement("canvas");
  temp.width = width;
  temp.height = height;
  const tempCtx = temp.getContext("2d");
  if (!tempCtx) return;
  tempCtx.drawImage(image, 0, 0, width, height);

  const src = tempCtx.getImageData(0, 0, width, height).data;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;
  const dstImage = ctx.createImageData(width, height);
  const dst = dstImage.data;

  for (let i = 0; i < src.length; i += 4) {
    const gray = Math.max(src[i], src[i + 1], src[i + 2]);
    const alphaMask = 255 - src[i + 3];
    const value = Math.max(gray, alphaMask) >= 128 ? 255 : 0;
    dst[i] = value;
    dst[i + 1] = value;
    dst[i + 2] = value;
    dst[i + 3] = 255;
  }
  ctx.putImageData(dstImage, 0, 0);
}

function binarizeCanvasInPlace(canvas, threshold = 128) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const value = Math.max(data[i], data[i + 1], data[i + 2]) >= threshold ? 255 : 0;
    data[i] = value;
    data[i + 1] = value;
    data[i + 2] = value;
    data[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function computeBoardCanvasScale(canvas) {
  const maxW = Math.max(280, Math.floor(window.innerWidth * 0.62));
  const maxH = Math.max(280, Math.floor(window.innerHeight * 0.72));
  const byW = Math.floor(maxW / canvas.width);
  const byH = Math.floor(maxH / canvas.height);
  return Math.max(1, Math.min(byW, byH));
}

function applyBoardCanvasDisplaySize(canvas) {
  const scale = computeBoardCanvasScale(canvas);
  canvas.style.width = `${canvas.width * scale}px`;
  canvas.style.height = `${canvas.height * scale}px`;
}

function getCanvasPointerPixel(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  return {
    x: Math.max(0, Math.min(canvas.width - 1, x)),
    y: Math.max(0, Math.min(canvas.height - 1, y)),
  };
}

function updateNodePreviewFromCanvas(node, canvas) {
  const dataUrl = canvas.toDataURL("image/png");
  const preview = new Image();
  preview.onload = () => {
    node.imageIndex = 0;
    node.imgs = [preview];
    app.graph?.setDirtyCanvas?.(true, true);
  };
  preview.src = dataUrl;
}

async function openStandaloneMaskBoardEditor(node, idx, image, defaultImage, slotWidget, imageWidget, seedKey = null) {
  ensureBoardEditorStyle();
  if (typeof activeBoardEditorCleanup === "function") {
    activeBoardEditorCleanup();
    activeBoardEditorCleanup = null;
  }

  const overlay = document.createElement("div");
  overlay.className = "ff-masktap-board-overlay";

  const dialog = document.createElement("div");
  dialog.className = "ff-masktap-board-dialog";
  overlay.appendChild(dialog);

  const toolbar = document.createElement("div");
  toolbar.className = "ff-masktap-board-toolbar";
  dialog.appendChild(toolbar);

  const title = document.createElement("div");
  title.className = "ff-masktap-board-title";
  title.textContent = `FreeFuse Mask Editor - Slot ${idx}`;
  toolbar.appendChild(title);

  const brushBtn = document.createElement("button");
  brushBtn.type = "button";
  brushBtn.className = "ff-masktap-board-btn active";
  brushBtn.textContent = "Brush";
  toolbar.appendChild(brushBtn);

  const eraserBtn = document.createElement("button");
  eraserBtn.type = "button";
  eraserBtn.className = "ff-masktap-board-btn";
  eraserBtn.textContent = "Eraser";
  toolbar.appendChild(eraserBtn);

  const sizeLabel = document.createElement("span");
  sizeLabel.textContent = "Size";
  toolbar.appendChild(sizeLabel);

  const sizeRange = document.createElement("input");
  sizeRange.type = "range";
  sizeRange.className = "ff-masktap-board-size";
  sizeRange.min = "1";
  sizeRange.max = "64";
  sizeRange.step = "1";
  sizeRange.value = "6";
  toolbar.appendChild(sizeRange);

  const sizeNumber = document.createElement("input");
  sizeNumber.type = "number";
  sizeNumber.className = "ff-masktap-board-size-num";
  sizeNumber.min = "1";
  sizeNumber.max = "64";
  sizeNumber.step = "1";
  sizeNumber.value = "6";
  toolbar.appendChild(sizeNumber);

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "ff-masktap-board-btn ghost";
  clearBtn.textContent = "Clear";
  toolbar.appendChild(clearBtn);

  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.className = "ff-masktap-board-btn ghost";
  resetBtn.textContent = "Reset to Default";
  toolbar.appendChild(resetBtn);

  const spacer = document.createElement("div");
  spacer.className = "ff-masktap-board-spacer";
  toolbar.appendChild(spacer);

  const meta = document.createElement("div");
  meta.className = "ff-masktap-board-meta";
  toolbar.appendChild(meta);

  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "ff-masktap-board-btn ghost";
  cancelBtn.textContent = "Cancel";
  toolbar.appendChild(cancelBtn);

  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.className = "ff-masktap-board-btn";
  saveBtn.textContent = "Save";
  toolbar.appendChild(saveBtn);

  const content = document.createElement("div");
  content.className = "ff-masktap-board-content";
  dialog.appendChild(content);

  const canvasWrap = document.createElement("div");
  canvasWrap.className = "ff-masktap-board-canvas-wrap";
  content.appendChild(canvasWrap);

  const canvas = document.createElement("canvas");
  canvas.className = "ff-masktap-board-canvas";
  canvasWrap.appendChild(canvas);

  normalizeMaskCanvasFromImage(canvas, image);
  const defaultCanvas = document.createElement("canvas");
  normalizeMaskCanvasFromImage(
    defaultCanvas,
    defaultImage || image,
    canvas.width,
    canvas.height
  );

  const defaultCtx = defaultCanvas.getContext("2d", { willReadFrequently: true });
  const defaultImageData = defaultCtx
    ? defaultCtx.getImageData(0, 0, defaultCanvas.width, defaultCanvas.height)
    : null;
  const defaultPixels = defaultImageData ? new Uint8ClampedArray(defaultImageData.data) : null;

  const applyDefaultMask = () => {
    if (!defaultPixels) {
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const restored = new ImageData(new Uint8ClampedArray(defaultPixels), canvas.width, canvas.height);
    ctx.putImageData(restored, 0, 0);
  };

  applyBoardCanvasDisplaySize(canvas);

  meta.textContent = `${canvas.width}x${canvas.height}  black=0 white=1`;

  const state = {
    tool: BOARD_TOOL_BRUSH,
    drawing: false,
    last: null,
    brushSize: Number(sizeRange.value),
  };

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return false;
  }
  ctx.imageSmoothingEnabled = false;

  const setTool = (tool) => {
    state.tool = tool;
    brushBtn.classList.toggle("active", tool === BOARD_TOOL_BRUSH);
    eraserBtn.classList.toggle("active", tool === BOARD_TOOL_ERASER);
  };

  const clampBrushSize = (v) => {
    const n = Number.parseInt(String(v), 10);
    if (!Number.isFinite(n)) return state.brushSize;
    return Math.max(1, Math.min(64, n));
  };

  const setBrushSize = (value) => {
    const size = clampBrushSize(value);
    state.brushSize = size;
    sizeRange.value = String(size);
    sizeNumber.value = String(size);
  };

  const drawSegment = (from, to) => {
    ctx.strokeStyle = state.tool === BOARD_TOOL_ERASER ? "#000000" : "#ffffff";
    ctx.lineWidth = state.brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    binarizeCanvasInPlace(canvas);
  };

  const handlePointerDown = (event) => {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    state.drawing = true;
    const p = getCanvasPointerPixel(canvas, event);
    state.last = p;
    drawSegment(p, p);
  };

  const handlePointerMove = (event) => {
    if (!state.drawing) return;
    const p = getCanvasPointerPixel(canvas, event);
    drawSegment(state.last || p, p);
    state.last = p;
  };

  const handlePointerUp = (event) => {
    if (state.drawing) {
      const p = getCanvasPointerPixel(canvas, event);
      drawSegment(state.last || p, p);
    }
    state.drawing = false;
    state.last = null;
  };

  let closed = false;
  const cleanups = [];
  const addEvent = (target, type, handler, options) => {
    target.addEventListener(type, handler, options);
    cleanups.push(() => target.removeEventListener(type, handler, options));
  };

  const closeEditor = () => {
    if (closed) return;
    closed = true;
    for (const fn of cleanups.splice(0)) {
      try {
        fn();
      } catch {
        // no-op
      }
    }
    overlay.remove();
    if (activeBoardEditorCleanup === closeEditor) {
      activeBoardEditorCleanup = null;
    }
  };

  activeBoardEditorCleanup = closeEditor;

  addEvent(overlay, "click", (event) => {
    if (event.target === overlay) {
      closeEditor();
    }
  });
  addEvent(window, "resize", () => applyBoardCanvasDisplaySize(canvas));
  addEvent(window, "keydown", (event) => {
    if (event.key === "Escape") {
      closeEditor();
      return;
    }
    if (event.key === "b" || event.key === "B") {
      setTool(BOARD_TOOL_BRUSH);
      return;
    }
    if (event.key === "e" || event.key === "E") {
      setTool(BOARD_TOOL_ERASER);
    }
  });
  addEvent(canvas, "pointerdown", handlePointerDown);
  addEvent(canvas, "pointermove", handlePointerMove);
  addEvent(canvas, "pointerup", handlePointerUp);
  addEvent(canvas, "pointercancel", handlePointerUp);
  addEvent(canvas, "lostpointercapture", handlePointerUp);
  addEvent(brushBtn, "click", () => setTool(BOARD_TOOL_BRUSH));
  addEvent(eraserBtn, "click", () => setTool(BOARD_TOOL_ERASER));
  addEvent(sizeRange, "input", () => setBrushSize(sizeRange.value));
  addEvent(sizeNumber, "change", () => setBrushSize(sizeNumber.value));
  addEvent(clearBtn, "click", () => {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });
  addEvent(resetBtn, "click", () => applyDefaultMask());
  addEvent(cancelBtn, "click", () => closeEditor());
  addEvent(saveBtn, "click", async () => {
    if (saveBtn.disabled) return;
    saveBtn.disabled = true;
    cancelBtn.disabled = true;
    try {
      binarizeCanvasInPlace(canvas);
      const editPrefix = seedKey
        ? `freefuse-masktap-edit-s${seedKey}`
        : "freefuse-masktap-edit";
      const newRef = await uploadCanvasAsImageRef(canvas, editPrefix);
      const formatted = formatWidgetRef(newRef);

      slotWidget.value = formatted;
      imageWidget.value = formatted;
      node.__ff_masktap_active_slot = idx;
      syncImageProxyToActiveSlot(node, formatted);

      if (!Array.isArray(node.images)) {
        node.images = [];
      }
      node.images[idx] = newRef;
      updateNodePreviewFromCanvas(node, canvas);
      closeEditor();
    } catch (error) {
      console.warn("[FreeFuseMaskTapUI] Failed to save edited mask:", error);
      saveBtn.disabled = false;
      cancelBtn.disabled = false;
    }
  });

  document.body.appendChild(overlay);
  return true;
}

function ensureImageProxyWidget(node) {
  let imageWidget = getWidget(node, "image");
  if (!imageWidget) {
    imageWidget = node.addWidget("text", "image", "", null, { multiline: false });
  }
  hideWidget(imageWidget);
  return imageWidget;
}

function getSelectedSlotIndex(node) {
  const slotWidget = getAnyWidget(node, [UI_SLOT_WIDGET, LEGACY_UI_SLOT_WIDGET]);
  if (!slotWidget) return 0;
  const n = Number.parseInt(String(slotWidget.value ?? "0"), 10);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(SLOT_COUNT - 1, n));
}

function normalizeRefObject(ref) {
  if (!ref || !ref.filename) return null;
  return {
    filename: String(ref.filename),
    subfolder: ref.subfolder ? String(ref.subfolder) : "",
    type: ref.type ? String(ref.type) : "input",
  };
}

function isLikelyLegacyBlankRef(ref) {
  if (!ref || !ref.filename) return false;
  const name = String(ref.filename).toLowerCase();
  return name.startsWith("freefuse-masktap-slot-") || name.startsWith("freefuse-masktap-blank-");
}

function isUserEditedRef(ref) {
  if (!ref || !ref.filename) return false;
  const name = String(ref.filename).toLowerCase();
  return name.startsWith(USER_EDIT_REF_PREFIX);
}

function extractUserEditSeedKey(ref) {
  if (!isUserEditedRef(ref)) return null;
  const name = String(ref.filename).toLowerCase();
  const match = name.match(USER_EDIT_SEED_RE);
  if (!match) return null;
  return String(match[1]);
}

function getGraphLinkRecord(linkId) {
  if (!Number.isFinite(Number(linkId))) return null;
  const links = app.graph?.links;
  if (!links) return null;

  if (Array.isArray(links)) {
    const asIndex = links[Number(linkId)];
    if (asIndex) return asIndex;
    return links.find((link) => Array.isArray(link) && Number(link[0]) === Number(linkId)) || null;
  }

  if (typeof links === "object") {
    return links[linkId] || links[String(linkId)] || null;
  }
  return null;
}

function getOriginNodeIdFromLink(link) {
  if (!link) return null;
  if (Array.isArray(link)) {
    const id = Number(link[1]);
    return Number.isFinite(id) ? id : null;
  }
  if (typeof link === "object") {
    const id = Number(link.origin_id ?? link.originId ?? link.from_id ?? link.fromId);
    return Number.isFinite(id) ? id : null;
  }
  return null;
}

function getNodeByIdSafe(nodeId) {
  const id = Number(nodeId);
  if (!Number.isFinite(id)) return null;
  const graph = app.graph;
  if (!graph) return null;
  if (typeof graph.getNodeById === "function") {
    const node = graph.getNodeById(id);
    if (node) return node;
  }
  if (Array.isArray(graph._nodes)) {
    return graph._nodes.find((node) => Number(node?.id) === id) || null;
  }
  return null;
}

function getPhase1SeedKey(node) {
  if (!node || !Array.isArray(node.inputs)) return null;
  const maskBankInput = node.inputs.find((input) => input?.name === "mask_bank");
  if (!maskBankInput || maskBankInput.link == null) return null;

  const link = getGraphLinkRecord(maskBankInput.link);
  const upstreamId = getOriginNodeIdFromLink(link);
  if (!Number.isFinite(upstreamId)) return null;

  const upstreamNode = getNodeByIdSafe(upstreamId);
  if (!upstreamNode || !Array.isArray(upstreamNode.widgets)) return null;

  const seedWidget = upstreamNode.widgets.find(
    (widget) => String(widget?.name || "").toLowerCase() === "seed"
  );
  if (!seedWidget) return null;

  const raw = seedWidget.value;
  if (raw == null) return null;
  const asNumber = Number(raw);
  if (Number.isFinite(asNumber)) {
    return String(Math.trunc(asNumber));
  }
  const text = String(raw).trim();
  return text || null;
}

function shouldUseEditedRefForSeed(ref, seedKey) {
  if (!isUserEditedRef(ref)) return true;
  if (!seedKey) return true;
  const refSeed = extractUserEditSeedKey(ref);
  if (!refSeed) return false;
  return refSeed === seedKey;
}

function syncSeedContext(node) {
  const seedKey = getPhase1SeedKey(node) || "";
  if (node.__ff_masktap_seed_key === seedKey) {
    return seedKey || null;
  }

  const prevSeedKey = node.__ff_masktap_seed_key;
  node.__ff_masktap_seed_key = seedKey;
  node.__ff_masktap_default_slot_refs = {};

  // Seed changed: clear stale edited refs so next run/open reflects fresh defaults.
  if (prevSeedKey !== undefined && prevSeedKey !== seedKey) {
    for (let i = 0; i < SLOT_COUNT; i++) {
      const slotWidget = getSlotRefWidget(node, i);
      if (!slotWidget) continue;
      const ref = parseWidgetRef(slotWidget.value);
      if (!isUserEditedRef(ref)) continue;
      if (!shouldUseEditedRefForSeed(ref, seedKey || null)) {
        slotWidget.value = "";
      }
    }
  }
  return seedKey || null;
}

function inferSlotRefFromNodeOutputs(node, slotIndex) {
  const idx = Math.max(0, Math.min(SLOT_COUNT - 1, Number(slotIndex) || 0));

  if (Array.isArray(node.images)) {
    const fromNodeImages = normalizeRefObject(node.images[idx]);
    if (fromNodeImages) return fromNodeImages;
  }

  const output = app.nodeOutputs?.[String(node.id)];
  if (output && Array.isArray(output.images)) {
    const fromOutputs = normalizeRefObject(output.images[idx]);
    if (fromOutputs) return fromOutputs;
  }

  return null;
}

function syncImageProxyToActiveSlot(node, value) {
  const idx = Number(node.__ff_masktap_active_slot);
  if (!Number.isFinite(idx) || idx < 0 || idx >= SLOT_COUNT) return;
  const slotWidget = getSlotRefWidget(node, idx);
  if (!slotWidget) return;
  slotWidget.value = value || "";
}

function hasWidgetValue(widget) {
  const value = widget?.value;
  if (value == null) return false;
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (typeof value === "object" && value.filename) {
    return true;
  }
  return String(value).trim().length > 0;
}

function getDefaultSlotRefStore(node) {
  if (!node.__ff_masktap_default_slot_refs || typeof node.__ff_masktap_default_slot_refs !== "object") {
    node.__ff_masktap_default_slot_refs = {};
  }
  return node.__ff_masktap_default_slot_refs;
}

function getDefaultRefForSlot(node, idx, slotWidget) {
  const store = getDefaultSlotRefStore(node);
  const inferred = normalizeRefObject(inferSlotRefFromNodeOutputs(node, idx));
  if (inferred && !hasWidgetValue(slotWidget)) {
    store[idx] = inferred;
  }
  const stored = normalizeRefObject(store[idx]);
  return stored || inferred || null;
}

async function openMaskEditorForSlot(node, slotIndex) {
  const seedKey = syncSeedContext(node);
  const idx = Math.max(0, Math.min(SLOT_COUNT - 1, Number(slotIndex) || 0));
  const slotWidget = getSlotRefWidget(node, idx);
  if (!slotWidget) {
    console.warn(`[FreeFuseMaskTapUI] Missing slot widget ${slotRefName(idx)}.`);
    return;
  }

  let ref = parseWidgetRef(slotWidget.value);
  let inferredFromOutputs = false;
  if (ref && isUserEditedRef(ref) && !shouldUseEditedRefForSeed(ref, seedKey)) {
    // Edited mask came from a different seed (or legacy untagged edit): discard it.
    ref = null;
    slotWidget.value = "";
  }
  if (ref && isLikelyLegacyBlankRef(ref)) {
    const currentMaskRef = inferSlotRefFromNodeOutputs(node, idx);
    if (currentMaskRef) {
      ref = currentMaskRef;
    }
  }
  if (!ref) {
    ref = inferSlotRefFromNodeOutputs(node, idx);
    inferredFromOutputs = !!ref;
  }
  if (ref && inferredFromOutputs) {
    ref = normalizeRefObject(ref);
  }
  if (!ref) {
    ref = await uploadBlankImageRef();
  }
  const defaultRef = getDefaultRefForSlot(node, idx, slotWidget);

  const imageWidget = ensureImageProxyWidget(node);

  node.__ff_masktap_active_slot = idx;

  let image = null;
  try {
    image = await loadImage(buildViewUrl(ref));
  } catch (error) {
    console.warn("[FreeFuseMaskTapUI] Failed to load slot image, using blank canvas:", error);
  }

  if (!image) {
    const fallback = document.createElement("canvas");
    fallback.width = 512;
    fallback.height = 512;
    const fallbackCtx = fallback.getContext("2d");
    if (fallbackCtx) {
      fallbackCtx.fillStyle = "#000000";
      fallbackCtx.fillRect(0, 0, fallback.width, fallback.height);
    }
    image = await loadImage(fallback.toDataURL("image/png"));
  }

  let defaultImage = image;
  if (defaultRef) {
    try {
      defaultImage = await loadImage(buildViewUrl(defaultRef));
    } catch (error) {
      console.warn("[FreeFuseMaskTapUI] Failed to load default slot image:", error);
    }
  }

  await openStandaloneMaskBoardEditor(node, idx, image, defaultImage, slotWidget, imageWidget, seedKey);
}

function clearCurrentSlotRef(node) {
  const idx = getSelectedSlotIndex(node);
  const slotWidget = getSlotRefWidget(node, idx);
  if (slotWidget) {
    slotWidget.value = "";
  }
}

function setupMaskTapUI(node) {
  syncSeedContext(node);

  if (node.__ff_masktap_ui_initialized) {
    compactLegacyMaskOutputs(node);
    compactNodeSize(node);
    return;
  }

  compactLegacyMaskOutputs(node);

  for (let i = 0; i < SLOT_COUNT; i++) {
    hideWidget(getSlotRefWidget(node, i));
  }

  const imageWidget = ensureImageProxyWidget(node);
  const prevImageCallback = imageWidget.callback;
  imageWidget.callback = function (...args) {
    try {
      if (typeof prevImageCallback === "function") {
        prevImageCallback.apply(this, args);
      }
    } finally {
      syncImageProxyToActiveSlot(node, imageWidget.value);
    }
  };

  const slotValues = Array.from({ length: SLOT_COUNT }, (_, i) => `${i}`);
  if (!getAnyWidget(node, [UI_SLOT_WIDGET, LEGACY_UI_SLOT_WIDGET])) {
    node.addWidget("combo", UI_SLOT_WIDGET, "0", () => {}, { values: slotValues });
  }

  if (!getAnyWidget(node, [UI_OPEN_BUTTON, LEGACY_UI_OPEN_BUTTON])) {
    node.addWidget("button", UI_OPEN_BUTTON, "Open Mask Editor", () => {
      const idx = getSelectedSlotIndex(node);
      void openMaskEditorForSlot(node, idx);
    });
  }

  if (!getAnyWidget(node, [UI_CLEAR_BUTTON, LEGACY_UI_CLEAR_BUTTON])) {
    node.addWidget("button", UI_CLEAR_BUTTON, "Clear Current Slot", () => {
      clearCurrentSlotRef(node);
    });
  }

  const slotWidget = getAnyWidget(node, [UI_SLOT_WIDGET, LEGACY_UI_SLOT_WIDGET]);
  if (slotWidget) {
    slotWidget.callback = () => {
      const idx = getSelectedSlotIndex(node);
      node.__ff_masktap_active_slot = idx;
      const currentRef = getSlotRefWidget(node, idx)?.value || "";
      imageWidget.value = currentRef;
    };
  }

  node.__ff_masktap_active_slot = 0;
  node.__ff_masktap_ui_initialized = true;
  compactNodeSize(node);
}

function setupMaskReassembleUI(node) {
  if (node.__ff_maskreassemble_ui_initialized) {
    compactLegacyMaskInputs(node);
    compactNodeSize(node);
    return;
  }
  compactLegacyMaskInputs(node);
  node.__ff_maskreassemble_ui_initialized = true;
  compactNodeSize(node);
}

app.registerExtension({
  name: "FreeFuse.MaskTapEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === TAP_NODE_NAME) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        setupMaskTapUI(this);
        return result;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        setupMaskTapUI(this);
        return result;
      };

      const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        if (getExtraMenuOptions) {
          getExtraMenuOptions.apply(this, arguments);
        }
        options.push(null);
        options.push({
          content: "FreeFuse: Open Mask Editor (Current Slot)",
          callback: () => {
            const idx = getSelectedSlotIndex(this);
            void openMaskEditorForSlot(this, idx);
          },
        });
        options.push({
          content: "FreeFuse: Clear Current Slot Edit",
          callback: () => {
            clearCurrentSlotRef(this);
          },
        });
      };
      return;
    }

    if (nodeData?.name === REASSEMBLE_NODE_NAME) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        setupMaskReassembleUI(this);
        return result;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        setupMaskReassembleUI(this);
        return result;
      };
    }
  },
});
