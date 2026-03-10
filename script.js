import * as THREE from 'https://esm.sh/three@0.164.1';
import { GLTFLoader } from 'https://esm.sh/three@0.164.1/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'https://esm.sh/three@0.164.1/examples/jsm/loaders/DRACOLoader.js';
import { FBXLoader } from 'https://esm.sh/three@0.164.1/examples/jsm/loaders/FBXLoader.js';
import { TRACKS, computeTrackBounds, createCheckpoints, sampleClosedCurveFrame, nearestPointOnCurve } from './track.js';
import { installBrowserBenchmarkHooks } from './tools/browser_benchmark.js';

const speedText = document.getElementById('speedText');
const gearText = document.getElementById('gearText');
const lapText = document.getElementById('lapText');
const positionText = document.getElementById('positionText');
const leaderboardText = document.getElementById('leaderboardText');
const bestText = document.getElementById('bestText');
const nowText = document.getElementById('nowText');
const penaltyText = document.getElementById('penaltyText');
const damageText = document.getElementById('damageText');
const partsText = document.getElementById('partsText');
const tyreText = document.getElementById('tyreText');
const slipText = document.getElementById('slipText');
const pitText = document.getElementById('pitText');
const buildText = document.getElementById('buildText');
const throttleBar = document.getElementById('throttleBar');
const brakeBar = document.getElementById('brakeBar');
const rpmBar = document.getElementById('rpmBar');
const minimap = document.getElementById('minimap');
const mm = minimap.getContext('2d');
const gArc = document.getElementById('gArc');
const gNeedle = document.getElementById('gNeedle');
const gSpeedText = document.getElementById('gSpeedText');
const carSelect = document.getElementById('carSelect');
const carStats = document.getElementById('carStats');
const startMenu = document.getElementById('startMenu');
const startCarSelect = document.getElementById('startCarSelect');
const trackSelect = document.getElementById('trackSelect');
const carPreview = document.getElementById('carPreview');
const trackPreview = document.getElementById('trackPreview');
const startRaceBtn = document.getElementById('startRaceBtn');
const btnLeft = document.getElementById('btnLeft');
const btnRight = document.getElementById('btnRight');
const btnBrake = document.getElementById('btnBrake');
const btnAccel = document.getElementById('btnAccel');
const finishPanel = document.getElementById('finishPanel');
const finishSummary = document.getElementById('finishSummary');
const restartBtn = document.getElementById('restartBtn');

const BUILD_VERSION = 'racing v2026.03.11-02';
if (buildText) buildText.textContent = BUILD_VERSION;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#a9d0ff');
scene.fog = new THREE.Fog('#a9d0ff', 180, 760);

const frustum = 95;
const camera = new THREE.OrthographicCamera(
  -frustum * (innerWidth / innerHeight),
  frustum * (innerWidth / innerHeight),
  frustum,
  -frustum,
  0.1,
  2500
);
camera.up.set(0, 1, 0);

const hemi = new THREE.HemisphereLight(0xf2f8ff, 0x6f9b63, 1.35);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 1.25);
dir.position.set(120, 220, 80);
scene.add(dir);

const fill = new THREE.DirectionalLight(0xbfd8ff, 0.55);
fill.position.set(-120, 140, -60);
scene.add(fill);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(2200, 2200),
  new THREE.MeshStandardMaterial({ color: '#6aa55a', roughness: 0.95 })
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);


let currentTrackKey = 'stadium';
let curve = null;
let mapBounds = { minX: -1, maxX: 1, minZ: -1, maxZ: 1 };
let checkpoints = [];
let trackLength = 1;
let startP = new THREE.Vector3();
let startDir = new THREE.Vector3(0, 0, 1);
let roadMesh = null;
let curbMesh = null;
let startLine = null;
let pitZoneMesh = null;
let pitZoneCenter = new THREE.Vector3();

// Continuous road strips (no segment gaps)
const roadMat = new THREE.MeshStandardMaterial({ color: '#3a4150', roughness: .88, metalness: .03 });
const edgeMat = new THREE.MeshStandardMaterial({ color: '#c2c9d3', roughness: 0.95, metalness: 0.0 });
const curbW = 48;
const roadW = 40;
const segCount = 900;
const shoulderWidth = 24; // reserved shoulder width
const guardWallLimit = roadW * 0.5 + shoulderWidth;

function buildRibbon(width, y, mat) {
  const positions = [];
  const uvs = [];
  const indices = [];

  for (let i = 0; i <= segCount; i++) {
    const t = i / segCount;
    const p = curve.getPointAt(t);
    const p2 = curve.getPointAt((t + 1 / segCount) % 1);
    const dir = new THREE.Vector3().subVectors(p2, p).normalize();
    const right = new THREE.Vector3(-dir.z, 0, dir.x);

    const l = p.clone().addScaledVector(right, -width / 2);
    const r = p.clone().addScaledVector(right, width / 2);

    positions.push(l.x, y, l.z, r.x, y, r.z);
    uvs.push(0, t * 14, 1, t * 14);

    if (i < segCount) {
      const b = i * 2;
      indices.push(b, b + 1, b + 2, b + 1, b + 3, b + 2);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(indices);
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, mat);
  scene.add(m);
  return m;
}


function setTrack(trackKey = 'classic') {
  currentTrackKey = TRACKS[trackKey] ? trackKey : 'classic';
  const pts = TRACKS[currentTrackKey];
  curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal', 0.08);
  trackLength = Math.max(1, curve.getLength());
  mapBounds = computeTrackBounds(pts);

  if (curbMesh) scene.remove(curbMesh);
  if (roadMesh) scene.remove(roadMesh);
  if (startLine) scene.remove(startLine);
  if (pitZoneMesh) scene.remove(pitZoneMesh);

  curbMesh = buildRibbon(curbW, 0.22, edgeMat);
  roadMesh = buildRibbon(roadW, 0.5, roadMat);

  startP = curve.getPointAt(0);
  const startP2 = curve.getPointAt(1 / segCount);
  startDir = new THREE.Vector3().subVectors(startP2, startP).normalize();
  const startRight = new THREE.Vector3(-startDir.z, 0, startDir.x);

  startLine = new THREE.Mesh(
    new THREE.PlaneGeometry(roadW * 0.92, 2.8),
    new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.95, side: THREE.DoubleSide })
  );
  startLine.rotation.x = -Math.PI / 2;
  startLine.rotation.z = Math.atan2(startRight.y, startRight.x);
  startLine.position.set(startP.x, 0.72, startP.z);
  scene.add(startLine);

  pitZoneCenter = startP.clone().add(startDir.clone().multiplyScalar(45));
  pitZoneMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(roadW * 0.8, 16),
    new THREE.MeshBasicMaterial({ color: '#5ee0ff', transparent: true, opacity: 0.35, side: THREE.DoubleSide })
  );
  pitZoneMesh.rotation.x = -Math.PI / 2;
  pitZoneMesh.position.set(pitZoneCenter.x, 0.74, pitZoneCenter.z);
  scene.add(pitZoneMesh);

  checkpoints = createCheckpoints(curve, 8);
  drawTrackPreview();
  resetCar();
}

// Car mesh (real sample model via GLB)
const carRoot = new THREE.Group();
scene.add(carRoot);

// fallback if model fails
const fallback = new THREE.Mesh(
  new THREE.BoxGeometry(8.2, 1.6, 15.0),
  new THREE.MeshStandardMaterial({ color: '#dfe7f4', roughness: .4, metalness: .25 })
);
fallback.position.y = 1.4;
carRoot.add(fallback);

const AI_DRIVERS = [
  { name: 'NOVA', color: '#ff8b7a', style: { pace: 98, corner: 0.85, jitter: 0.28, bravery: 1.04, drift: 1.08 } },
  { name: 'RUNE', color: '#7ab6ff', style: { pace: 96, corner: 1.2, jitter: 0.18, bravery: 0.98, drift: 0.9 } },
  { name: 'ECHO', color: '#9bff96', style: { pace: 94, corner: 0.95, jitter: 0.24, bravery: 0.95, drift: 0.96 } },
  { name: 'BLITZ', color: '#ffd86b', style: { pace: 101, corner: 0.75, jitter: 0.22, bravery: 1.08, drift: 1.14 } },
  { name: 'MIRA', color: '#d69bff', style: { pace: 95, corner: 1.05, jitter: 0.16, bravery: 1.0, drift: 0.98 } },
];

function createAICar(color) {
  const root = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(7.8, 1.4, 14.2),
    new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.2 })
  );
  body.position.y = 1.3;
  const canopy = new THREE.Mesh(
    new THREE.BoxGeometry(5.2, 1.1, 5.8),
    new THREE.MeshStandardMaterial({ color: '#1f2430', roughness: 0.3, metalness: 0.45 })
  );
  canopy.position.set(0, 2.05, -0.2);
  root.add(body, canopy);
  return root;
}

const rivals = AI_DRIVERS.map((d, i) => {
  const root = createAICar(d.color);
  scene.add(root);
  return {
    id: `ai-${i + 1}`,
    name: d.name,
    root,
    t: (0.95 - i * 0.045 + 1) % 1,
    speed: d.style.pace,
    speedCurrent: d.style.pace,
    x: 0,
    z: 0,
    heading: 0,
    yawOffset: 0,
    ox: 0,
    oz: 0,
    ovx: 0,
    ovz: 0,
    laneOffset: 0,
    laneTarget: 0,
    steer: 0,
    throttleCmd: 0,
    vx: 0,
    vz: 0,
    yawRate: 0,
    brakePedal: 0,
    lap: 1,
    nextCp: 0,
    cpCycleReady: false,
    learn: Array.from({ length: 8 }, () => ({ attack: 1, samples: 0, mistakes: 0 })),
    lastLearnIdx: 0,
    driftAmount: 0,
    trackError: 0,
    lastProgress: 0,
    recovery: 0,
    stability: 1,
    lastContactTelemetry: 0,
    style: d.style,
  };
});

const loader = new GLTFLoader();
const draco = new DRACOLoader();
draco.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');
loader.setDRACOLoader(draco);
const fbxLoader = new FBXLoader();
const texLoader = new THREE.TextureLoader();

const CAR_MODELS = {
  ferrari: {
    label: 'Ferrari',
    type: 'gltf',
    url: 'https://threejs.org/examples/models/gltf/ferrari.glb',
    stats: { topSpeed: 220, accel: 150, brake: 165, handling: 1.05 },
    rotFix: { x: 0, y: 0, z: 0 },
    yawOffset: 0,
  },
  lowpoly1: {
    label: 'Low Poly Car 1',
    type: 'fbx',
    url: './assets/cars/lowpoly/car_1.fbx',
    texture: './assets/cars/lowpoly/car_texture_1.png',
    stats: { topSpeed: 192, accel: 128, brake: 148, handling: 1.15 },
    rotFix: { x: -Math.PI / 2, y: 0, z: 0 },
    yawOffset: -Math.PI / 2,
  },
  lowpoly2: {
    label: 'Low Poly Car 2',
    type: 'fbx',
    url: './assets/cars/lowpoly/car_2.fbx',
    texture: './assets/cars/lowpoly/car_texture_2.png',
    stats: { topSpeed: 205, accel: 136, brake: 155, handling: 0.98 },
    rotFix: { x: -Math.PI / 2, y: 0, z: 0 },
    yawOffset: -Math.PI / 2,
  },
};

let currentCarKey = carSelect?.value || 'ferrari';
let currentCarStats = CAR_MODELS[currentCarKey]?.stats || CAR_MODELS.ferrari.stats;
let currentCarYawOffset = CAR_MODELS[currentCarKey]?.yawOffset || 0;


function updateCarStatsUI() {
  const cfg = CAR_MODELS[currentCarKey] || CAR_MODELS.ferrari;
  const s = cfg.stats;
  if (!carStats) return;
  carStats.textContent = `${cfg.label}\nTOP ${s.topSpeed} km/h · ACC ${s.accel}\nBRK ${s.brake} · HDL ${(s.handling * 100).toFixed(0)}%`;
}

function prepareModel(model, extraRotationY = 0, rotFix = null) {
  model.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = false;
      obj.receiveShadow = false;
    }
  });

  const box = new THREE.Box3().setFromObject(model);
  const size = new THREE.Vector3();
  box.getSize(size);
  const targetLength = 15.0;
  const scale = targetLength / Math.max(size.x, size.z, 0.001);
  model.scale.setScalar(scale);
  model.rotation.y += extraRotationY;
  if (rotFix) {
    model.rotation.x += rotFix.x || 0;
    model.rotation.y += rotFix.y || 0;
    model.rotation.z += rotFix.z || 0;
  }

  const box2 = new THREE.Box3().setFromObject(model);
  model.position.y += -box2.min.y + 0.15;
  return model;
}

function setFallbackVisible(visible) {
  fallback.visible = visible;
}


function loadCarModel(key = 'ferrari') {
  currentCarKey = CAR_MODELS[key] ? key : 'ferrari';
  const cfg = CAR_MODELS[currentCarKey] || CAR_MODELS.ferrari;
  currentCarStats = cfg.stats;
  currentCarYawOffset = cfg.yawOffset || 0;
  updateCarStatsUI();

  if (cfg.type === 'gltf') {
    loader.load(
      cfg.url,
      (gltf) => {
        carRoot.clear();
        const model = prepareModel(gltf.scene, Math.PI, cfg.rotFix);
        carRoot.add(model);
        setFallbackVisible(false);
      },
      undefined,
      () => {
        carRoot.clear();
        carRoot.add(fallback);
        setFallbackVisible(true);
      }
    );
    return;
  }

  if (cfg.type === 'fbx') {
    fbxLoader.load(
      cfg.url,
      (model) => {
        const tex = cfg.texture ? texLoader.load(cfg.texture) : null;
        if (tex) {
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.flipY = false;
        }

        model.traverse((obj) => {
          if (!obj.isMesh) return;
          obj.castShadow = false;
          obj.receiveShadow = false;

          const baseMat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
          obj.material = new THREE.MeshStandardMaterial({
            map: tex || null,
            color: baseMat?.color || new THREE.Color('#e6edf8'),
            roughness: 0.7,
            metalness: 0.15,
          });
        });

        carRoot.clear();
        const normalized = prepareModel(model, Math.PI, cfg.rotFix);
        carRoot.add(normalized);
        setFallbackVisible(false);
      },
      undefined,
      () => {
        carRoot.clear();
        carRoot.add(fallback);
        setFallbackVisible(true);
      }
    );
  }
}

function syncCarSelectors(value) {
  if (carSelect) carSelect.value = value;
  if (startCarSelect) startCarSelect.value = value;
}

carSelect?.addEventListener('change', () => {
  syncCarSelectors(carSelect.value);
  loadCarModel(carSelect.value);
  drawCarPreview();
});

startCarSelect?.addEventListener('change', () => {
  syncCarSelectors(startCarSelect.value);
  loadCarModel(startCarSelect.value);
  drawCarPreview();
});

trackSelect?.addEventListener('change', () => {
  drawTrackPreview();
});

startRaceBtn?.addEventListener('click', () => {
  setTrack(trackSelect?.value || 'classic');
  const selectedCar = startCarSelect?.value || 'ferrari';
  syncCarSelectors(selectedCar);
  loadCarModel(selectedCar);
  raceStarted = true;
  raceFinished = false;
  if (finishPanel) finishPanel.style.display = 'none';
  if (startMenu) startMenu.style.display = 'none';
  setupAudio();
});

restartBtn?.addEventListener('click', () => {
  raceFinished = false;
  raceStarted = false;
  keys.clear();
  if (finishPanel) finishPanel.style.display = 'none';
  if (startMenu) startMenu.style.display = 'grid';
  resetCar();
});

// skid marks
const skidGroup = new THREE.Group();
scene.add(skidGroup);
const skidMarks = [];

// rear smoke particles
const smokeGroup = new THREE.Group();
scene.add(smokeGroup);
const smokeParticles = [];
let raceStarted = false;

const state = {
  x: 0, z: 0,
  heading: 0,
  steer: 0,
  vx: 0,
  vz: 0,
  yawRate: 0,
  skidCd: 0,
  brakePedal: 0,
  gear: 1,
  rpm: 0.25,
  shiftLock: 0,
  lap: 1,
  nextCp: 0,
  lapStart: performance.now(),
  bestLap: null,
  lapPenaltyMs: 0,
  offroadAllWheels: false,
  prevGateDist: 0,
  startGateArmed: false,
  cpCycleReady: false,
  damage: 0,
  damageEngine: 0,
  damageTire: 0,
  damageAero: 0,
  repairTimer: 0,
  slipstreamBoost: 0,
  pitStopsUsed: 0,
  canUsePitThisLap: true,
  tireWear: 0,
};

const aiTelemetry = {
  telemetryVersion: 2,
  episodes: [],
  current: null,
  maxEpisodes: 24,
  sampleEveryMs: 220,
  maxStepSamplesPerDriver: 6000,
  lastSampleAt: 0,
};

let playerTrackT = 0;
let raceOrder = [];
let raceFinished = false;

const keys = new Set();
addEventListener('keydown', (e) => {
  keys.add(e.key.toLowerCase());
  if (e.key.toLowerCase() === 'r') resetCar();
  setupAudio();
});
addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

function bindHoldButton(el, key) {
  if (!el) return;
  const down = (ev) => {
    ev.preventDefault();
    if (!raceStarted) return;
    keys.add(key);
    setupAudio();
  };
  const up = (ev) => {
    ev.preventDefault();
    keys.delete(key);
  };

  // Pointer events
  el.addEventListener('pointerdown', down);
  el.addEventListener('pointerup', up);
  el.addEventListener('pointercancel', up);
  el.addEventListener('pointerleave', up);

  // Touch fallback (iOS/Safari)
  el.addEventListener('touchstart', down, { passive: false });
  el.addEventListener('touchend', up, { passive: false });
  el.addEventListener('touchcancel', up, { passive: false });

  // Mouse fallback
  el.addEventListener('mousedown', down);
  el.addEventListener('mouseup', up);
  el.addEventListener('mouseleave', up);
}

bindHoldButton(btnLeft, 'a');
bindHoldButton(btnRight, 'd');
bindHoldButton(btnBrake, 's');
bindHoldButton(btnAccel, 'w');



function fmt(ms) { return `${(ms / 1000).toFixed(3)}s`; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function angleDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}
function sampleTrackFrame(t) {
  return sampleClosedCurveFrame(curve, t);
}

function mapToMini(x, z) {
  const pad = 14;
  const w = minimap.width - pad * 2;
  const h = minimap.height - pad * 2;
  const nx = (x - mapBounds.minX) / Math.max(1, (mapBounds.maxX - mapBounds.minX));
  const nz = (z - mapBounds.minZ) / Math.max(1, (mapBounds.maxZ - mapBounds.minZ));
  return { x: pad + nx * w, y: pad + nz * h };
}

function polarToXY(cx, cy, r, deg) {
  const a = (deg - 90) * Math.PI / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function describeArc(cx, cy, r, startDeg, endDeg) {
  const s = polarToXY(cx, cy, r, endDeg);
  const e = polarToXY(cx, cy, r, startDeg);
  const large = endDeg - startDeg <= 180 ? 0 : 1;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 0 ${e.x} ${e.y}`;
}

function drawCarPreview() {
  if (!carPreview) return;
  const ctx = carPreview.getContext('2d');
  if (!ctx) return;
  const w = carPreview.width;
  const h = carPreview.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#121926';
  ctx.fillRect(0, 0, w, h);

  const cfg = CAR_MODELS[currentCarKey] || CAR_MODELS.ferrari;
  const color = currentCarKey === 'ferrari' ? '#e94f55' : currentCarKey === 'lowpoly1' ? '#58a7ff' : '#7be08a';
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(70, 44, 180, 34, 12);
  ctx.fill();
  ctx.fillStyle = '#1f2430';
  ctx.fillRect(116, 38, 88, 20);
  ctx.fillStyle = '#dce8ff';
  ctx.font = '12px Inter, Arial';
  ctx.fillText(cfg.label, 12, 20);
  ctx.fillText(`TOP ${cfg.stats.topSpeed} | ACC ${cfg.stats.accel} | HDL ${(cfg.stats.handling * 100).toFixed(0)}%`, 12, 106);
}

function drawTrackPreview() {
  if (!trackPreview) return;
  const ctx = trackPreview.getContext('2d');
  if (!ctx) return;
  const w = trackPreview.width;
  const h = trackPreview.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#121926';
  ctx.fillRect(0, 0, w, h);

  const pts = TRACKS[trackSelect?.value || currentTrackKey] || TRACKS.stadium;
  const bounds = pts.reduce((acc, p) => ({
    minX: Math.min(acc.minX, p.x), maxX: Math.max(acc.maxX, p.x),
    minZ: Math.min(acc.minZ, p.z), maxZ: Math.max(acc.maxZ, p.z),
  }), { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity });

  const pad = 14;
  const sx = (w - pad * 2) / Math.max(1, (bounds.maxX - bounds.minX));
  const sz = (h - pad * 2) / Math.max(1, (bounds.maxZ - bounds.minZ));

  ctx.strokeStyle = '#9ec4ff';
  ctx.lineWidth = 4;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = pad + (p.x - bounds.minX) * sx;
    const y = pad + (p.z - bounds.minZ) * sz;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.stroke();

  const sp = pts[0];
  const x0 = pad + (sp.x - bounds.minX) * sx;
  const y0 = pad + (sp.z - bounds.minZ) * sz;
  ctx.fillStyle = '#fff';
  ctx.fillRect(x0 - 3, y0 - 3, 6, 6);
}

function drawMiniMap() {
  mm.clearRect(0, 0, minimap.width, minimap.height);
  mm.fillStyle = 'rgba(18,24,38,.78)';
  mm.fillRect(0, 0, minimap.width, minimap.height);

  mm.strokeStyle = '#a6b5cc';
  mm.lineWidth = 4;
  mm.beginPath();
  for (let i = 0; i <= 180; i++) {
    const p = curve.getPointAt(i / 180);
    const m = mapToMini(p.x, p.z);
    if (i === 0) mm.moveTo(m.x, m.y);
    else mm.lineTo(m.x, m.y);
  }
  mm.closePath();
  mm.stroke();

  const startM = mapToMini(startP.x, startP.z);
  mm.fillStyle = '#ffffff';
  mm.fillRect(startM.x - 4, startM.y - 4, 8, 8);

  const cp = checkpoints[state.nextCp];
  const cpM = mapToMini(cp.x, cp.z);
  mm.fillStyle = '#ffd86b';
  mm.beginPath(); mm.arc(cpM.x, cpM.y, 4.5, 0, Math.PI * 2); mm.fill();

  rivals.forEach((ai) => {
    const rivalM = mapToMini(ai.x, ai.z);
    mm.fillStyle = ai.root.children[0]?.material?.color?.getStyle?.() || '#ff8b7a';
    mm.beginPath(); mm.arc(rivalM.x, rivalM.y, 3.3, 0, Math.PI * 2); mm.fill();
  });

  const carM = mapToMini(state.x, state.z);
  mm.fillStyle = '#5fe4ff';
  mm.beginPath(); mm.arc(carM.x, carM.y, 4, 0, Math.PI * 2); mm.fill();
}


function resetCar() {
  state.x = startP.x;
  state.z = startP.z;
  state.heading = Math.atan2(startDir.x, startDir.z);
  state.vx = 0; state.vz = 0; state.yawRate = 0; state.steer = 0;
  state.skidCd = 0;
  state.brakePedal = 0;
  state.gear = 1; state.rpm = 0.25; state.shiftLock = 0;
  state.lap = 1; state.nextCp = 0;
  state.lapStart = performance.now();
  state.lapPenaltyMs = 0;
  state.offroadAllWheels = false;
  state.prevGateDist = 0;
  state.startGateArmed = false;
  state.cpCycleReady = false;
  state.damage = 0;
  state.damageEngine = 0;
  state.damageTire = 0;
  state.damageAero = 0;
  state.repairTimer = 0;
  state.slipstreamBoost = 0;
  state.pitStopsUsed = 0;
  state.canUsePitThisLap = true;
  state.tireWear = 0;

  const gateFwd = new THREE.Vector2(startDir.x, startDir.z).normalize();
  const gateDelta = new THREE.Vector2(state.x - startP.x, state.z - startP.z);
  state.prevGateDist = gateDelta.dot(gateFwd);

  rivals.forEach((ai, i) => {
    ai.t = (0.95 - i * 0.045 + 1) % 1;
    ai.ox = 0;
    ai.oz = 0;
    ai.ovx = 0;
    ai.ovz = 0;
    ai.laneOffset = 0;
    ai.laneTarget = 0;
    ai.speedCurrent = ai.speed;
    ai.steer = 0;
    ai.throttleCmd = 0;
    ai.vx = 0;
    ai.vz = 0;
    ai.yawRate = 0;
    ai.brakePedal = 0;
    ai.lap = 1;
    ai.nextCp = 0;
    ai.cpCycleReady = false;
    ai.learn = Array.from({ length: checkpoints.length }, () => ({ attack: 1, samples: 0, mistakes: 0 }));
    ai.lastLearnIdx = 0;
    ai.driftAmount = 0;
    ai.trackError = 0;
    ai.lastProgress = 0;
    ai.recovery = 0;
    ai.stability = 1;
    ai.lastContactTelemetry = 0;
    const rp = curve.getPointAt(ai.t);
    const rp2 = curve.getPointAt((ai.t + 0.002) % 1);
    ai.x = rp.x;
    ai.z = rp.z;
    ai.heading = Math.atan2(rp2.x - rp.x, rp2.z - rp.z);
  });
  raceOrder = [];
  beginAITelemetrySession();

  // clear skid marks + smoke
  skidMarks.length = 0;
  while (skidGroup.children.length) skidGroup.remove(skidGroup.children[0]);
  smokeParticles.length = 0;
  while (smokeGroup.children.length) smokeGroup.remove(smokeGroup.children[0]);
}

syncCarSelectors(currentCarKey);
loadCarModel(currentCarKey);
if (trackSelect) trackSelect.value = 'stadium';
drawCarPreview();
drawTrackPreview();
setTrack('stadium');
installBrowserBenchmarkHooks({ buildVersion: BUILD_VERSION });

function spawnSkidMark(x, z, heading, alpha = 0.25) {
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(0.7, 2.1),
    new THREE.MeshBasicMaterial({
      color: '#111',
      transparent: true,
      opacity: alpha,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -4,
    })
  );
  m.rotation.x = -Math.PI / 2;
  m.rotation.z = heading;
  m.position.set(x, 0.71, z); // sit on top of road to avoid z-fighting artifacts
  m.renderOrder = 10;
  skidGroup.add(m);
  skidMarks.push({ mesh: m, life: 2.8 });

  if (skidMarks.length > 260) {
    const old = skidMarks.shift();
    if (old?.mesh) skidGroup.remove(old.mesh);
  }
}

function updateSkidMarks(dt) {
  for (let i = skidMarks.length - 1; i >= 0; i--) {
    const s = skidMarks[i];
    s.life -= dt;
    if (s.life <= 0) {
      skidGroup.remove(s.mesh);
      skidMarks.splice(i, 1);
      continue;
    }
    s.mesh.material.opacity = Math.max(0, Math.min(0.35, s.life * 0.12));
  }
}

function spawnSmoke(x, z, intensity = 1) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.55 + Math.random() * 0.25, 8, 8),
    new THREE.MeshBasicMaterial({ color: '#cfd3d9', transparent: true, opacity: 0.22 * intensity, depthWrite: false })
  );
  m.position.set(x + (Math.random() - 0.5) * 0.6, 1.3 + Math.random() * 0.3, z + (Math.random() - 0.5) * 0.6);
  smokeGroup.add(m);
  smokeParticles.push({ mesh: m, life: 0.9 + Math.random() * 0.4, vx: (Math.random() - 0.5) * 0.5, vz: 0.6 + Math.random() * 1.0 });

  if (smokeParticles.length > 180) {
    const old = smokeParticles.shift();
    if (old?.mesh) smokeGroup.remove(old.mesh);
  }
}

function updateSmoke(dt) {
  for (let i = smokeParticles.length - 1; i >= 0; i--) {
    const p = smokeParticles[i];
    p.life -= dt;
    if (p.life <= 0) {
      smokeGroup.remove(p.mesh);
      smokeParticles.splice(i, 1);
      continue;
    }
    p.mesh.position.x += p.vx * dt;
    p.mesh.position.z += p.vz * dt;
    p.mesh.position.y += 0.6 * dt;
    p.mesh.scale.multiplyScalar(1 + dt * 0.8);
    p.mesh.material.opacity = Math.max(0, p.life * 0.18);
  }
}

function nearestTrackSample(x, z, samples = 180, aroundT = null, window = 0.12) {
  const hit = nearestPointOnCurve(curve, x, z, { samples, aroundT, window });
  return {
    t: hit.t,
    distSq: hit.distSq,
    point: hit.frame.p,
    right: new THREE.Vector2(hit.frame.rightX, hit.frame.rightZ),
    tangent: new THREE.Vector2(hit.frame.tangentX, hit.frame.tangentZ),
    signedOffset: hit.signedOffset,
  };
}

function roadDistanceSq(x, z, aroundT = null) {
  return nearestTrackSample(x, z, 180, aroundT).distSq;
}

function trackTAt(x, z, samples = 260, aroundT = null, window = 0.12) {
  return nearestTrackSample(x, z, samples, aroundT, window).t;
}

function buildTelemetryObservation(ai, progress) {
  const nearest = nearestTrackSample(ai.x, ai.z, 220, ai.t, 0.12);
  const headingTrack = Math.atan2(nearest.tangent.x, nearest.tangent.y);
  const headingErr = angleDiff(ai.heading, headingTrack);
  const speed = Math.hypot(ai.vx, ai.vz);
  const fwd = new THREE.Vector2(Math.sin(ai.heading), Math.cos(ai.heading));
  const right = new THREE.Vector2(fwd.y, -fwd.x);
  const lateralSpeed = ai.vx * right.x + ai.vz * right.y;
  const slipAngle = Math.atan2(lateralSpeed, Math.max(1, Math.abs(speed)));
  const offroadMargin = (Math.abs(nearest.signedOffset) - roadW * 0.5) / Math.max(1, roadW * 0.5);
  const future = [0.01, 0.025, 0.05, 0.085, 0.12].map((delta) => {
    const frame = sampleTrackFrame(nearest.t + delta);
    const futureHeading = Math.atan2(frame.tangentX, frame.tangentZ);
    return angleDiff(futureHeading, headingTrack);
  });
  return {
    obs: [
      speed / 180,
      lateralSpeed / 30,
      ai.yawRate / 2.5,
      ai.steer,
      ai.throttleCmd || 0,
      ai.brakePedal,
      nearest.signedOffset / Math.max(1, roadW * 0.5),
      headingErr / Math.PI,
      slipAngle / 1.2,
      offroadMargin,
      ...future,
      Math.sin(ai.heading),
      Math.cos(ai.heading),
      nearest.t,
      progress - Math.floor(progress),
      clamp((performance.now() - state.lapStart) / 120000, 0, 1),
      Math.abs(nearest.signedOffset) > (roadW * 0.5) ? 1 : 0,
    ].map((value) => clamp(value, -5, 5)),
    speed,
    headingErr,
    slipAngle,
    offroadMargin,
  };
}

function beginAITelemetrySession() {
  aiTelemetry.current = {
    version: aiTelemetry.telemetryVersion,
    startedAt: Date.now(),
    laps: [],
    drivers: rivals.map((ai) => ({
      name: ai.name,
      style: { ...ai.style },
      stepSamples: [],
      segments: Array.from({ length: checkpoints.length || 8 }, (_, idx) => ({
        idx,
        reward: 0,
        samples: 0,
        speedSum: 0,
        throttleSum: 0,
        brakeSum: 0,
        driftSum: 0,
        stabilityLoss: 0,
        progressDelta: 0,
        offroad: 0,
        contact: 0,
      })),
    })),
  };
  aiTelemetry.lastSampleAt = 0;
}

function saveAITelemetrySnapshot(reason = 'manual') {
  if (!aiTelemetry.current) return null;
  const payload = {
    version: aiTelemetry.telemetryVersion,
    reason,
    savedAt: new Date().toISOString(),
    current: aiTelemetry.current,
    history: aiTelemetry.episodes,
  };
  try {
    localStorage.setItem('racingAIDebugTelemetry', JSON.stringify(payload));
  } catch (err) {
    console.warn('telemetry save failed', err);
  }
  return payload;
}

function finalizeAITelemetrySession(reason = 'finished') {
  if (!aiTelemetry.current) return;
  const snapshot = saveAITelemetrySnapshot(reason);
  aiTelemetry.episodes.unshift(snapshot.current);
  aiTelemetry.episodes = aiTelemetry.episodes.slice(0, aiTelemetry.maxEpisodes);
  saveAITelemetrySnapshot(`${reason}-history`);
}

function sampleAIRLStep(now) {
  if (!raceStarted || !aiTelemetry.current || (now - aiTelemetry.lastSampleAt) < aiTelemetry.sampleEveryMs) return;
  aiTelemetry.lastSampleAt = now;
  rivals.forEach((ai, idx) => {
    const entry = aiTelemetry.current.drivers[idx];
    if (!entry) return;
    const segIdx = ai.nextCp % entry.segments.length;
    const seg = entry.segments[segIdx];
    const speed = Math.hypot(ai.vx, ai.vz);
    const offroad = roadDistanceSq(ai.x, ai.z, ai.t) > (roadW * roadW * 0.32);
    const progress = lapProgress(ai.lap, ai.nextCp, ai.t);
    const progressDelta = Math.max(0, progress - (ai.lastProgress || 0));
    ai.lastProgress = progress;
    const reward = (
      speed * 0.028
      + progressDelta * 26
      - Math.abs(ai.trackError) * 0.12
      - Math.abs(ai.steer) * 0.58
      - ai.brakePedal * 1.4
      - (1 - ai.stability) * 1.8
      - (offroad ? 3.2 : 0)
      - ai.driftAmount * 0.38
    );
    seg.reward += reward;
    seg.samples += 1;
    seg.speedSum += speed;
    seg.throttleSum += ai.throttleCmd || 0;
    seg.brakeSum += ai.brakePedal;
    seg.driftSum += ai.driftAmount;
    seg.stabilityLoss += Math.max(0, 1 - ai.stability);
    seg.progressDelta += progressDelta;
    if (offroad) seg.offroad += 1;

    const { obs, headingErr, slipAngle, offroadMargin } = buildTelemetryObservation(ai, progress);
    entry.stepSamples.push({
      tMs: Math.round(now),
      lap: ai.lap,
      segIdx,
      obs,
      action: [clamp(ai.steer / 0.62, -1, 1), clamp(ai.throttleCmd || 0, 0, 1), clamp(ai.brakePedal, 0, 1), clamp(ai.driftAmount, 0, 1)],
      reward,
      progress,
      progressDelta,
      speed,
      headingErr,
      slipAngle,
      stability: ai.stability,
      offroadMargin,
      offroad: offroad ? 1 : 0,
      contact: ai.lastContactTelemetry,
    });
    if (entry.stepSamples.length > aiTelemetry.maxStepSamplesPerDriver) entry.stepSamples.shift();
  });
}

window.dumpRacingAITelemetry = () => saveAITelemetrySnapshot('window-dump');
window.getRacingAITelemetry = () => ({ ...aiTelemetry, current: aiTelemetry.current, historySize: aiTelemetry.episodes.length });
window.downloadRacingAITelemetry = () => {
  const payload = saveAITelemetrySnapshot('window-download');
  if (!payload) return null;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `racing-ai-telemetry-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  return payload;
};

function lapProgress(lap, nextCp, t) {
  const cpFrac = clamp(nextCp / checkpoints.length, 0, 1);
  const mix = cpFrac * 0.65 + t * 0.35;
  return (lap - 1) + mix;
}

function updateRaceOrder() {
  const entries = [{ name: 'YOU', progress: lapProgress(state.lap, state.nextCp, playerTrackT), isPlayer: true }];
  rivals.forEach((ai) => {
    entries.push({
      name: ai.name,
      progress: lapProgress(ai.lap, ai.nextCp, ai.t),
      isPlayer: false,
    });
  });

  entries.sort((a, b) => b.progress - a.progress);
  raceOrder = entries;

  const playerPos = entries.findIndex((e) => e.isPlayer) + 1;
  if (positionText) positionText.textContent = `P${playerPos}/${entries.length}`;
  if (leaderboardText) {
    leaderboardText.textContent = entries.map((e, i) => `${i + 1}. ${e.name}`).join('\n');
  }
}

function showFinishPanel() {
  const playerPos = raceOrder.findIndex((e) => e.isPlayer) + 1;
  const best = state.bestLap ? fmt(state.bestLap) : '--';
  if (finishSummary) {
    finishSummary.textContent = `결과: P${playerPos}/${raceOrder.length}\nBEST: ${best}\nPIT STOPS: ${state.pitStopsUsed}\n\n${raceOrder.map((e, i) => `${i + 1}. ${e.name}`).join('\n')}`;
  }
  if (finishPanel) finishPanel.style.display = 'grid';
}

// sound (engine-like layered synth)
let ac, oscLow, oscHigh, gainLow, gainHigh, masterGain, filter;
async function setupAudio() {
  if (ac) {
    if (ac.state === 'suspended') await ac.resume();
    return;
  }
  ac = new AudioContext();

  oscLow = ac.createOscillator();
  oscHigh = ac.createOscillator();
  gainLow = ac.createGain();
  gainHigh = ac.createGain();
  masterGain = ac.createGain();
  filter = ac.createBiquadFilter();

  oscLow.type = 'sawtooth';
  oscHigh.type = 'square';

  oscLow.frequency.value = 55;
  oscHigh.frequency.value = 110;

  gainLow.gain.value = 0.0001;
  gainHigh.gain.value = 0.0001;
  masterGain.gain.value = 0.0001;

  filter.type = 'lowpass';
  filter.frequency.value = 900;
  filter.Q.value = 1.2;

  oscLow.connect(gainLow).connect(filter);
  oscHigh.connect(gainHigh).connect(filter);
  filter.connect(masterGain).connect(ac.destination);

  oscLow.start();
  oscHigh.start();
}

let last = performance.now();
function tick(now) {
  const dt = Math.min(0.033, (now - last) / 1000);
  last = now;

  const throttle = raceStarted && (keys.has('arrowup') || keys.has('w')) ? 1 : 0;
  const brakeTarget = raceStarted && (keys.has('arrowdown') || keys.has('s') || keys.has(' ')) ? 1 : 0;
  // gradual brake pedal response
  state.brakePedal += (brakeTarget - state.brakePedal) * Math.min(1, dt * (brakeTarget ? 4.5 : 3.2));
  const brake = state.brakePedal;

  // invert fixed: left key should steer left on screen
  const steerIn = raceStarted
    ? ((keys.has('arrowleft') || keys.has('a') ? 1 : 0) + (keys.has('arrowright') || keys.has('d') ? -1 : 0))
    : 0;

  const fwd = new THREE.Vector2(Math.sin(state.heading), Math.cos(state.heading));
  const right = new THREE.Vector2(fwd.y, -fwd.x);
  let vForward = state.vx * fwd.x + state.vz * fwd.y;
  let vLateral = state.vx * right.x + state.vz * right.y;

  const speed = Math.hypot(state.vx, state.vz);
  const distSq = roadDistanceSq(state.x, state.z);
  const onRoad = distSq < (roadW * 0.9) ** 2;

  let slipstreamTarget = null;
  let nearestAhead = Infinity;
  rivals.forEach((ai) => {
    const dx = ai.x - state.x;
    const dz = ai.z - state.z;
    const along = dx * fwd.x + dz * fwd.y;
    const side = Math.abs(dx * right.x + dz * right.y);
    if (along > 8 && along < 42 && side < 6 && along < nearestAhead) {
      nearestAhead = along;
      slipstreamTarget = ai;
    }
  });

  const targetSlipBoost = slipstreamTarget ? clamp((42 - nearestAhead) / 42, 0.12, 1) : 0;
  state.slipstreamBoost += (targetSlipBoost - state.slipstreamBoost) * Math.min(1, dt * 3.2);

  const wheelHalf = 2.4;
  const leftWheelDistSq = roadDistanceSq(state.x + right.x * wheelHalf, state.z + right.y * wheelHalf);
  const rightWheelDistSq = roadDistanceSq(state.x - right.x * wheelHalf, state.z - right.y * wheelHalf);
  const bothWheelsOffRoad = leftWheelDistSq > (roadW * 0.92) ** 2 && rightWheelDistSq > (roadW * 0.92) ** 2;
  if (bothWheelsOffRoad && !state.offroadAllWheels) {
    state.lapPenaltyMs += 2000;
  }
  state.offroadAllWheels = bothWheelsOffRoad;

  const engineFactor = 1 - clamp(state.damageEngine / 170, 0, 0.5);
  const tireFactor = 1 - clamp(state.damageTire / 130, 0, 0.55);
  const aeroFactor = 1 - clamp(state.damageAero / 180, 0, 0.35);
  const wearFactor = 1 - clamp(state.tireWear / 180, 0, 0.45);
  const damageFactor = (engineFactor * 0.5) + (tireFactor * 0.3) + (aeroFactor * 0.2);

  const grip = onRoad
    ? (2.4 * currentCarStats.handling * tireFactor * wearFactor)
    : (1.0 * currentCarStats.handling * 0.72 * tireFactor * wearFactor);

  const steerLimit = 0.62 * currentCarStats.handling * tireFactor * wearFactor * (0.35 + 0.65 * Math.max(0, 1 - speed / 120));
  const steerTarget = steerIn * steerLimit;
  state.steer += (steerTarget - state.steer) * Math.min(1, dt * 7.5);

  let aLong = 0;
  const reverseIntent = brake > 0.2 && throttle === 0 && Math.abs(vForward) < 9;

  if (throttle) {
    aLong += currentCarStats.accel * damageFactor + 20 * state.slipstreamBoost;
  } else if (reverseIntent) {
    // hold brake at low speed -> engage reverse
    aLong -= (currentCarStats.accel * 0.62) * brake;
  }

  if (brake && !reverseIntent) {
    // normal braking while rolling
    aLong -= currentCarStats.brake * brake * Math.sign(vForward || 1);
  }

  aLong -= 1.05 * vForward;
  aLong -= 0.0048 * vForward * Math.abs(vForward);
  if (!onRoad) aLong -= 45 * Math.sign(vForward || 0);

  vForward += aLong * dt;
  const reverseMax = -136;
  const topSpeedWithEffects = (currentCarStats.topSpeed * damageFactor + 16 * state.slipstreamBoost) / 1.18;
  vForward = clamp(vForward, reverseMax, topSpeedWithEffects);
  vLateral *= Math.exp(-grip * dt);

  const targetYaw = Math.abs(vForward) > 0.5 ? (vForward / 3.4) * Math.tan(state.steer) : 0;
  state.yawRate += (targetYaw - state.yawRate) * Math.min(1, dt * 5.2);
  state.yawRate *= Math.exp(-1.5 * dt);
  state.heading += state.yawRate * dt;

  state.vx = fwd.x * vForward + right.x * vLateral;
  state.vz = fwd.y * vForward + right.y * vLateral;

  const prevX = state.x;
  const prevZ = state.z;
  state.x += state.vx * dt;
  state.z += state.vz * dt;

  rivals.forEach((ai, i) => {
    ai.lastContactTelemetry = 0;
    const aiFwd = new THREE.Vector2(Math.sin(ai.heading), Math.cos(ai.heading));
    const aiRight = new THREE.Vector2(aiFwd.y, -aiFwd.x);
    let aiVForward = ai.vx * aiFwd.x + ai.vz * aiFwd.y;
    let aiVLateral = ai.vx * aiRight.x + ai.vz * aiRight.y;

    const speedNorm = clamp(Math.abs(aiVForward) / 180, 0, 1);
    const lookBase = 0.012 + speedNorm * 0.028;
    const frameNear = sampleTrackFrame(ai.t);
    const frameMid = sampleTrackFrame((ai.t + lookBase) % 1);
    const frameFar = sampleTrackFrame((ai.t + lookBase * 2.1) % 1);
    const headingNear = Math.atan2(frameNear.tangentX, frameNear.tangentZ);
    const headingMid = Math.atan2(frameMid.tangentX, frameMid.tangentZ);
    const headingFar = Math.atan2(frameFar.tangentX, frameFar.tangentZ);
    const cornerNow = Math.abs(angleDiff(headingMid, headingNear));
    const cornerAhead = Math.abs(angleDiff(headingFar, headingNear));
    const cornerNeed = Math.max(cornerNow * 0.7 + cornerAhead * 1.15, 0);
    const cornerSign = Math.sign(frameNear.tangentX * frameFar.tangentZ - frameNear.tangentZ * frameFar.tangentX) || 1;
    const segmentIdx = ai.nextCp % checkpoints.length;
    const learn = ai.learn[segmentIdx] || { attack: 1, samples: 0, mistakes: 0 };
    const attack = clamp(learn.attack * ai.style.bravery, 0.82, 1.18);

    const nearest = nearestTrackSample(ai.x, ai.z, 180, ai.t, 0.14);
    const signedTrackErr = nearest.signedOffset;
    ai.trackError = signedTrackErr;

    let blockAhead = false;
    let avoidBias = 0;
    let crowding = 0;
    let closingSpeed = 0;
    const checkTraffic = (x, z, vx = 0, vz = 0, weight = 1) => {
      const dx = x - ai.x;
      const dz = z - ai.z;
      const along = dx * aiFwd.x + dz * aiFwd.y;
      const side = dx * aiRight.x + dz * aiRight.y;
      const nearAhead = along > 1 && along < lerp(20, 34, speedNorm);
      const sideWindow = lerp(6.2, 8.6, Math.min(1, Math.abs(along) / 34));
      if (nearAhead && Math.abs(side) < sideWindow) {
        blockAhead = true;
        const sideDir = Math.abs(side) > 0.35 ? -Math.sign(side) : (signedTrackErr >= 0 ? -1 : 1);
        const urgency = (1 - along / Math.max(1, lerp(20, 34, speedNorm))) * weight;
        avoidBias += sideDir * urgency;
        closingSpeed = Math.max(closingSpeed, (ai.vx - vx) * aiFwd.x + (ai.vz - vz) * aiFwd.y);
      }
      const dist = Math.hypot(dx, dz) || 1;
      if (dist < 18) {
        crowding += (1 - dist / 18) * weight;
        avoidBias += (-side / dist) * 0.55 * weight;
      }
    };
    checkTraffic(state.x, state.z, state.vx, state.vz, 1.2);
    rivals.forEach((other) => {
      if (other === ai) return;
      checkTraffic(other.x, other.z, other.vx, other.vz, 1);
    });
    avoidBias = clamp(avoidBias, -1.25, 1.25);

    const apexOffset = clamp(cornerNeed * roadW * 0.62, 0, roadW * 0.28) * -cornerSign;
    const exitOffset = clamp(cornerNeed * roadW * 0.5, 0, roadW * 0.24) * cornerSign;
    const racingLineOffset = lerp(apexOffset, exitOffset, 0.38 + speedNorm * 0.28);
    const jitterScale = (1 - clamp(cornerNeed * 1.1 + crowding * 0.9, 0, 0.82));
    const jitter = Math.sin(now * 0.00018 * (0.85 + ai.style.jitter) + i * 1.17) * (roadW * 0.025 * ai.style.jitter * jitterScale);
    const avoidOffset = blockAhead ? avoidBias * (roadW * 0.22) : 0;
    const recoveryNeed = clamp((Math.abs(signedTrackErr) / (roadW * 0.42)) + Math.abs(aiVLateral) / 18 + Math.abs(ai.yawRate) / 3.4 - 0.8, 0, 1.2);
    ai.recovery += (recoveryNeed - ai.recovery) * Math.min(1, dt * (recoveryNeed > ai.recovery ? 3.6 : 1.8));
    ai.laneTarget = clamp(racingLineOffset + avoidOffset + jitter - signedTrackErr * ai.recovery * 0.42, -(roadW * 0.3), roadW * 0.3);
    ai.laneOffset += (ai.laneTarget - ai.laneOffset) * Math.min(1, dt * (blockAhead ? 4.8 : 2.7));

    const targetX = frameMid.p.x + frameMid.rightX * ai.laneOffset;
    const targetZ = frameMid.p.z + frameMid.rightZ * ai.laneOffset;
    const toTX = targetX - ai.x;
    const toTZ = targetZ - ai.z;
    const toTL = Math.hypot(toTX, toTZ) || 1;
    const desiredX = toTX / toTL;
    const desiredZ = toTZ / toTL;

    const cross = aiFwd.x * desiredZ - aiFwd.y * desiredX;
    const dot = clamp(aiFwd.x * desiredX + aiFwd.y * desiredZ, -1, 1);
    const headingErr = Math.atan2(cross, dot);

    const linePenalty = clamp(Math.abs(signedTrackErr) / (roadW * 0.36), 0, 1.1);
    const trafficPenalty = blockAhead ? (0.12 + crowding * 0.26 + clamp(closingSpeed / 42, 0, 0.4)) : crowding * 0.16;
    const risk = clamp(cornerNeed * 1.05 + linePenalty * 0.6 + trafficPenalty + ai.recovery * 0.75, 0, 2.4);
    const baseCornerSpeed = 194 - cornerNeed * 126 - linePenalty * 16 - trafficPenalty * 22;
    const targetSpeed = clamp(baseCornerSpeed * attack * ai.style.corner * (1 - ai.recovery * 0.18), 66, currentCarStats.topSpeed / 1.09);
    const speedError = aiVForward - targetSpeed;
    const driftWindow = cornerNeed > 0.24 && aiVForward > targetSpeed * (0.94 - ai.recovery * 0.08) && ai.recovery < 0.58;
    const desiredDrift = driftWindow
      ? clamp((cornerNeed - 0.22) * 1.8 + Math.max(0, speedError) / 92 - crowding * 0.55, 0, 1) * ai.style.drift * (1 - ai.recovery * 0.72)
      : 0;
    ai.driftAmount += (desiredDrift - ai.driftAmount) * Math.min(1, dt * (desiredDrift > ai.driftAmount ? 3.3 : 2.8));

    const throttleTarget = blockAhead
      ? clamp((targetSpeed - 12 - closingSpeed * 0.3) / Math.max(1, currentCarStats.topSpeed), 0.18, 0.76)
      : clamp((targetSpeed - Math.max(0, speedError) * 0.32) / Math.max(1, currentCarStats.topSpeed * 0.96), 0.18, 1);
    ai.throttleCmd = clamp(throttleTarget * (1 - ai.recovery * 0.38), 0.12, 1);
    const brakeTargetAI = speedError > 5
      ? clamp(speedError / 50 + risk * 0.22 + clamp(closingSpeed / 35, 0, 0.32), 0, 1)
      : clamp((risk - 0.55) * 0.55, 0, 0.42);

    ai.brakePedal += (brakeTargetAI - ai.brakePedal) * Math.min(1, dt * (brakeTargetAI > ai.brakePedal ? 5.8 : 3.4));
    const steerAssist = ai.driftAmount * cornerSign * 0.2;
    const steerGain = 1.42 + cornerNeed * 0.48 + ai.recovery * 0.24;
    const steerDamping = clamp(aiVLateral / 28, -0.28, 0.28);
    const steerTargetAI = clamp(-headingErr * steerGain - steerDamping - signedTrackErr / (roadW * 0.9) + steerAssist, -0.62, 0.62);
    ai.steer += (steerTargetAI - ai.steer) * Math.min(1, dt * (5.1 + ai.recovery * 0.8));

    const aiGrip = (2.95 - ai.driftAmount * 1.3 - ai.recovery * 0.18) * currentCarStats.handling;
    let aiALong = currentCarStats.accel * ai.throttleCmd;
    aiALong -= currentCarStats.brake * ai.brakePedal * Math.sign(aiVForward || 1);
    aiALong -= 0.98 * aiVForward;
    aiALong -= 0.0043 * aiVForward * Math.abs(aiVForward);
    if (Math.abs(signedTrackErr) > roadW * 0.52) aiALong -= 16 * Math.sign(aiVForward || 1);

    const desiredLat = clamp(-headingErr * Math.abs(aiVForward) * 0.24, -24, 24) * (0.58 + ai.driftAmount * 0.82);
    aiVLateral += (desiredLat - aiVLateral) * Math.min(1, dt * (2.1 + ai.driftAmount * 0.9 + ai.recovery * 0.6));
    aiVForward += aiALong * dt;
    aiVForward = clamp(aiVForward, 0, currentCarStats.topSpeed / 1.12);
    aiVLateral *= Math.exp(-aiGrip * dt);

    const yawLead = ai.driftAmount * cornerSign * clamp(Math.abs(aiVLateral) / 28, 0, 0.26);
    const aiTargetYaw = aiVForward > 0.5 ? (aiVForward / 3.55) * Math.tan(ai.steer + yawLead) : 0;
    ai.yawRate += (aiTargetYaw - ai.yawRate) * Math.min(1, dt * (4.8 + ai.driftAmount * 0.55));
    ai.yawRate *= Math.exp(-(2.05 - ai.driftAmount * 0.35 - ai.recovery * 0.1) * dt);
    ai.heading += ai.yawRate * dt;

    const aiFwd2 = new THREE.Vector2(Math.sin(ai.heading), Math.cos(ai.heading));
    const aiRight2 = new THREE.Vector2(aiFwd2.y, -aiFwd2.x);
    ai.vx = aiFwd2.x * aiVForward + aiRight2.x * aiVLateral;
    ai.vz = aiFwd2.y * aiVForward + aiRight2.y * aiVLateral;

    ai.x += ai.vx * dt;
    ai.z += ai.vz * dt;
    const nearestAfter = nearestTrackSample(ai.x, ai.z, 220, ai.t, 0.1);
    ai.t = nearestAfter.t;
    ai.trackError = nearestAfter.signedOffset;

    const roadError = Math.abs(nearestAfter.signedOffset) / Math.max(1, roadW * 0.5);
    const instability = Math.abs(aiVLateral) / 32 + Math.abs(headingErr) / 1.1 + roadError + ai.recovery * 0.2;
    ai.stability = clamp(1 - instability * 0.42, 0, 1);
    if (instability > 1.28) {
      learn.attack = clamp(learn.attack - dt * 0.28, 0.82, 1.18);
      learn.mistakes += dt;
    } else if (!blockAhead && speedError < 2 && cornerNeed < 0.7) {
      learn.attack = clamp(learn.attack + dt * 0.05, 0.82, 1.18);
    }
    learn.samples += dt;

    const aiCp = checkpoints[ai.nextCp];
    if (new THREE.Vector2(ai.x, ai.z).distanceTo(new THREE.Vector2(aiCp.x, aiCp.z)) < 30) {
      ai.lastLearnIdx = segmentIdx;
      ai.nextCp += 1;
      if (ai.nextCp >= checkpoints.length) {
        ai.nextCp = 0;
        ai.cpCycleReady = true;
      }
    }

    if (ai.cpCycleReady && ai.t < 0.03) {
      ai.lap += 1;
      ai.cpCycleReady = false;
      ai.learn.forEach((seg) => {
        if (seg.samples > 0.8) {
          const settle = clamp(seg.mistakes / seg.samples, 0, 1.4);
          seg.attack = clamp(seg.attack + (0.035 - settle * 0.06), 0.82, 1.18);
          seg.samples = 0;
          seg.mistakes = 0;
        }
      });
    }

    const dxPR = state.x - ai.x;
    const dzPR = state.z - ai.z;
    const distPR = Math.hypot(dxPR, dzPR) || 0.0001;
    const minDist = 12.6;
    if (distPR < minDist) {
      const nx = dxPR / distPR;
      const nz = dzPR / distPR;
      const push = minDist - distPR;

      state.x += nx * push * 0.52;
      state.z += nz * push * 0.52;
      ai.x -= nx * push * 0.48;
      ai.z -= nz * push * 0.48;

      const playerV = new THREE.Vector2(state.vx, state.vz);
      const rivalV = new THREE.Vector2(ai.vx, ai.vz);
      const relN = (playerV.x - rivalV.x) * nx + (playerV.y - rivalV.y) * nz;

      if (relN < 0) {
        const j = -relN * 0.52;
        state.vx += nx * j;
        state.vz += nz * j;
        ai.vx -= nx * j * 0.78;
        ai.vz -= nz * j * 0.78;
        ai.recovery = clamp(ai.recovery + 0.18, 0, 1.2);
        learn.attack = clamp(learn.attack - 0.02, 0.82, 1.18);
        const telemetryDriver = aiTelemetry.current?.drivers?.[i];
        const telemetrySeg = telemetryDriver?.segments?.[ai.nextCp % telemetryDriver.segments.length];
        if (telemetrySeg) telemetrySeg.contact += 1;
        ai.lastContactTelemetry = 1;
        state.yawRate += (Math.random() - 0.5) * 0.18;
        const impact = Math.abs(relN) * 0.2;
        state.damageEngine = clamp(state.damageEngine + impact * 0.45, 0, 100);
        state.damageTire = clamp(state.damageTire + impact * 0.75, 0, 100);
        state.damageAero = clamp(state.damageAero + impact * 0.6, 0, 100);
      }
    }
  });

  if (!onRoad && Math.abs(vForward) > 40) {
    state.damageEngine = clamp(state.damageEngine + dt * 0.8, 0, 100);
    state.damageTire = clamp(state.damageTire + dt * 1.8, 0, 100);
    state.damageAero = clamp(state.damageAero + dt * 1.2, 0, 100);
  }

  const pitDist = Math.hypot(state.x - pitZoneCenter.x, state.z - pitZoneCenter.z);
  const inPitZone = pitDist < 15;
  const pitActive = inPitZone && Math.abs(vForward) < 9 && state.canUsePitThisLap;
  if (pitActive) {
    state.repairTimer += dt;
    state.damageEngine = clamp(state.damageEngine - dt * 11, 0, 100);
    state.damageTire = clamp(state.damageTire - dt * 16, 0, 100);
    state.damageAero = clamp(state.damageAero - dt * 13, 0, 100);
    state.tireWear = clamp(state.tireWear - dt * 24, 0, 100);
    if (state.repairTimer > 1.8) {
      state.canUsePitThisLap = false;
      state.pitStopsUsed += 1;
      state.repairTimer = 0;
    }
  } else {
    state.repairTimer = Math.max(0, state.repairTimer - dt * 0.8);
  }

  // skid mark trigger (hard brake / lateral slip)
  state.skidCd = Math.max(0, state.skidCd - dt);
  const slip = Math.abs(vLateral);
  const skidStrength = (brake * Math.max(0, Math.abs(vForward) - 12) * 0.028) + (slip * 0.095);
  const cornerLoad = Math.abs(state.steer) * Math.abs(vForward) * 0.0014;
  state.tireWear = clamp(state.tireWear + dt * (cornerLoad + skidStrength * 0.18 + (onRoad ? 0.015 : 0.07)), 0, 100);

  if (onRoad && skidStrength > 0.28 && state.skidCd <= 0) {
    const hw = 1.6;
    const wx1 = state.x + right.x * hw;
    const wz1 = state.z + right.y * hw;
    const wx2 = state.x - right.x * hw;
    const wz2 = state.z - right.y * hw;
    const alpha = Math.min(0.34, 0.14 + skidStrength * 0.08);
    spawnSkidMark(wx1, wz1, state.heading, alpha);
    spawnSkidMark(wx2, wz2, state.heading, alpha);
    state.skidCd = 0.04;
  }

  // rear smoke on throttle and stronger when sliding/braking
  const rearOffset = 6.2;
  const sx = state.x - fwd.x * rearOffset;
  const sz = state.z - fwd.y * rearOffset;
  const smokeIntensity = clamp(throttle * 0.55 + brake * 0.35 + skidStrength * 0.5, 0, 1.2);
  if (smokeIntensity > 0.08 && Math.random() < 0.65) {
    spawnSmoke(sx, sz, smokeIntensity);
  }

  // checkpoints: must pass all in order before lap counts
  const cp = checkpoints[state.nextCp];
  if (new THREE.Vector2(state.x, state.z).distanceTo(new THREE.Vector2(cp.x, cp.z)) < 32) {
    state.nextCp += 1;
    if (state.nextCp >= checkpoints.length) {
      state.nextCp = 0;
      state.cpCycleReady = true;
    }
  }

  // lap timing by start line crossing
  const gateFwd = new THREE.Vector2(startDir.x, startDir.z).normalize();
  const gateRight = new THREE.Vector2(gateFwd.y, -gateFwd.x);
  const gateDelta = new THREE.Vector2(state.x - startP.x, state.z - startP.z);
  const gateDist = gateDelta.dot(gateFwd);
  const lateral = Math.abs(gateDelta.dot(gateRight));

  if (Math.abs(gateDist) > 14) state.startGateArmed = true;

  const crossedForward = state.startGateArmed
    && state.cpCycleReady
    && state.prevGateDist < 0
    && gateDist >= 0
    && lateral < (roadW * 0.62)
    && vForward > 12
    && (now - state.lapStart) > 2500;

  if (crossedForward) {
    const lapMs = now - state.lapStart + state.lapPenaltyMs;
    state.bestLap = state.bestLap ? Math.min(state.bestLap, lapMs) : lapMs;
    aiTelemetry.current?.laps?.push({
      lap: state.lap,
      lapMs,
      damage: state.damage,
      tireWear: state.tireWear,
      penaltyMs: state.lapPenaltyMs,
    });
    state.lapStart = now;
    state.lapPenaltyMs = 0;
    state.offroadAllWheels = false;
    state.cpCycleReady = false;
    state.lap++;
    state.canUsePitThisLap = true;
    if (state.lap > 7 && !raceFinished) {
      raceFinished = true;
      raceStarted = false;
      keys.clear();
      finalizeAITelemetrySession('race-finished');
      updateRaceOrder();
      showFinishPanel();
    }
  }

  state.damage = clamp((state.damageEngine * 0.4) + (state.damageTire * 0.35) + (state.damageAero * 0.25), 0, 100);
  state.prevGateDist = gateDist;

  // place meshes
  carRoot.position.set(state.x, 0, state.z);
  carRoot.rotation.y = state.heading + currentCarYawOffset;
  rivals.forEach((ai) => {
    ai.root.position.set(ai.x, 0, ai.z);
    ai.root.rotation.y = ai.heading + ai.yawOffset;
  });
  playerTrackT = trackTAt(state.x, state.z);
  sampleAIRLStep(now);
  updateRaceOrder();
  updateSkidMarks(dt);
  updateSmoke(dt);

  // slight-tilt fixed camera (~10deg-ish) while following car
  const camH = 120;
  const camBack = 70;
  const camSide = 0;
  camera.position.lerp(new THREE.Vector3(state.x + camSide, camH, state.z + camBack), 1 - Math.exp(-dt * 8));
  camera.lookAt(state.x, 0, state.z);

  // gearbox + rpm model (with shift drop)
  const kmh = Math.max(0, vForward * 2.8);
  const upAt = [0, 24, 47, 72, 100];
  const downAt = [0, 14, 32, 52, 78];

  state.shiftLock = Math.max(0, state.shiftLock - dt);
  if (state.shiftLock <= 0) {
    if (state.gear < 5 && kmh > upAt[state.gear]) {
      state.gear += 1;
      state.rpm *= 0.62; // drop on upshift
      state.shiftLock = 0.18;
    } else if (state.gear > 1 && kmh < downAt[state.gear - 1]) {
      state.gear -= 1;
      state.rpm = Math.max(state.rpm, 0.52); // jump on downshift
      state.shiftLock = 0.14;
    }
  }

  const gearRatios = [0, 2.9, 2.1, 1.55, 1.2, 0.98];
  const normSpeed = clamp(kmh / (205 / gearRatios[state.gear]), 0, 1.2);
  const targetRpm = clamp(0.22 + normSpeed * 0.68 + throttle * 0.18 - brake * 0.08, 0.08, 1);
  state.rpm += (targetRpm - state.rpm) * Math.min(1, dt * 8.5);

  speedText.textContent = `${kmh.toFixed(0)} km/h`;
  gearText.textContent = state.gear;
  lapText.textContent = `${Math.min(7, state.lap)}/7`;
  bestText.textContent = state.bestLap ? fmt(state.bestLap) : '--';
  nowText.textContent = fmt((now - state.lapStart) + state.lapPenaltyMs);
  penaltyText.textContent = `+${(state.lapPenaltyMs / 1000).toFixed(3)}s`;
  if (damageText) damageText.textContent = `${state.damage.toFixed(0)}%${state.repairTimer > 0.2 ? ' (REPAIR)' : ''}`;
  if (partsText) partsText.textContent = `ENG ${state.damageEngine.toFixed(0)} / TIR ${state.damageTire.toFixed(0)} / AERO ${state.damageAero.toFixed(0)}`;
  if (tyreText) tyreText.textContent = `${state.tireWear.toFixed(0)}%`;
  if (slipText) slipText.textContent = state.slipstreamBoost > 0.1 ? `ON +${(state.slipstreamBoost * 100).toFixed(0)}%` : 'OFF';
  if (pitText) pitText.textContent = state.canUsePitThisLap ? `READY (${state.pitStopsUsed})` : `USED (${state.pitStopsUsed})`;
  throttleBar.style.width = `${throttle * 100}%`;
  brakeBar.style.width = `${brake * 100}%`;
  rpmBar.style.width = `${state.rpm * 100}%`;

  // side minimap + speed gauge
  drawMiniMap();
  const speedNorm = clamp(kmh / Math.max(140, currentCarStats.topSpeed), 0, 1);
  const startDeg = 210;
  const endDeg = startDeg + speedNorm * 300;
  gArc.setAttribute('d', describeArc(110, 110, 86, startDeg, endDeg));

  // keep needle centered and within dial sweep
  const needleDeg = -150 + speedNorm * 300;
  gNeedle.setAttribute('transform', `rotate(${needleDeg} 110 110)`);
  gSpeedText.textContent = `${kmh.toFixed(0)}`;

  // engine sound (mute at idle, stronger under load)
  if (ac) {
    const absFwd = Math.abs(vForward);
    const moving = absFwd > 2;
    const load = clamp(0.35 + throttle * 0.85 - brake * 0.18, 0, 1.25);

    const base = 32 + state.rpm * 86 + load * 8;
    const high = 72 + state.rpm * 190 + load * 16;

    const targetMaster = moving || throttle > 0
      ? (0.04 + state.rpm * 0.14 + throttle * 0.06)
      : 0.0001;

    oscLow.frequency.setTargetAtTime(base, ac.currentTime, 0.025);
    oscHigh.frequency.setTargetAtTime(high, ac.currentTime, 0.025);

    gainLow.gain.setTargetAtTime(0.045 * load, ac.currentTime, 0.04);
    gainHigh.gain.setTargetAtTime(0.022 * load, ac.currentTime, 0.04);
    filter.frequency.setTargetAtTime(650 + state.rpm * 1700, ac.currentTime, 0.05);
    masterGain.gain.setTargetAtTime(targetMaster, ac.currentTime, 0.06);
  }

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

addEventListener('resize', () => {
  const aspect = innerWidth / innerHeight;
  camera.left = -frustum * aspect;
  camera.right = frustum * aspect;
  camera.top = frustum;
  camera.bottom = -frustum;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
