import * as THREE from 'https://esm.sh/three@0.164.1';
import { GLTFLoader } from 'https://esm.sh/three@0.164.1/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'https://esm.sh/three@0.164.1/examples/jsm/loaders/DRACOLoader.js';

const speedText = document.getElementById('speedText');
const gearText = document.getElementById('gearText');
const lapText = document.getElementById('lapText');
const bestText = document.getElementById('bestText');
const nowText = document.getElementById('nowText');
const throttleBar = document.getElementById('throttleBar');
const brakeBar = document.getElementById('brakeBar');
const rpmBar = document.getElementById('rpmBar');
const minimap = document.getElementById('minimap');
const mm = minimap.getContext('2d');
const gArc = document.getElementById('gArc');
const gNeedle = document.getElementById('gNeedle');
const gSpeedText = document.getElementById('gSpeedText');

const steerPad = document.getElementById('steerPad');
const steerThumb = document.getElementById('steerThumb');
const btnThrottle = document.getElementById('btnThrottle');
const btnBrake = document.getElementById('btnBrake');

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

// Track path
// Reference-like winding course (top-right hairpin + lower loop)
const pts = [
  new THREE.Vector3(-320, 0, -40),
  new THREE.Vector3(-250, 0, -190),
  new THREE.Vector3(-40, 0, -250),
  new THREE.Vector3(180, 0, -200),
  new THREE.Vector3(315, 0, -70),
  new THREE.Vector3(320, 0, 70),
  new THREE.Vector3(210, 0, 145),
  new THREE.Vector3(45, 0, 155),
  new THREE.Vector3(-60, 0, 105),
  new THREE.Vector3(10, 0, 20),
  new THREE.Vector3(180, 0, 40),
  new THREE.Vector3(260, 0, 190),
  new THREE.Vector3(120, 0, 300),
  new THREE.Vector3(-90, 0, 315),
  new THREE.Vector3(-280, 0, 260),
  new THREE.Vector3(-355, 0, 120),
];
const curve = new THREE.CatmullRomCurve3(pts, true, 'catmullrom', 0.2);

const mapBounds = pts.reduce((acc, p) => ({
  minX: Math.min(acc.minX, p.x),
  maxX: Math.max(acc.maxX, p.x),
  minZ: Math.min(acc.minZ, p.z),
  maxZ: Math.max(acc.maxZ, p.z),
}), { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity });

// Continuous road strips (no segment gaps)
const roadMat = new THREE.MeshStandardMaterial({ color: '#3a4150', roughness: .88, metalness: .03 });
const edgeMat = new THREE.MeshStandardMaterial({ color: '#c2c9d3', roughness: 0.95, metalness: 0.0 });
const curbW = 48;
const roadW = 40;
const segCount = 900;

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

buildRibbon(curbW, 0.22, edgeMat);
buildRibbon(roadW, 0.5, roadMat);

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

const loader = new GLTFLoader();
const draco = new DRACOLoader();
draco.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');
loader.setDRACOLoader(draco);
loader.load(
  'https://threejs.org/examples/models/gltf/ferrari.glb',
  (gltf) => {
    carRoot.clear();
    const model = gltf.scene;

    model.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = false;
        obj.receiveShadow = false;
      }
    });

    // normalize model size and orientation for this game scale
    const box = new THREE.Box3().setFromObject(model);
    const size = new THREE.Vector3();
    box.getSize(size);
    const targetLength = 15.0;
    const scale = targetLength / Math.max(size.x, size.z, 0.001);
    model.scale.setScalar(scale);

    // rotate so car faces +Z in our movement convention
    model.rotation.y = Math.PI;

    // place so wheels/body sit near y=0 plane
    const box2 = new THREE.Box3().setFromObject(model);
    model.position.y += -box2.min.y + 0.15;

    carRoot.add(model);
  },
  undefined,
  () => {
    // keep fallback silently
  }
);

// skid marks
const skidGroup = new THREE.Group();
scene.add(skidGroup);
const skidMarks = [];

// rear smoke particles
const smokeGroup = new THREE.Group();
scene.add(smokeGroup);
const smokeParticles = [];
const state = {
  x: -320, z: -40,
  heading: Math.PI * 0.12,
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
};

const keys = new Set();
const mobileInput = { steer: 0, throttle: 0, brake: 0 };
let steerPadActiveId = null;

addEventListener('keydown', (e) => {
  keys.add(e.key.toLowerCase());
  if (e.key.toLowerCase() === 'r') resetCar();
  setupAudio();
});
addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

// avoid long-press text selection/copy popup on mobile controls
['contextmenu', 'selectstart'].forEach((evt) => {
  steerPad.addEventListener(evt, (e) => e.preventDefault());
  btnThrottle.addEventListener(evt, (e) => e.preventDefault());
  btnBrake.addEventListener(evt, (e) => e.preventDefault());
});

function setBtnHold(btn, key) {
  const down = () => { mobileInput[key] = 1; setupAudio(); };
  const up = () => { mobileInput[key] = 0; };
  btn.addEventListener('pointerdown', (e) => { e.preventDefault(); down(); });
  btn.addEventListener('pointerup', up);
  btn.addEventListener('pointercancel', up);
  btn.addEventListener('pointerleave', up);
}
setBtnHold(btnThrottle, 'throttle');
setBtnHold(btnBrake, 'brake');

function updateSteerFromPointer(clientX, clientY) {
  const rect = steerPad.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const dx = clientX - cx;
  const dy = clientY - cy;
  const maxR = rect.width * 0.34;
  const len = Math.hypot(dx, dy) || 0.0001;
  const clamped = Math.min(maxR, len);
  const nx = (dx / len) * clamped;
  const ny = (dy / len) * clamped;
  steerThumb.style.transform = `translate(${nx}px, ${ny}px)`;

  // match PC-style steering: discrete left/neutral/right
  const raw = clamp(nx / maxR, -1, 1);
  const threshold = 0.22;
  if (raw > threshold) mobileInput.steer = 1;
  else if (raw < -threshold) mobileInput.steer = -1;
  else mobileInput.steer = 0;
}

steerPad.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  steerPadActiveId = e.pointerId;
  steerPad.setPointerCapture(e.pointerId);
  setupAudio();
  updateSteerFromPointer(e.clientX, e.clientY);
});
steerPad.addEventListener('pointermove', (e) => {
  if (e.pointerId !== steerPadActiveId) return;
  updateSteerFromPointer(e.clientX, e.clientY);
});
function resetSteerPad() {
  steerPadActiveId = null;
  mobileInput.steer = 0;
  steerThumb.style.transform = 'translate(0px, 0px)';
}
steerPad.addEventListener('pointerup', resetSteerPad);
steerPad.addEventListener('pointercancel', resetSteerPad);

const checkpoints = Array.from({ length: 8 }, (_, i) => curve.getPointAt(i / 8));

function fmt(ms) { return `${(ms / 1000).toFixed(3)}s`; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

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

  const cp = checkpoints[state.nextCp];
  const cpM = mapToMini(cp.x, cp.z);
  mm.fillStyle = '#ffd86b';
  mm.beginPath(); mm.arc(cpM.x, cpM.y, 4.5, 0, Math.PI * 2); mm.fill();

  const carM = mapToMini(state.x, state.z);
  mm.fillStyle = '#5fe4ff';
  mm.beginPath(); mm.arc(carM.x, carM.y, 4, 0, Math.PI * 2); mm.fill();
}


function resetCar() {
  state.x = -320; state.z = -40;
  state.heading = Math.PI * 0.12;
  state.vx = 0; state.vz = 0; state.yawRate = 0; state.steer = 0;
  state.skidCd = 0;
  state.brakePedal = 0;
  state.gear = 1; state.rpm = 0.25; state.shiftLock = 0;
  state.lap = 1; state.nextCp = 0;
  state.lapStart = performance.now();

  // clear skid marks + smoke
  skidMarks.length = 0;
  while (skidGroup.children.length) skidGroup.remove(skidGroup.children[0]);
  smokeParticles.length = 0;
  while (smokeGroup.children.length) smokeGroup.remove(smokeGroup.children[0]);
}

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

function roadDistanceSq(x, z) {
  let min = Infinity;
  for (let i = 0; i < 90; i++) {
    const p = curve.getPointAt(i / 90);
    const d = (p.x - x) ** 2 + (p.z - z) ** 2;
    if (d < min) min = d;
  }
  return min;
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

  const keyThrottle = (keys.has('arrowup') || keys.has('w')) ? 1 : 0;
  const keyBrake = (keys.has('arrowdown') || keys.has('s') || keys.has(' ')) ? 1 : 0;
  const throttle = Math.max(keyThrottle, mobileInput.throttle);
  const brakeTarget = Math.max(keyBrake, mobileInput.brake);
  // gradual brake pedal response
  state.brakePedal += (brakeTarget - state.brakePedal) * Math.min(1, dt * (brakeTarget ? 4.5 : 3.2));
  const brake = state.brakePedal;

  // keyboard + touch steer (left positive in this coordinate setup)
  const keySteer = (keys.has('arrowleft') || keys.has('a') ? 1 : 0) + (keys.has('arrowright') || keys.has('d') ? -1 : 0);
  const steerIn = clamp(keySteer + mobileInput.steer, -1, 1);

  const fwd = new THREE.Vector2(Math.sin(state.heading), Math.cos(state.heading));
  const right = new THREE.Vector2(fwd.y, -fwd.x);
  let vForward = state.vx * fwd.x + state.vz * fwd.y;
  let vLateral = state.vx * right.x + state.vz * right.y;

  const speed = Math.hypot(state.vx, state.vz);
  const distSq = roadDistanceSq(state.x, state.z);
  const onRoad = distSq < (roadW * 0.9) ** 2;
  const grip = onRoad ? 2.4 : 1.0;

  const steerLimit = 0.62 * (0.35 + 0.65 * Math.max(0, 1 - speed / 120));
  const steerTarget = steerIn * steerLimit;
  state.steer += (steerTarget - state.steer) * Math.min(1, dt * 7.5);

  let aLong = 0;
  if (throttle) aLong += 140;
  if (brake) aLong -= 155 * brake * Math.sign(vForward || 1);
  aLong -= 1.05 * vForward;
  aLong -= 0.0048 * vForward * Math.abs(vForward);
  if (!onRoad) aLong -= 45 * Math.sign(vForward || 0);

  vForward += aLong * dt;
  vForward = clamp(vForward, -45, 170);
  vLateral *= Math.exp(-grip * dt);

  const targetYaw = Math.abs(vForward) > 0.5 ? (vForward / 3.4) * Math.tan(state.steer) : 0;
  state.yawRate += (targetYaw - state.yawRate) * Math.min(1, dt * 5.2);
  state.yawRate *= Math.exp(-1.5 * dt);
  state.heading += state.yawRate * dt;

  state.vx = fwd.x * vForward + right.x * vLateral;
  state.vz = fwd.y * vForward + right.y * vLateral;
  state.x += state.vx * dt;
  state.z += state.vz * dt;

  // skid mark trigger (hard brake / lateral slip)
  state.skidCd = Math.max(0, state.skidCd - dt);
  const slip = Math.abs(vLateral);
  const skidStrength = (brake * Math.max(0, Math.abs(vForward) - 12) * 0.028) + (slip * 0.095);
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

  // checkpoints/lap
  const cp = checkpoints[state.nextCp];
  if (new THREE.Vector2(state.x, state.z).distanceTo(new THREE.Vector2(cp.x, cp.z)) < 24) {
    state.nextCp++;
    if (state.nextCp >= checkpoints.length) {
      state.nextCp = 0;
      const lapMs = now - state.lapStart;
      state.bestLap = state.bestLap ? Math.min(state.bestLap, lapMs) : lapMs;
      state.lapStart = now;
      state.lap++;
    }
  }

  // place meshes
  carRoot.position.set(state.x, 0, state.z);
  carRoot.rotation.y = state.heading;
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
  nowText.textContent = fmt(now - state.lapStart);
  throttleBar.style.width = `${throttle * 100}%`;
  brakeBar.style.width = `${brake * 100}%`;
  rpmBar.style.width = `${state.rpm * 100}%`;

  // side minimap + speed gauge
  drawMiniMap();
  const speedNorm = clamp(kmh / 220, 0, 1);
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
