import * as THREE from 'https://esm.sh/three@0.164.1';

export function makeProceduralTrack({ n = 40, cx = 0, cz = 0, sx = 1, sz = 1, f }) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const r = f(a);
    pts.push(new THREE.Vector3(cx + Math.cos(a) * r * sx, 0, cz + Math.sin(a) * r * sz));
  }
  return pts;
}

export const TRACKS = {
  classic: [
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
  ],
  stadium: makeProceduralTrack({
    n: 40, cx: -120, cz: -20, sx: 1.12, sz: 0.78,
    f: (a) => {
      let r = 520 + 170 * Math.cos(a) - 110 * Math.cos(2 * a) + 55 * Math.sin(3 * a);
      if (a > 1.1 && a < 2.05) r += 130;
      if (a > 4.2 && a < 5.0) r -= 90;
      return r;
    }
  }),
  coastal: makeProceduralTrack({
    n: 42, cx: 0, cz: 20, sx: 1.08, sz: 0.82,
    f: (a) => 500 + 135 * Math.sin(a + 0.35) + 95 * Math.cos(2.6 * a) + 40 * Math.sin(4.2 * a),
  }),
  canyon: makeProceduralTrack({
    n: 38, cx: -40, cz: 0, sx: 1.0, sz: 0.86,
    f: (a) => {
      let r = 470 + 190 * Math.cos(a * 1.1) - 75 * Math.sin(a * 2.8);
      if (a > 2.0 && a < 3.15) r -= 120;
      return r;
    }
  }),
  nightcity: makeProceduralTrack({
    n: 44, cx: -20, cz: -10, sx: 1.05, sz: 0.8,
    f: (a) => 510 + 110 * Math.cos(3 * a) + 80 * Math.sin(2 * a + 0.8),
  }),
};

export function computeTrackBounds(points) {
  return points.reduce((acc, p) => ({
    minX: Math.min(acc.minX, p.x),
    maxX: Math.max(acc.maxX, p.x),
    minZ: Math.min(acc.minZ, p.z),
    maxZ: Math.max(acc.maxZ, p.z),
  }), { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity });
}

export function createCheckpoints(curve, count = 8) {
  return Array.from({ length: count }, (_, i) => curve.getPointAt(i / count));
}

export function sampleClosedCurveFrame(curve, t, delta = 0.003) {
  const wrappedT = ((t % 1) + 1) % 1;
  const p = curve.getPointAt(wrappedT);
  const pPrev = curve.getPointAt((wrappedT - delta + 1) % 1);
  const pNext = curve.getPointAt((wrappedT + delta) % 1);
  const tx = pNext.x - pPrev.x;
  const tz = pNext.z - pPrev.z;
  const len = Math.hypot(tx, tz) || 1;
  const tangentX = tx / len;
  const tangentZ = tz / len;
  const rightX = -tangentZ;
  const rightZ = tangentX;
  return { t: wrappedT, p, tangentX, tangentZ, rightX, rightZ };
}

export function nearestPointOnCurve(curve, x, z, options = {}) {
  const {
    samples = 180,
    aroundT = null,
    window = 0.12,
    refinePasses = 2,
  } = options;

  let best = { t: 0, frame: sampleClosedCurveFrame(curve, 0), distSq: Infinity };
  let start = 0;
  let end = 1;

  if (Number.isFinite(aroundT)) {
    start = aroundT - window;
    end = aroundT + window;
  }

  const searchRange = (rangeStart, rangeEnd, count) => {
    for (let i = 0; i <= count; i++) {
      const alpha = i / Math.max(1, count);
      const t = rangeStart + (rangeEnd - rangeStart) * alpha;
      const frame = sampleClosedCurveFrame(curve, t);
      const dx = x - frame.p.x;
      const dz = z - frame.p.z;
      const distSq = dx * dx + dz * dz;
      if (distSq < best.distSq) best = { t: frame.t, frame, distSq };
    }
  };

  searchRange(start, end, samples);

  let radius = Number.isFinite(aroundT) ? window * 0.45 : 0.08;
  for (let pass = 0; pass < refinePasses; pass++) {
    searchRange(best.t - radius, best.t + radius, Math.max(24, Math.floor(samples * 0.35)));
    radius *= 0.35;
  }

  const dx = x - best.frame.p.x;
  const dz = z - best.frame.p.z;
  const signedOffset = dx * best.frame.rightX + dz * best.frame.rightZ;
  return { ...best, signedOffset };
}
