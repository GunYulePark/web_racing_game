import * as THREE from 'https://esm.sh/three@0.164.1';
import { TRACKS, createCheckpoints, nearestPointOnCurve, sampleClosedCurveFrame } from '../track.js';
import { actionFromPolicy, createHeuristicPolicyStub } from './policy_inference.js';

const ROAD_HALF_WIDTH = 20;
const MAX_SPEED = 78;
const MAX_REVERSE = -8;
const ACCEL_RATE = 18;
const BRAKE_RATE = 24;
const DRAG_LINEAR = 0.55;
const DRAG_QUAD = 0.0038;
const BASE_GRIP = 3.2;
const CORNER_STIFFNESS = 0.24;

const DEFAULT_DRIVERS = [
  { name: 'NOVA', pace: 1.0, corner: 0.92, drift: 1.0, jitter: 0.08 },
  { name: 'RUNE', pace: 0.97, corner: 1.02, drift: 0.86, jitter: 0.04 },
  { name: 'ECHO', pace: 0.95, corner: 0.98, drift: 0.9, jitter: 0.06 },
  { name: 'BLITZ', pace: 1.04, corner: 0.9, drift: 1.08, jitter: 0.05 },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function wrapAngle(angle) {
  let out = angle;
  while (out > Math.PI) out -= Math.PI * 2;
  while (out < -Math.PI) out += Math.PI * 2;
  return out;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function createTrack(trackKey) {
  const pts = TRACKS[trackKey] || TRACKS.stadium;
  const curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal', 0.08);
  return { curve, checkpoints: createCheckpoints(curve, 8) };
}

function nearest(curve, x, z, aroundT = null) {
  return nearestPointOnCurve(curve, x, z, { samples: 220, aroundT, window: 0.12 });
}

function buildObservation(curve, driver) {
  const hit = nearest(curve, driver.x, driver.z, driver.trackT);
  const headingTrack = Math.atan2(hit.frame.tangentX, hit.frame.tangentZ);
  const headingErr = wrapAngle(driver.heading - headingTrack);
  const slipAngle = Math.atan2(driver.lateralSpeed, Math.max(1, Math.abs(driver.speed)));
  const offroadMargin = (Math.abs(hit.signedOffset) - ROAD_HALF_WIDTH) / ROAD_HALF_WIDTH;

  const future = [];
  for (const delta of [0.01, 0.025, 0.05, 0.085, 0.12]) {
    const frame = sampleClosedCurveFrame(curve, hit.t + delta);
    const futureHeading = Math.atan2(frame.tangentX, frame.tangentZ);
    future.push(wrapAngle(futureHeading - headingTrack));
  }

  const obs = [
    driver.speed / MAX_SPEED,
    driver.lateralSpeed / 30,
    driver.yawRate / 2.5,
    driver.steer,
    driver.throttle,
    driver.brake,
    hit.signedOffset / ROAD_HALF_WIDTH,
    headingErr / Math.PI,
    slipAngle / 1.2,
    offroadMargin,
    ...future,
    Math.sin(driver.heading),
    Math.cos(driver.heading),
    hit.t,
    driver.progress - Math.floor(driver.progress),
    Math.min(1, driver.step / driver.maxSteps),
    Math.abs(hit.signedOffset) > ROAD_HALF_WIDTH ? 1 : 0,
  ];

  return {
    obs: obs.map((value) => clamp(value, -5, 5)),
    nearest: hit,
    headingErr,
    slipAngle,
    offroad: Math.abs(hit.signedOffset) > ROAD_HALF_WIDTH,
  };
}

function heuristicAction(obs, style, rng, step) {
  const speed = obs[0];
  const headingErr = obs[7] * Math.PI;
  const slipAngle = obs[8] * 1.2;
  const curvatureNear = obs[10];
  const curvatureFar = obs[12];
  const offroadMargin = obs[9];
  const cornerNeed = Math.abs(curvatureNear) * 0.9 + Math.abs(curvatureFar) * 1.1 + Math.abs(headingErr) * 0.55;
  const noise = (rng() - 0.5) * style.jitter * (0.2 + Math.min(1, step / 600));
  const steer = clamp(-(headingErr * (1.35 + (1 - style.corner) * 0.25) + slipAngle * 0.72) + noise, -1, 1);
  const throttle = clamp(0.94 * style.pace - cornerNeed * 0.78 - Math.max(0, offroadMargin) * 0.5 - Math.max(0, speed - 0.84) * 0.85, 0.12, 1);
  const brake = clamp(cornerNeed * (0.72 + (1 - style.corner) * 0.1) + Math.max(0, speed - 0.8) * 0.82 + Math.max(0, offroadMargin) * 0.45, 0, 1);
  const drift = clamp(cornerNeed * 0.82 * style.drift - Math.max(0, offroadMargin) * 1.2, 0, 1);
  return [steer, throttle, brake, drift];
}

function stepDriver(curve, driver, action, dt) {
  const steerCmd = clamp(action[0], -1, 1);
  const throttle = clamp(action[1], 0, 1);
  const brake = clamp(action[2], 0, 1);
  const drift = clamp(action[3], 0, 1);

  driver.throttle = throttle;
  driver.brake = brake;
  const steerLimit = 0.72 * (0.35 + 0.65 * Math.max(0, 1 - Math.abs(driver.speed) / 55));
  const steerTarget = steerCmd * steerLimit;
  const steerDelta = steerTarget - driver.steer;
  driver.steer += steerDelta * Math.min(1, dt * 7);

  const before = nearest(curve, driver.x, driver.z, driver.trackT);
  driver.trackT = before.t;
  const headingTrack = Math.atan2(before.frame.tangentX, before.frame.tangentZ);
  const headingErr = wrapAngle(driver.heading - headingTrack);
  const grip = BASE_GRIP - drift * 1.7;

  let aLong = ACCEL_RATE * throttle;
  aLong -= BRAKE_RATE * brake * (driver.speed >= 0 ? 1 : -1);
  aLong -= DRAG_LINEAR * driver.speed;
  aLong -= DRAG_QUAD * driver.speed * Math.abs(driver.speed);
  if (Math.abs(before.signedOffset) > ROAD_HALF_WIDTH) {
    aLong -= 10 * (driver.speed >= 0 ? 1 : -1);
  }

  const desiredLat = clamp(-headingErr * Math.abs(driver.speed) * CORNER_STIFFNESS, -24, 24) * (0.65 + drift * 0.95);
  driver.lateralSpeed += (desiredLat - driver.lateralSpeed) * Math.min(1, dt * (2 + drift));
  driver.speed = clamp(driver.speed + aLong * dt, MAX_REVERSE, MAX_SPEED * 1.05);
  driver.lateralSpeed *= Math.exp(-grip * dt);

  const yawLead = drift * clamp(Math.abs(driver.lateralSpeed) / 28, 0, 0.35) * Math.sign(headingErr || 1);
  const targetYaw = Math.abs(driver.speed) > 0.4 ? (driver.speed / 3.5) * Math.tan(driver.steer + yawLead) : 0;
  driver.yawRate += (targetYaw - driver.yawRate) * Math.min(1, dt * (4.8 + drift));
  driver.yawRate *= Math.exp(-(1.9 - drift * 0.4) * dt);
  driver.heading = wrapAngle(driver.heading + driver.yawRate * dt);

  const fwdX = Math.sin(driver.heading);
  const fwdZ = Math.cos(driver.heading);
  const rightX = fwdZ;
  const rightZ = -fwdX;
  driver.x += (fwdX * driver.speed + rightX * driver.lateralSpeed) * dt;
  driver.z += (fwdZ * driver.speed + rightZ * driver.lateralSpeed) * dt;

  const after = nearest(curve, driver.x, driver.z, driver.trackT);
  const newProgress = driver.lapCount + after.t;
  if (newProgress + 0.5 < driver.progress) {
    driver.lapCount += 1;
    driver.lapTimes.push(Number((driver.step * dt).toFixed(3)) - (driver.lastLapStamp || 0));
    driver.lastLapStamp = Number((driver.step * dt).toFixed(3));
    driver.progress = driver.lapCount + after.t;
  } else {
    driver.progress = newProgress;
  }
  driver.trackT = after.t;

  const headingTrack2 = Math.atan2(after.frame.tangentX, after.frame.tangentZ);
  const headingErr2 = wrapAngle(driver.heading - headingTrack2);
  const slipAngle = Math.abs(Math.atan2(driver.lateralSpeed, Math.max(1, Math.abs(driver.speed))));
  const offroad = Math.abs(after.signedOffset) > ROAD_HALF_WIDTH;
  const stability = clamp(1 - (Math.abs(headingErr2) / 1.6 + slipAngle / 1.15 + Math.abs(driver.yawRate) / 3.0 + Math.abs(after.signedOffset) / (ROAD_HALF_WIDTH * 1.4)) * 0.35, 0, 1);
  driver.offroadSteps += offroad ? 1 : 0;
  driver.maxAbsOffset = Math.max(driver.maxAbsOffset, Math.abs(after.signedOffset));
  driver.stabilitySum += stability;
  driver.stabilityMin = Math.min(driver.stabilityMin, stability);
  driver.reward += driver.progress - driver.prevProgress;
  driver.prevProgress = driver.progress;

  return { offroad, stability, headingErr: headingErr2, slipAngle, signedOffset: after.signedOffset };
}

function resolveContacts(drivers) {
  let contacts = 0;
  for (let i = 0; i < drivers.length; i += 1) {
    for (let j = i + 1; j < drivers.length; j += 1) {
      const a = drivers[i];
      const b = drivers[j];
      const dx = b.x - a.x;
      const dz = b.z - a.z;
      const dist = Math.hypot(dx, dz) || 1e-6;
      const minDist = 10.8;
      if (dist < minDist) {
        const nx = dx / dist;
        const nz = dz / dist;
        const push = (minDist - dist) * 0.5;
        a.x -= nx * push;
        a.z -= nz * push;
        b.x += nx * push;
        b.z += nz * push;
        a.contactCount += 1;
        b.contactCount += 1;
        contacts += 1;
      }
    }
  }
  return contacts;
}

export function runDeterministicAIBenchmark(config = {}) {
  const trackKey = config.track || 'stadium';
  const lapsTarget = Math.max(1, config.laps || 1);
  const dt = config.dt || 0.1;
  const maxSteps = config.maxSteps || 3200;
  const seed = Number.isFinite(config.seed) ? config.seed : 1337;
  const rng = mulberry32(seed);
  const { curve, checkpoints } = createTrack(trackKey);
  const policy = config.policy || null;
  const drivers = (config.drivers || DEFAULT_DRIVERS).map((base, index) => {
    const frame = sampleClosedCurveFrame(curve, (0.94 - index * 0.035 + 1) % 1);
    return {
      ...base,
      x: frame.p.x,
      z: frame.p.z,
      heading: Math.atan2(frame.tangentX, frame.tangentZ),
      speed: 8 + index * 0.4,
      lateralSpeed: 0,
      yawRate: 0,
      steer: 0,
      throttle: 0,
      brake: 0,
      lapCount: 0,
      lapTimes: [],
      trackT: frame.t,
      progress: frame.t,
      prevProgress: frame.t,
      reward: 0,
      step: 0,
      maxSteps,
      offroadSteps: 0,
      contactCount: 0,
      stabilitySum: 0,
      stabilityMin: 1,
      maxAbsOffset: 0,
      lastLapStamp: 0,
    };
  });

  let totalContacts = 0;
  for (let step = 1; step <= maxSteps; step += 1) {
    for (const driver of drivers) {
      driver.step = step;
      const { obs } = buildObservation(curve, driver);
      const action = policy && Array.isArray(policy.layers) && policy.layers.length
        ? actionFromPolicy(obs, policy)
        : heuristicAction(obs, driver, rng, step);
      stepDriver(curve, driver, action, dt);
    }
    totalContacts += resolveContacts(drivers);
    if (drivers.every((driver) => driver.lapCount >= lapsTarget)) break;
  }

  const results = drivers.map((driver) => ({
    name: driver.name,
    lapsCompleted: driver.lapCount,
    finished: driver.lapCount >= lapsTarget,
    lapTimes: driver.lapTimes,
    meanLapTime: driver.lapTimes.length ? driver.lapTimes.reduce((sum, value) => sum + value, 0) / driver.lapTimes.length : null,
    offroadRate: driver.offroadSteps / Math.max(1, driver.step),
    contactCount: driver.contactCount,
    stabilityMean: driver.stabilitySum / Math.max(1, driver.step),
    stabilityMin: driver.stabilityMin,
    maxAbsOffset: driver.maxAbsOffset,
    progress: driver.progress,
    rewardProxy: driver.reward,
  }));

  const summary = {
    benchmark: 'deterministic-browser-ai-v1',
    seed,
    track: trackKey,
    checkpoints: checkpoints.length,
    lapsTarget,
    dt,
    stepsRun: Math.max(...drivers.map((driver) => driver.step)),
    totalContacts,
    meanLapTime: results.filter((item) => item.meanLapTime != null).reduce((sum, item, _, arr) => sum + item.meanLapTime / Math.max(1, arr.length), 0),
    meanOffroadRate: results.reduce((sum, item) => sum + item.offroadRate, 0) / Math.max(1, results.length),
    meanStability: results.reduce((sum, item) => sum + item.stabilityMean, 0) / Math.max(1, results.length),
    finishedDrivers: results.filter((item) => item.finished).length,
    results,
  };
  return summary;
}

export function downloadDeterministicAIBenchmark(config = {}) {
  const payload = runDeterministicAIBenchmark(config);
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `deterministic-browser-benchmark-${payload.track}-${payload.seed}.json`;
  a.click();
  URL.revokeObjectURL(url);
  return payload;
}

export function installBrowserBenchmarkHooks(extra = {}) {
  window.runDeterministicAIBenchmark = (config = {}) => runDeterministicAIBenchmark({ ...extra, ...config });
  window.downloadDeterministicAIBenchmark = (config = {}) => downloadDeterministicAIBenchmark({ ...extra, ...config });
  window.createHeuristicPolicyStub = createHeuristicPolicyStub;
}
