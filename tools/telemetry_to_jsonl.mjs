#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

function usage() {
  console.log('Usage: node tools/telemetry_to_jsonl.mjs <telemetry.json> [output-prefix]');
}

const inputPath = process.argv[2];
if (!inputPath) {
  usage();
  process.exit(1);
}

const outputPrefix = process.argv[3] || path.join(path.dirname(inputPath), path.basename(inputPath, path.extname(inputPath)));
const raw = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const history = [];
if (raw.current) history.push(raw.current);
if (Array.isArray(raw.history)) history.push(...raw.history);

const episodeRows = [];
const segmentRows = [];

history.forEach((episode, episodeIndex) => {
  const startedAt = episode.startedAt || null;
  const laps = Array.isArray(episode.laps) ? episode.laps : [];
  const drivers = Array.isArray(episode.drivers) ? episode.drivers : [];
  drivers.forEach((driver) => {
    const segments = Array.isArray(driver.segments) ? driver.segments : [];
    const totals = segments.reduce((acc, seg) => {
      acc.reward += seg.reward || 0;
      acc.samples += seg.samples || 0;
      acc.speedSum += seg.speedSum || 0;
      acc.throttleSum += seg.throttleSum || 0;
      acc.brakeSum += seg.brakeSum || 0;
      acc.driftSum += seg.driftSum || 0;
      acc.stabilityLoss += seg.stabilityLoss || 0;
      acc.progressDelta += seg.progressDelta || 0;
      acc.offroad += seg.offroad || 0;
      acc.contact += seg.contact || 0;
      return acc;
    }, {
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
    });

    episodeRows.push({
      episodeIndex,
      startedAt,
      driver: driver.name,
      lapsCompleted: laps.length,
      totalReward: totals.reward,
      totalSamples: totals.samples,
      avgSpeed: totals.samples ? totals.speedSum / totals.samples : 0,
      avgThrottle: totals.samples ? totals.throttleSum / totals.samples : 0,
      avgBrake: totals.samples ? totals.brakeSum / totals.samples : 0,
      avgDrift: totals.samples ? totals.driftSum / totals.samples : 0,
      avgStabilityLoss: totals.samples ? totals.stabilityLoss / totals.samples : 0,
      totalProgressDelta: totals.progressDelta,
      offroadCount: totals.offroad,
      contactCount: totals.contact,
      style: driver.style || {},
    });

    segments.forEach((seg) => {
      const samples = seg.samples || 0;
      segmentRows.push({
        episodeIndex,
        startedAt,
        driver: driver.name,
        segmentIdx: seg.idx,
        samples,
        reward: seg.reward || 0,
        avgSpeed: samples ? (seg.speedSum || 0) / samples : 0,
        avgThrottle: samples ? (seg.throttleSum || 0) / samples : 0,
        avgBrake: samples ? (seg.brakeSum || 0) / samples : 0,
        avgDrift: samples ? (seg.driftSum || 0) / samples : 0,
        avgStabilityLoss: samples ? (seg.stabilityLoss || 0) / samples : 0,
        progressDelta: seg.progressDelta || 0,
        offroadCount: seg.offroad || 0,
        contactCount: seg.contact || 0,
        style: driver.style || {},
      });
    });
  });
});

fs.writeFileSync(`${outputPrefix}.episodes.jsonl`, episodeRows.map((row) => JSON.stringify(row)).join('\n') + (episodeRows.length ? '\n' : ''));
fs.writeFileSync(`${outputPrefix}.segments.jsonl`, segmentRows.map((row) => JSON.stringify(row)).join('\n') + (segmentRows.length ? '\n' : ''));

console.log(JSON.stringify({
  input: inputPath,
  episodes: episodeRows.length,
  segments: segmentRows.length,
  outputs: [`${outputPrefix}.episodes.jsonl`, `${outputPrefix}.segments.jsonl`],
}, null, 2));
