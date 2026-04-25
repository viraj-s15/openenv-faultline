#!/usr/bin/env node
/**
 * Convert WarGames output runs to EpisodeLog JSON for the dashboard.
 *
 * Usage:
 *   node scripts/csv-to-episode.cjs <run-dir>                    # single run
 *   node scripts/csv-to-episode.cjs                               # all runs in outputs/
 *
 * Reads from each run directory:
 *   - summary.csv  (episode metadata)
 *   - steps.csv    (per-step detail)
 *
 * Writes to dashboard/public/scenarios/<run-id>.json
 * Also generates dashboard/public/scenarios/index.json (run listing)
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const OUTPUTS_DIR = path.join(ROOT, 'outputs');
const SCENARIOS_DIR = path.join(ROOT, 'dashboard', 'public', 'scenarios');

// ─── CSV parser (handles quoted fields with commas) ───
function parseCSV(text) {
  const rows = [];
  let i = 0;
  while (i < text.length) {
    const row = [];
    while (i < text.length) {
      if (text[i] === '"') {
        let field = ''; i++;
        while (i < text.length) {
          if (text[i] === '"') { if (text[i+1] === '"') { field += '"'; i += 2; } else { i++; break; } }
          else { field += text[i]; i++; }
        }
        row.push(field);
        if (text[i] === ',') i++;
      } else {
        let field = '';
        while (i < text.length && text[i] !== ',' && text[i] !== '\n') { field += text[i]; i++; }
        row.push(field);
        if (i < text.length && text[i] === ',') i++;
      }
      if (i < text.length && text[i] === '\n') { i++; break; }
    }
    if (row.length > 0) rows.push(row);
  }
  return rows;
}

function col(row, headers, name) {
  const i = headers.indexOf(name);
  return i === -1 ? '' : (row[i] || '');
}

// ─── Red action severity for metric synthesis ───
function redSeverity(cmd) {
  if (!cmd) return 0;
  const c = cmd.toLowerCase();
  if (c.includes('kill -9') || c.includes('kill -sigkill')) return 1.0;
  if (c.includes('flushdb') || c.includes('rm ')) return 0.95;
  if (c.includes('echo') && c.includes('>')) return 0.7;
  if (c.includes('setnx') || c.includes('lock')) return 0.5;
  if (c.includes('lpush') || c.includes('flood') || c.includes('garbage')) return 0.6;
  if (c.includes('kill -hup')) return 0.8;
  if (c.includes('kill')) return 0.7;
  if (c.includes('redis-cli set')) return 0.5;
  if (c.includes('ps aux') || c.includes('cat ') || c.includes('curl')) return 0.05;
  return 0.15;
}

// ─── Adaptive metric synthesis ───
// Build a trajectory from start → end based on cumulative damage
function buildTrajectory(steps) {
  const n = steps.length;
  if (n === 0) return [];

  // Accumulate severity over time
  let totalDamage = 0;
  const damages = steps.map(s => {
    const sev = redSeverity(s.red_command);
    totalDamage += sev;
    return { step: s.step, sev, cumulative: totalDamage };
  });

  const maxDamage = Math.max(totalDamage, 1);

  return steps.map((s, idx) => {
    const progress = (idx + 1) / n;
    const damageRatio = damages[idx].cumulative / maxDamage;
    const d = damageRatio;

    return {
      step: s.step,
      success_rate: Math.max(0.01, +(0.99 * (1 - d * 0.9)).toFixed(3)),
      p99_latency: Math.round(32 + d * 13000 * progress),
      queue_depth: Math.round(3 + d * 20000 * progress),
      worker_restarts: Math.round(d * 8),
      consumer_stalls: Math.round(d * 58),
    };
  });
}

// ─── Convert a single run ───
function convertRun(runDir) {
  const runId = path.basename(runDir);
  const summaryPath = path.join(runDir, 'summary.csv');
  const stepsPath = path.join(runDir, 'steps.csv');

  if (!fs.existsSync(summaryPath) || !fs.existsSync(stepsPath)) {
    console.error(`Skipping ${runId}: missing summary.csv or steps.csv`);
    return null;
  }

  // Parse summary
  const sumParsed = parseCSV(fs.readFileSync(summaryPath, 'utf-8').trim());
  const sumHeaders = sumParsed[0].map(h => h.trim());
  const sumRow = sumParsed[1];
  const summary = {
    task: col(sumRow, sumHeaders, 'task_name'),
    red_model: col(sumRow, sumHeaders, 'red_model'),
    blue_model: col(sumRow, sumHeaders, 'blue_model'),
    max_steps: parseInt(col(sumRow, sumHeaders, 'max_steps_cap')) || 30,
    actual_steps: parseInt(col(sumRow, sumHeaders, 'actual_steps')) || 0,
    success: col(sumRow, sumHeaders, 'success') === 'true',
    final_score: parseFloat(col(sumRow, sumHeaders, 'final_score')) || 0,
    max_reward: parseFloat(col(sumRow, sumHeaders, 'max_reward')) || 0,
    avg_reward: parseFloat(col(sumRow, sumHeaders, 'avg_reward')) || 0,
    error: col(sumRow, sumHeaders, 'error') || null,
  };

  // Parse steps
  const stepsParsed = parseCSV(fs.readFileSync(stepsPath, 'utf-8').trim());
  const stepsHeaders = stepsParsed[0].map(h => h.trim());
  const stepsRows = stepsParsed.slice(1);

  let cumulativeRed = 0, cumulativeBlue = 0;
  const rawSteps = stepsRows.map(row => {
    const step = parseInt(col(row, stepsHeaders, 'step'));
    const reward = parseFloat(col(row, stepsHeaders, 'reward'));
    const done = col(row, stepsHeaders, 'done') === 'true';
    const termination = col(row, stepsHeaders, 'termination_reason') || null;
    const redCmd = col(row, stepsHeaders, 'red_command');
    const redReasoning = col(row, stepsHeaders, 'red_reasoning');
    const redExit = parseInt(col(row, stepsHeaders, 'red_exit_code')) || 0;
    const blueKind = col(row, stepsHeaders, 'blue_kinds');
    const blueTarget = col(row, stepsHeaders, 'blue_commands');
    const blueStatus = col(row, stepsHeaders, 'blue_statuses');
    const blueDetail = col(row, stepsHeaders, 'blue_details');

    cumulativeRed += reward;
    cumulativeBlue += Math.max(0, 1 - reward);

    const blue = blueTarget ? { kind: blueKind || 'llm_command', target: blueTarget, status: blueStatus || 'applied', detail: blueDetail || '' } : null;

    return {
      step, reward, done, termination,
      red_command: redCmd, red_reasoning: redReasoning, red_exit: redExit,
      blue, cumulativeRed: Math.round(cumulativeRed * 100) / 100,
      cumulativeBlue: Math.round(cumulativeBlue * 100) / 100,
    };
  });

  // Build adaptive metric trajectory
  const trajectory = buildTrajectory(rawSteps);

  // Assemble steps with metrics
  const steps = rawSteps.map((s, idx) => {
    const t = trajectory[idx] || trajectory[trajectory.length - 1] || { success_rate: 0.01, p99_latency: 13000, queue_depth: 20362, worker_restarts: 8, consumer_stalls: 58 };
    const prevT = idx > 0 ? trajectory[idx - 1] : { success_rate: 0.99, p99_latency: 32, queue_depth: 3, worker_restarts: 0, consumer_stalls: 0 };

    // After red: degrade from prev baseline
    const dmg = s.red_command ? redSeverity(s.red_command) * 0.12 : 0;
    const afterRed = {
      gateway_success_rate: Math.max(0, +(prevT.success_rate - dmg).toFixed(3)),
      gateway_p99_latency_ms: Math.round(prevT.p99_latency * (1 + dmg * 1.5)),
      queue_depth: Math.round(prevT.queue_depth * (1 + dmg * 0.3)),
      worker_restart_count: t.worker_restarts,
      consumer_stall_count: t.consumer_stalls,
    };

    // After blue: trajectory target
    const afterBlue = {
      gateway_success_rate: t.success_rate,
      gateway_p99_latency_ms: t.p99_latency,
      queue_depth: t.queue_depth,
      worker_restart_count: t.worker_restarts,
      consumer_stall_count: t.consumer_stalls,
    };

    // Derive process status from success rate
    const sr = t.success_rate;
    let processStatus;
    if (sr >= 0.85) processStatus = { gateway: 'running', auth: 'running', worker: 'running', job_generator: 'running' };
    else if (sr >= 0.6) processStatus = { gateway: 'running', auth: 'running', worker: sr < 0.7 ? 'degraded' : 'running', job_generator: 'running' };
    else if (sr >= 0.3) processStatus = { gateway: 'running', auth: sr < 0.4 ? 'stopped' : 'degraded', worker: 'stopped', job_generator: 'running' };
    else processStatus = { gateway: sr < 0.05 ? 'stopped' : 'degraded', auth: 'stopped', worker: 'stopped', job_generator: 'running' };

    return {
      step: s.step,
      red: { command: s.red_command, reasoning: s.red_reasoning || '', output: '', exitCode: s.red_exit },
      blue: s.blue,
      metrics_before: {
        gateway_success_rate: prevT.success_rate,
        gateway_p99_latency_ms: prevT.p99_latency,
        queue_depth: prevT.queue_depth,
        worker_restart_count: prevT.worker_restarts,
        consumer_stall_count: prevT.consumer_stalls,
      },
      metrics_after_red: afterRed,
      metrics_after_blue: afterBlue,
      process_status: processStatus,
      reward: s.reward,
      red_score: s.cumulativeRed,
      blue_score: s.cumulativeBlue,
      done: s.done,
      termination_reason: s.done ? (s.termination || 'max_steps') : null,
    };
  });

  return {
    run_id: runId,
    episode_id: runId,
    task: summary.task,
    max_steps: summary.max_steps,
    red_model: summary.red_model,
    blue_model: summary.blue_model,
    final_score: summary.final_score,
    avg_reward: summary.avg_reward,
    success: summary.success,
    error: summary.error,
    initial_metrics: { gateway_success_rate: 0.99, gateway_p99_latency_ms: 32, queue_depth: 3, worker_restart_count: 0, consumer_stall_count: 0 },
    initial_process_status: { gateway: 'running', auth: 'running', worker: 'running', job_generator: 'running' },
    steps,
  };
}

// ─── Main ───
fs.mkdirSync(SCENARIOS_DIR, { recursive: true });

const runDir = process.argv[2];
const dirs = runDir
  ? [path.resolve(runDir)]
  : fs.readdirSync(OUTPUTS_DIR).filter(d => d.startsWith('docker_')).map(d => path.join(OUTPUTS_DIR, d));

const index = [];

for (const dir of dirs) {
  const result = convertRun(dir);
  if (!result) continue;

  const outPath = path.join(SCENARIOS_DIR, `${result.run_id}.json`);
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.log(`✓ ${result.run_id}: ${result.steps.length} steps, ${result.red_model} vs ${result.blue_model}, score=${result.final_score}`);

  index.push({
    run_id: result.run_id,
    task: result.task,
    red_model: result.red_model,
    blue_model: result.blue_model,
    max_steps: result.max_steps,
    actual_steps: result.steps.length,
    success: result.success,
    final_score: result.final_score,
    avg_reward: result.avg_reward,
    error: result.error,
  });
}

// Write index
fs.writeFileSync(path.join(SCENARIOS_DIR, 'index.json'), JSON.stringify(index, null, 2));
console.log(`\nWrote ${index.length} runs to scenarios/index.json`);