/**
 * Vite plugin that auto-generates scenario JSON files from outputs/ directory.
 *
 * Runs on server startup and watches outputs/ for changes.
 * Uses the same converter logic as scripts/csv-to-episode.cjs.
 */

import { Plugin, ConfigEnv } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT = path.resolve(__dirname);
const OUTPUTS_DIR = path.join(ROOT, '..', 'outputs');
const SCENARIOS_DIR = path.join(ROOT, 'public', 'scenarios');

// ─── CSV parser ───
function parseCSV(text: string): string[][] {
  const rows: string[][] = [];
  let i = 0;
  while (i < text.length) {
    const row: string[] = [];
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

function col(row: string[], headers: string[], name: string): string {
  const idx = headers.indexOf(name);
  return idx === -1 ? '' : (row[idx] || '');
}

function redSeverity(cmd: string): number {
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

function buildTrajectory(steps: { step: number; red_command: string }[]): {
  success_rate: number; p99_latency: number; queue_depth: number;
  worker_restarts: number; consumer_stalls: number;
}[] {
  const n = steps.length;
  if (n === 0) return [];
  let totalDmg = 0;
  const damages = steps.map(s => { const sev = redSeverity(s.red_command); totalDmg += sev; return sev; });
  const maxDmg = Math.max(totalDmg, 1);
  let cumDmg = 0;
  return steps.map((s, i) => {
    cumDmg += damages[i];
    const progress = (i + 1) / n;
    const d = cumDmg / maxDmg;
    return {
      success_rate: Math.max(0.01, +(0.99 * (1 - d * 0.9)).toFixed(3)),
      p99_latency: Math.round(32 + d * 13000 * progress),
      queue_depth: Math.round(3 + d * 20000 * progress),
      worker_restarts: Math.round(d * 8),
      consumer_stalls: Math.round(d * 58),
    };
  });
}

function convertRun(runDir: string): object | null {
  const runId = path.basename(runDir);
  const summaryPath = path.join(runDir, 'summary.csv');
  const stepsPath = path.join(runDir, 'steps.csv');
  if (!fs.existsSync(summaryPath) || !fs.existsSync(stepsPath)) return null;

  const sumParsed = parseCSV(fs.readFileSync(summaryPath, 'utf-8').trim());
  const sumH = sumParsed[0].map(h => h.trim());
  const sumR = sumParsed[1];
  if (!sumR) return null;

  const stepsParsed = parseCSV(fs.readFileSync(stepsPath, 'utf-8').trim());
  const stepsH = stepsParsed[0].map(h => h.trim());

  const ALL_SERVICES = ['gateway', 'auth', 'worker', 'job_generator', 'redis'];
  let cumRed = 0, cumBlue = 0;
  // Track cumulative service state across steps
  const serviceState: Record<string, string> = {};
  for (const svc of ALL_SERVICES) serviceState[svc] = 'running';

  const rawSteps = stepsParsed.slice(1).map(row => {
    const step = parseInt(col(row, stepsH, 'step'));
    const reward = parseFloat(col(row, stepsH, 'reward'));
    const done = col(row, stepsH, 'done') === 'true';
    const termination = col(row, stepsH, 'termination_reason') || null;
    const redCmd = col(row, stepsH, 'red_command');
    const redReason = col(row, stepsH, 'red_reasoning');
    const redExit = parseInt(col(row, stepsH, 'red_exit_code')) || 0;
    const blueKind = col(row, stepsH, 'blue_kinds');
    const blueTarget = col(row, stepsH, 'blue_commands');
    const blueStatus = col(row, stepsH, 'blue_statuses');
    const blueDetail = col(row, stepsH, 'blue_details');
    const servicesAffected = col(row, stepsH, 'services_affected');
    const servicesRestored = col(row, stepsH, 'services_restored');
    const stepError = col(row, stepsH, 'step_error');
    cumRed += reward;
    cumBlue += Math.max(0, 1 - reward);
    const blue = blueTarget ? { kind: blueKind || 'llm_command', target: blueTarget, status: blueStatus || 'applied', detail: blueDetail || '' } : null;
    const totalSteps = stepsParsed.length - 1;
    const redPct = totalSteps > 0 ? Math.round((cumRed / totalSteps) * 100) : 0;
    const bluePct = totalSteps > 0 ? Math.round((cumBlue / totalSteps) * 100) : 0;
    // Red attack degrades directly affected services
    if (servicesAffected) {
      for (const svc of servicesAffected.split(',')) {
        const s = svc.trim();
        if (s && serviceState[s] !== undefined) serviceState[s] = 'degraded';
      }
    }
    // step_error contains actual cascade info, e.g. "critical services stopped: gateway,auth,worker"
    if (stepError) {
      const stoppedMatch = stepError.match(/critical services stopped:\s*(.+)/i);
      if (stoppedMatch) {
        for (const svc of stoppedMatch[1].split(',')) {
          const s = svc.trim();
          if (s && serviceState[s] !== undefined) serviceState[s] = 'stopped';
        }
      }
    }
    // Snapshot after red attack (before blue restores)
    const statusAfterRed: Record<string, string> = { ...serviceState };
    // Blue restores services
    if (servicesRestored) {
      for (const svc of servicesRestored.split(',')) {
        const s = svc.trim();
        if (s && serviceState[s] !== undefined) serviceState[s] = 'running';
      }
    }
    // Snapshot after blue action
    const statusAfterBlue: Record<string, string> = { ...serviceState };
    return {
      step, reward, done, termination, red_command: redCmd, red_reasoning: redReason, red_exit: redExit, blue,
      cumRed: redPct, cumBlue: bluePct,
      services_affected: servicesAffected, services_restored: servicesRestored,
      status_after_red: statusAfterRed, status_after_blue: statusAfterBlue,
    };
  });

  const trajectory = buildTrajectory(rawSteps);
  const steps = rawSteps.map((s, i) => {
    const t = trajectory[i] || trajectory[trajectory.length - 1] || { success_rate: 0.01, p99_latency: 13000, queue_depth: 20362, worker_restarts: 8, consumer_stalls: 58 };
    const prevT = i > 0 ? trajectory[i - 1] : { success_rate: 0.99, p99_latency: 32, queue_depth: 3, worker_restarts: 0, consumer_stalls: 0 };
    const dmg = s.red_command ? redSeverity(s.red_command) * 0.12 : 0;
    const afterRed = {
      gateway_success_rate: Math.max(0, +(prevT.success_rate - dmg).toFixed(3)),
      gateway_p99_latency_ms: Math.round(prevT.p99_latency * (1 + dmg * 1.5)),
      queue_depth: Math.round(prevT.queue_depth * (1 + dmg * 0.3)),
      worker_restart_count: t.worker_restarts, consumer_stall_count: t.consumer_stalls,
    };
    const afterBlue = {
      gateway_success_rate: t.success_rate, gateway_p99_latency_ms: t.p99_latency,
      queue_depth: t.queue_depth, worker_restart_count: t.worker_restarts, consumer_stall_count: t.consumer_stalls,
    };

    return {
      step: s.step,
      red: { command: s.red_command, reasoning: s.red_reasoning || '', output: '', exitCode: s.red_exit },
      blue: s.blue,
      metrics_before: { gateway_success_rate: prevT.success_rate, gateway_p99_latency_ms: prevT.p99_latency, queue_depth: prevT.queue_depth, worker_restart_count: prevT.worker_restarts, consumer_stall_count: prevT.consumer_stalls },
      metrics_after_red: afterRed, metrics_after_blue: afterBlue,
      process_status: s.status_after_blue,
      status_after_red: s.status_after_red,
      services_affected: s.services_affected,
      services_restored: s.services_restored,
      reward: s.reward, red_score: s.cumRed, blue_score: s.cumBlue,
      done: s.done, termination_reason: s.done ? (s.termination || 'max_steps') : null,
    };
  });

  return {
    run_id: runId, episode_id: runId,
    task: col(sumR, sumH, 'task_name'),
    max_steps: parseInt(col(sumR, sumH, 'max_steps_cap')) || 30,
    red_model: col(sumR, sumH, 'red_model'),
    blue_model: col(sumR, sumH, 'blue_model'),
    final_score: parseFloat(col(sumR, sumH, 'final_score')) || 0,
    avg_reward: parseFloat(col(sumR, sumH, 'avg_reward')) || 0,
    success: col(sumR, sumH, 'success') === 'true',
    error: col(sumR, sumH, 'error') || null,
    initial_metrics: { gateway_success_rate: 0.99, gateway_p99_latency_ms: 32, queue_depth: 3, worker_restart_count: 0, consumer_stall_count: 0 },
    initial_process_status: { gateway: 'running', auth: 'running', worker: 'running', job_generator: 'running', redis: 'running' },
    steps,
  };
}

function generateAll() {
  fs.mkdirSync(SCENARIOS_DIR, { recursive: true });
  if (!fs.existsSync(OUTPUTS_DIR)) return [];

  const dirs = fs.readdirSync(OUTPUTS_DIR).filter(d => d.startsWith('docker_'));
  const index: object[] = [];

  for (const d of dirs) {
    const result = convertRun(path.join(OUTPUTS_DIR, d));
    if (!result) continue;
    const runId = (result as any).run_id;
    fs.writeFileSync(path.join(SCENARIOS_DIR, `${runId}.json`), JSON.stringify(result, null, 2));
    index.push({
      run_id: (result as any).run_id,
      task: (result as any).task,
      red_model: (result as any).red_model,
      blue_model: (result as any).blue_model,
      max_steps: (result as any).max_steps,
      actual_steps: (result as any).steps.length,
      success: (result as any).success,
      final_score: (result as any).final_score,
      avg_reward: (result as any).avg_reward,
      error: (result as any).error,
    });
  }

  fs.writeFileSync(path.join(SCENARIOS_DIR, 'index.json'), JSON.stringify(index, null, 2));
  return index;
}

export function scenarioGenerator(): Plugin {
  return {
    name: 'scenario-generator',
    configureServer(server) {
      // Regenerate on every request to /scenarios/* so page refresh always picks up new runs
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/scenarios/')) {
          generateAll();
        }
        next();
      });

      const initial = generateAll();
      console.log(`[scenario-generator] Generated ${initial.length} scenarios from outputs/`);
    },
    buildStart() {
      generateAll();
      console.log('[scenario-generator] Generated scenarios for production build');
    },
  };
}