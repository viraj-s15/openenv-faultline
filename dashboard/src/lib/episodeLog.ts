// ─── Core data types ───

export interface MetricsSnapshot {
  gateway_success_rate: number;
  gateway_p99_latency_ms: number;
  queue_depth: number;
  worker_restart_count: number;
  consumer_stall_count: number;
}

// ─── Episode log format ─── (single source of truth for playback) ───

export interface EpisodeLog {
  run_id?: string;
  episode_id: string;
  task: string;
  max_steps: number;
  red_model?: string;
  blue_model?: string;
  final_score?: number;
  avg_reward?: number;
  success?: boolean;
  error?: string | null;
  initial_metrics: MetricsSnapshot;
  initial_process_status: Record<string, string>;
  steps: StepRecord[];
}

export interface StepRecord {
  step: number;
  red: {
    command: string;
    reasoning: string;
    output: string;
    exitCode: number;
  };
  blue: {
    kind: string;
    target: string;
    status: string;
    detail: string;
  } | null;
  metrics_before: MetricsSnapshot;
  metrics_after_red: MetricsSnapshot;
  metrics_after_blue: MetricsSnapshot;
  process_status: Record<string, string>;
  status_after_red?: Record<string, string>;
  services_affected?: string;
  services_restored?: string;
  reward: number;
  red_score: number;
  blue_score: number;
  done: boolean;
  termination_reason: string | null;
}

// ─── UI-friendly derived types ───

export interface TimelineEvent {
  step: number;
  agent: 'red' | 'blue';
  command: string;
  reasoning?: string;
  output: string;
  exitCode: number;
  reward?: number;
}

export interface MetricsHistoryPoint {
  step: number;
  successRate: number;
  p99Latency: number;
  queueDepth: number;
  workerRestarts: number;
  consumerStalls: number;
  reward: number;
}

export interface PlaybackSnapshot {
  stepIndex: number;
  totalSteps: number;
  isComplete: boolean;
  isInitial: boolean;
  metrics: MetricsSnapshot;
  processStatus: Record<string, string>;
  redScore: number;
  blueScore: number;
  reward: number;
  redEvents: TimelineEvent[];
  blueEvents: TimelineEvent[];
  metricsHistory: MetricsHistoryPoint[];
}

function stepToRedEvent(s: StepRecord): TimelineEvent {
  return {
    step: s.step,
    agent: 'red',
    command: s.red.command,
    reasoning: s.red.reasoning,
    output: s.red.output,
    exitCode: s.red.exitCode,
    reward: s.reward,
  };
}

function parseBlueReasoning(detail: string): { reasoning: string; output: string } {
  // Blue detail format: "exit_code=N reasoning=... output=..." or "exit_code=N output=..."
  const rMatch = detail.match(/reasoning=(.+?)(?: output=|$)/);
  const oMatch = detail.match(/output=(.+)$/);
  return {
    reasoning: rMatch && rMatch[1] ? rMatch[1].trim() : '',
    output: oMatch && oMatch[1] ? oMatch[1].trim() : detail,
  };
}

function stepToBlueEvent(s: StepRecord): TimelineEvent {
  const parsed = parseBlueReasoning(s.blue!.detail);
  return {
    step: s.step,
    agent: 'blue',
    command: s.blue!.target,
    reasoning: parsed.reasoning || '',
    output: parsed.output,
    exitCode: s.blue!.status === 'applied' ? 0 : 1,
  };
}

function metricsToHistoryPoint(step: number, reward: number, m: MetricsSnapshot): MetricsHistoryPoint {
  return {
    step,
    reward,
    successRate: m.gateway_success_rate,
    p99Latency: m.gateway_p99_latency_ms,
    queueDepth: m.queue_depth,
    workerRestarts: m.worker_restart_count,
    consumerStalls: m.consumer_stall_count,
  };
}

export function getSnapshot(log: EpisodeLog, cursor: number): PlaybackSnapshot {
  const slice = log.steps.slice(0, cursor);
  const lastStep = slice.length > 0 ? slice[slice.length - 1]! : null;
  const metrics = lastStep ? lastStep.metrics_after_blue : log.initial_metrics;
  const processStatus = lastStep ? lastStep.status_after_red ?? lastStep.process_status : log.initial_process_status;

  const redEvents: TimelineEvent[] = slice.map(stepToRedEvent);
  const blueEvents: TimelineEvent[] = slice
    .filter((s) => s.blue !== null)
    .map(stepToBlueEvent);

  const metricsHistory: MetricsHistoryPoint[] = [
    metricsToHistoryPoint(0, 0, log.initial_metrics),
    ...slice.map((s) => metricsToHistoryPoint(s.step, s.reward, s.metrics_after_blue)),
  ];

  return {
    stepIndex: cursor,
    totalSteps: log.max_steps,
    isComplete: lastStep?.done ?? false,
    isInitial: cursor === 0,
    metrics,
    processStatus,
    redScore: Math.round((lastStep?.reward ?? 0) * 100),
    blueScore: Math.round((1 - (lastStep?.reward ?? 0)) * 100),
    reward: lastStep?.reward ?? 0,
    redEvents,
    blueEvents,
    metricsHistory,
  };
}