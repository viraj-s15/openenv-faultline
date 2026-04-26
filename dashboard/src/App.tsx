import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  Play, Pause, SkipForward, SkipBack, RotateCcw,
  Shield, Skull, Activity, Cpu, ArrowLeft,
} from 'lucide-react';
import { usePlayback } from './hooks/usePlayback';
import { useScenarioData } from './hooks/useScenarioData';
import RunSelector from './RunSelector';
import type { TimelineEvent, MetricsHistoryPoint } from './lib/episodeLog';

/* ─── color tokens ─── */
const C = {
  red: '#f97066',
  blue: '#53b1fd',
  emerald: '#34d399',
  amber: '#f59e0b',
  slate950: '#0b0d12',
  slate900: '#0f1117',
  slate850: '#141821',
  slate800: '#1a1d27',
  slate750: '#1e2130',
  slate700: '#2a2d3a',
  slate600: '#3a3f52',
  slate500: '#4a5068',
  slate400: '#6b7084',
  slate300: '#8b91a5',
  slate100: '#d0d5e2',
  slate50: '#e4e6eb',
} as const;

const CHART_MARGIN = { top: 4, right: 4, bottom: 4, left: -30 } as const;
const CHART_TOOLTIP = {
  contentStyle: {
    background: C.slate800, border: `1px solid ${C.slate700}`,
    borderRadius: 6, fontSize: 11, fontFamily: 'var(--font-mono)',
  },
  labelStyle: { color: C.slate100 },
} as const;

/* ─── feed component ─── */
function Feed({ events, agent }: { events: TimelineEvent[]; agent: 'red' | 'blue' }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [events.length]);

  const color = agent === 'red' ? C.red : C.blue;
  const label = agent === 'red' ? 'RED TEAM' : 'BLUE TEAM';
  const icon = agent === 'red' ? <Skull size={14} /> : <Shield size={14} />;

  return (
    <div className="panel flex flex-col h-full">
      <div className="panel-header" style={{ color }}>
        {icon}
        {label}
        <span style={{ color: C.slate400, marginLeft: 'auto', fontWeight: 400, fontSize: 11, letterSpacing: 0 }}>
          {events.length} {events.length === 1 ? 'entry' : 'entries'}
        </span>
      </div>
      <div ref={ref} className="feed-scroll flex-1 overflow-y-auto p-3">
        {events.length === 0 && (
          <div style={{ color: C.slate500 }} className="text-center pt-8 text-sm">
            {agent === 'red' ? 'Awaiting red commands…' : 'Blue standing by…'}
          </div>
        )}
        <AnimatePresence initial={false}>
          {events.map((evt, i) => (
            <motion.div
              key={`${evt.step}-${evt.agent}-${i}`}
              initial={{ opacity: 0, x: agent === 'red' ? -12 : 12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.25 }}
              className="mb-3"
            >
              <div className="flex items-start gap-2">
                <span
                  style={{ background: `${color}18`, color, borderRadius: 4, padding: '1px 6px', fontSize: 10, fontWeight: 600 }}
                >
                  STEP {evt.step}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-[12px] break-all" style={{ color: C.slate50 }}>
                    {evt.command}
                  </div>
                  {evt.reasoning && (
                    <div style={{
                      marginTop: 4,
                      paddingLeft: 8,
                      borderLeft: `2px solid ${agent === 'red' ? 'rgba(249,112,102,0.4)' : 'rgba(83,177,253,0.4)'}`,
                      color: C.slate300,
                      fontSize: 11,
                      lineHeight: '1.6',
                      fontFamily: 'var(--font-sans)',
                      fontStyle: 'italic',
                    }}>
                      {evt.reasoning}
                    </div>
                  )}
                  {evt.output && (
                    <pre style={{ color: C.slate300, background: C.slate900, borderRadius: 4, padding: '4px 6px', marginTop: 4, fontSize: 11 }} className="whitespace-pre-wrap break-all leading-[1.5]">
                      {evt.output.length > 300 ? evt.output.slice(0, 300) + '…' : evt.output}
                    </pre>
                  )}
                  <div className="flex items-center gap-3 mt-1">
                    {evt.reward !== undefined && (
                      <span style={{
                        color: evt.reward >= 0.5 ? C.emerald : evt.reward >= 0.2 ? C.amber : C.red,
                        fontSize: 10, fontWeight: 600,
                      }}>
                        reward {evt.reward.toFixed(2)}
                      </span>
                    )}
                    <span style={{ color: evt.exitCode === 0 ? C.emerald : C.red, fontSize: 10 }}>
                      exit {evt.exitCode}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ─── metric card ─── */
function MetricCard({ label, value, sub, color }: {
  label: string; value: string; sub?: string; color: string;
}) {
  return (
    <div style={{ background: C.slate850, border: `1px solid ${C.slate700}`, borderRadius: 8 }} className="p-3">
      <div style={{ color: C.slate400 }} className="text-[10px] font-semibold tracking-wider uppercase mb-1">{label}</div>
      <div className="metric-value text-xl font-bold" style={{ color }}>{value}</div>
      {sub && <div style={{ color: C.slate500 }} className="text-[11px] mt-0.5">{sub}</div>}
    </div>
  );
}

/* ─── mini chart ─── */
function MiniChart({ data, dataKey, color, type, yDomain }: {
  data: MetricsHistoryPoint[]; dataKey: keyof MetricsHistoryPoint;
  color: string; type: 'line' | 'area' | 'bar'; yDomain?: [number, number];
}) {
  if (type === 'line') {
    return (
      <ResponsiveContainer width="100%" height={80}>
        <LineChart data={data} margin={CHART_MARGIN}>
          <XAxis dataKey="step" tick={{ fontSize: 9, fill: C.slate500 }} axisLine={false} tickLine={false} />
          <YAxis domain={yDomain} tick={{ fontSize: 9, fill: C.slate500 }} axisLine={false} tickLine={false} />
          <Tooltip {...CHART_TOOLTIP} />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    );
  }
  if (type === 'area') {
    return (
      <ResponsiveContainer width="100%" height={80}>
        <AreaChart data={data} margin={CHART_MARGIN}>
          <XAxis dataKey="step" tick={{ fontSize: 9, fill: C.slate500 }} axisLine={false} tickLine={false} />
          <YAxis domain={yDomain} tick={{ fontSize: 9, fill: C.slate500 }} axisLine={false} tickLine={false} />
          <Tooltip {...CHART_TOOLTIP} />
          <Area type="monotone" dataKey={dataKey} stroke={color} fill={`${color}25`} strokeWidth={2} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={80}>
      <BarChart data={data} margin={CHART_MARGIN}>
        <XAxis dataKey="step" tick={{ fontSize: 9, fill: C.slate500 }} axisLine={false} tickLine={false} />
        <YAxis domain={yDomain} tick={{ fontSize: 9, fill: C.slate500 }} axisLine={false} tickLine={false} />
        <Tooltip {...CHART_TOOLTIP} />
        <Bar dataKey={dataKey} fill={color} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}



/* ─── ROUTER ─── */

function useHashRouter(): { view: 'selector' | 'dashboard'; runId: string | null } {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const handler = () => setHash(window.location.hash);
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);

  const match = hash.match(/^#\/run\/(.+)$/);
  if (match && match[1]) return { view: 'dashboard', runId: decodeURIComponent(match[1]) };
  return { view: 'selector', runId: null };
}

/* ─── APP ─── */
export default function App() {
  const { view, runId } = useHashRouter();

  if (view === 'selector') {
    return <RunSelector onSelect={(id) => { window.location.hash = `#/run/${encodeURIComponent(id)}`; }} />;
  }

  return <Dashboard runId={runId!} onBack={() => { window.location.hash = '#/'; }} />;
}

/* ─── DASHBOARD ─── */
function Dashboard({ runId, onBack }: { runId: string; onBack: () => void }) {
  const { log, loading, error } = useScenarioData(runId);
  const { isPlaying, snapshot, play, pause, stepForward, stepBack, reset, restart } = usePlayback(log);
  // Keyboard shortcuts: ← rewind, → advance, Space toggle play/pause
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepBack(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); stepForward(); }
      else if (e.key === ' ') { e.preventDefault(); isPlaying ? pause() : play(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [stepBack, stepForward, pause, play, isPlaying]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: C.slate950 }}>
        <Activity size={24} className="animate-spin" style={{ color: C.blue }} />
        <span style={{ color: C.slate300, marginLeft: 12, fontFamily: 'var(--font-mono)' }}>Loading scenario…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: C.slate950 }}>
        <span style={{ color: C.red, fontFamily: 'var(--font-mono)' }}>Failed to load scenario: {error}</span>
      </div>
    );
  }



  const m = snapshot.metrics;
  const s = snapshot;

  const rewardPct = s.redScore;
  const latency = m.gateway_p99_latency_ms;
  const queueDepth = m.queue_depth;

  const srColor = s.isInitial ? C.slate500 : rewardPct >= 50 ? C.emerald : rewardPct >= 20 ? C.amber : C.red;
  const latColor = s.isInitial ? C.slate500 : latency < 100 ? C.emerald : latency < 500 ? C.amber : C.red;

  const episodeLabel = log ? `${log.red_model || 'Red'} vs ${log.blue_model || 'Blue'}` : '';

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: C.slate950 }}>
      {/* ── HEADER ── */}
      <motion.header
        className="flex items-center gap-4 px-5 py-3"
        style={{ background: C.slate900, borderBottom: `1px solid ${C.slate700}` }}
        initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}
      >
        <div className="flex items-center gap-2.5">
          <button onClick={onBack} className="p-1 rounded-md hover:bg-slate-750 transition-colors" style={{ color: C.slate400 }} title="Back to runs">
            <ArrowLeft size={18} />
          </button>
          <h1 className="text-base font-bold tracking-wide" style={{ fontFamily: 'var(--font-sans)', color: C.slate50 }}>
            FAULTLINE
          </h1>
          {episodeLabel && (
            <span style={{ color: C.slate500, fontSize: 11, fontFamily: 'var(--font-mono)', marginLeft: 4 }}>
              {episodeLabel}
            </span>
          )}
        </div>

        {/* step counter + playback */}
        <div className="ml-auto flex items-center gap-4">
          <div style={{ color: C.slate300, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
            STEP <span style={{ color: C.slate50, fontWeight: 700 }}>{s.stepIndex}</span>
            <span style={{ color: C.slate500 }}>/{s.totalSteps}</span>
          </div>

          <div className="flex items-center gap-1">
            <button onClick={reset} className="p-1.5 rounded-md hover:bg-slate-750 transition-colors" style={{ color: C.slate300 }} title="Reset">
              <RotateCcw size={15} />
            </button>
            <button onClick={stepBack} disabled={s.stepIndex <= 0} className="p-1.5 rounded-md hover:bg-slate-750 transition-colors disabled:opacity-30" style={{ color: C.slate300 }} title="Step back">
              <SkipBack size={16} />
            </button>
            {isPlaying ? (
              <button onClick={pause} className="p-1.5 rounded-md hover:bg-slate-750 transition-colors" style={{ color: C.slate50 }} title="Pause">
                <Pause size={18} />
              </button>
            ) : (
              <button onClick={s.isComplete ? restart : play} className="p-1.5 rounded-md transition-colors" style={{ color: C.slate50, background: C.slate750 }} title={s.isComplete ? 'Restart' : 'Play'}>
                <Play size={18} />
              </button>
            )}
            <button onClick={stepForward} disabled={s.isComplete} className="p-1.5 rounded-md hover:bg-slate-750 transition-colors disabled:opacity-30" style={{ color: C.slate300 }} title="Step forward">
              <SkipForward size={16} />
            </button>
          </div>


        </div>
      </motion.header>

      {/* ── MAIN 3-COLUMN ── */}
      <div className="flex-1 grid grid-cols-[1fr_300px_1fr] gap-3 p-3 min-h-0">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4, delay: 0.1 }} className="min-h-0">
          <Feed events={s.redEvents} agent="red" />
        </motion.div>

        {/* ── CENTER ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.2 }} className="flex flex-col gap-3 min-h-0">
          {/* SCOREBOARD */}
          <div style={{ background: C.slate850, border: `1px solid ${C.slate700}`, borderRadius: 10 }} className="flex items-center justify-center gap-6 p-4">
            <div className="flex flex-col items-center gap-0.5">
              <div className="flex items-center gap-1.5">
                <Skull size={16} style={{ color: C.red }} />
                <span style={{ color: C.slate400, fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Red</span>
              </div>
              <div className="metric-value text-3xl font-bold" style={{ color: C.red }}>
                {s.redScore}%
              </div>
            </div>
            <div className="flex flex-col items-center gap-0.5" style={{ color: C.slate600 }}>
              <span style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em' }}>vs</span>
            </div>
            <div className="flex flex-col items-center gap-0.5">
              <div className="flex items-center gap-1.5">
                <Shield size={16} style={{ color: C.blue }} />
                <span style={{ color: C.slate400, fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Blue</span>
              </div>
              <div className="metric-value text-3xl font-bold" style={{ color: C.blue }}>
                {s.blueScore}%
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Success Rate"
              value={`${rewardPct}%`}
              sub={`p99 ${latency.toFixed(0)}ms`}
              color={srColor}
            />
            <MetricCard
              label="Latency P99"
              value={`${latency.toFixed(0)}ms`}
              color={latColor}
            />
            <MetricCard
              label="Queue Depth"
              value={String(queueDepth)}
              sub={`restarts ${m.worker_restart_count}`}
              color={queueDepth < 10 ? C.blue : queueDepth < 30 ? C.amber : C.red}
            />
            <MetricCard
              label="Reward"
              value={s.reward.toFixed(3)}
              color={s.reward >= 0.5 ? C.emerald : s.reward >= 0.2 ? C.amber : C.red}
            />
          </div>

          <div className="flex-1 flex flex-col gap-2 min-h-0 overflow-y-auto">
            <div className="panel p-2 flex-shrink-0">
              <div className="flex items-center justify-between mb-1 px-1">
                <span style={{ color: C.slate400 }} className="text-[10px] font-semibold tracking-wider uppercase">REWARD %</span>
                {log?.avg_reward != null && (
                  <span style={{ color: C.emerald, fontSize: 10, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    AVG {(log.avg_reward * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <MiniChart data={s.metricsHistory} dataKey="reward" color={C.emerald} type="line" yDomain={[0, 1]} />
            </div>
            <div className="panel p-2 flex-shrink-0">
              <div style={{ color: C.slate400 }} className="text-[10px] font-semibold tracking-wider uppercase mb-1 px-1">P99 LATENCY (ms)</div>
              <MiniChart data={s.metricsHistory} dataKey="p99Latency" color={C.amber} type="area" />
            </div>
            <div className="panel p-2 flex-shrink-0">
              <div style={{ color: C.slate400 }} className="text-[10px] font-semibold tracking-wider uppercase mb-1 px-1">QUEUE DEPTH</div>
              <MiniChart data={s.metricsHistory} dataKey="queueDepth" color={C.blue} type="bar" />
            </div>
          </div>

          {/* services */}
          <div style={{ background: C.slate850, borderRadius: 8, border: `1px solid ${C.slate700}` }} className="p-3 flex-shrink-0">
            <div style={{ color: C.slate400 }} className="text-[10px] font-semibold tracking-wider uppercase mb-2">SERVICES</div>
            <div className="flex items-center gap-3 flex-wrap">
              {Object.entries(s.processStatus).map(([name, status]) => {
                const st = status.toLowerCase();
                const isRunning = st === 'running';
                const isDegraded = st === 'degraded';
                const dotColor = isRunning ? C.emerald : isDegraded ? C.amber : C.red;
                const labelColor = isRunning ? C.slate50 : isDegraded ? C.amber : C.red;
                return (
                  <div key={name} className="flex items-center gap-1.5" style={{ fontSize: 11 }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: dotColor,
                      boxShadow: `0 0 6px ${dotColor}50`,
                    }} />
                    <span style={{ color: labelColor, fontFamily: 'var(--font-mono)' }}>{name}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4, delay: 0.3 }} className="min-h-0">
          <Feed events={s.blueEvents} agent="blue" />
        </motion.div>
      </div>

      {/* ── BOTTOM BAR ── */}
      <motion.footer
        className="flex items-center gap-4 px-5 py-2"
        style={{ background: C.slate900, borderTop: `1px solid ${C.slate700}` }}
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
      >
        <div className="flex items-center gap-1.5">
          <Activity size={12} style={{ color: !s.isInitial ? C.emerald : C.slate500 }} />
          <span style={{ color: !s.isInitial ? C.slate300 : C.slate500, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
            {s.isComplete ? 'EPISODE COMPLETE' : s.isInitial ? 'EPISODE LOADED' : `STEP ${s.stepIndex}`}
          </span>
        </div>


        <div className="ml-auto flex items-center gap-1" style={{ color: C.slate500, fontSize: 10 }}>
          <Cpu size={11} />
          <span>Faultline Dashboard</span>
        </div>
      </motion.footer>
    </div>
  );
}