import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Skull, Shield, ChevronRight, AlertTriangle, CheckCircle, Cpu } from 'lucide-react';

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
};

export interface RunInfo {
  run_id: string;
  task: string;
  red_model: string;
  blue_model: string;
  max_steps: number;
  actual_steps: number;
  success: boolean;
  final_score: number;
  avg_reward: number;
  error: string | null;
}

interface Props {
  onSelect: (runId: string) => void;
}

export default function RunSelector({ onSelect }: Props) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/scenarios/index.json')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((data: RunInfo[]) => setRuns(data))
      .catch(e => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: C.slate950 }}>
        <span style={{ color: C.red, fontFamily: 'var(--font-mono)' }}>Failed to load runs: {error}</span>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: C.slate950 }}>
        <Cpu size={24} className="animate-spin" style={{ color: C.blue }} />
        <span style={{ color: C.slate300, marginLeft: 12, fontFamily: 'var(--font-mono)' }}>Loading runs…</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen" style={{ background: C.slate950 }}>
      <header className="flex items-center gap-4 px-6 py-4" style={{ background: C.slate900, borderBottom: `1px solid ${C.slate700}` }}>
        <h1 className="text-lg font-bold tracking-wide" style={{ color: C.slate50 }}>FAULTLINE</h1>
        <span style={{ color: C.slate500, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
          Episode Replay
        </span>
        <span style={{ color: C.slate600, fontFamily: 'var(--font-mono)', fontSize: 11, marginLeft: 'auto' }}>
          {runs.length} run{runs.length !== 1 ? 's' : ''}
        </span>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-3">
          {runs.map((run, i) => {
            const hasError = !!run.error;
            const shortRed = run.red_model.split('/').pop();
            const shortBlue = run.blue_model.split('/').pop();
            return (
              <motion.button
                key={run.run_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: i * 0.05 }}
                onClick={() => onSelect(run.run_id)}
                className="w-full text-left rounded-lg p-4 transition-colors"
                style={{
                  background: C.slate850,
                  border: `1px solid ${C.slate700}`,
                  cursor: 'pointer',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = C.slate500; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = C.slate700; }}
              >
                <div className="flex items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <Skull size={13} style={{ color: C.red }} />
                      <span style={{ color: C.red, fontWeight: 600, fontSize: 13 }}>{shortRed}</span>
                      <span style={{ color: C.slate500, fontSize: 11 }}>vs</span>
                      <Shield size={13} style={{ color: C.blue }} />
                      <span style={{ color: C.blue, fontWeight: 600, fontSize: 13 }}>{shortBlue}</span>
                    </div>
                    <div className="flex items-center gap-3" style={{ color: C.slate400, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                      <span>{run.actual_steps}/{run.max_steps} steps</span>
                      <span style={{ color: C.slate600 }}>|</span>
                      <span>avg_reward {run.avg_reward.toFixed(4)}</span>
                      <span style={{ color: C.slate600 }}>|</span>
                      <span>score {run.final_score.toFixed(4)}</span>
                      {hasError && (
                        <>
                          <span style={{ color: C.slate600 }}>|</span>
                          <span style={{ color: C.amber }} className="flex items-center gap-1">
                            <AlertTriangle size={11} /> {(run.error || '').length > 60 ? (run.error || '').substring(0, 60) + '…' : run.error}
                          </span>
                        </>
                      )}
                    </div>
                    <div style={{ color: C.slate500, fontSize: 10, marginTop: 2 }}>{run.run_id}</div>
                  </div>
                  <div className="flex items-center gap-3">
                    {run.success ? (
                      <span className="flex items-center gap-1" style={{ color: C.emerald, fontSize: 11 }}>
                        <CheckCircle size={13} /> Success
                      </span>
                    ) : (
                      <span style={{ color: C.slate500, fontSize: 11 }}>Failed</span>
                    )}
                    <ChevronRight size={18} style={{ color: C.slate600 }} />
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>
      </div>
    </div>
  );
}