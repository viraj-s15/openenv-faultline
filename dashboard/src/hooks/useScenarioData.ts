import { useState, useEffect } from 'react';
import type { EpisodeLog } from '../lib/episodeLog';

export function useScenarioData(runId: string | null) {
  const [log, setLog] = useState<EpisodeLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) { setLog(null); return; }
    let cancelled = false;
    setLog(null);
    setError(null);
    fetch(`/scenarios/${runId}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load scenario: ${res.status}`);
        return res.json() as Promise<EpisodeLog>;
      })
      .then((data) => {
        if (!cancelled) setLog(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, [runId]);

  return { log, error, loading: log === null && error === null };
}