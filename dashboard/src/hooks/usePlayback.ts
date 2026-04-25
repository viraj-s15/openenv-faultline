import { useState, useEffect, useRef, useCallback } from 'react';
import { getSnapshot, type EpisodeLog, type PlaybackSnapshot } from '../lib/episodeLog';

export interface PlaybackState {
  cursor: number;
  isPlaying: boolean;
  snapshot: PlaybackSnapshot;
}

const INTERVAL_MS = 2500;

const EMPTY_LOG: EpisodeLog = {
  episode_id: '',
  task: '',
  max_steps: 0,
  initial_metrics: { gateway_success_rate: 0, gateway_p99_latency_ms: 0, queue_depth: 0, worker_restart_count: 0, consumer_stall_count: 0 },
  initial_process_status: {},
  steps: [],
};

export function usePlayback(log: EpisodeLog | null) {
  const effective = log ?? EMPTY_LOG;
  const [cursor, setCursor] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const snapshot = getSnapshot(effective, cursor);

  // Reset cursor when log changes
  useEffect(() => {
    setCursor(0);
    setIsPlaying(log !== null);
  }, [log]);

  // Auto-advance when playing
  useEffect(() => {
    if (!isPlaying || snapshot.isComplete || cursor >= effective.max_steps) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = setTimeout(() => {
      setCursor((c) => Math.min(c + 1, effective.steps.length));
    }, INTERVAL_MS);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [isPlaying, cursor, snapshot.isComplete, effective]);

  const play = useCallback(() => {
    if (snapshot.isComplete) {
      setCursor(0);
      setIsPlaying(true);
    } else {
      setIsPlaying(true);
    }
  }, [snapshot.isComplete]);

  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const stepForward = useCallback(() => {
    setIsPlaying(false);
    setCursor((c) => Math.min(c + 1, effective.steps.length));
  }, [effective]);

  const stepBack = useCallback(() => {
    setIsPlaying(false);
    setCursor((c) => Math.max(c - 1, 0));
  }, []);

  const goToStep = useCallback((step: number) => {
    setIsPlaying(false);
    setCursor(Math.max(0, Math.min(step, effective.steps.length)));
  }, [effective]);

  const reset = useCallback(() => {
    setIsPlaying(false);
    setCursor(0);
  }, []);

  const restart = useCallback(() => {
    setCursor(0);
    setIsPlaying(true);
  }, []);

  return {
    isPlaying,
    snapshot,
    play,
    pause,
    stepForward,
    stepBack,
    goToStep,
    reset,
    restart,
  };
}