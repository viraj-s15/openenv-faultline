"""Trainer callbacks: curriculum hard-switch and W&B artifact logging."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import wandb
except ImportError:  # pragma: no cover - optional dep
    wandb = None

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - exercised when training extras are absent
    class TrainerCallback:  # type: ignore[no-redef]
        """Minimal stub so this module imports without transformers installed."""

        def on_step_begin(self, args, state, control, **kwargs): ...

        def on_save(self, args, state, control, **kwargs): ...

        def on_train_end(self, args, state, control, **kwargs): ...


def _stage_for_step(schedule: Sequence[dict[str, Any]], step: int) -> int:
    """Return the schedule index whose `until_step` covers `step`."""
    for idx, stage in enumerate(schedule):
        if step <= int(stage["until_step"]):
            return idx
    return max(0, len(schedule) - 1)


class CurriculumCallback(TrainerCallback):
    """Hard-switches `trainer.train_dataset` when the curriculum stage changes.

    On every step the callback compares the current schedule index against the
    one used to build the active dataset; on a change it asks the dataset_builder
    to produce a new `Dataset` for the current stage and swaps it onto the
    trainer instance.
    """

    def __init__(
        self,
        schedule: Sequence[dict[str, Any]],
        dataset_builder: Callable[[Sequence[str]], Any],
        trainer_ref: Callable[[], Any],
        initial_step: int = 0,
    ) -> None:
        self._schedule = list(schedule)
        self._build_dataset = dataset_builder
        self._trainer_ref = trainer_ref
        self._active_stage = _stage_for_step(self._schedule, initial_step)

    def on_step_begin(self, args, state, control, **kwargs):
        if not self._schedule:
            return control
        next_stage = _stage_for_step(self._schedule, int(state.global_step))
        if next_stage == self._active_stage:
            return control
        tasks = [str(t) for t in self._schedule[next_stage]["tasks"]]
        new_dataset = self._build_dataset(tasks)
        trainer = self._trainer_ref()
        if trainer is not None and new_dataset is not None:
            trainer.train_dataset = new_dataset
        self._active_stage = next_stage
        return control


class WandbArtifactCallback(TrainerCallback):
    """Logs the trainer output dir as a W&B artifact at every checkpoint and at end of training."""

    def __init__(self, artifact_name: str, artifact_type: str = "model") -> None:
        self._artifact_name = artifact_name
        self._artifact_type = artifact_type

    def _wandb_active(self) -> bool:
        if wandb is None:
            return False
        if os.environ.get("WANDB_MODE", "online") == "disabled":
            return False
        return wandb.run is not None

    def _log_dir(self, dir_path: Path, alias: str) -> None:
        if not self._wandb_active() or not dir_path.exists():
            return
        artifact = wandb.Artifact(self._artifact_name, type=self._artifact_type)
        artifact.add_dir(str(dir_path))
        wandb.log_artifact(artifact, aliases=[alias])

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        if ckpt_dir.exists():
            self._log_dir(ckpt_dir, alias=f"step-{int(state.global_step)}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._log_dir(Path(args.output_dir), alias="final")
        return control
