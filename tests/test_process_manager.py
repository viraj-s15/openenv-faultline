import os
import subprocess

from wargames_env.server.process_manager import ProcessManager


def test_is_pid_alive_treats_zombie_process_as_dead(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="Z\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ProcessManager._is_pid_alive(1234) is False


def test_is_pid_alive_accepts_running_process(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="Sl\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ProcessManager._is_pid_alive(1234) is True
