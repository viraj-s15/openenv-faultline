from dataclasses import dataclass
from pathlib import Path

TRACKED_CONFIGS = [
    Path("gateway/config.json"),
    Path("gateway/blocked_routes.json"),
    Path("auth/config.json"),
    Path("worker/config.json"),
    Path("worker/job_generator_config.json"),
    Path("registry.json"),
]


SERVICE_BY_CONFIG = {
    Path("gateway/config.json"): "gateway",
    Path("gateway/blocked_routes.json"): "gateway",
    Path("auth/config.json"): "auth",
    Path("worker/config.json"): "worker",
    Path("worker/job_generator_config.json"): "job_generator",
    Path("registry.json"): "gateway",
}


@dataclass(frozen=True)
class ConfigBaseline:
    mesh_root: Path
    contents: dict[Path, str]

    @classmethod
    def capture(cls, mesh_root: Path) -> "ConfigBaseline":
        contents = {}
        for relative_path in TRACKED_CONFIGS:
            path = mesh_root / relative_path
            if path.exists():
                contents[relative_path] = path.read_text(encoding="utf-8")
        return cls(mesh_root=mesh_root, contents=contents)

    def restore_modified(self) -> list[Path]:
        restored = []
        for relative_path, baseline in self.contents.items():
            path = self.mesh_root / relative_path
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current == baseline:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(baseline, encoding="utf-8")
            restored.append(path)
        return restored

    def service_for_path(self, path: Path) -> str:
        return SERVICE_BY_CONFIG[path.relative_to(self.mesh_root)]
