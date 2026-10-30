"""Record the project's resolved dependency closure without unrelated local packages."""

from importlib.metadata import distribution
from pathlib import Path
import platform

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    pending = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "wandb", "pytest", "huggingface_hub"]
    packages = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in packages:
            continue
        installed = distribution(name)
        packages[name] = installed.version
        for raw_requirement in installed.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    destination = Path("artifacts/environment-lock.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Captured on Python {platform.python_version()}, {platform.system()} {platform.machine()}.",
             "# Resolved project dependencies; CUDA wheels require the matching PyTorch index.",
             "--extra-index-url https://download.pytorch.org/whl/cu121",
             *[f"{name}=={version}" for name, version in sorted(packages.items())]]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recorded {len(packages)} packages in {destination}")


if __name__ == "__main__":
    main()
