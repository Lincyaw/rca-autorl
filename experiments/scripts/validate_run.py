"""Validate experiment run record completeness and traceability."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def validate_run(run_dir: Path) -> list[str]:
    """Return list of validation errors (empty = all good)."""
    errors: list[str] = []

    # --- meta.yaml ---
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        errors.append("missing meta.yaml")
        return errors

    if yaml is not None:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    else:
        # Fallback: just check file is non-empty
        meta = {"_raw": meta_path.read_text(encoding="utf-8")}
        if not meta["_raw"].strip():
            errors.append("meta.yaml is empty")
            return errors
        # Can't validate fields without yaml parser
        print("  [warn] pyyaml not installed, skipping field validation")
        meta = None

    if meta is not None:
        required_fields = ["run_id", "plan", "date", "status"]
        for field in required_fields:
            value = meta.get(field)
            if not value or (isinstance(value, str) and not value.strip()):
                errors.append(f"meta.yaml missing required field: {field}")

        code = meta.get("code", {})
        if isinstance(code, dict):
            commit = code.get("commit", "")
            if not commit or not str(commit).strip():
                errors.append("meta.yaml code.commit is empty")
        else:
            errors.append("meta.yaml code section missing or malformed")

        config = meta.get("config", {})
        if isinstance(config, dict):
            config_file = config.get("file", "")
            if not config_file or not str(config_file).strip():
                errors.append("meta.yaml config.file is empty")
        else:
            errors.append("meta.yaml config section missing or malformed")

    # --- metrics.json ---
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        errors.append("missing metrics.json")
    else:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            non_null = {k: v for k, v in metrics.items() if v is not None and not k.startswith("_")}
            if not non_null:
                errors.append("metrics.json has no filled values (all null)")
            for key, value in non_null.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"metrics.json[{key}] is not numeric: {type(value).__name__}")
        except json.JSONDecodeError as exc:
            errors.append(f"metrics.json is not valid JSON: {exc}")

    # --- notes.md ---
    notes_path = run_dir / "notes.md"
    if not notes_path.exists():
        errors.append("missing notes.md")

    return errors


def validate_traceability(run_dir: Path) -> list[str]:
    """Cross-check meta.yaml references against repo state."""
    errors: list[str] = []
    meta_path = run_dir / "meta.yaml"
    if yaml is None or not meta_path.exists():
        return errors

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    code = meta.get("code", {})
    if not isinstance(code, dict):
        return errors

    commit = str(code.get("commit", "")).strip()
    if commit:
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            errors.append(f"code.commit {commit} not found in git history")

    config = meta.get("config", {})
    config_file = str(config.get("file", "")).strip() if isinstance(config, dict) else ""
    if config_file and commit:
        result = subprocess.run(
            ["git", "show", f"{commit}:{config_file}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            errors.append(f"config.file {config_file} not found at commit {commit}")

    return errors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate experiment run records")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to validate")
    parser.add_argument("--trace", action="store_true", help="Also check traceability")
    args = parser.parse_args()

    total_errors = 0
    for path_str in args.run_dirs:
        run_dir = Path(path_str).resolve()
        if not run_dir.is_dir():
            print(f"SKIP {run_dir} (not a directory)")
            continue

        errors = validate_run(run_dir)
        if args.trace:
            errors.extend(validate_traceability(run_dir))

        status = "PASS" if not errors else "FAIL"
        print(f"{status} {run_dir.name}")
        for error in errors:
            print(f"  - {error}")
        total_errors += len(errors)

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
