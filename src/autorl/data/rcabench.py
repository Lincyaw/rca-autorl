from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autorl.rca_utils import normalize_root_causes


def iter_case_dirs(source_root: str | Path) -> list[Path]:
    root = Path(source_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"source root not found: {root}")
    if (root / "injection.json").exists():
        return [root]

    case_dirs = sorted({path.parent for path in root.rglob("injection.json")})
    if not case_dirs:
        raise ValueError(f"no RCABench case directories found under: {root}")
    return case_dirs


def load_case_bundle(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir).expanduser().resolve()
    injection = _read_json(path / "injection.json")
    env = _read_json(path / "env.json", default={})
    display_config = _parse_maybe_json(injection.get("display_config"))

    ground_truth = injection.get("ground_truth") if isinstance(injection, dict) else None
    target_graph = _load_target_graph(path, ground_truth)
    root_causes = _extract_root_causes(target_graph)

    incident = _build_incident_text(
        case_dir=path,
        injection=injection,
        env=env,
        display_config=display_config,
    )
    sft_prompt = _build_sft_prompt(
        case_dir=path,
        injection=injection,
        env=env,
        display_config=display_config,
    )

    sample_id = str(
        injection.get("injection_name")
        or injection.get("task_id")
        or injection.get("id")
        or path.name
    )
    return {
        "id": sample_id,
        "case_name": path.name,
        "case_dir": str(path),
        "benchmark": str(injection.get("benchmark", "")),
        "incident": incident,
        "question": incident,
        "sft_prompt": sft_prompt,
        "messages": [{"role": "user", "content": incident}],
        "sft_messages": [{"role": "user", "content": sft_prompt}],
        "root_causes": root_causes,
        "answer": target_graph,
        "target_graph": target_graph,
        "metadata": {
            "namespace": env.get("NAMESPACE"),
            "timezone": env.get("TIMEZONE"),
            "abnormal_start": env.get("ABNORMAL_START"),
            "abnormal_end": env.get("ABNORMAL_END"),
            "normal_start": env.get("NORMAL_START"),
            "normal_end": env.get("NORMAL_END"),
            "task_id": injection.get("task_id"),
            "fault_type": injection.get("fault_type"),
            "labels": injection.get("labels", {}),
        },
    }


def build_rl_manifest_samples(source_root: str | Path, max_cases: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for case_dir in iter_case_dirs(source_root)[: max_cases or None]:
        bundle = load_case_bundle(case_dir)
        sample = {
            "id": bundle["id"],
            "case_name": bundle["case_name"],
            "incident": bundle["incident"],
            "question": bundle["question"],
            "data_dir": bundle["case_dir"],
            "root_causes": bundle["root_causes"],
            "answer": bundle["answer"],
            "target_graph": bundle["target_graph"],
            "messages": bundle["messages"],
            **bundle["metadata"],
        }
        samples.append(sample)
    return samples


def build_sft_manifest_samples(source_root: str | Path, max_cases: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for case_dir in iter_case_dirs(source_root)[: max_cases or None]:
        bundle = load_case_bundle(case_dir)
        assistant_response = json.dumps(bundle["target_graph"], ensure_ascii=False, sort_keys=True)
        sample = {
            "id": bundle["id"],
            "case_name": bundle["case_name"],
            "incident": bundle["incident"],
            "question": bundle["sft_prompt"],
            "messages": bundle["sft_messages"],
            "assistant_response": assistant_response,
            "answer": bundle["target_graph"],
            "root_causes": bundle["root_causes"],
            "data_dir": bundle["case_dir"],
            **bundle["metadata"],
        }
        samples.append(sample)
    return samples


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return target


def build_standardized_manifests(
    source_root: str | Path,
    output_dir: str | Path,
    max_cases: int | None = None,
) -> dict[str, str]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_samples = build_rl_manifest_samples(source_root, max_cases=max_cases)
    sft_samples = build_sft_manifest_samples(source_root, max_cases=max_cases)
    if not rl_samples or not sft_samples:
        raise ValueError("no standardized samples were generated")

    rl_path = write_jsonl(out_dir / "rl.jsonl", rl_samples)
    sft_path = write_jsonl(out_dir / "sft.jsonl", sft_samples)
    metadata = {
        "source_root": str(Path(source_root).expanduser().resolve()),
        "output_dir": str(out_dir),
        "num_cases": len(rl_samples),
        "rl_manifest": str(rl_path),
        "sft_manifest": str(sft_path),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rl_manifest": str(rl_path),
        "sft_manifest": str(sft_path),
        "metadata": str(metadata_path),
    }


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _extract_root_causes(ground_truth: Any) -> list[str]:
    return normalize_root_causes(ground_truth)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            lowered = text.lower()
            if text and lowered not in seen:
                result.append(text)
                seen.add(lowered)
        return result
    text = str(value).strip()
    return [text] if text else []


def _build_incident_text(
    *,
    case_dir: Path,
    injection: dict[str, Any],
    env: dict[str, Any],
    display_config: Any,
) -> str:
    injection_name = str(injection.get("injection_name") or case_dir.name)
    benchmark = str(injection.get("benchmark", "unknown"))
    namespace = str(env.get("NAMESPACE", "unknown"))
    abnormal_window = f"{env.get('ABNORMAL_START', '')} -> {env.get('ABNORMAL_END', '')}".strip()
    point_bits: list[str] = []
    if isinstance(display_config, dict):
        point = display_config.get("injection_point")
        if isinstance(point, dict):
            for key in ("app_name", "server_address", "method", "route"):
                value = point.get(key)
                if value:
                    point_bits.append(f"{key}={value}")
    point_summary = ", ".join(point_bits) if point_bits else "unavailable"
    return (
        f"Investigate the RCABench incident '{injection_name}'. "
        f"Benchmark={benchmark}. Namespace={namespace}. "
        f"Abnormal window={abnormal_window}. Injection point: {point_summary}. "
        "Use the provided telemetry data directory to identify the causal graph and root causes."
    )


def _build_sft_prompt(
    *,
    case_dir: Path,
    injection: dict[str, Any],
    env: dict[str, Any],
    display_config: Any,
) -> str:
    incident = _build_incident_text(
        case_dir=case_dir,
        injection=injection,
        env=env,
        display_config=display_config,
    )
    return (
        f"{incident}\n"
        "Return a JSON object with keys: nodes, edges, root_causes, component_to_service. "
        "Keep the output concise and machine-readable."
    )


def _load_target_graph(case_dir: Path, ground_truth: Any) -> dict[str, Any]:
    conclusion_graph = _load_graph_from_conclusion(case_dir / "conclusion.parquet")
    if conclusion_graph is not None:
        return conclusion_graph

    root_causes = _extract_root_causes(ground_truth)
    component_to_service = _build_component_mapping(ground_truth)
    nodes = [
        {"component": service, "state": ["anomalous"], "timestamp": ""}
        for service in root_causes
    ]
    return {
        "nodes": nodes,
        "edges": [],
        "root_causes": nodes,
        "component_to_service": component_to_service,
    }


def _build_component_mapping(ground_truth: Any) -> dict[str, str]:
    if not isinstance(ground_truth, dict):
        return {}
    pods = _normalize_string_list(ground_truth.get("pod"))
    services = _normalize_string_list(ground_truth.get("service"))
    if not pods or not services:
        return {}
    if len(services) == 1:
        return {pod: services[0] for pod in pods}
    mapping: dict[str, str] = {}
    for idx, pod in enumerate(pods):
        mapping[pod] = services[min(idx, len(services) - 1)]
    return mapping


def _load_graph_from_conclusion(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    try:
        table = pq.read_table(path)
        rows = table.to_pylist()
    except Exception:
        return None
    if not rows:
        return None

    for row in rows:
        graph = _extract_graph_from_record(row)
        if graph is not None:
            return graph
    return None


def _extract_graph_from_record(record: Any) -> dict[str, Any] | None:
    if isinstance(record, dict):
        for value in record.values():
            parsed = _parse_maybe_json(value)
            graph = _normalize_graph_candidate(parsed)
            if graph is not None:
                return graph
        root_causes = []
        for key in (
            "root_causes",
            "root_cause",
            "root_cause_services",
            "root_cause_service",
            "service",
            "services",
        ):
            root_causes = _normalize_string_list(record.get(key))
            if root_causes:
                break
        if root_causes:
            nodes = [
                {"component": service, "state": ["anomalous"], "timestamp": ""}
                for service in root_causes
            ]
            return {
                "nodes": nodes,
                "edges": [],
                "root_causes": nodes,
                "component_to_service": {},
            }
    return None


def _normalize_graph_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if not any(key in value for key in ("nodes", "edges", "root_causes", "component_to_service")):
        return None

    def _node_list(items: Any) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    component = (
                        item.get("component")
                        or item.get("service")
                        or item.get("service_name")
                        or item.get("component_name")
                        or item.get("name")
                    )
                    if component:
                        nodes.append(
                            {
                                "component": str(component),
                                "state": list(item.get("state", ["anomalous"])),
                                "timestamp": str(item.get("timestamp", "")),
                            }
                        )
                elif item is not None:
                    nodes.append({"component": str(item), "state": ["anomalous"], "timestamp": ""})
        return nodes

    nodes = _node_list(value.get("nodes"))
    root_causes = _node_list(value.get("root_causes"))
    if not nodes and root_causes:
        nodes = list(root_causes)
    if not root_causes and nodes:
        root_causes = list(nodes)

    edges: list[dict[str, str]] = []
    raw_edges = value.get("edges")
    if isinstance(raw_edges, list):
        for item in raw_edges:
            if isinstance(item, dict):
                source = item.get("source")
                target = item.get("target")
                if source and target:
                    edges.append({"source": str(source), "target": str(target)})

    component_to_service = value.get("component_to_service", {})
    if isinstance(component_to_service, list):
        component_to_service = {
            str(item.get("component_name")): str(item.get("service_name"))
            for item in component_to_service
            if isinstance(item, dict) and item.get("component_name") and item.get("service_name")
        }
    if not isinstance(component_to_service, dict):
        component_to_service = {}

    return {
        "nodes": nodes,
        "edges": edges,
        "root_causes": root_causes,
        "component_to_service": component_to_service,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build standardized RL/SFT manifests from RCABench case directories")
    parser.add_argument("source_root", help="Case directory or root directory containing RCABench cases")
    parser.add_argument("--output-dir", required=True, help="Directory to write rl.jsonl / sft.jsonl manifests")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap on the number of cases to convert")
    args = parser.parse_args(argv)

    outputs = build_standardized_manifests(
        source_root=args.source_root,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
    )
    print(json.dumps(outputs, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
