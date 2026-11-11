"""Prepare the complete xLAM source, retaining all records and M1 assignments.

Original encoded arrays are embedded verbatim to preserve numeric literals.
Quality flags and gold-derived task labels are stored in a separate sidecar.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import same_value, strict_json
from task_types import describe
from audit_full_data import audit
from data_utils import normalized_query, query_template

VERSION = "xlam_full_v1"
SPLITS = ("train", "dev", "test")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def group_id(record):
    return hashlib.sha256(query_template(record["query"]).encode("utf-8")).hexdigest()


def anchors_from_archive(path):
    anchors = {}
    with tarfile.open(path, "r:gz") as archive:
        for split in SPLITS:
            member = archive.getmember(f"data/{split}.jsonl")
            if not member.isfile():
                raise ValueError("M1 split is not a regular archive member")
            for line in archive.extractfile(member):
                row = json.loads(line)
                if row["id"] in anchors:
                    raise ValueError("Repeated M1 anchor ID")
                anchors[row["id"]] = split
    return anchors


def allocate(records, anchors, seed):
    groups = defaultdict(list)
    seen = set()
    for record in records:
        if type(record["id"]) is not int or record["id"] in seen:
            raise ValueError("Invalid or repeated source ID")
        seen.add(record["id"])
        groups[group_id(record)].append(record)
    if anchors.keys() - seen:
        raise ValueError("M1 archive contains IDs absent from source")
    n = len(records)
    targets = {"train": round(.8 * n), "dev": round(.1 * n)}
    targets["test"] = n - sum(targets.values())
    partitions, group_splits = {s: [] for s in SPLITS}, {}
    unassigned = []
    for key, rows in groups.items():
        locked = {anchors[r["id"]] for r in rows if r["id"] in anchors}
        if len(locked) > 1:
            raise ValueError("A query group has conflicting M1 assignments")
        if locked:
            split = locked.pop()
            partitions[split].extend(rows)
            group_splits[key] = split
        else:
            unassigned.append((key, rows))
    random.Random(seed).shuffle(unassigned)
    for key, rows in unassigned:
        split = max(SPLITS, key=lambda s: targets[s] - len(partitions[s]))
        partitions[split].extend(rows)
        group_splits[key] = split
    for index, split in enumerate(SPLITS):
        random.Random(seed + index + 1).shuffle(partitions[split])
    return partitions, group_splits, targets


def source_line(record):
    # Both columns in the original release are JSON text. Embedding them avoids
    # rounding decimal values through a float load/dump cycle.
    if not all(isinstance(record[k], str) for k in ("tools", "answers")):
        raise ValueError("Expected the original JSON-encoded xLAM columns")
    for key in ("tools", "answers"):
        if not isinstance(strict_json(record[key]), list):
            raise ValueError(f"Source {key} must decode to an array")
    return (f'{{"id":{record["id"]},"query":' + json.dumps(record["query"], ensure_ascii=False)
            + ',"tools":' + record["tools"] + ',"answers":' + record["answers"] + '}\n')


def verify(source, output, anchors):
    by_id = {r["id"]: r for r in source}
    queries, templates, ids = {}, {}, {}
    for split in SPLITS:
        ids[split], queries[split], templates[split] = set(), set(), set()
        with (output / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = strict_json(line)
                key = row["id"]
                if key in ids[split]:
                    raise ValueError("Duplicate ID within a split")
                expected = by_id[key]
                if row["query"] != expected["query"] or any(not same_value(row[k], strict_json(expected[k])) for k in ("tools", "answers")):
                    raise ValueError(f"Source content changed: {key}")
                if key in anchors and anchors[key] != split:
                    raise ValueError("M1 assignment changed")
                ids[split].add(key)
                queries[split].add(normalized_query(row["query"]))
                templates[split].add(query_template(row["query"]))
    if set.union(*ids.values()) != by_id.keys() or sum(map(len, ids.values())) != len(source):
        raise ValueError("Source coverage or disjointness failed")
    overlaps = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{left}_{right}"] = {"ids": len(ids[left] & ids[right]), "normalized_queries": len(queries[left] & queries[right]),
                                         "recognized_templates": len(templates[left] & templates[right])}
        if any(overlaps[f"{left}_{right}"].values()):
            raise ValueError("Cross-split overlap")
    observed_meta = {}
    with (output / "metadata.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in observed_meta:
                raise ValueError("Duplicate metadata ID")
            observed_meta[row["id"]] = row
            expected = by_id[row["id"]]
            if row["id"] not in ids[row["split"]] or row["group_id"] != group_id(expected):
                raise ValueError("Metadata assignment mismatch")
            if any(row.get(k) != v for k, v in describe(expected).items()):
                raise ValueError("Metadata task label mismatch")
    if observed_meta.keys() != by_id.keys():
        raise ValueError("Metadata coverage mismatch")
    return {"passed": True, "source_content_preserved": len(source), "m1_assignments_preserved": len(anchors), "cross_split_overlap": overlaps}


def documentation(manifest, profile):
    lines = ["# Data", "",
             "We use the complete [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) release from [APIGen](https://arxiv.org/abs/2406.18518v1), licensed CC BY 4.0. All 60,000 synthetic requests are retained, with their original candidate definitions, order, numeric values, and full call arrays. Compound requests remain intact.", "",
             f"Active version: `{VERSION}`. Source and generated-file hashes are recorded in [the split summary](../analysis/full_split_summary.json).", "",
             "## Splits", "", "| Split | Records | Flagged records |", "|---|---:|---:|"]
    for split in SPLITS:
        entry = manifest["splits"][split]
        lines.append(f"| {split} | {entry['records']:,} | {entry['flagged_records']:,} |")
    lines += ["", f"Seed {manifest['seed']}; {manifest['query_groups']:,} request-template groups are assigned toward 80/10/10. All {manifest['verification']['m1_assignments_preserved']:,} M1 IDs keep their original split, along with other members of their recognized groups. Remaining groups are shuffled and assigned to the largest target deficit.", "",
              "Verification checks all source content and IDs, metadata consistency, and zero cross-split normalized-request/template overlap. Tools may recur across splits; this tests new requests rather than enforcing unseen tools. The grouping rule cannot identify every paraphrase.", "",
              "## Format and storage", "",
              "`data/xlam_full_v1/{train,dev,test}.jsonl` uses UTF-8 JSON Lines. Each row contains `id`, `query`, the complete `tools` array, and the complete `answers` array. Calls contain `name` and `arguments`. Original encoded arrays are embedded directly to preserve numeric literals.", "",
              "The separate `metadata.jsonl` holds split/group IDs, reference-derived task labels, and quality flags. It is for offline analysis and must not enter model inputs. `manifest.json` records provenance and file hashes. Data files are ignored by Git; the original download and M1 submission remain unchanged.", "",
              "## Corpus profile", "",
              f"The source has {profile['unique_candidate_names']:,} tool names and {profile['unique_candidate_schemas']:,} distinct full schemas. Name equality alone does not establish schema equality.", "",
              "| Calls per request | Records |", "|---|---:|"]
    call_counts = {int(k): v for k, v in profile["reference_call_count_histogram"].items()}
    for count in (1, 2, 3):
        lines.append(f"| {count} | {call_counts.get(count, 0):,} |")
    lines += [f"| 4 or more | {sum(v for k, v in call_counts.items() if k >= 4):,} |", "",
              "| Candidate tools | Records |", "|---|---:|"]
    for count, total in sorted((int(k), v) for k, v in profile["candidate_count_histogram"].items()):
        lines.append(f"| {count} | {total:,} |")
    lines += ["", "| Length in characters | Median | P95 | Maximum |", "|---|---:|---:|---:|"]
    for name, values in profile["lengths"].items():
        lines.append(f"| {name} | {values['median']:g} | {values['p95']:g} | {values['max']:g} |")
    lines += ["", "Lengths use compact JSON before prompts and chat templates. Measure actual tokens with the selected model before choosing context limits; long call lists remain in the data.", "",
              "## Quality checks", "",
              f"The audit flags {profile['flagged_record_count']:,} records for recognized type conflicts or repeated candidate names. A flag may identify a schema issue rather than a wrong answer. All flagged records remain; report flagged/unflagged diagnostics and version any later corrections.", "",
              f"The conservative task fingerprint finds {profile['duplicate_complete_task_extra_records']:,} copies beyond the first occurrence. They remain in the data but share their request group. The largest recognized group contains {profile['max_query_template_group_size']} records. Exact distributions and per-record flags are in [full_data_profile.json](../analysis/full_data_profile.json).", "",
              "Evaluation reports single/multiple candidates and calls, repeated/distinct tools, and unused candidates separately. See [scoring.md](../scoring.md) for definitions. The earlier 98.1% lexical-selection diagnostic applies only to the historical M1 subset; no baseline scores are available for the full release.", "",
              "## Reproduction", "", "Run from the repository root. Create the files once:", "", "```powershell",
              "python scripts/prepare_full_data.py --input ../xlam_function_calling_60k.json --m1-archive ../ms1/data.tar.gz --output data/xlam_full_v1 --summary analysis/full_split_summary.json --documentation docs/data.md", "```", "",
              "For existing files, verify without rewriting:", "", "```powershell",
              "python scripts/prepare_full_data.py --input ../xlam_function_calling_60k.json --m1-archive ../ms1/data.tar.gz --output data/xlam_full_v1 --verify-only", "```", "",
              "To refresh the detailed source audit:", "", "```powershell",
              "python scripts/audit_full_data.py --input ../xlam_function_calling_60k.json --output analysis", "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--m1-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--documentation", type=Path)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    inputs = {args.input.resolve(), args.m1_archive.resolve()}
    outputs = [(args.output / f).resolve() for f in ("train.jsonl", "dev.jsonl", "test.jsonl", "metadata.jsonl", "manifest.json")]
    outputs.extend(p.resolve() for p in (args.summary, args.documentation) if p is not None)
    if inputs.intersection(outputs) or len(outputs) != len(set(outputs)):
        parser.error("Output paths must be distinct and must not overwrite source inputs")
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    anchors = anchors_from_archive(args.m1_archive)
    if args.verify_only:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != sha(args.input) or manifest["m1_archive_sha256"] != sha(args.m1_archive):
            raise ValueError("Source or anchor hash changed")
        for file, digest in manifest["files_sha256"].items():
            if sha(args.output / file) != digest:
                raise ValueError(f"Generated file changed: {file}")
        print(json.dumps(verify(source, args.output, anchors)))
        return
    if any((args.output / file).exists() for file in ("train.jsonl", "dev.jsonl", "test.jsonl", "metadata.jsonl", "manifest.json")):
        parser.error("Generated files already exist; choose another output directory or use --verify-only")
    profile = audit(source)
    flags = {r["id"]: r["flags"] for r in profile["flagged_records"]}
    partitions, group_splits, targets = allocate(source, anchors, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with (args.output / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as metadata:
        for split, rows in partitions.items():
            category_counts, pattern_counts = Counter(), Counter()
            with (args.output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(source_line(row))
                    labels = describe(row)
                    category_counts[labels["category"]] += 1
                    pattern_counts[labels["call_pattern"]] += 1
                    entry = {"id": row["id"], "split": split, "group_id": group_id(row),
                             **labels, "quality_flags": flags.get(row["id"], [])}
                    metadata.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            summaries[split] = {"records": len(rows), "flagged_records": sum(r["id"] in flags for r in rows),
                                "categories": dict(category_counts), "call_patterns": dict(pattern_counts)}
    manifest = {"version": VERSION, "seed": args.seed, "source_sha256": sha(args.input), "m1_archive_sha256": sha(args.m1_archive),
                "target_counts": targets, "splits": summaries, "query_groups": len(group_splits),
                "grouping": "M1 NFKC/casefold/whitespace normalization and conservative quote/number masking",
                "allocation": "preserve all M1 group assignments; seeded shuffle of remaining groups; largest remaining count deficit",
                "records_removed": 0, "records_rewritten": 0,
                "files_sha256": {f: sha(args.output / f) for f in ("train.jsonl", "dev.jsonl", "test.jsonl", "metadata.jsonl")},
                "verification": verify(source, args.output, anchors)}
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (args.output / "manifest.json").write_text(rendered, encoding="utf-8")
    for path, content in ((args.summary, rendered), (args.documentation, documentation(manifest, profile))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(json.dumps({"splits": {s: v["records"] for s, v in summaries.items()}, "groups": len(group_splits), "verification": manifest["verification"]}))


if __name__ == "__main__":
    main()
