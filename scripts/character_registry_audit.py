"""Character registry audit v2.

Answers four questions about the existing annotated corpus, using
characters.yaml as the reference universe:

  A. BLAST RADIUS  - what did historical text rewriting actually change?
                     (diff raw_text vs preprocessed_text for every unit;
                     classify substitutions: safe expansion, descriptor
                     substitution, mangling, possessive shift)
  B. EXCLUSION GAPS - which characters are mentioned in a unit's ORIGINAL
                     text but absent from characters_present? (Rachel-class)
  C. PHANTOMS       - which characters_present have no surface support in the
                     original text? (Cottard-class if the support only exists
                     in the rewritten text; otherwise possible coreference)
  D. CHIMERAS       - targeted adjudication: Octave/tante Léonie, and the four
                     unit-annotations listing M. de Marsantes as present.

Plus: an inventory of honorific-pattern names in the corpus that no registry
entity claims (candidate missing characters, ranked by frequency).

Unit selection: "latest" = highest run number holding an annotation for the
unit. This mirrors the propose/accept rerun pattern; flagged in the report so
it can be aligned with build_chapter_overlay_data's selection if they differ.

Usage:  python scripts/character_registry_audit.py [--repo PATH]
Writes: outputs/character-registry-audit-v2.md / .json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from proust.registry import Registry, normalize_text  # noqa: E402

DESCRIPTOR_LEAD = re.compile(
    r"^(le|la|les|l'|un|une|du|de|des)\s", re.IGNORECASE
)
POSSESSIVE_LEAD = re.compile(r"^(ma|mon|mes|sa|son|ses|notre|votre|leur)\s",
                             re.IGNORECASE)
MANGLE = re.compile(r"\b(M\.|Mme|Mlle|Madame)\s+de\s+(M\.|Mme|Mlle|Madame)\s")

HONORIFIC = re.compile(
    r"(?:M\.|Mme|Madame|Mlle|Mademoiselle|Mgr|baronne?|duc(?:hesse)?|"
    r"princesse?|marquise?|comtesse?|vicomtesse?|docteur|maître|général|"
    r"colonel|capitaine|abbé|oncle|tante)"
    r"\s+(?:de\s+|d'|du\s+|des\s+)?[A-ZÀ-Þ][\w'\u2019-]+"
    r"(?:[-\s]+[A-ZÀ-Þ][\w'\u2019-]+)?"
)

PLACE_STOPLIST = {
    "Paris", "Balbec", "Combray", "France", "Venise", "Doncières", "Rivebelle",
    "Méséglise", "Montjouvain", "Tansonville", "Roussainville", "Martinville",
    "Hudimesnil", "Incarville", "Parville", "Maineville", "Guermantes",
    "Champs-Élysées", "Bois", "Boulogne", "Normandie", "Bretagne", "Florence",
    "Padoue", "Trocadéro", "Opéra", "Odéon", "Sorbonne", "Figaro", "Gaulois",
    "Dieu", "Monsieur", "Madame", "Mademoiselle", "Saint-André", "Saint-Hilaire",
    "Saint-Esprit", "Notre-Dame", "Vivonne",
}


def run_number(run_id: str) -> int:
    return int(run_id.rsplit("-", 1)[1])


def load_corpus(repo: Path):
    units, annotations = {}, {}
    for path in sorted(repo.glob("outputs/run-*/units/*.json")):
        run_id = path.parts[-3]
        unit = json.loads(path.read_text(encoding="utf-8"))
        units[(run_id, unit["unit_id"])] = unit
    for path in sorted(repo.glob("outputs/run-*/annotations/*.json")):
        run_id = path.parts[-3]
        annotation = json.loads(path.read_text(encoding="utf-8"))
        annotations[(run_id, annotation["unit_id"])] = annotation
    latest = {}
    for (run_id, unit_id) in annotations:
        current = latest.get(unit_id)
        if current is None or run_number(run_id) > run_number(current):
            latest[unit_id] = run_id
    return units, annotations, latest


# ---------------------------------------------------------------- A. rewriting

def diff_substitutions(raw: str, preprocessed: str):
    """Phrase-level substitutions between raw and preprocessed text.

    difflib fragments a name expansion ("Charlus" -> "baron de Charlus") into
    a bare insert of "baron de". To recover the actual rule, adjacent non-equal
    opcodes separated by short equal runs are merged, and one equal token of
    context is kept on each side.
    """
    raw_tokens = raw.split()
    pre_tokens = preprocessed.split()
    matcher = difflib.SequenceMatcher(a=raw_tokens, b=pre_tokens, autojunk=False)
    opcodes = [op for op in matcher.get_opcodes()]
    groups, current = [], []
    for index, (op, a1, a2, b1, b2) in enumerate(opcodes):
        if op == "equal":
            if current and (a2 - a1) <= 3 and index != len(opcodes) - 1:
                current.append(opcodes[index])
            elif current:
                groups.append(current)
                current = []
        else:
            current.append(opcodes[index])
    if current:
        groups.append(current)
    for group in groups:
        while group and group[-1][0] == "equal":
            group.pop()
        if not group:
            continue
        a1, b1 = group[0][1], group[0][3]
        a2, b2 = group[-1][2], group[-1][4]
        core = (" ".join(raw_tokens[a1:a2]), " ".join(pre_tokens[b1:b2]))
        if a1 > 0 and b1 > 0:
            a1, b1 = a1 - 1, b1 - 1
        if a2 < len(raw_tokens) and b2 < len(pre_tokens):
            a2, b2 = a2 + 1, b2 + 1
        contextual = (" ".join(raw_tokens[a1:a2]), " ".join(pre_tokens[b1:b2]))
        yield core, contextual


PUNCT = " «»\"'’,;.:!?—–-()"


def _is_subsequence(short: list, long: list) -> bool:
    iterator = iter(long)
    return all(token in iterator for token in short)


def classify_substitution(core: tuple, contextual: tuple) -> str:
    core_orig = core[0].strip(PUNCT)
    core_new = core[1].strip(PUNCT)
    ctx_orig = contextual[0].strip(PUNCT)
    ctx_new = contextual[1].strip(PUNCT)
    if not core_orig and not core_new:
        return "typographic"
    if MANGLE.search(ctx_new):
        return "mangled"
    if not core_orig:  # pure insertion beside preserved tokens
        return "expansion"
    if not core_new:
        return "deletion"
    if POSSESSIVE_LEAD.match(core_orig):
        return "possessive_shift"
    if DESCRIPTOR_LEAD.match(core_orig) or core_orig[0].islower():
        return "descriptor_substitution"
    if _is_subsequence(ctx_orig.split(), ctx_new.split()):
        return "expansion"
    orig_tail = ctx_orig.split()[-1] if ctx_orig.split() else ""
    new_tail = ctx_new.split()[-1] if ctx_new.split() else ""
    if orig_tail and orig_tail == new_tail:
        return "variant_flattening"
    return "other"


def audit_rewriting(units, annotations):
    rule_counts = Counter()
    rule_class = {}
    rule_examples = defaultdict(list)
    corrupted_annotated_units = set()
    rewritten_units = 0
    for (run_id, unit_id), unit in units.items():
        raw = unit.get("raw_text") or ""
        pre = unit.get("preprocessed_text") or ""
        if not raw or raw == pre:
            continue
        rewritten_units += 1
        for core, contextual in diff_substitutions(raw, pre):
            key = contextual
            rule_counts[key] += 1
            kind = classify_substitution(core, contextual)
            rule_class[key] = kind
            if len(rule_examples[key]) < 3:
                rule_examples[key].append(f"{run_id}/{unit_id}")
            if kind in ("mangled", "descriptor_substitution"):
                if (run_id, unit_id) in annotations:
                    corrupted_annotated_units.add(("severe", run_id, unit_id))
            elif kind in ("possessive_shift", "variant_flattening"):
                if (run_id, unit_id) in annotations:
                    corrupted_annotated_units.add((kind, run_id, unit_id))
    severity_units = defaultdict(set)
    for kind, run_id, unit_id in corrupted_annotated_units:
        severity_units[kind].add(f"{run_id}/{unit_id}")
    return {
        "rewritten_unit_files": rewritten_units,
        "distinct_substitutions": len(rule_counts),
        "annotated_units_by_severity": {
            kind: sorted(paths) for kind, paths in severity_units.items()
        },
        "rules": [
            {
                "orig": orig, "new": new, "count": count,
                "class": rule_class[(orig, new)],
                "examples": rule_examples[(orig, new)],
            }
            for (orig, new), count in rule_counts.most_common()
        ],
    }


# ------------------------------------------------------- B/C. gaps & phantoms

def audit_presence(registry, units, annotations, latest):
    scanner = registry.compile_scanner()
    gap_agg = defaultdict(lambda: {"units": 0, "strong_units": 0,
                                   "mentions": 0, "examples": []})
    phantom_agg = defaultdict(lambda: {"substitution_induced": 0,
                                       "no_surface_match": 0, "examples": []})
    unresolved_canonicals = Counter()
    ambiguous_surfaces = Counter()
    units_scanned = 0

    for unit_id, run_id in sorted(latest.items()):
        annotation = annotations[(run_id, unit_id)]
        unit = units.get((run_id, unit_id))
        if unit is None:
            continue
        units_scanned += 1
        chapter_id = unit.get("chapter_id")
        raw_mentions = scanner.scan(unit.get("raw_text") or "", chapter_id)
        pre_mentions = scanner.scan(unit.get("preprocessed_text") or "", chapter_id)
        for key in [k for k in raw_mentions if k.startswith("__ambiguous__:")]:
            ambiguous_surfaces[key.split(":", 1)[1]] += len(raw_mentions.pop(key))
        for key in [k for k in pre_mentions if k.startswith("__ambiguous__:")]:
            pre_mentions.pop(key)

        present = set()
        for character in annotation.get("characters_present", []):
            resolution = registry.resolve(character.get("canonical_name", ""),
                                          chapter_id=chapter_id)
            if resolution.status == "resolved":
                present.add(resolution.entity_id)
            else:
                unresolved_canonicals[character.get("canonical_name", "?")] += 1

        for entity_id, hits in raw_mentions.items():
            if entity_id in present or entity_id == "le-narrateur":
                continue
            entry = gap_agg[entity_id]
            entry["units"] += 1
            entry["mentions"] += len(hits)
            if len(hits) >= 2:
                entry["strong_units"] += 1
            if len(entry["examples"]) < 6:
                entry["examples"].append(
                    {"unit": f"{run_id}/{unit_id}", "hits": len(hits),
                     "forms": sorted(set(hits))}
                )

        for entity_id in present:
            if raw_mentions.get(entity_id):
                continue
            entry = phantom_agg[entity_id]
            if pre_mentions.get(entity_id):
                entry["substitution_induced"] += 1
                tag = "substitution_induced"
            else:
                entry["no_surface_match"] += 1
                tag = "no_surface_match"
            if len(entry["examples"]) < 6:
                entry["examples"].append({"unit": f"{run_id}/{unit_id}",
                                          "kind": tag})

    display = {e.id: e.display_name for e in registry.entities.values()}
    gaps = sorted(
        ({"entity": display.get(k, k), "id": k, **v} for k, v in gap_agg.items()),
        key=lambda row: (-row["strong_units"], -row["mentions"]),
    )
    phantoms = sorted(
        ({"entity": display.get(k, k), "id": k, **v}
         for k, v in phantom_agg.items()),
        key=lambda row: -(row["substitution_induced"] + row["no_surface_match"]),
    )
    return {
        "latest_units_scanned": units_scanned,
        "exclusion_gaps": gaps,
        "phantoms": phantoms,
        "unresolved_annotation_canonicals": dict(unresolved_canonicals),
        "ambiguous_surface_hits": dict(ambiguous_surfaces),
    }


# ------------------------------------------------------------- D. chimeras

def audit_chimeras(units, annotations):
    marsantes, octave = [], []
    pattern = re.compile(r"(M(?:me|adame)?\.?)\s+de\s+Marsantes")
    for (run_id, unit_id), annotation in sorted(annotations.items()):
        names = [c.get("canonical_name")
                 for c in annotation.get("characters_present", [])]
        unit = units.get((run_id, unit_id)) or {}
        raw = normalize_text(unit.get("raw_text") or "")
        if "M. de Marsantes" in names:
            excerpts = []
            for match in pattern.finditer(raw):
                start = max(0, match.start() - 90)
                excerpts.append(
                    {"honorific": match.group(1),
                     "context": "…" + raw[start:match.end() + 90] + "…"}
                )
            marsantes.append({"unit": f"{run_id}/{unit_id}",
                              "raw_mentions": excerpts})
        if "Octave" in names:
            bare = len(re.findall(r"(?<!Mme )(?<!Madame )\bOctave\b", raw))
            addressed = len(re.findall(r"\b(?:Mme|Madame) Octave\b", raw))
            octave.append({"unit": f"{run_id}/{unit_id}",
                           "bare_octave_mentions": bare,
                           "mme_octave_mentions": addressed})
    return {"m_de_marsantes_units": marsantes, "octave_units": octave}


# ------------------------------------------------- candidate missing persons

def audit_candidates(registry, repo: Path):
    known = {normalize_text(f.form) for f in registry.forms}
    counts = Counter()
    for chapter_path in sorted(
        repo.glob("data/islt/editions/fr-original/chapters/*.json")
    ):
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        for paragraph in chapter.get("paragraphs", []):
            text = normalize_text(paragraph.get("text", ""))
            for match in HONORIFIC.finditer(text):
                name = match.group(0).strip()
                if name in known:
                    continue
                tail = name.split(None, 1)[-1]
                if tail in PLACE_STOPLIST or tail in known:
                    continue
                counts[name] += 1
    return [{"name": name, "count": count}
            for name, count in counts.most_common() if count >= 3]


# ----------------------------------------------------------------- reporting

def write_markdown(path: Path, report: dict):
    lines = ["# Character Registry Audit v2", ""]
    rewriting = report["rewriting"]
    lines += [
        "## A. Historical text-rewriting blast radius",
        "",
        f"- Unit files where preprocessed differs from raw: "
        f"**{rewriting['rewritten_unit_files']}**",
        f"- Distinct substitution rules observed: "
        f"**{rewriting['distinct_substitutions']}**",
        f"- Annotated unit files with SEVERE substitutions "
        f"(descriptor/mangled): "
        f"**{len(rewriting['annotated_units_by_severity'].get('severe', []))}**",
        f"- Annotated unit files with possessive normalization only: "
        f"**{len(rewriting['annotated_units_by_severity'].get('possessive_shift', []))}**",
        f"- Annotated unit files with name-variant flattening "
        f"(e.g. Mme de Guermantes -> duchesse de Guermantes): "
        f"**{len(rewriting['annotated_units_by_severity'].get('variant_flattening', []))}**",
        "",
        "| class | orig | new | count | example |",
        "| --- | --- | --- | --- | --- |",
    ]
    interesting = [r for r in rewriting["rules"]
                   if r["class"] != "expansion"][:40]
    for rule in interesting:
        lines.append(
            f"| {rule['class']} | `{rule['orig'][:60]}` | `{rule['new'][:60]}` "
            f"| {rule['count']} | {rule['examples'][0]} |"
        )
    lines += ["", f"(showing {len(interesting)} non-expansion rules; full list "
                  f"in the JSON)", ""]

    presence = report["presence"]
    lines += [
        "## B. Exclusion gaps (mentioned in original text, absent from "
        "characters_present)",
        "",
        f"Latest-run annotations scanned: {presence['latest_units_scanned']} "
        f"(selection rule: highest run number per unit)",
        "",
        "| entity | units w/ gap | strong (≥2 mentions) | total mentions |",
        "| --- | --- | --- | --- |",
    ]
    for row in presence["exclusion_gaps"][:30]:
        lines.append(f"| {row['entity']} | {row['units']} | "
                     f"{row['strong_units']} | {row['mentions']} |")
    lines += ["", "## C. Phantom presences (listed present, no support in "
                  "original text)", "",
              "| entity | substitution-induced | no-surface-match |",
              "| --- | --- | --- |"]
    for row in presence["phantoms"][:30]:
        lines.append(f"| {row['entity']} | {row['substitution_induced']} | "
                     f"{row['no_surface_match']} |")
    if presence["unresolved_annotation_canonicals"]:
        lines += ["", "Unresolved annotation canonicals (registry gap!): "
                  + ", ".join(f"{k} ({v})" for k, v in
                              presence["unresolved_annotation_canonicals"].items())]
    if presence["ambiguous_surface_hits"]:
        lines += ["", "Ambiguous surfaces routed to triage: "
                  + ", ".join(f"{k} ({v})" for k, v in
                              presence["ambiguous_surface_hits"].items())]

    chimeras = report["chimeras"]
    lines += ["", "## D. Chimera adjudication", "",
              "### M. de Marsantes — units listing him present", ""]
    for row in chimeras["m_de_marsantes_units"]:
        lines.append(f"**{row['unit']}**")
        for excerpt in row["raw_mentions"]:
            lines.append(f"- [{excerpt['honorific']}] {excerpt['context']}")
        if not row["raw_mentions"]:
            lines.append("- (no 'de Marsantes' string in raw text at all)")
        lines.append("")
    lines += ["### Octave — units listing him present", "",
              "| unit | bare 'Octave' | 'Mme Octave' |", "| --- | --- | --- |"]
    for row in chimeras["octave_units"]:
        lines.append(f"| {row['unit']} | {row['bare_octave_mentions']} | "
                     f"{row['mme_octave_mentions']} |")

    lines += ["", "## E. Candidate names not in the registry "
                  "(honorific patterns, count ≥ 3)", "",
              "| candidate | count |", "| --- | --- |"]
    for row in report["candidates"][:60]:
        lines.append(f"| {row['name']} | {row['count']} |")
    lines += ["", f"(full candidate list in the JSON: "
                  f"{len(report['candidates'])} entries)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    repo = args.repo

    registry = Registry.load(repo / "characters.yaml")
    units, annotations, latest = load_corpus(repo)
    report = {
        "rewriting": audit_rewriting(units, annotations),
        "presence": audit_presence(registry, units, annotations, latest),
        "chimeras": audit_chimeras(units, annotations),
        "candidates": audit_candidates(registry, repo),
    }
    json_path = repo / "outputs" / "character-registry-audit-v2.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    md_path = repo / "outputs" / "character-registry-audit-v2.md"
    write_markdown(md_path, report)
    presence = report["presence"]
    print(f"units (files): {len(units)}  annotations: {len(annotations)}  "
          f"latest-units: {presence['latest_units_scanned']}")
    severity = report["rewriting"]["annotated_units_by_severity"]
    print(f"rewritten unit files: {report['rewriting']['rewritten_unit_files']}  "
          f"annotated units severe/possessive/flattening: "
          f"{len(severity.get('severe', []))}/"
          f"{len(severity.get('possessive_shift', []))}/"
          f"{len(severity.get('variant_flattening', []))}")
    print(f"top exclusion gaps: "
          + ", ".join(f"{r['entity']}({r['strong_units']})"
                      for r in presence["exclusion_gaps"][:8]))
    print(f"top phantoms: "
          + ", ".join(f"{r['entity']}({r['substitution_induced']}+"
                      f"{r['no_surface_match']})"
                      for r in presence["phantoms"][:6]))
    print(f"candidates: {len(report['candidates'])}  "
          f"wrote {md_path.name} / {json_path.name}")


if __name__ == "__main__":
    main()
