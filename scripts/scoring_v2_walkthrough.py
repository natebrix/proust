"""Sample walkthrough for the scoring v2 adoption checkpoint.

For a small set of well-known units, shows the annotator's status effects
verbatim, the v2 movement arithmetic written out per lens, the v1 net for
contrast, the labels, and every pairwise comparison with weight and outcome.
Written for joint human verification against the text before the corpus-wide
re-score is trusted.

Writes outputs/scoring-v2/sample-walkthrough.md.
"""
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proust import scoring_v2  # noqa: E402

LENSES = ("advantage", "prestige", "inclusion")

SAMPLE = [
    ("v1-p1-combray#p-101-p-105", "The Legrandin garden scene: basic arithmetic check"),
    ("v3-p2#p-91-p-95", "The grandmother's death: advantage internal weights, solemn passage"),
    ("v2-p2-noms-de-pays-le-pays#p-101-p-105", "Villeparisis at Balbec: the adjudicated prestige/inclusion divergence"),
    ("v4-p2#p-406-p-410", "The 17-character Verdurin salon: ambiguity note weighs, never pushes"),
    (None, "SEARCH:Rachel+Berma"),
    (None, "SEARCH:Norpois+Bergotte"),
]


def find_unit(needles):
    for run_dir in sorted(glob.glob("outputs/foundation-run-0*")):
        for ann_path in sorted(glob.glob(f"{run_dir}/annotations/*.json")):
            ann = json.loads(Path(ann_path).read_text())
            names = {c["canonical_name"] for c in ann["characters_present"]}
            if all(any(needle in name for name in names) for needle in needles):
                return Path(ann_path).stem, run_dir
    return None, None


def locate(unit_id):
    for run_dir in sorted(glob.glob("outputs/foundation-run-0*")):
        path = Path(run_dir) / "annotations" / f"{unit_id}.json"
        if path.exists():
            return run_dir
    raise SystemExit(f"unit not found: {unit_id}")


def main():
    lines = ["# Scoring v2 sample walkthrough", "",
             "Per unit: the annotator's effects verbatim, the v2 movement arithmetic, "
             "labels, and weighted comparisons. v1 nets shown for contrast where the "
             "scorer exposes them.", ""]

    resolved = []
    for unit_id, note in SAMPLE:
        if unit_id is None:
            needles = note.split(":", 1)[1].split("+")
            found, run_dir = find_unit(needles)
            if found:
                resolved.append((found, run_dir, f"Search hit for {' & '.join(needles)}"))
        else:
            resolved.append((unit_id, locate(unit_id), note))

    # a zero-effect bystander example: first unit with a present character with no effects
    for run_dir in sorted(glob.glob("outputs/foundation-run-0*"))[:5]:
        done = False
        for ann_path in sorted(glob.glob(f"{run_dir}/annotations/*.json")):
            ann = json.loads(Path(ann_path).read_text())
            with_effects = {e["character"] for e in ann["status_effects"]}
            bystanders = [c["canonical_name"] for c in ann["characters_present"]
                          if c["canonical_name"] not in with_effects]
            if bystanders and len(ann["characters_present"]) >= 3:
                resolved.append((Path(ann_path).stem, run_dir,
                                 f"Zero-effect bystander case ({bystanders[0]})"))
                done = True
                break
        if done:
            break

    for unit_id, run_dir, note in resolved:
        ann = json.loads((Path(run_dir) / "annotations" / f"{unit_id}.json").read_text())
        lines.append(f"## {unit_id}")
        lines.append(f"*{note}*  (run: {Path(run_dir).name})")
        lines.append("")
        lines.append(f"- ambiguity notes: {len(ann.get('ambiguities') or [])}"
                     f" -> comparison weight factor rho = {scoring_v2.ambiguity_weight(ann):.2f}")
        lines.append("")
        lines.append("**Effects as annotated:**")
        for e in ann["status_effects"]:
            lines.append(f"- {e['character']}: {e['dimension']} {e['delta']:+d} "
                         f"(confidence {e['confidence']}) — {e['explanation'][:110]}")
        if not ann["status_effects"]:
            lines.append("- (none)")
        lines.append("")
        for lens in LENSES:
            movements = scoring_v2.unit_movements(ann, lens)
            labels = scoring_v2.unit_labels(ann, lens)
            comparisons = scoring_v2.unit_comparisons(ann, lens)
            lines.append(f"**{lens}**: " + (", ".join(
                f"{c} = {m:+.2f} [{labels.get(c, '?')}]" for c, m in sorted(movements.items())
            ) or "(no scored characters)"))
            shown = 0
            draws = 0
            for comp in comparisons:
                a, b = comp["character_a"], comp["character_b"]
                if comp["observed_a"] == 0.5:
                    draws += 1
                    continue
                winner = a if comp["observed_a"] == 1.0 else b
                lines.append(f"  - {a} vs {b}: {winner} wins "
                             f"(m {comp['movement_a']:+.2f} vs {comp['movement_b']:+.2f}), "
                             f"weight {comp['weight']:.2f}")
                shown += 1
            if draws:
                lines.append(f"  - (+ {draws} draws among characters with no relative movement)")
        lines.append("")

    out = Path("outputs/scoring-v2/sample-walkthrough.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(resolved)} units)")


if __name__ == "__main__":
    main()
