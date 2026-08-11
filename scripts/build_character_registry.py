"""Build characters.yaml from the three legacy alias layers + a curated overlay.

Sources harvested (provenance is preserved per surface form):
  1. root aliases.csv            (pre-LLM manual layer)
  2. outputs/run-*/run.json      (per-run alias maps grown during annotation)
  3. outputs/run-*/annotations/  (canonical names + surface forms the model used)

The OVERLAY below is the hand-authored interpretive layer: identity chains,
characters missing from all legacy layers (Rachel-class), scoped appellations
("le docteur" -> Cottard only in the Verdurin salon), and known-bug corrections
(Mme Octave is tante Léonie, not Octave). Entries with status "proposed" or
"review", and every review_questions item, are decisions awaiting Nathan.

Auto-policy for harvested forms not covered by the overlay:
  - descriptor-led or lowercase-led forms  -> rewrite: never (reference only)
  - everything else                        -> rewrite: allow

Usage:  python scripts/build_character_registry.py [--repo PATH]
Writes: characters.yaml, outputs/aliases.generated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DESCRIPTOR_LEAD = re.compile(
    r"^(le|la|les|l'|un|une|du|de|des|ma|mon|mes|sa|son|ses|notre|votre|leur)\s",
    re.IGNORECASE,
)

TITLE_LEAD = re.compile(
    r"^(?:M\.|Mme|Madame|Mlle|Mademoiselle|Mgr|baronne?|duc(?:hesse)?|"
    r"princesse?|marquise?|comtesse?|vicomtesse?|docteur|maître|général|"
    r"colonel|capitaine|abbé|oncle|tante|la Berma)\b"
)

KNOWN_BAD_CSV_ROWS = {
    # alias -> reason it must not be harvested as-is
    "Mme Octave": "Mme Octave is tante Léonie (Combray), not Octave the Balbec dandy",
    "Madame Octave": "Madame Octave is tante Léonie, not Octave",
    "Charles": "bare first name; collides with Charlie/Charlus contexts — reference only",
    "Robert": "bare first name; unsafe as a global rewrite — reference only",
    "Charlie": "bare first name; reference only",
}


def slugify(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "unnamed"


def is_descriptor(form: str) -> bool:
    return bool(DESCRIPTOR_LEAD.match(form)) or form[:1].islower()


# --------------------------------------------------------------------------
# OVERLAY: the interpretive layer. ids are stable; display_name matches the
# canonical name used in accepted annotations wherever one exists, so that
# resolution of existing annotation output is lossless.
# --------------------------------------------------------------------------

def F(form, scope="global", rewrite=None, scan=None, notes="",
      sources=("overlay",)):
    if rewrite is None:
        rewrite = "never" if (is_descriptor(form) or scope != "global") else "allow"
    if scan is None:
        scan = bool(not DESCRIPTOR_LEAD.match(form)
                    and (form[:1].isupper() or " " in form))
    return {
        "form": form,
        "scope": scope,
        "rewrite": rewrite,
        "scan": bool(scan),
        "sources": list(sources),
        **({"notes": notes} if notes else {}),
    }


OVERLAY = [
    # ---------------------------------------------------------------- family
    dict(id="le-narrateur", display_name="le narrateur", status="confirmed",
         annotation_names=["le narrateur"],
         notes="First-person narrator; never a rewrite target.",
         surface_forms=[F("le narrateur", rewrite="never", scan=False),
                        F("Marcel", scope="mention_only", rewrite="never",
                          notes="Named twice in La Prisonnière; review whether to scan.")]),
    dict(id="la-grand-mere", display_name="la grand-mère", status="confirmed",
         annotation_names=["la grand-mère", "la grand-mère du narrateur"],
         surface_forms=[F("la grand-mère", rewrite="never", scan=True),
                        F("ma grand-mère", rewrite="never", scan=True),
                        F("ma grand'mère", rewrite="never", scan=True,
                          notes="apostrophe variant used in the corpus"),
                        F("Bathilde", scope="mention_only"),
                        F("Mme Amédée", scope="mention_only")]),
    dict(id="la-mere", display_name="la mère du narrateur", status="confirmed",
         annotation_names=["la mère du narrateur"],
         surface_forms=[F("la mère du narrateur", rewrite="never"),
                        F("maman", rewrite="never", scan=True), F("ma mère", rewrite="never", scan=True)]),
    dict(id="le-pere", display_name="le père du narrateur", status="confirmed",
         annotation_names=["le père du narrateur"],
         surface_forms=[F("le père du narrateur", rewrite="never"),
                        F("mon père", rewrite="never", scan=True)]),
    dict(id="le-grand-pere", display_name="le grand-père du narrateur",
         status="confirmed",
         annotation_names=["le grand-père du narrateur"],
         surface_forms=[F("le grand-père du narrateur", rewrite="never"),
                        F("mon grand-père", rewrite="never", scan=True),
                        F("Amédée", scope="mention_only")]),
    dict(id="tante-leonie", display_name="tante Léonie", status="proposed",
         notes="MISSING from all legacy layers as herself; root aliases.csv "
               "wrongly merged her forms of address into Octave.",
         review_questions=["Confirm creating this entity and retiring the "
                           "aliases.csv rows 'Mme Octave,Octave' and "
                           "'Madame Octave,Octave'."],
         surface_forms=[F("tante Léonie", rewrite="never", scan=True),
                        F("Léonie"),
                        F("ma tante Léonie", rewrite="never", scan=True),
                        F("Mme Octave", scope="mention_only", rewrite="never",
                          notes="How Françoise addresses her — NOT Octave"),
                        F("Madame Octave", scope="mention_only", rewrite="never")]),
    dict(id="oncle-adolphe", display_name="oncle Adolphe", status="confirmed",
         annotation_names=["oncle Adolphe"],
         surface_forms=[F("oncle Adolphe", rewrite="never", scan=True),
                        F("mon oncle Adolphe", rewrite="never", scan=True)]),

    # ------------------------------------------------------------ swann side
    dict(id="swann", display_name="Swann", status="confirmed",
         annotation_names=["Swann"],
         surface_forms=[F("Swann"), F("M. Swann"), F("Charles Swann"),
                        F("Charles", scope="mention_only", rewrite="never",
                          notes="bare first name — unsafe as rewrite")]),
    dict(id="odette", display_name="Odette", status="confirmed",
         annotation_names=["Odette", "Mme Swann"],
         eras=[{"name": "Mme de Crécy / la dame en rose", "note": "pre-marriage"},
               {"name": "Odette / Mme Swann", "note": "Swann marriage"},
               {"name": "Mme de Forcheville", "note": "remarriage"}],
         surface_forms=[F("Odette"), F("Mme Swann"), F("Madame Swann"),
                        F("Odette de Crécy"), F("Mme de Crécy", scope="mention_only",
                          notes="also her first husband's name-sphere; review"),
                        F("la dame en rose", rewrite="never", scan=True),
                        F("Miss Sacripant", scope="mention_only",
                          notes="Elstir portrait identity"),
                        F("Mme de Forcheville")]),
    dict(id="gilberte", display_name="Gilberte", status="confirmed",
         annotation_names=["Gilberte"],
         eras=[{"name": "Mlle Swann"}, {"name": "Mlle de Forcheville"},
               {"name": "Mme de Saint-Loup"}],
         review_questions=["Standings show 'Mlle d'Éporcheville' as a separate "
                           "provisional character (2 matches); it is Gilberte "
                           "misheard — merge?"],
         surface_forms=[F("Gilberte"), F("Mlle Swann"), F("Gilberte Swann"),
                        F("Mlle de Forcheville"), F("Mme de Saint-Loup"),
                        F("Mlle d'Éporcheville", scope="mention_only",
                          notes="misheard name; V5 incident")]),

    # ------------------------------------------------------------ guermantes
    dict(id="duchesse-de-guermantes", display_name="duchesse de Guermantes",
         status="confirmed",
         annotation_names=["duchesse de Guermantes", "princesse des Laumes"],
         notes="princesse des Laumes -> duchesse merge already reviewed and "
               "accepted upstream in Nathan's normalization plan.",
         eras=[{"name": "princesse des Laumes", "note": "before the old duke dies"},
               {"name": "duchesse de Guermantes"}],
         surface_forms=[F("duchesse de Guermantes", rewrite="never",
                          notes="already canonical"),
                        F("Mme de Guermantes"), F("Madame de Guermantes"),
                        F("Oriane"), F("princesse des Laumes"),
                        F("Mme des Laumes"),
                        F("la duchesse", rewrite="never", scope="mention_only",
                          notes="descriptor — was a live rewrite rule in 404 runs; "
                                "must remain reference-only")]),
    dict(id="duc-de-guermantes", display_name="duc de Guermantes",
         status="proposed",
         annotation_names=["duc de Guermantes"],
         review_questions=["Annotations also use canonical 'prince des Laumes' "
                           "(separate provisional standings row, 4 matches). Same "
                           "person, earlier title — merge to match the accepted "
                           "princesse-des-Laumes precedent?"],
         eras=[{"name": "prince des Laumes",
                "note": "see the prince-des-laumes era entity"},
               {"name": "duc de Guermantes"}],
         surface_forms=[F("duc de Guermantes", rewrite="never"),
                        F("M. de Guermantes",
                          notes="occasionally the prince; review ambiguity"),
                        F("Basin")]),
    dict(id="prince-des-laumes", display_name="prince des Laumes",
         status="review",
         annotation_names=["prince des Laumes"],
         notes="Era identity of the future duc de Guermantes (Un amour de "
               "Swann). Kept split to mirror the current annotation corpus.",
         same_person_links=[{"entity": "duc-de-guermantes",
                             "relation": "same person, pre-succession title",
                             "policy": "person_view_merge", "review": True}],
         review_questions=["Merge into duc de Guermantes to match the accepted "
                           "princesse-des-Laumes precedent, or keep the era "
                           "ledger split?"],
         surface_forms=[F("prince des Laumes", rewrite="never"),
                        F("M. des Laumes",
                          notes="Laumes-era address form (Un amour de Swann)")]),
    dict(id="prince-de-guermantes", display_name="prince de Guermantes",
         status="proposed",
         notes="MISSING from standings despite major scenes (Dreyfus confession, "
               "V7 matinée host).",
         surface_forms=[F("prince de Guermantes", rewrite="never"),
                        F("Gilbert", scope="mention_only")]),
    dict(id="princesse-de-guermantes", display_name="princesse de Guermantes",
         status="review",
         annotation_names=["princesse de Guermantes"],
         same_person_links=[{"entity": "mme-verdurin",
                             "relation": "the post-V7 princesse de Guermantes IS "
                                         "Mme Verdurin after her remarriage",
                             "policy": "keep_separate",
                             "review": True}],
         review_questions=["Split by era (Marie-Gilbert vs Verdurin-era) or key "
                           "standings on person with a name-era ledger? Current "
                           "standings row may blend the two people."],
         surface_forms=[F("princesse de Guermantes", rewrite="never"),
                        F("Marie-Gilbert", scope="mention_only"),
                        F("la princesse", rewrite="never", scope="mention_only",
                          notes="bare descriptor — was a live rewrite rule in "
                                "runs 067-071; reference-only forever")]),
    dict(id="charlus", display_name="baron de Charlus", status="confirmed",
         annotation_names=["baron de Charlus", "Charlus"],
         surface_forms=[F("baron de Charlus", rewrite="never"),
                        F("M. de Charlus"), F("Charlus"), F("Palamède"),
                        F("Mémé", scope="mention_only")]),
    dict(id="saint-loup", display_name="Robert de Saint-Loup", status="confirmed",
         annotation_names=["Robert de Saint-Loup", "Saint-Loup"],
         surface_forms=[F("Robert de Saint-Loup", rewrite="never"),
                        F("Saint-Loup"),
                        F("marquis de Saint-Loup-en-Bray"),
                        F("Robert", scope="mention_only", rewrite="never",
                          notes="bare first name — unsafe as rewrite"),
                        F("Bobbey", scope="mention_only",
                          notes="Rachel-circle nickname; review")]),
    dict(id="mme-de-marsantes", display_name="Mme de Marsantes", status="proposed",
         notes="Robert's mother. MISSING from all legacy layers; her mentions "
               "risk mis-resolving to her dead husband.",
         surface_forms=[F("Mme de Marsantes"), F("comtesse de Marsantes")]),
    dict(id="m-de-marsantes", display_name="M. de Marsantes", status="review",
         annotation_names=["M. de Marsantes"],
         notes="Robert's father, dead before the novel opens; legitimate only as "
               "posthumous/associational mention.",
         review_questions=["Audit adjudicates the 4 unit-annotations listing him "
                           "present: genuine posthumous mentions vs substituted "
                           "Mme de Marsantes."],
         surface_forms=[F("M. de Marsantes")]),

    # --------------------------------------------------------- verdurin clan
    dict(id="mme-verdurin", display_name="Mme Verdurin", status="confirmed",
         annotation_names=["Mme Verdurin"],
         eras=[{"name": "Mme Verdurin"}, {"name": "duchesse de Duras"},
               {"name": "princesse de Guermantes", "note": "see same_person_links"}],
         surface_forms=[F("Mme Verdurin"), F("Madame Verdurin"),
                        F("la Patronne", scope="context:verdurin-salon",
                          rewrite="never")]),
    dict(id="m-verdurin", display_name="M. Verdurin", status="confirmed",
         annotation_names=["M. Verdurin"],
         surface_forms=[F("M. Verdurin"),
                        F("le Patron", scope="context:verdurin-salon",
                          rewrite="never")]),
    dict(id="docteur-cottard", display_name="docteur Cottard", status="confirmed",
         annotation_names=["docteur Cottard"],
         surface_forms=[F("docteur Cottard", rewrite="never"),
                        F("Cottard"), F("le docteur Cottard", rewrite="never"),
                        F("le docteur", scope="context:verdurin-salon",
                          rewrite="never",
                          notes="THE regression case: a historical rewrite rule "
                                "inserted Cottard into a V2 brothel simile")]),
    dict(id="princesse-sherbatoff", display_name="princesse Sherbatoff",
         status="proposed",
         notes="Faithful of the little clan; MISSING from all legacy layers.",
         surface_forms=[F("princesse Sherbatoff"), F("Mme Sherbatoff")]),
    dict(id="saniette", display_name="Saniette", status="confirmed",
         annotation_names=["Saniette"], surface_forms=[F("Saniette")]),
    dict(id="brichot", display_name="Brichot", status="confirmed",
         annotation_names=["Brichot"], surface_forms=[F("Brichot")]),
    dict(id="ski", display_name="M. Ski", status="confirmed",
         annotation_names=["M. Ski"],
         surface_forms=[F("M. Ski"), F("Ski"),
                        F("Viradobetski", scope="mention_only")]),
    dict(id="morel", display_name="Morel", status="confirmed",
         annotation_names=["Morel"],
         surface_forms=[F("Morel"), F("Charles Morel"),
                        F("Charlie", scope="mention_only", rewrite="never",
                          notes="bare first name — unsafe as rewrite")]),

    # ------------------------------------------------------------- the missing
    dict(id="rachel", display_name="Rachel", status="proposed",
         notes="MISSING from every legacy layer -> structurally invisible to the "
               "annotator under scope rule 17 despite co-lead scenes in V3 and "
               "the V7 Berma duel.",
         eras=[{"name": "Rachel quand du Seigneur",
                "note": "maison de passe period (V2)"},
               {"name": "Rachel", "note": "actress; Saint-Loup's mistress"},
               {"name": "Rachel", "note": "celebrated reciter, V7"}],
         surface_forms=[F("Rachel"),
                        F("Rachel quand du Seigneur", rewrite="never", scan=True,
                          notes="narrator's Halévy joke-name; scan, never rewrite"),
                        F("Zézette", scope="mention_only",
                          notes="Saint-Loup's pet name for her")]),
    dict(id="mlle-vinteuil", display_name="Mlle Vinteuil", status="proposed",
         notes="MISSING from all legacy layers despite Montjouvain and the "
               "posthumous transcription arc.",
         surface_forms=[F("Mlle Vinteuil")]),
    dict(id="amie-de-mlle-vinteuil", display_name="l'amie de Mlle Vinteuil",
         status="review",
         notes="Genuinely unnamed but individuated and consequential; the "
               "named-characters-only rule needs a decision for her.",
         review_questions=["Admit descriptor-identified recurring figures as "
                           "entities (mention-only, never rewritten)?"],
         surface_forms=[F("l'amie de Mlle Vinteuil", rewrite="never", scan=True,
                          scope="mention_only")]),
    dict(id="celeste-albaret", display_name="Céleste Albaret", status="proposed",
         notes="MISSING from all legacy layers.",
         surface_forms=[F("Céleste Albaret"), F("Céleste")]),
    dict(id="gisele", display_name="Gisèle", status="proposed",
         notes="Petite bande member; missing from legacy layers.",
         surface_forms=[F("Gisèle")]),
    dict(id="rosemonde", display_name="Rosemonde", status="proposed",
         notes="Petite bande member; missing from legacy layers.",
         surface_forms=[F("Rosemonde")]),
    dict(id="theodore", display_name="Théodore", status="proposed",
         surface_forms=[F("Théodore")]),
    dict(id="eulalie", display_name="Eulalie", status="proposed",
         surface_forms=[F("Eulalie")]),
    dict(id="mme-sazerat", display_name="Mme Sazerat", status="proposed",
         surface_forms=[F("Mme Sazerat")]),

    # ------------------------------------------------------------ artists etc
    dict(id="elstir", display_name="Elstir", status="confirmed",
         annotation_names=["Elstir"],
         eras=[{"name": "M. Biche / Tiche", "note": "Verdurin salon period"},
               {"name": "Elstir"}],
         surface_forms=[F("Elstir"), F("M. Elstir"),
                        F("M. Biche", scope="mention_only"),
                        F("Biche", scope="mention_only"),
                        F("Tiche", scope="mention_only")]),
    dict(id="le-peintre", display_name="le peintre", status="review",
         annotation_names=["le peintre"],
         same_person_links=[{"entity": "elstir",
                             "relation": "'le peintre' of the Verdurin salon is "
                                         "Elstir pre-revelation",
                             "policy": "person_view_merge", "review": True}],
         review_questions=["Merge the 29-match 'le peintre' standings row into "
                           "Elstir (person view), keep split (name view), or "
                           "both via era ledger?"],
         surface_forms=[F("le peintre", rewrite="never",
                          scope="context:verdurin-salon")]),
    dict(id="la-berma", display_name="la Berma", status="confirmed",
         annotation_names=["la Berma"],
         surface_forms=[F("la Berma", rewrite="never", scan=True), F("Berma")]),
    dict(id="bergotte", display_name="Bergotte", status="confirmed",
         annotation_names=["Bergotte"], surface_forms=[F("Bergotte")]),
    dict(id="m-vinteuil", display_name="M. Vinteuil", status="confirmed",
         annotation_names=["M. Vinteuil", "Vinteuil"],
         surface_forms=[F("M. Vinteuil", rewrite="never",
                          notes="already canonical"), F("Vinteuil")]),

    # ---------------------------------------------------------------- others
    dict(id="bloch", display_name="Bloch", status="confirmed",
         annotation_names=["Bloch"],
         eras=[{"name": "Bloch"}, {"name": "Jacques du Rozier", "note": "V7"}],
         surface_forms=[F("Bloch"),
                        F("M. Bloch", scope="mention_only", rewrite="never",
                          notes="ambiguous with Bloch père — resolver must flag"),
                        F("Jacques du Rozier", scope="mention_only")]),
    dict(id="bloch-pere", display_name="Bloch père", status="confirmed",
         annotation_names=["Bloch père"],
         surface_forms=[F("Bloch père", rewrite="never"),
                        F("M. Bloch", scope="mention_only", rewrite="never",
                          notes="shared with Bloch fils — resolver must flag")]),
    dict(id="albertine", display_name="Albertine", status="confirmed",
         annotation_names=["Albertine"],
         surface_forms=[F("Albertine"), F("Albertine Simonet"),
                        F("Mlle Simonet")]),
    dict(id="francoise", display_name="Françoise", status="confirmed",
         annotation_names=["Françoise"], surface_forms=[F("Françoise")]),
    dict(id="jupien", display_name="Jupien", status="confirmed",
         annotation_names=["Jupien"], surface_forms=[F("Jupien")]),
    dict(id="aime", display_name="Aimé", status="confirmed",
         annotation_names=["Aimé"], surface_forms=[F("Aimé")]),
    dict(id="norpois", display_name="Norpois", status="confirmed",
         annotation_names=["Norpois"],
         surface_forms=[F("Norpois"), F("M. de Norpois"),
                        F("marquis de Norpois")]),
    dict(id="legrandin", display_name="Legrandin", status="confirmed",
         annotation_names=["Legrandin"],
         eras=[{"name": "Legrandin"},
               {"name": "comte de Méséglise", "note": "late self-ennoblement"}],
         surface_forms=[F("Legrandin"), F("M. Legrandin"),
                        F("comte de Méséglise", scope="mention_only")]),
    dict(id="mme-de-villeparisis", display_name="Mme de Villeparisis",
         status="confirmed",
         annotation_names=["Mme de Villeparisis"],
         surface_forms=[F("Mme de Villeparisis"), F("Madame de Villeparisis"),
                        F("marquise de Villeparisis")]),
    dict(id="remi", display_name="Rémi", status="proposed",
         annotation_names=["Rémi", "Remi"],
         notes="Swann's coachman; standings currently split him across a "
               "diacritic variant (Rémi 4 matches / Remi 2 matches).",
         surface_forms=[F("Rémi"), F("Remi", notes="diacritic variant")]),
    dict(id="octave", display_name="Octave", status="review",
         annotation_names=["Octave"],
         notes="The Balbec dandy ('dans les choux'), later revealed a genius; "
               "must NOT absorb tante Léonie's 'Mme Octave'.",
         review_questions=["Audit whether the existing 4-0-0 record includes "
                           "units where only Mme Octave (= Léonie) appears."],
         surface_forms=[F("Octave")]),
]


# --------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------

def harvest_csv(repo: Path):
    rows, issues = [], []
    csv_path = repo / "aliases.csv"
    with csv_path.open(encoding="utf-8") as handle:
        for alias, canonical in csv.reader(handle):
            stripped = (alias.strip(), canonical.strip())
            if alias != stripped[0] or canonical != stripped[1]:
                issues.append(f"whitespace in row {alias!r} -> {canonical!r}")
            if stripped[0] in KNOWN_BAD_CSV_ROWS:
                issues.append(
                    f"excluded row {stripped[0]!r} -> {stripped[1]!r}: "
                    f"{KNOWN_BAD_CSV_ROWS[stripped[0]]}"
                )
                continue
            rows.append(stripped)
    return rows, issues


def harvest_run_maps(repo: Path):
    """canonical -> {alias -> run_count}"""
    counts = defaultdict(lambda: defaultdict(int))
    for run_json in sorted(repo.glob("outputs/run-*/run.json")):
        data = json.loads(run_json.read_text(encoding="utf-8"))
        for canonical, entry in (data.get("alias_map") or {}).items():
            for alias in entry.get("aliases", []):
                counts[canonical][alias] += 1
    return counts


def harvest_annotations(repo: Path):
    """canonical -> {"units": n, "surface_forms": {form: count}}"""
    usage = defaultdict(lambda: {"units": 0, "surface_forms": defaultdict(int)})
    for path in sorted(repo.glob("outputs/run-*/annotations/*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        for character in annotation.get("characters_present", []):
            canonical = character.get("canonical_name")
            if not canonical:
                continue
            usage[canonical]["units"] += 1
            for surface in character.get("surface_forms", []) or []:
                usage[canonical]["surface_forms"][surface] += 1
    return usage


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------

def build(repo: Path):
    csv_rows, csv_issues = harvest_csv(repo)
    run_maps = harvest_run_maps(repo)
    annotation_usage = harvest_annotations(repo)

    entities = {e["id"]: json.loads(json.dumps(e)) for e in OVERLAY}
    claimed_annotation_names = {
        name for e in entities.values() for name in e.get("annotation_names", [])
    }
    claimed_forms = {
        (e["id"], f["form"]) for e in entities.values()
        for f in e.get("surface_forms", [])
    }
    forms_anywhere = {f["form"] for e in entities.values()
                      for f in e.get("surface_forms", [])}

    def entity_for_canonical(canonical):
        for entity in entities.values():
            if canonical in entity.get("annotation_names", []):
                return entity
        entity = dict(
            id=slugify(canonical), display_name=canonical, status="confirmed",
            annotation_names=[canonical],
            surface_forms=[F(canonical, sources=("annotations",))],
        )
        if entity["id"] in entities:  # slug collision — keep distinct
            entity["id"] = entity["id"] + "-2"
        entities[entity["id"]] = entity
        claimed_annotation_names.add(canonical)
        claimed_forms.add((entity["id"], canonical))
        forms_anywhere.add(canonical)
        return entity

    # annotation canonicals first: these are load-bearing for resolution.
    for canonical in sorted(annotation_usage):
        entity_for_canonical(canonical)

    def add_form(entity, form, source, notes=""):
        if (entity["id"], form) in claimed_forms:
            for existing in entity["surface_forms"]:
                if existing["form"] == form and source not in existing["sources"]:
                    existing["sources"].append(source)
            return
        if form in forms_anywhere:
            return  # claimed by another entity (deliberate, e.g. M. Bloch)
        entity["surface_forms"].append(F(form, sources=(source,), notes=notes))
        claimed_forms.add((entity["id"], form))
        forms_anywhere.add(form)

    # per-run alias maps
    for canonical, aliases in run_maps.items():
        if canonical not in claimed_annotation_names and canonical not in {
            e["display_name"] for e in entities.values()
        }:
            entity_for_canonical(canonical)
        entity = next(
            e for e in entities.values()
            if canonical in e.get("annotation_names", [])
            or e["display_name"] == canonical
        )
        for alias, run_count in sorted(aliases.items()):
            add_form(entity, alias, f"run_maps({run_count} runs)")

    # root csv
    for alias, canonical in csv_rows:
        entity = next(
            (e for e in entities.values()
             if canonical in e.get("annotation_names", [])
             or e["display_name"] == canonical
             or any(f["form"] == canonical for f in e["surface_forms"])),
            None,
        ) or entity_for_canonical(canonical)
        add_form(entity, alias, "aliases_csv")

    # model-observed surface forms
    for canonical, usage in annotation_usage.items():
        entity = next(e for e in entities.values()
                      if canonical in e.get("annotation_names", []))
        for surface, count in sorted(usage["surface_forms"].items()):
            if count < 2 or len(surface) < 3:
                continue
            if DESCRIPTOR_LEAD.match(surface):
                continue  # coreference record ("la petite", "sa mère")
            if not surface[:1].isupper() and not TITLE_LEAD.match(surface):
                continue  # pronouns/descriptors ("elle", "votre ami")
            add_form(entity, surface, f"annotation_surface_forms({count}x)")

    ordered = sorted(entities.values(),
                     key=lambda e: (e["status"] != "review",
                                    e["status"] != "proposed",
                                    e["display_name"].lower()))
    return ordered, csv_issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()

    ordered, csv_issues = build(args.repo)
    payload = {
        "version": 1,
        "generated_by": "scripts/build_character_registry.py",
        "notes": (
            "Single source of truth for character identity. Layers merged: "
            "root aliases.csv, per-run alias maps, annotation usage, plus a "
            "hand-authored overlay. status=proposed/review entries and all "
            "review_questions await Nathan's decision. Scope grammar and "
            "safety rules: see proust/registry.py docstring."
        ),
        "known_source_issues": csv_issues,
        "entities": ordered,
    }
    out = args.repo / "characters.yaml"
    out.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=88),
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(ordered)} entities)")

    from proust.registry import Registry  # validate + emit legacy csv
    registry = Registry.load(out)
    generated_csv = args.repo / "outputs" / "aliases.generated.csv"
    with generated_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for alias, canonical in sorted(registry.rewrite_map().items()):
            writer.writerow([alias, canonical])
    print(f"validated OK; wrote {generated_csv} "
          f"({len(registry.rewrite_map())} safe rewrite rules)")


if __name__ == "__main__":
    main()
