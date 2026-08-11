"""Character registry: single source of truth for character identity.

Replaces the three-layer alias system (root aliases.csv, per-run alias maps,
downstream reviewed merges) with one governed file: characters.yaml.

Design principles (see proust/docs/character_registry_design.md):

1. One registry, everything else generated. The root aliases.csv and per-run
   prompt reference sheets become build artifacts of characters.yaml.
2. Annotate the text Proust wrote. ``rewrite_map`` only ever emits forms
   explicitly marked ``rewrite: allow``; descriptor phrases and common-noun
   appellations ("le docteur", "la duchesse", "princesse") are structurally
   excluded from rewriting and exist only as resolution/reference knowledge.
3. Open world with triage. ``resolve`` never silently drops a name: unknown
   surfaces return an Unresolved marker that callers must record.

Scope grammar for surface forms (string-valued for YAML simplicity):

    global                    valid everywhere
    mention_only              never rewritten; used for resolution/reference
    context:<tag>             valid where the tag applies (e.g. verdurin-salon)
    chapters:<id>,<id>,...    valid only in the listed chapter ids
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "characters.yaml"

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'"})

# Forms that begin with these lowercase words are descriptors/appellations,
# never safe for context-free text rewriting.
_DESCRIPTOR_LEAD = re.compile(
    r"^(le|la|les|l'|un|une|du|de|des|ma|mon|mes|sa|son|ses|notre|votre|leur)\s",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """NFC-normalize and unify apostrophe variants (matching only)."""
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHES)


@dataclass(frozen=True)
class SurfaceForm:
    form: str
    entity_id: str
    scope: str = "global"
    rewrite: str = "never"  # "allow" | "never"
    scan: bool = True  # eligible as string evidence for mention scanning
    sources: tuple = ()
    notes: str = ""

    @property
    def scope_kind(self) -> str:
        return self.scope.split(":", 1)[0]

    @property
    def scope_arg(self) -> str:
        parts = self.scope.split(":", 1)
        return parts[1] if len(parts) == 2 else ""

    def applies_in(self, chapter_id: str | None, context_tags: tuple) -> bool:
        kind = self.scope_kind
        if kind in ("global", "mention_only"):
            return True
        if kind == "context":
            return self.scope_arg in context_tags
        if kind == "chapters":
            allowed = {c.strip() for c in self.scope_arg.split(",")}
            return chapter_id in allowed
        raise ValueError(f"Unknown scope {self.scope!r} on form {self.form!r}")


@dataclass(frozen=True)
class Entity:
    id: str
    display_name: str
    status: str = "confirmed"  # confirmed | proposed | review
    notes: str = ""
    eras: tuple = ()
    same_person_links: tuple = ()
    annotation_names: tuple = ()  # canonical names used in accepted annotations
    review_questions: tuple = ()


@dataclass
class Resolution:
    entity_id: str | None
    matched_form: str | None
    status: str  # "resolved" | "ambiguous" | "unresolved"
    candidates: tuple = ()


@dataclass
class Registry:
    entities: dict = field(default_factory=dict)  # id -> Entity
    forms: list = field(default_factory=list)  # [SurfaceForm], longest-first
    _by_norm_form: dict = field(default_factory=dict)  # norm form -> [SurfaceForm]
    _by_annotation_name: dict = field(default_factory=dict)  # name -> entity_id

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "Registry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        registry = cls()
        for raw in data.get("entities", []):
            entity = Entity(
                id=raw["id"],
                display_name=raw["display_name"],
                status=raw.get("status", "confirmed"),
                notes=raw.get("notes", "") or "",
                eras=tuple(raw.get("eras", []) or ()),
                same_person_links=tuple(raw.get("same_person_links", []) or ()),
                annotation_names=tuple(raw.get("annotation_names", []) or ()),
                review_questions=tuple(raw.get("review_questions", []) or ()),
            )
            if entity.id in registry.entities:
                raise ValueError(f"Duplicate entity id {entity.id!r}")
            registry.entities[entity.id] = entity
            for name in entity.annotation_names:
                existing = registry._by_annotation_name.get(name)
                if existing and existing != entity.id:
                    raise ValueError(
                        f"Annotation name {name!r} claimed by {existing!r} and {entity.id!r}"
                    )
                registry._by_annotation_name[name] = entity.id
            for form_raw in raw.get("surface_forms", []) or ():
                form_text = form_raw["form"]
                default_scan = bool(
                    not _DESCRIPTOR_LEAD.match(form_text)
                    and (form_text[:1].isupper() or " " in form_text)
                )
                form = SurfaceForm(
                    form=form_text,
                    entity_id=entity.id,
                    scope=form_raw.get("scope", "global"),
                    rewrite=form_raw.get("rewrite", "never"),
                    scan=bool(form_raw.get("scan", default_scan)),
                    sources=tuple(form_raw.get("sources", []) or ()),
                    notes=form_raw.get("notes", "") or "",
                )
                registry.forms.append(form)
                registry._by_norm_form.setdefault(
                    normalize_text(form.form), []
                ).append(form)
        registry.forms.sort(key=lambda f: len(f.form), reverse=True)
        registry.validate()
        return registry

    # -------------------------------------------------------------- validate
    def validate(self) -> None:
        """Structural safety invariants. Raises ValueError on violation."""
        problems = []
        for form in self.forms:
            if form.rewrite == "allow":
                if _DESCRIPTOR_LEAD.match(form.form) or form.form[:1].islower():
                    problems.append(
                        f"rewrite:allow on descriptor/lowercase form {form.form!r} "
                        f"({form.entity_id}) — descriptor phrases must be "
                        f"mention_only or context-scoped"
                    )
                if form.scope_kind == "mention_only":
                    problems.append(
                        f"form {form.form!r} is mention_only but rewrite:allow"
                    )
        # A form rewritable by two different entities in overlapping scope is
        # inherently unsafe for text rewriting.
        for norm, claimants in self._by_norm_form.items():
            rewriters = [f for f in claimants if f.rewrite == "allow"]
            if len({f.entity_id for f in rewriters}) > 1:
                globalish = [f for f in rewriters if f.scope_kind == "global"]
                if len(globalish) >= 1 and len(rewriters) > 1:
                    ids = sorted({f.entity_id for f in rewriters})
                    problems.append(
                        f"form {claimants[0].form!r} rewrite-claimed by multiple "
                        f"entities with overlapping scope: {ids}"
                    )
        if problems:
            raise ValueError(
                "characters.yaml failed safety validation:\n- " + "\n- ".join(problems)
            )

    # --------------------------------------------------------------- resolve
    def resolve(
        self,
        surface: str,
        chapter_id: str | None = None,
        context_tags: tuple = (),
    ) -> Resolution:
        """Resolve a surface form or annotation canonical name to an entity.

        Never raises on unknown names; returns status="unresolved" so callers
        can triage instead of dropping.
        """
        surface_norm = normalize_text(surface.strip())
        if surface in self._by_annotation_name:
            return Resolution(self._by_annotation_name[surface], surface, "resolved")
        candidates = [
            f
            for f in self._by_norm_form.get(surface_norm, [])
            if f.applies_in(chapter_id, context_tags)
        ]
        entity_ids = sorted({f.entity_id for f in candidates})
        if len(entity_ids) == 1:
            return Resolution(entity_ids[0], surface, "resolved")
        if len(entity_ids) > 1:
            return Resolution(None, surface, "ambiguous", tuple(entity_ids))
        return Resolution(None, surface, "unresolved")

    # ----------------------------------------------------------- rewrite map
    def rewrite_map(
        self, chapter_id: str | None = None, context_tags: tuple = ()
    ) -> dict:
        """alias -> canonical display name, ONLY for forms marked rewrite:allow.

        Compatible with corpus.apply_alias_replacements. By construction this
        can never contain descriptor phrases, so the docteur-Cottard class of
        corruption is structurally impossible.
        """
        mapping = {}
        for form in self.forms:
            if form.rewrite != "allow":
                continue
            if not form.applies_in(chapter_id, context_tags):
                continue
            if self._embedded_in_foreign_form(form):
                continue
            canonical = self.entities[form.entity_id].display_name
            if form.form != canonical:
                mapping[form.form] = canonical
        return mapping

    def _embedded_in_foreign_form(self, form: "SurfaceForm") -> bool:
        """True if this form occurs (word-bounded) inside a longer form that
        belongs to a different entity. Rewriting such an alias would corrupt
        the longer name — the 'Marsantes' -> 'Mme de M. de Marsantes' bug."""
        needle = re.compile(rf"(?<!\w){re.escape(form.form)}(?!\w)")
        for other in self.forms:
            if other.entity_id == form.entity_id:
                continue
            if len(other.form) <= len(form.form):
                continue
            if needle.search(other.form):
                return True
        return False

    # ------------------------------------------------------- reference sheet
    def reference_sheet(
        self, chapter_id: str | None = None, context_tags: tuple = ()
    ) -> dict:
        """Dramatis-personae reference for prompt injection (no rewriting).

        canonical display name -> {"aliases": [...], "notes": ...}, matching
        the shape the current prompt template expects for {{ALIAS_MAP}}.
        """
        sheet = {}
        for form in self.forms:
            if not form.applies_in(chapter_id, context_tags):
                continue
            entity = self.entities[form.entity_id]
            entry = sheet.setdefault(
                entity.display_name, {"aliases": [], "notes": entity.notes}
            )
            if form.form not in entry["aliases"]:
                entry["aliases"].append(form.form)
        return sheet

    # -------------------------------------------------------------- scanning
    def compile_scanner(self) -> "MentionScanner":
        return MentionScanner(self)


class MentionScanner:
    """Finds registry surface forms in text (word-boundary, longest-first,
    non-overlapping). Used by the audit for mention inventories."""

    def __init__(self, registry: Registry):
        self.registry = registry
        forms = sorted({f.form for f in registry.forms if f.scan},
                       key=len, reverse=True)
        pattern = "|".join(re.escape(normalize_text(f)) for f in forms)
        self._regex = re.compile(rf"(?<!\w)(?:{pattern})(?!\w)") if forms else None

    def scan(self, text: str, chapter_id: str | None = None) -> dict:
        """Return entity_id -> [matched surface strings]."""
        if self._regex is None:
            return {}
        found: dict = {}
        for match in self._regex.finditer(normalize_text(text)):
            resolution = self._resolve_scannable(match.group(0), chapter_id)
            if resolution.status == "resolved":
                found.setdefault(resolution.entity_id, []).append(match.group(0))
            elif resolution.status == "ambiguous":
                found.setdefault("__ambiguous__:" + match.group(0), []).append(
                    match.group(0)
                )
        return found

    def _resolve_scannable(self, surface: str, chapter_id: str | None):
        surface_norm = normalize_text(surface.strip())
        candidates = [
            f
            for f in self.registry._by_norm_form.get(surface_norm, [])
            if f.scan and f.applies_in(chapter_id, ())
        ]
        entity_ids = sorted({f.entity_id for f in candidates})
        if len(entity_ids) == 1:
            return Resolution(entity_ids[0], surface, "resolved")
        if len(entity_ids) > 1:
            return Resolution(None, surface, "ambiguous", tuple(entity_ids))
        return Resolution(None, surface, "unresolved")
