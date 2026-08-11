# Prompt-v2 A/B report

Arm A = prompt v1 + legacy alias map (lifted from the accepted run). Arm B = prompt v2 + registry reference sheet. Both on the current (Wikisource) canonical text; `accepted` is the legacy annotation, kept as a reference point, not ground truth. Direction = sign(net_score) with a 0.25 neutral band; net_score reuses proust.runner's scoring weights per lens.

## Aggregates

- Units: 10
- Present: accepted=10, A=10, B=10, fully complete=10
- Open-world discoveries (B-only, not in accepted or A): 11 instances, 10 distinct names: M. d'Argencourt, Mme Verdurin, Morel, Odette, Rachel, Swann, baron de Charlus, cousine Poictiers, le narrateur, le père du narrateur
- Unresolved names in B: 2 total -- legitimate_off_sheet=2, possible_registry_gap=0, registry_miss_model_error=0 (heuristic triage; final call is human, see the design doc's decision queue)

### Direction agreement rates

| lens | A vs accepted | B vs accepted | A vs B |
| --- | --- | --- | --- |
| advantage | 54% (7/13) | 46% (6/13) | 79% (15/19) |
| prestige | 62% (8/13) | 46% (6/13) | 79% (15/19) |
| inclusion | 54% (7/13) | 46% (6/13) | 79% (15/19) |

## Units

### v7-p4-le-bal-de-tetes#p-96-p-100 -> v7-p4-le-bal-de-tetes#p-84-p-87

la Berma / Rachel duel -- the Rachel-exclusion test

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): la Berma
- characters_present (A): duchesse de Guermantes, la Berma
- characters_present (B): duchesse de Guermantes[resolved], Rachel[resolved], la Berma[resolved]
- B-only discoveries: Rachel[resolved]
- direction agreement [advantage]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [prestige]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [inclusion]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)

### v7-p2-m-de-charlus-pendant-la-guerre#p-76-p-80 -> v7-p2-m-de-charlus-pendant-la-guerre#p-62-p-66

the Marsantes chimera scene (Françoise's grief-reaction to Mme de Marsantes)

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): M. de Marsantes
- characters_present (A): Robert de Saint-Loup, duchesse de Guermantes
- characters_present (B): Robert de Saint-Loup[resolved], Morel[resolved], baron de Charlus[resolved], M. d'Argencourt[unresolved]
- B-only discoveries: Morel[resolved], baron de Charlus[resolved], M. d'Argencourt[unresolved]
- missing vs accepted (A): M. de Marsantes
- missing vs accepted (B): M. de Marsantes
- unresolved in B: M. d'Argencourt (legitimate_off_sheet)
- direction agreement [advantage]: A-vs-accepted=n/a (0 comparable), B-vs-accepted=n/a (0 comparable), A-vs-B=100% (1/1)
- direction agreement [prestige]: A-vs-accepted=n/a (0 comparable), B-vs-accepted=n/a (0 comparable), A-vs-B=100% (1/1)
- direction agreement [inclusion]: A-vs-accepted=n/a (0 comparable), B-vs-accepted=n/a (0 comparable), A-vs-B=100% (1/1)

### v1-p1-combray#p-111-p-115 -> v1-p1-combray#p-111-p-115

oncle Adolphe / la dame en rose (introduction of oncle Adolphe, immediately preceding the dame-en-rose reveal)

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): oncle Adolphe
- characters_present (A): oncle Adolphe
- characters_present (B): oncle Adolphe[resolved]
- direction agreement [advantage]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (1/1)
- direction agreement [prestige]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (1/1)
- direction agreement [inclusion]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (1/1)

### v1-p1-combray#p-311-p-315 -> v1-p1-combray#p-307-p-311

Montjouvain -- l'amie de Mlle Vinteuil test

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): M. Vinteuil
- characters_present (A): M. Vinteuil, Swann
- characters_present (B): M. Vinteuil[resolved], Swann[resolved]
- direction agreement [advantage]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [prestige]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [inclusion]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)

### v3-p1#p-51-p-55 -> v3-p1#p-51-p-55

heavy variant-flattening zone per the audit

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): la Berma
- characters_present (A): Elstir, la Berma
- characters_present (B): la Berma[resolved], Elstir[resolved]
- direction agreement [advantage]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [prestige]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [inclusion]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)

### v3-p1#p-306-p-310 -> v3-p1#p-305-p-309

Saint-Loup / duchesse control with accepted scores

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): Robert de Saint-Loup, duchesse de Guermantes
- characters_present (A): Robert de Saint-Loup, duchesse de Guermantes
- characters_present (B): duchesse de Guermantes[resolved], Robert de Saint-Loup[resolved], cousine Poictiers[unresolved]
- B-only discoveries: cousine Poictiers[unresolved]
- unresolved in B: cousine Poictiers (legitimate_off_sheet)
- direction agreement [advantage]: A-vs-accepted=50% (1/2), B-vs-accepted=0% (0/2), A-vs-B=50% (1/2)
- direction agreement [prestige]: A-vs-accepted=50% (1/2), B-vs-accepted=0% (0/2), A-vs-B=50% (1/2)
- direction agreement [inclusion]: A-vs-accepted=50% (1/2), B-vs-accepted=0% (0/2), A-vs-B=50% (1/2)

### v1-p2-un-amour-de-swann#p-105-p-106 -> v1-p2-un-amour-de-swann#p-105-p-106

third-person control

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): M. Verdurin, Mme Verdurin, Swann
- characters_present (A): M. Verdurin, Mme Verdurin, Swann
- characters_present (B): Mme Verdurin[resolved], Odette[resolved], Swann[resolved], M. Verdurin[resolved]
- B-only discoveries: Odette[resolved]
- direction agreement [advantage]: A-vs-accepted=0% (0/3), B-vs-accepted=0% (0/3), A-vs-B=67% (2/3)
- direction agreement [prestige]: A-vs-accepted=33% (1/3), B-vs-accepted=0% (0/3), A-vs-B=67% (2/3)
- direction agreement [inclusion]: A-vs-accepted=0% (0/3), B-vs-accepted=0% (0/3), A-vs-B=67% (2/3)

### v4-p2#p-346-p-350 -> v4-p2#p-334-p-336

roster-rich Verdurin salon

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): marquis de Cambremer
- characters_present (A): Mme de Cambremer, marquis de Cambremer
- characters_present (B): marquis de Cambremer[resolved], Mme de Cambremer[resolved], Mme Verdurin[resolved]
- B-only discoveries: Mme Verdurin[resolved]
- direction agreement [advantage]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [prestige]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)
- direction agreement [inclusion]: A-vs-accepted=100% (1/1), B-vs-accepted=100% (1/1), A-vs-B=100% (2/2)

### v5#p-121-p-125 -> v5#p-117-p-121

narrator-intimate Albertine unit

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): Albertine
- characters_present (A): Albertine
- characters_present (B): le narrateur[resolved], Albertine[resolved]
- B-only discoveries: le narrateur[resolved]
- direction agreement [advantage]: A-vs-accepted=0% (0/1), B-vs-accepted=100% (1/1), A-vs-B=0% (0/1)
- direction agreement [prestige]: A-vs-accepted=0% (0/1), B-vs-accepted=100% (1/1), A-vs-B=0% (0/1)
- direction agreement [inclusion]: A-vs-accepted=0% (0/1), B-vs-accepted=100% (1/1), A-vs-B=0% (0/1)

### v2-p1-autour-de-mme-swann#p-241-p-250 -> v2-p1-autour-de-mme-swann#p-236-p-244

Norpois demolishes Bergotte -- known contested reading

present: accepted=yes, A=yes, B=yes

- characters_present (accepted): Bergotte, Norpois, la mère du narrateur
- characters_present (A): Bergotte, docteur Cottard, la mère du narrateur
- characters_present (B): le narrateur[resolved], Bergotte[resolved], docteur Cottard[resolved], Swann[resolved], le père du narrateur[resolved], la mère du narrateur[resolved]
- B-only discoveries: le narrateur[resolved], Swann[resolved], le père du narrateur[resolved]
- missing vs accepted (A): Norpois
- missing vs accepted (B): Norpois
- direction agreement [advantage]: A-vs-accepted=50% (1/2), B-vs-accepted=0% (0/2), A-vs-B=67% (2/3)
- direction agreement [prestige]: A-vs-accepted=50% (1/2), B-vs-accepted=0% (0/2), A-vs-B=67% (2/3)
- direction agreement [inclusion]: A-vs-accepted=50% (1/2), B-vs-accepted=0% (0/2), A-vs-B=67% (2/3)

