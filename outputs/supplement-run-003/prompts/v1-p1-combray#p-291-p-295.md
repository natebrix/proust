You are annotating a French passage from Marcel Proust's *À la recherche du temps perdu* for **local appraisal events** and **character status effects**.

This is a **supplemental coverage pass**. The passage has already been annotated once. That accepted annotation captured the dominant local movement and its focal characters, and it is **fixed** — you must not re-score, revise, or contradict it.

Your job is narrower: judge whether any of the **additional candidate characters** listed below are **materially involved** in the local social or evaluative dynamics of the passage, and score **only those characters**.

## Inputs

You will be given:

1. A French passage.
2. An alias map for named characters.
3. The **accepted annotation** for this passage (characters already scored, with their events and status effects). This is fixed context, not a draft to improve.
4. A **candidate list** of additional characters detected in the passage text but not scored in the accepted annotation. The candidate list may include `le narrateur`.
5. Optionally, brief prior context from the immediately preceding window.

## Scope rules

* Score **only** characters from the candidate list. Never emit events or status effects whose target is an already-scored character.
* An already-scored character **may** appear as the `source` of an event targeting a candidate character.
* The candidate list is a mechanical screen, not a quota. Most candidates are peripheral mentions and should be **omitted**.
* Include a candidate only if omitting them would misrepresent how the passage locally positions its participants.
* Resolve references to the **canonical character name** using the alias map.
* Work primarily from the passage itself. Use prior context only for local disambiguation.
* Do not invent motives, unstated events, or long-run arc interpretations.
* Prefer the **smallest sufficient reading** of the passage.
* An **empty result** (`appraisal_events: []`, `status_effects: []`, and only trivially-present `characters_present`) is a valid, common, and expected outcome. Do not manufacture weak events to justify a candidate.

## The narrator as participant

`le narrateur` may appear in the candidate list. Distinguish carefully between two roles:

* **The narrating voice** — the retrospective "I" who tells, evaluates, and ironizes. This voice remains an evaluation `source` (use `"source": "narrator"` as in the accepted annotation). The voice is **never** a scored character.
* **The in-scene self** — the protagonist as a participant in the staged scene: he is received or snubbed, favored or dismissed, gains or loses composure, standing, or emotional leverage relative to the people in the room. This in-scene self is scored as the character `le narrateur`.

Score `le narrateur` only when the passage **stages** him as a social participant:

* he is included in or excluded from valued company
* another character defers to, favors, dismisses, or dominates him
* he gains or loses emotional leverage in a staged interaction (e.g., with Albertine or Gilberte)
* the scene's social outcome lands on him as a participant, not merely through him as a lens

Do **not** score `le narrateur` when:

* he is only the perceiving or remembering consciousness
* the passage is essayistic reflection, description, or generalization
* his "loss" or "gain" exists only at the level of retrospective commentary

In third-person stretches (notably *Un amour de Swann*), `le narrateur` should almost never be scored.

## What to detect

For candidate characters, track the same local shifts as the first pass:

* praise, blame, admiration, snub
* prestige or discredit by association
* narrated elevation or diminishment
* inclusion in or exclusion from valued social space
* signs that another character depends on, yields to, or dismisses them

## Interpretive principles

All interpretive rules of the first pass apply unchanged:

* judge only the local evaluative and social dynamics of the supplied passage
* do not judge morality, factual correctness, long-term importance, or desert
* distinguish who evaluates, who is targeted, and whether the passage endorses, neutrally reports, ironizes, or leaves uncertain that evaluation
* respect quoted speech, free indirect style, irony, and narrator distance
* do not force zero-sum logic — a candidate can gain or lose independently of the already-scored characters
* the consummation-and-renewal rule from the first pass applies: do not collapse attained intimacy or narrator-endorsed renewal into diminishment merely because the path was hesitant or dependent

## Relation to the accepted annotation

* The accepted annotation defines the dominant local movement. Do not restate it.
* Your events should cover the **remaining** participants' positioning, which is often quieter: a hostess's successful reception, a rival's eclipse, a servant's competence acknowledged, the narrator's admission or exclusion.
* If a candidate's only involvement is as part of the movement already captured (e.g., a collective source of an existing snub), and the passage gives them no distinct local outcome of their own, omit them.
* Never emit an event that reverses the direction of an accepted event for the same interaction. If you believe the accepted annotation is wrong, record that in `ambiguities` — do not correct it through scoring.

## Task

1. From the candidate list, identify which characters (if any) are materially involved in the local movement.
2. Extract only the **significant** appraisal or status-relevant events involving them.
3. Record only the dominant local status effects for those characters.
4. Note ambiguity only when it materially changes the reading.
5. Prefer fewer, high-quality events. Default to **0 or 1** events. Never more than **3** events total, and only reach 3 when distinct candidates have genuinely distinct movements.
6. Never more than **2 status effects** for a single character.

## Output

Return valid JSON only, in exactly the first-pass schema:

{
"characters_present": [
{
"canonical_name": "string",
"surface_forms": ["string"],
"presence_type": "explicit | implicit",
"presence_confidence": 0.0
}
],
"appraisal_events": [
{
"event_id": "S1",
"source": "canonical character name | narrator | collective_social_voice | unknown",
"target": "canonical character name",
"type": "praise | blame | admiration | snub | prestige_association | discredit_association | narrated_elevation | narrated_diminishment | other",
"polarity": "positive | negative | mixed",
"narrative_stance": "endorsed | neutral_report | ironized | uncertain",
"confidence": 0.0,
"evidence": "brief quotation or paraphrase from the passage",
"explanation": "1-2 sentence explanation in English"
}
],
"status_effects": [
{
"character": "canonical character name",
"dimension": "general_appraisal | social_status | rhetorical_position | emotional_position | inclusion_exclusion",
"delta": -2,
"based_on_events": ["S1"],
"confidence": 0.0,
"explanation": "brief explanation in English"
}
],
"ambiguities": [
"string"
]
}

Schema guidance:

* `characters_present` lists only the candidate characters you actually scored (or judged explicitly implicit-but-material). Do not relist already-scored characters.
* Event ids use the `S` prefix (`S1`, `S2`, ...) so supplement events are distinguishable from first-pass events (`E1`, ...).
* `status_effects` targets must be candidate characters only.
* Delta scale, dimensions, stance values, and confidence conventions are identical to the first pass:
  * delta: -2 clearly diminished ... +2 clearly elevated
  * be conservative when irony, layered narration, or reference resolution makes interpretation unstable
* `explanation` fields must be written in English.
* `ambiguities` defaults to an empty list.

## Important rules

* Candidate characters only. Canonical names only.
* The accepted annotation is fixed; never re-score its characters.
* An empty supplement is a good supplement when the candidates are peripheral.
* Do not add a winner/loser verdict, a summary object, or fields beyond the schema.
* Do not turn one movement into a chain of micro-events.
* Do not add balancing effects unless both directions are central for that candidate.

## Inputs begin below

### Alias map

{
  "Swann": {
    "aliases": [
      "Swann",
      "M. Swann",
      "Charles Swann"
    ]
  },
  "Legrandin": {
    "aliases": [
      "Legrandin",
      "M. Legrandin"
    ]
  },
  "Mme de Villeparisis": {
    "aliases": [
      "Mme de Villeparisis",
      "Madame de Villeparisis"
    ]
  },
  "Mme de Cambremer": {
    "aliases": [
      "Mme de Cambremer",
      "Madame de Cambremer"
    ]
  },
  "M. Vinteuil": {
    "aliases": [
      "M. Vinteuil",
      "Vinteuil"
    ]
  },
  "la mère du narrateur": {
    "aliases": [
      "maman",
      "ma mère"
    ]
  },
  "Odette": {
    "aliases": [
      "Odette",
      "Odette de Crécy",
      "Odette de Crecy",
      "Mme de Crécy",
      "Mme de Crecy"
    ]
  },
  "Mme Verdurin": {
    "aliases": [
      "Mme Verdurin",
      "Madame Verdurin"
    ]
  },
  "M. Verdurin": {
    "aliases": [
      "M. Verdurin",
      "Monsieur Verdurin",
      "Verdurin"
    ]
  },
  "comte de Forcheville": {
    "aliases": [
      "Forcheville",
      "comte de Forcheville",
      "M. de Forcheville"
    ]
  },
  "Brichot": {
    "aliases": [
      "Brichot",
      "M. Brichot"
    ]
  },
  "docteur Cottard": {
    "aliases": [
      "Cottard",
      "docteur Cottard",
      "le docteur"
    ]
  },
  "Mme Cottard": {
    "aliases": [
      "Mme Cottard",
      "Madame Cottard"
    ]
  },
  "Saniette": {
    "aliases": [
      "Saniette"
    ]
  },
  "le peintre": {
    "aliases": [
      "le peintre",
      "peintre"
    ]
  },
  "marquis de Forestelle": {
    "aliases": [
      "marquis de Forestelle",
      "M. de Forestelle",
      "Forestelle"
    ]
  },
  "baron de Charlus": {
    "aliases": [
      "baron de Charlus",
      "Charlus"
    ]
  },
  "oncle Adolphe": {
    "aliases": [
      "mon oncle Adolphe",
      "oncle Adolphe",
      "Adolphe"
    ]
  },
  "marquise de Saint-Euverte": {
    "aliases": [
      "marquise de Saint-Euverte",
      "Mme de Saint-Euverte",
      "Saint-Euverte"
    ]
  },
  "général de Froberville": {
    "aliases": [
      "général de Froberville",
      "general de Froberville",
      "Froberville"
    ]
  },
  "marquis de Bréauté": {
    "aliases": [
      "marquis de Bréauté",
      "marquis de Breaute",
      "Bréauté",
      "Breaute"
    ]
  },
  "marquise de Gallardon": {
    "aliases": [
      "marquise de Gallardon",
      "Mme de Gallardon",
      "Gallardon"
    ]
  },
  "duc de Guermantes": {
    "aliases": [
      "duc de Guermantes"
    ]
  },
  "princesse de Parme": {
    "aliases": [
      "princesse de Parme"
    ]
  },
  "M. d'Orsan": {
    "aliases": [
      "M. d'Orsan",
      "d'Orsan",
      "Orsan"
    ]
  },
  "Rémi": {
    "aliases": [
      "Rémi",
      "Remi"
    ]
  },
  "comtesse de Monteriender": {
    "aliases": [
      "comtesse de Monteriender",
      "Mme de Monteriender",
      "Monteriender"
    ]
  },
  "Napoléon III": {
    "aliases": [
      "Napoléon III",
      "Napoleon III"
    ]
  },
  "Gilberte": {
    "aliases": [
      "Gilberte"
    ]
  },
  "Françoise": {
    "aliases": [
      "Françoise",
      "Francoise"
    ]
  },
  "la Berma": {
    "aliases": [
      "la Berma",
      "Berma"
    ]
  },
  "Bergotte": {
    "aliases": [
      "Bergotte"
    ]
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "le marquis de Norpois"
    ]
  },
  "la grand-mère": {
    "aliases": [
      "ma grand-mère",
      "grand-mère",
      "ma grand'mère",
      "grand'mère",
      "la grand-mère"
    ]
  },
  "M. de Stermaria": {
    "aliases": [
      "M. de Stermaria",
      "de Stermaria",
      "Stermaria"
    ]
  },
  "Aimé": {
    "aliases": [
      "Aimé",
      "Aime"
    ]
  },
  "Mlle de Stermaria": {
    "aliases": [
      "Mlle de Stermaria"
    ]
  },
  "marquis de Cambremer": {
    "aliases": [
      "marquis de Cambremer",
      "M. de Cambremer"
    ]
  },
  "princesse de Luxembourg": {
    "aliases": [
      "princesse de Luxembourg",
      "La princesse de Luxembourg"
    ]
  },
  "le père du narrateur": {
    "aliases": [
      "mon père",
      "votre père"
    ]
  },
  "Mme Blandais": {
    "aliases": [
      "Mme Blandais",
      "Madame Blandais"
    ]
  },
  "Mme Poncin": {
    "aliases": [
      "Mme Poncin",
      "Madame Poncin"
    ]
  },
  "Robert de Saint-Loup": {
    "aliases": [
      "Saint-Loup",
      "Robert de Saint-Loup",
      "marquis de Saint-Loup-en-Bray",
      "le neveu de Mme de Villeparisis"
    ]
  },
  "M. de Marsantes": {
    "aliases": [
      "M. de Marsantes",
      "Marsantes",
      "Saint-Loup de Saint-Loup"
    ]
  },
  "Bloch": {
    "aliases": [
      "Bloch",
      "Bloch fils"
    ]
  },
  "prince des Laumes": {
    "aliases": [
      "prince des Laumes"
    ]
  },
  "Bloch père": {
    "aliases": [
      "Bloch père"
    ]
  },
  "le directeur": {
    "aliases": [
      "le directeur",
      "directeur"
    ]
  },
  "Dreyfus": {
    "aliases": [
      "Dreyfus"
    ]
  },
  "jeune blonde de Rivebelle": {
    "aliases": [
      "jeune blonde",
      "jeune blonde à l'air triste"
    ]
  },
  "duchesse de Guermantes": {
    "aliases": [
      "duchesse de Guermantes",
      "Mme de Guermantes",
      "Madame de Guermantes",
      "la duchesse"
    ]
  },
  "Jupien": {
    "aliases": [
      "Jupien"
    ]
  },
  "princesse de Guermantes": {
    "aliases": [
      "princesse de Guermantes",
      "princesse de Guermantes-Bavière",
      "Mme de Guermantes-Bavière"
    ]
  },
  "duc de Châtellerault": {
    "aliases": [
      "duc de Châtellerault",
      "M. de Châtellerault",
      "Châtellerault"
    ]
  },
  "M. de Vaugoubert": {
    "aliases": [
      "M. de Vaugoubert",
      "Vaugoubert"
    ]
  },
  "Mme de Vaugoubert": {
    "aliases": [
      "Mme de Vaugoubert",
      "Madame de Vaugoubert"
    ]
  },
  "Albertine": {
    "aliases": [
      "Albertine"
    ]
  },
  "Andrée": {
    "aliases": [
      "Andrée",
      "Andree"
    ]
  },
  "Mme Bontemps": {
    "aliases": [
      "Mme Bontemps",
      "Madame Bontemps"
    ]
  },
  "Morel": {
    "aliases": [
      "Morel"
    ]
  },
  "Elstir": {
    "aliases": [
      "Elstir"
    ]
  },
  "prince de Léon": {
    "aliases": [
      "prince de Léon",
      "prince de Leon",
      "Léon",
      "Leon"
    ]
  },
  "marquis du Lau": {
    "aliases": [
      "marquis du Lau",
      "du Lau"
    ]
  },
  "Mme de Chaussepierre": {
    "aliases": [
      "Mme de Chaussepierre",
      "Madame de Chaussepierre",
      "Chaussepierre"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "la propriété de Swann",
        "parc de Swann",
        "mariage de Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "Swann",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.82,
      "evidence": "« mes parents n’allant plus à Tansonville depuis le mariage de Swann »; « Nous pourrions longer le parc, puisque ces dames ne sont pas là »; et ils prenaient un autre chemin « pour ne pas avoir l’air de regarder dans le parc ».",
      "explanation": "The family’s practice of avoiding Tansonville after Swann’s marriage and only approaching the park when his wife and daughter are absent functions as a local social snub; Swann is diminished by association with his household."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "He is locally excluded as the family avoids visiting his estate and even detours to avoid seeming to look into his park when his wife and daughter are present."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-291-p-295"
}

### Candidate characters

[
  "Françoise",
  "le grand-père du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Mais je croyais que vous le saviez, Léonie, disait la mère du narrateur. Je pensais que Françoise nous avait vus sortir par la petite porte du potager.

### Passage

Car il y avait autour de Combray deux « côtés » pour les promenades, et si opposés qu'on ne sortait pas en effet de chez nous par la même porte, quand on voulait aller d'un côté ou de l'autre : le côté de Méséglise-la-Vineuse, qu'on appelait aussi le côté de chez Swann parce qu'on passait devant la propriété de Swann pour aller par là, et le côté de Guermantes. De Méséglise-la-Vineuse, à vrai dire, je n'ai jamais connu que le « côté » et des gens étrangers qui venaient le dimanche se promener à Combray, des gens que, cette fois, ma tante elle-même et nous tous ne « connaissions point » et qu'à ce signe on tenait pour « des gens qui seront venus de Méséglise ». Quant à Guermantes je devais un jour en connaître davantage, mais bien plus tard seulement ; et pendant toute mon adolescence, si Méséglise était pour moi quelque chose d'inaccessible comme l'horizon, dérobé à la vue, si loin qu'on allât, par les plis d'un terrain qui ne ressemblait déjà plus à celui de Combray, Guermantes, lui, ne m'est apparu que comme le terme plutôt idéal que réel de son propre « côté », une sorte d'expression géographique abstraite comme la ligne de l'équateur, comme le pôle, comme l'orient. Alors, « prendre par Guermantes » pour aller à Méséglise, ou le contraire, m'eût semblé une expression aussi dénuée de sens que prendre par l'est pour aller à l'ouest. Comme mon père parlait toujours du côté de Méséglise comme de la plus belle vue de la plaine qu'il connût et du côté de Guermantes comme du type de paysage de rivière, je leur donnais, en les concevant ainsi comme deux entités, cette cohésion, cette unité qui n'appartiennent qu'aux créations de notre esprit ; la moindre parcelle de chacun d'eux me semblait précieuse et manifester leur excellence particulière, tandis qu'à côté d'eux, avant qu'on fût arrivé sur le sol sacré de l'un ou de l'autre, les chemins purement matériels au milieu desquels ils étaient posés comme l'idéal de la vue de plaine et l'idéal du paysage de rivière, ne valaient pas plus la peine d'être regardés que par le spectateur épris d'art dramatique les petites rues qui avoisinent un théâtre. Mais surtout je mettais entre eux, bien plus que leurs distances kilométriques, la distance qu'il y avait entre les deux parties de mon cerveau où je pensais à eux, une de ces distances dans l'esprit qui ne font pas qu'éloigner, qui séparent et mettent dans un autre plan. Et cette démarcation était rendue plus absolue encore parce que cette habitude que nous avions de n'aller jamais vers les deux côtés un même jour, dans une seule promenade, mais une fois du côté de Méséglise, une fois du côté de Guermantes, les enfermait pour ainsi dire loin l'un de l'autre, inconnaissables l'un à l'autre, dans les vases clos et sans communication entre eux d'après-midi différents.

Quand on voulait aller du côté de Méséglise, on sortait (pas trop tôt et même si le ciel était couvert, parce que la promenade n'était pas bien longue et n'entraînait pas trop) comme pour aller n'importe où, par la grande porte de la maison de ma tante sur la rue du Saint-Esprit. On était salué par l'armurier, on jetait ses lettres à la boîte, on disait en passant à Théodore, de la part de Françoise, qu'elle n'avait plus d'huile ou de café, et l'on sortait de la ville par le chemin qui passait le long de la barrière blanche du parc de Swann. Avant d'y arriver, nous rencontrions, venue au-devant des étrangers, l'odeur de ses lilas. Eux-mêmes, d'entre les petits coeurs verts et frais de leurs feuilles, levaient curieusement au-dessus de la barrière du parc leurs panaches de plumes mauves ou blanches que lustrait, même à l'ombre, le soleil où elles avaient baigné. Quelques-uns, à demi cachés par la petite maison en tuiles appelée maison des Archers, où logeait le gardien, dépassaient son pignon gothique de leur rose minaret. Les Nymphes du printemps eussent semblé vulgaires, auprès de ces jeunes houris qui gardaient dans ce jardin français les tons vifs et purs des miniatures de la Perse. Malgré mon désir d'enlacer leur taille souple et d'attirer à moi les boucles étoilées de leur tête odorante, nous passions sans nous arrêter, mes parents n'allant plus à Tansonville depuis le mariage de Swann, et, pour ne pas avoir l'air de regarder dans le parc, au lieu de prendre le chemin qui longe sa clôture et qui monte directement aux champs, nous en prenions un autre qui y conduit aussi, mais obliquement, et nous faisait déboucher trop loin. Un jour, mon grand-père dit à mon père :

– Vous rappelez-vous que Swann a dit hier que, comme sa femme et sa fille partaient pour Reims, il en profiterait pour aller passer vingt-quatre heures à Paris ? Nous pourrions longer le parc, puisque ces dames ne sont pas là, cela nous abrégerait d'autant.

Nous nous arrêtâmes un moment devant la barrière. Le temps des lilas approchait de sa fin ; quelques-uns effusaient encore en hauts lustres mauves les bulles délicates de leurs fleurs, mais dans bien des parties du feuillage où déferlait, il y avait seulement une semaine, leur mousse embaumée, se flétrissait, diminuée et noircie, une écume creuse, sèche et sans parfum. Mon grand-père montrait à mon père en quoi l'aspect des lieux était resté le même, et en quoi il avait changé, depuis la promenade qu'il avait faite avec Swann le jour de la mort de sa femme, et il saisit cette occasion pour raconter cette promenade une fois de plus.

Devant nous, une allée bordée de capucines montait en plein soleil vers le château. À droite, au contraire, le parc s'étendait en terrain plat. Obscurcie par l'ombre des grands arbres qui l'entouraient, une pièce d'eau avait été creusée par les parents de Swann ; mais dans ses créations les plus factices, c'est sur la nature que l'homme travaille ; certains lieux font toujours régner autour d'eux leur empire particulier, arborent leurs insignes immémoriaux au milieu d'un parc comme ils auraient fait loin de toute intervention humaine, dans une solitude qui revient partout les entourer, surgie des nécessités de leur exposition et superposée à l'oeuvre humaine. C'est ainsi qu'au pied de l'allée qui dominait l'étang artificiel, s'était composée sur deux rangs, tressés de fleurs de myosotis et de pervenches, la couronne naturelle, délicate et bleue qui ceint le front clair-obscur des eaux, et que le glaïeul, laissant fléchir ses glaives avec un abandon royal, étendait sur l'eupatoire et la grenouillette au pied mouillé les fleurs de lis en lambeaux, violettes et jaunes, de son sceptre lacustre.
