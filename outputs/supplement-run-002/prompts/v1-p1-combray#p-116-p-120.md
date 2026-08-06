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
      "canonical_name": "oncle Adolphe",
      "surface_forms": [
        "mon oncle",
        "oncle"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "oncle Adolphe",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.76,
      "evidence": "« sa trop grande facilité … à des comtesses de nom ronflant … la politesse de les présenter à la grand-mère … l'avait déjà brouillé plus d'une fois avec mon grand-père »; le père du narrateur, en souriant : « Une amie de ton oncle »; et l’oncle « tâchait … d’éviter tout trait d’union entre sa famille et ce genre de relations »",
      "explanation": "The narrator frames the uncle as socially compromised by his relations with actresses/cocottes, causes of family conflicts, and shows his embarrassment by refusing to name or to connect this woman to the family. The father’s quip confirms the social discredit attached to these liaisons."
    }
  ],
  "status_effects": [
    {
      "character": "oncle Adolphe",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.76,
      "explanation": "Locally, his status is lowered: his acquaintances are marked with a social stigma and he seeks to conceal the link with his family, a sign of embarrassment and discredit."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-116-p-120"
}

### Candidate characters

[
  "Remi",
  "la Berma",
  "la grand-mère",
  "la mère du narrateur",
  "le grand-père du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Toutes mes conversations avec mes camarades portaient sur ces acteurs dont l'art, bien qu'il me fût encore inconnu, était la première forme, entre toutes celles qu'il revêt, sous laquelle se laissait pressentir par moi l'Art. Entre la manière que l'un ou l'autre avait de débiter, de nuancer une tirade, les différences les plus minimes me semblaient avoir une importance incalculable. Et, d'après ce que l'on m'avait dit d'eux, je les classais par ordre de talent, dans des listes que je me récitais toute la journée, et qui avaient fini par durcir dans mon cerveau et par le gêner de leur inamovibilité.

### Passage

Plus tard, quand je fus au collège, chaque fois que pendant les classes je correspondais, aussitôt que le professeur avait la tête tournée, avec un nouvel ami, ma première question était toujours pour lui demander s'il était déjà allé au théâtre et s'il trouvait que le plus grand acteur était bien Got, le second Delaunay, etc. Et si, à son avis, Febvre ne venait qu'après Thiron, ou Delaunay qu'après Coquelin, la soudaine motilité que Coquelin, perdant la rigidité de la pierre, contractait dans mon esprit pour y passer au deuxième rang, et l'agilité miraculeuse, la féconde animation dont se voyait doué Delaunay pour reculer au quatrième, rendait la sensation du fleurissement et de la vie à mon cerveau assoupli et fertilisé.

Mais si les acteurs me préoccupaient ainsi, si la vue de Maubant sortant un après-midi du Théâtre-Français m'avait causé le saisissement et les souffrances de l'amour, combien le nom d'une étoile flamboyant à la porte d'un théâtre, combien, à la glace d'un coupé qui passait dans la rue avec ses chevaux fleuris de roses au frontail, la vue du visage d'une femme que je pensais être peut-être une actrice laissait en moi un trouble plus prolongé, un effort impuissant et douloureux pour me représenter sa vie. Je classais par ordre de talent les plus illustres : Sarah Bernhardt, la Berma, Bartet, Madeleine Brohan, Jeanne Samary, mais toutes m'intéressaient. Or mon oncle en connaissait beaucoup et aussi des cocottes que je ne distinguais pas nettement des actrices. Il les recevait chez lui. Et si nous n'allions le voir qu'à certains jours c'est que, les autres jours, venaient des femmes avec lesquelles sa famille n'aurait pas pu se rencontrer, du moins à son avis à elle, car, pour mon oncle, au contraire, sa trop grande facilité à faire à de jolies veuves qui n'avaient peut-être jamais été mariées, à des comtesses de nom ronflant, qui n'était sans doute qu'un nom de guerre, la politesse de les présenter à ma grand'mère ou même à leur donner des bijoux de famille, l'avait déjà brouillé plus d'une fois avec mon grand-père. Souvent, à un nom d'actrice qui venait dans la conversation, j'entendais mon père dire à ma mère, en souriant : « Une amie de ton oncle » ; et je pensais que le stage que peut-être pendant des années des hommes importants faisaient inutilement à la porte de telle femme qui ne répondait pas à leurs lettres et les faisait chasser par le concierge de son hôtel, mon oncle aurait pu en dispenser un gamin comme moi en le présentant chez lui à l'actrice, inapprochable à tant d'autres, qui était pour lui une intime amie.

Aussi – sous le prétexte qu'une leçon qui avait été déplacée tombait maintenant si mal qu'elle m'avait empêché plusieurs fois et m'empêcherait encore de voir mon oncle – un jour, autre que celui qui était réservé aux visites que nous lui faisions, profitant de ce que mes parents avaient déjeuné de bonne heure, je sortis et au lieu d'aller regarder la colonne d'affiches, pour quoi on me laissait aller seul, je courus jusqu'à lui. Je remarquai devant sa porte une voiture attelée de deux chevaux qui avaient aux oeillères un oeillet rouge comme avait le cocher à sa boutonnière. De l'escalier j'entendis un rire et une voix de femme, et dès que j'eus sonné, un silence, puis le bruit de portes qu'on fermait. Le valet de chambre vint ouvrir, et en me voyant parut embarrassé, me dit que mon oncle était très occupé, ne pourrait sans doute pas me recevoir, et, tandis qu'il allait pourtant le prévenir, la même voix que j'avais entendue disait : « Oh, si ! laisse-le entrer ; rien qu'une minute, cela m'amuserait tant. Sur la photographie qui est sur ton bureau, il ressemble tant à sa maman, ta nièce, dont la photographie est à côté de la sienne, n'est-ce pas ? Je voudrais le voir rien qu'un instant, ce gosse. »

J'entendis mon oncle grommeler, se fâcher ; finalement le valet de chambre me fit entrer.

Sur la table, il y avait la même assiette de massepains que d'habitude ; mon oncle avait sa vareuse de tous les jours, mais en face de lui, en robe de soie rose avec un grand collier de perles au cou, était assise une jeune femme qui achevait de manger une mandarine. L'incertitude où j'étais s'il fallait dire madame ou mademoiselle me fit rougir et, n'osant pas trop tourner les yeux de son côté de peur d'avoir à lui parler, j'allai embrasser mon oncle. Elle me regardait en souriant, mon oncle lui dit : « Mon neveu », sans lui dire mon nom, ni me dire le sien, sans doute parce que, depuis les difficultés qu'il avait eues avec mon grand-père, il tâchait autant que possible d'éviter tout trait d'union entre sa famille et ce genre de relations.
