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
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Mme Cottard",
      "surface_forms": [
        "Mme Cottard"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme Cottard",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Avant que « la gentillesse native » ait percé « l'empesé de la petite bourgeoise »; elle tient des « propos choisis » qu'elle « entendait et répétait »; ses dires sont « inspirés » par « la hauteur de son aigrette... le petit numéro tracé... dans ses gants ».",
      "explanation": "The narrator frames Mme Cottard as a stiff petty-bourgeois, repeating society clichés and guided by outward signs, which lowers her intellectually and socially in the scene."
    }
  ],
  "status_effects": [
    {
      "character": "Mme Cottard",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "She appears awkward, conventional, and unoriginal, which lowers her local estimation by the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-574-p-578"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette",
  "Swann",
  "docteur Cottard",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Quelquefois il allait dans des maisons de rendez-vous, espérant apprendre quelque chose d'elle, sans oser la nommer cependant. « J'ai une petite qui va vous plaire », disait l'entremetteuse. » Et il restait une heure à causer tristement avec quelque pauvre fille étonnée qu'il ne fît rien de plus. Une toute jeune et ravissante lui dit un jour : « Ce que je voudrais, c'est trouver un ami, alors il pourrait être sûr, je n'irais plus jamais avec personne. » – « Vraiment, crois-tu que ce soit possible qu'une femme soit touchée qu'on l'aime, ne vous trompe jamais ? » lui demanda Swann anxieusement. – « Pour sûr ! ça dépend des caractères ! » Swann ne pouvait s'empêcher de dire à ces filles les mêmes choses qui auraient plu à la princesse des Laumes. À celle qui cherchait un ami, il dit en souriant : « C'est gentil, tu as mis des yeux bleus de la couleur de ta ceinture. » – « Vous aussi, vous avez des manchettes bleues. » – « Comme nous avons une belle conversation, pour un endroit de ce genre ! Je ne t'ennuie pas, tu as peut-être à faire ? » – « Non, j'ai tout mon temps. Si vous m'aviez ennuyée, je vous l'aurais dit. Au contraire j'aime bien vous entendre causer. » – « Je suis très flatté. N'est-ce pas que nous causons gentiment ? » dit-il à l'entremetteuse qui venait d'entrer. – « Mais oui, c'est justement ce que je me disais. Comme ils sont sages ! Voilà ! on vient maintenant pour causer chez moi. Le Prince le disait, l'autre jour, c'est bien mieux ici que chez sa femme. Il paraît que maintenant dans le monde elles ont toutes un genre, c'est un vrai scandale ! Je vous quitte, je suis discrète. » Et elle laissa Swann avec la fille qui avait les yeux bleus. Mais bientôt il se leva et lui dit adieu, elle lui était indifférente, elle ne connaissait pas Odette.

### Passage

Le peintre ayant été malade, le docteur Cottard lui conseilla un voyage en mer ; plusieurs fidèles parlèrent de partir avec lui ; les Verdurin ne purent se résoudre à rester seuls, louèrent un yacht, puis s'en rendirent acquéreurs et ainsi Odette fit de fréquentes croisières. Chaque fois qu'elle était partie depuis un peu de temps, Swann sentait qu'il commençait à se détacher d'elle, mais comme si cette distance morale était proportionnée à la distance matérielle, dès qu'il savait Odette de retour, il ne pouvait pas rester sans la voir. Une fois, partis pour un mois seulement, croyaient-ils, soit qu'ils eussent été tentés en route, soit que M. Verdurin eût sournoisement arrangé les choses d'avance pour faire plaisir à sa femme et n'eût averti les fidèles qu'au fur et à mesure, d'Alger, ils allèrent à Tunis, puis en Italie, puis en Grèce, à Constantinople, en Asie Mineure. Le voyage durait depuis près d'un an. Swann se sentait absolument tranquille, presque heureux. Bien que M. Verdurin eût cherché à persuader au pianiste et au docteur Cottard que la tante de l'un et les malades de l'autre n'avaient aucun besoin d'eux, et, qu'en tous cas il était imprudent de laisser Mme Cottard rentrer à Paris que Mme Verdurin assurait être en révolution, il fut obligé de leur rendre leur liberté à Constantinople. Et le peintre partit avec eux. Un jour, peu après le retour de ces trois voyageurs, Swann voyant passer un omnibus pour le Luxembourg où il avait à faire, avait sauté dedans, et s'y était trouvé assis en face de Mme Cottard qui faisait sa tournée de visites « de jours » en grande tenue, plumet au chapeau, robe de soie, manchon, en-tout-cas, porte-cartes, et gants blancs nettoyés. Revêtue de ces insignes, quand il faisait sec elle allait à pied d'une maison à l'autre, dans un même quartier, mais pour passer ensuite dans un quartier différent usait de l'omnibus avec correspondance. Pendant les premiers instants, avant que la gentillesse native de la femme eût pu percer l'empesé de la petite bourgeoise, et ne sachant trop d'ailleurs si elle devait parler des Verdurin à Swann, elle tint tout naturellement, de sa voix lente, gauche et douce que par moments l'omnibus couvrait complètement de son tonnerre, des propos choisis parmi ceux qu'elle entendait et répétait dans les vingt-cinq maisons dont elle montait les étages dans une journée :

– Je ne vous demande pas, monsieur, si un homme dans le mouvement comme vous, a vu, aux Mirlitons, le portrait de Machard qui fait courir tout Paris. Eh bien ! qu'en dites-vous ? Êtes-vous dans le camp de ceux qui approuvent ou dans le camp de ceux qui blâment ? Dans tous les salons on ne parle que du portrait de Machard ; on n'est pas chic, on n'est pas pur, on n'est pas dans le train, si on ne donne pas son opinion sur le portrait de Machard.

Swann ayant répondu qu'il n'avait pas vu ce portrait, Mme Cottard eut peur de l'avoir blessé en l'obligeant à le confesser.

– Ah ! c'est très bien, au moins vous l'avouez franchement, vous ne vous croyez pas déshonoré parce que vous n'avez pas vu le portrait de Machard. Je trouve cela très beau de votre part. Hé bien, moi je l'ai vu, les avis sont partagés, il y en a qui trouvent que c'est un peu léché, un peu crème fouettée, moi, je le trouve idéal. Évidemment elle ne ressemble pas aux femmes bleues et jaunes de notre ami Biche. Mais je dois vous l'avouer franchement, vous ne me trouverez pas très fin de siècle, mais je le dis comme je le pense, je ne comprends pas. Mon Dieu je reconnais les qualités qu'il y a dans le portrait de mon mari, c'est moins étrange que ce qu'il fait d'habitude, mais il a fallu qu'il lui fasse des moustaches bleues. Tandis que Machard ! Tenez justement le mari de l'amie chez qui je vais en ce moment (ce qui me donne le très grand plaisir de faire route avec vous) lui a promis s'il est nommé à l'Académie (c'est un des collègues du docteur), de lui faire faire son portrait par Machard. Évidemment c'est un beau rêve ! j'ai une autre amie qui prétend qu'elle aime mieux Leloir. Je ne suis qu'une pauvre profane et Leloir est peut-être encore supérieur comme science. Mais je trouve que la première qualité d'un portrait, surtout quand il coûte 10.000 francs, est d'être ressemblant et d'une ressemblance agréable.

Ayant tenu ces propos que lui inspiraient la hauteur de son aigrette, le chiffre de son porte-cartes, le petit numéro tracé à l'encre dans ses gants par le teinturier et l'embarras de parler à Swann des Verdurin, Mme Cottard, voyant qu'on était encore loin du coin de la rue Bonaparte où le conducteur devait l'arrêter, écouta son coeur qui lui conseillait d'autres paroles.
