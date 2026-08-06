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
  "Remi": {
    "aliases": [
      "Rémi",
      "Remi",
      "le cocher"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« Ce simple croquis bouleversait Swann… »; il veut savoir à qui elle a cherché à plaire et se promet de l’interroger.",
      "explanation": "A casual report of Odette seen in a distinctive outfit exposes to Swann that she has a life beyond him, provoking jealousy and insecurity."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "He is unsettled and jealous upon realizing Odette’s independent daytime life."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-185-p-186"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Verdurin",
  "Odette"
]

### Prior local context (optional)

Chaque baiser appelle un autre baiser. Ah ! dans ces premiers temps où l'on aime, les baisers naissent si naturellement ! Ils foisonnent si pressés les uns contre les autres ; et l'on aurait autant de peine à compter les baisers qu'on s'est donnés pendant une heure que les fleurs d'un champ au mois de mai. Alors elle faisait mine de s'arrêter, disant : « Comment veux-tu que je joue comme cela si tu me tiens ? je ne peux tout faire à la fois ; sache au moins ce que tu veux ; est-ce que je dois jouer la phrase ou faire des petites caresses ? » ; lui se fâchait et elle éclatait d'un rire qui se changeait et retombait sur lui, en une pluie de baisers. Ou bien elle le regardait d'un air maussade, il revoyait un visage digne de figurer dans la Vie de Moïse de Botticelli, il l'y situait, il donnait au cou d'Odette l'inclinaison nécessaire ; et quand il l'avait bien peinte à la détrempe, au XVe siècle, sur la muraille de la Sixtine, l'idée qu'elle était cependant restée là, près du piano, dans le moment actuel, prête à être embrassée et possédée, l'idée de sa matérialité et de sa vie venait l'enivrer avec une telle force que, l'oeil égaré, les mâchoires tendues comme pour dévorer, il se précipitait sur cette vierge de Botticelli et se mettait à lui pincer les joues. Puis, une fois qu'il l'avait quittée, non sans être rentré pour l'embrasser encore parce qu'il avait oublié d'emporter dans son souvenir quelque particularité de son odeur ou de ses traits, il revenait dans sa victoria, bénissant Odette de lui permettre ces visites quotidiennes, dont il sentait qu'elles ne devaient pas lui causer à elle une bien grande joie, mais qui en le préservant de devenir jaloux – en lui ôtant l'occasion de souffrir de nouveau du mal qui s'était déclaré en lui le soir où il ne l'avait pas trouvée chez les Verdurin – l'aideraient à arriver, sans avoir plus d'autres de ces crises dont la première avait été si douloureuse et resterait la seule, au bout de ces heures singulières de sa vie, heures presque enchantées, à la façon de celles où il traversait Paris au clair de lune. Et, remarquant, pendant ce retour, que l'astre était maintenant déplacé par rapport à lui, et presque au bout de l'horizon, sentant que son amour obéissait, lui aussi, à des lois immuables et naturelles, il se demandait si cette période où il était entré durerait encore longtemps, si bientôt sa pensée ne verrait plus le cher visage qu'occupant une position lointaine et diminuée, et près de cesser de répandre du charme. Car Swann en trouvait aux choses, depuis qu'il était amoureux, comme au temps où, adolescent, il se croyait artiste ; mais ce n'était plus le même charme ; celui-ci, c'est Odette seule qui le leur conférait. Il sentait renaître en lui les inspirations de sa jeunesse qu'une vie frivole avait dissipées, mais elles portaient toutes le reflet, la marque d'un être particulier ; et, dans les longues heures qu'il prenait maintenant un plaisir délicat à passer chez lui, seul avec son âme en convalescence, il redevenait peu à peu lui-même, mais à une autre.

### Passage

Il n'allait chez elle que le soir, et il ne savait rien de l'emploi de son temps pendant le jour, pas plus que de son passé, au point qu'il lui manquait même ce petit renseignement initial qui, en nous permettant de nous imaginer ce que nous ne savons pas, nous donne envie de le connaître. Aussi ne se demandait-il pas ce qu'elle pouvait faire, ni quelle avait été sa vie. Il souriait seulement quelquefois en pensant qu'il y a quelques années, quand il ne la connaissait pas, on lui avait parlé d'une femme qui, s'il se rappelait bien, devait certainement être elle, comme d'une fille, d'une femme entretenue, une de ces femmes auxquelles il attribuait encore, comme il avait peu vécu dans leur société, le caractère entier, foncièrement pervers, dont les dota longtemps l'imagination de certains romanciers. Il se disait qu'il n'y a souvent qu'à prendre le contre-pied des réputations que fait le monde pour juger exactement une personne quand à un tel caractère il opposait celui d'Odette, bonne, naïve, éprise d'idéal, presque si incapable de ne pas dire la vérité, que l'ayant un jour priée, pour pouvoir dîner seul avec elle, d'écrire aux Verdurin qu'elle était souffrante, le lendemain, il l'avait vue, devant Mme Verdurin qui lui demandait si elle allait mieux, rougir, balbutier et refléter malgré elle, sur son visage, le chagrin, le supplice que cela lui était de mentir, et, tandis qu'elle multipliait dans sa réponse les détails inventés sur sa prétendue indisposition de la veille, avoir l'air de faire demander pardon par ses regards suppliants et sa voix désolée de la fausseté de ses paroles.

Certains jours pourtant, mais rares, elle venait chez lui dans l'après-midi, interrompre sa rêverie ou cette étude sur Ver Meer à laquelle il s'était remis dernièrement. On venait lui dire que Mme de Crécy était dans son petit salon. Il allait l'y retrouver, et quand il ouvrait la porte, au visage rosé d'Odette, dès qu'elle avait aperçu Swann, venait – changeant la forme de sa bouche, le regard de ses yeux, le modelé de ses joues – se mélanger un sourire. Une fois seul, il revoyait ce sourire, celui qu'elle avait eu la veille, un autre dont elle l'avait accueilli telle ou telle fois, celui qui avait été sa réponse, en voiture, quand il lui avait demandé s'il lui était désagréable en redressant les catleyas ; et la vie d'Odette pendant le reste du temps, comme il n'en connaissait rien, lui apparaissait avec son fond neutre et sans couleur, semblable à ces feuilles d'études de Watteau, où on voit çà et là, à toutes les places, dans tous les sens, dessinés aux trois crayons sur le papier chamois, d'innombrables sourires. Mais, parfois, dans un coin de cette vie que Swann voyait toute vide, si même son esprit lui disait qu'elle ne l'était pas, parce qu'il ne pouvait pas l'imaginer, quelque ami, qui, se doutant qu'ils s'aimaient, ne se fût pas risqué à lui rien dire d'elle que d'insignifiant, lui décrivait la silhouette d'Odette, qu'il avait aperçue, le matin même, montant à pied la rue Abbatucci dans une « visite » garnie de skunks, sous un chapeau « à la Rembrandt » et un bouquet de violettes à son corsage. Ce simple croquis bouleversait Swann parce qu'il lui faisait tout d'un coup apercevoir qu'Odette avait une vie qui n'était pas tout entière à lui ; il voulait savoir à qui elle avait cherché à plaire par cette toilette qu'il ne lui connaissait pas ; il se promettait de lui demander où elle allait, à ce moment-là, comme si dans toute la vie incolore – presque inexistante, parce qu'elle lui était invisible – de sa maîtresse, il n'y avait qu'une seule chose en dehors de tous ces sourires adressés à lui : sa démarche sous un chapeau à la Rembrandt, avec un bouquet de violettes au corsage.
