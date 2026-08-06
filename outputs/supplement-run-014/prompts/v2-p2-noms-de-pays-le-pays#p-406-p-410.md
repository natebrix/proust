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
      "canonical_name": "Albertine",
      "surface_forms": [
        "Albertine",
        "Mlle Simonet"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Albertine",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« Si peu plaisant que soit cet emploi de parfaitement, il indique un degré de civilisation et de culture... »; « je trouvai à Albertine l'air assez intimidé à la place d'implacable ; elle me sembla plus comme il faut que mal élevée »",
      "explanation": "The narrator reconfigures his image of Albertine: her words and manners transition her from the 'bacchante on a bicycle' to a more cultured and proper young girl. This elevates her social and cultural value locally in his eyes."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "Perceived as more 'proper' and with an unexpected degree of culture, Albertine gains local social status in the eyes of the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-406-p-410"
}

### Candidate characters

[
  "Elstir",
  "Mme Bontemps",
  "le narrateur"
]

### Prior local context (optional)

Si la connaissance du plaisir fut ainsi retardée pour moi de quelques heures, en revanche la gravité de cette présentation, je la ressentis tout de suite. Au moment de la présentation, nous avons beau nous sentir tout à coup gratifiés et porteurs d'un « bon », valable pour des plaisirs futurs, après lequel nous courions depuis des semaines, nous comprenons bien que son obtention met fin pour nous, non pas seulement à de pénibles recherches – ce qui ne pourrait que nous remplir de joie – mais aussi à l'existence d'un certain être, celui que notre imagination avait dénaturé, que notre crainte anxieuse de ne jamais pouvoir être connus de lui avait grandi. Au moment où notre nom résonne dans la bouche du présentateur, surtout si celui-ci l'entoure comme fit Elstir de commentaires élogieux, ce moment sacramentel, analogue à celui où, dans une féerie, le génie ordonne à une personne d'en être soudain une autre, celle que nous avons désiré d'approcher s'évanouit ; d'abord comment resterait-elle pareille à elle-même puisque – de par l'attention que l'inconnue est obligée de prêter à notre nom et de marquer à notre personne – dans les yeux situés à l'infini (et que nous croyions que les nôtres, errants, mal réglés, désespérés, divergents, ne parviendraient jamais à rencontrer) le regard conscient, la pensée inconnaissable que nous cherchions, vient d'être miraculeusement et tout simplement remplacée par notre propre image peinte comme au fond d'un miroir qui sourirait.

### Passage

Si l'incarnation de nous-même en ce qui nous semblait le plus différent est ce qui modifie le plus la personne à qui on vient de nous présenter, la forme de cette personne reste encore assez vague ; et nous pouvons nous demander si elle sera dieu, table ou cuvette. Mais, aussi agiles que ces ciroplastes qui font un buste devant nous en cinq minutes, les quelques mots que l'inconnue va nous dire préciseront cette forme et lui donneront quelque chose de définitif qui exclura toutes les hypothèses auxquelles se livraient la veille notre désir et notre imagination. Sans doute, même avant de venir à cette matinée, Albertine n'était plus tout à fait pour moi ce seul fantôme digne de hanter notre vie que reste une passante dont nous ne savons rien, que nous avons à peine discernée. Sa parenté avec Mme Bontemps avait déjà restreint ces hypothèses merveilleuses, en aveuglant une des voies par lesquelles elles pouvaient se répandre. Au fur et à mesure que je me rapprochais de la jeune fille, et la connaissais davantage, cette connaissance se faisait par soustraction, chaque partie d'imagination et de désir étant remplacée par une notion qui valait infiniment moins, notion à laquelle il est vrai que venait s'ajouter une sorte d'équivalent, dans le domaine de la vie, de ce que les Sociétés financières donnent après le remboursement de l'action primitive, et qu'elles appellent action de jouissance. Son nom, ses parentés avaient été une première limite apportée à mes suppositions. Son amabilité, tandis que tout près d'elle je retrouvais son petit grain de beauté sur la joue au-dessous de l'oeil fut une autre borne ; enfin je fus étonné de l'entendre se servir de l'adverbe « parfaitement » au lieu de « tout à fait », en parlant de deux personnes, disant de l'une « elle est parfaitement folle, mais très gentille tout de même » et de l'autre « c'est un monsieur parfaitement commun et parfaitement ennuyeux ». Si peu plaisant que soit cet emploi de parfaitement, il indique un degré de civilisation et de culture auquel je n'aurais pu imaginer qu'atteignait la bacchante à bicyclette, la muse orgiaque du golf. Il n'empêche d'ailleurs qu'après cette première métamorphose, Albertine devait changer encore bien des fois pour moi. Les qualités et les défauts qu'un être présente disposés au premier plan de son visage se rangent selon une formation tout autre si nous l'abordons par un côté différent – comme dans une ville les monuments répandus en ordre dispersé sur une seule ligne, d'un autre point de vue s'échelonnent en profondeur et échangent leurs grandeurs relatives. Pour commencer je trouvai à Albertine l'air assez intimidé à la place d'implacable ; elle me sembla plus comme il faut que mal élevée à en juger par les épithètes de « elle a un mauvais genre, elle a un drôle de genre », qu'elle appliqua à toutes les jeunes filles dont je lui parlai ; elle avait enfin comme point de mire du visage une tempe assez enflammée et peu agréable à voir, et non plus le regard singulier auquel j'avais toujours repensé jusque-là. Mais ce n'était qu'une seconde vue et il y en avait d'autres sans doute par lesquelles je devrais successivement passer. Ainsi ce n'est qu'après avoir reconnu non sans tâtonnements les erreurs d'optique du début qu'on pourrait arriver à la connaissance exacte d'un être si cette connaissance était possible. Mais elle ne l'est pas ; car tandis que se rectifie la vision que nous avons de lui, lui-même qui n'est pas un objectif inerte change pour son compte, nous pensons le rattraper, il se déplace, et, croyant le voir enfin plus clairement, ce n'est que les images anciennes que nous en avions prises que nous avons réussi à éclaircir, mais qui ne le représentent plus.

Pourtant, quelques déceptions inévitables qu'elle doive apporter, cette démarche vers ce qu'on n'a qu'entrevu, ce qu'on a eu le loisir d'imaginer, cette démarche est la seule qui soit saine pour les sens, qui y entretienne l'appétit. De quel morne ennui est empreinte la vie des gens qui par paresse ou timidité, se rendent directement en voiture chez des amis qu'ils ont connus sans avoir d'abord rêvé d'eux, sans jamais oser sur le parcours s'arrêter auprès de ce qu'ils désirent.

Je rentrai en pensant à cette matinée, en revoyant l'éclair au café que j'avais fini de manger avant de me laisser conduire par Elstir auprès d'Albertine, la rose que j'avais donnée au vieux monsieur, tous ces détails choisis à notre insu par les circonstances et qui composent pour nous, en un arrangement spécial et fortuit, le tableau d'une première rencontre. Mais ce tableau, j'eus l'impression de le voir d'un autre point de vue, de très loin de moi-même, comprenant qu'il n'avait pas existé que pour moi, quand quelques mois plus tard, à mon grand étonnement, comme je parlais à Albertine du premier jour où je l'avais connue, elle me rappela l'éclair, la fleur que j'avais donnée, tout ce que je croyais, je ne peux pas dire n'être important que pour moi, mais n'avoir été aperçu que de moi, que je retrouvais ainsi, transcrit en une version dont je ne soupçonnais l'existence, dans la pensée d'Albertine. Dès ce premier jour, quand en entrant je pus voir le souvenir que je rapportais, je compris quel tour de muscade avait été parfaitement exécuté, et comment j'avais causé un moment avec une personne qui, grâce à l'habileté du prestidigitateur, sans avoir rien de celle que j'avais suivie si longtemps au bord de la mer, lui avait été substituée. J'aurais du reste pu le deviner d'avance, puisque la jeune fille de la plage avait été fabriquée par moi. Malgré cela, comme je l'avais, dans mes conversations avec Elstir, identifiée à Albertine, je me sentais envers celle-ci l'obligation morale de tenir les promesses d'amour faites à l'Albertine imaginaire. On se fiance par procuration, et on se croit obligé d'épouser ensuite la personne interposée. D'ailleurs, si avait disparu provisoirement du moins de ma vie une angoisse qu'eût suffi à apaiser le souvenir des manières comme il faut, de cette expression « parfaitement commune » et de la tempe enflammée, ce souvenir éveillait en moi un autre genre de désir, qui bien que doux et nullement douloureux, semblable à un sentiment fraternel, pouvait à la longue devenir aussi dangereux en me faisant ressentir à tout moment le besoin d'embrasser cette personne nouvelle dont les bonnes façons et la timidité, la disponibilité inattendue, arrêtaient la course inutile de mon imagination, mais donnaient naissance à une gratitude attendrie. Et puis comme la mémoire commence tout de suite à prendre des clichés indépendants les uns des autres, supprime tout lien, tout progrès, entre les scènes qui y sont figurées, dans la collection de ceux qu'elle expose, le dernier ne détruit pas forcément les précédents. En face de la médiocre et touchante Albertine à qui j'avais parlé, je voyais la mystérieuse Albertine en face de la mer. C'étaient maintenant des souvenirs, c'est-à-dire des tableaux dont l'un ne me semblait pas plus vrai que l'autre. Pour en finir avec ce premier soir de présentation, en cherchant à revoir ce petit grain de beauté sur la joue au-dessous de l'oeil, je me rappelai que de chez Elstir, quand Albertine était partie, j'avais vu ce grain de beauté sur le menton. En somme, quand je la voyais, je remarquais qu'elle avait un grain de beauté, mais ma mémoire errante le promenait ensuite sur la figure d'Albertine et le plaçait tantôt ici tantôt là.

J'avais beau être assez désappointé d'avoir trouvé en Mlle Simonet une jeune fille trop peu différente de tout ce que je connaissais, de même que ma déception devant l'église de Balbec ne m'empêchait pas de désirer aller à Quimperlé, à Pont-Aven et à Venise, je me disais que par Albertine du moins, si elle-même n'était pas ce que j'avais espéré, je pourrais connaître ses amies de la petite bande.

Je crus d'abord que j'y échouerais. Comme elle devait rester fort longtemps encore à Balbec et moi aussi, j'avais trouvé que le mieux était de ne pas trop chercher à la voir et d'attendre une occasion qui me fît la rencontrer. Mais cela arrivât-il tous les jours, il était fort à craindre qu'elle se contentât de répondre de loin à mon salut, lequel dans ce cas, répété quotidiennement pendant toute la saison, ne m'avancerait à rien.
