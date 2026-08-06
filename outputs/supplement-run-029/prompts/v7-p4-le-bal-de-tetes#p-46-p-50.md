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
      "canonical_name": "Bloch",
      "surface_forms": [
        "Bloch"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bloch",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.84,
      "evidence": "Bloch « n'avait cessé de publier » des ouvrages qui donnaient « l'impression d'une hauteur intellectuelle »; dans une société reconstituée, il fit « une apparition de grand homme » et « on ne pensait pas qu'il eût jamais vécu ailleurs ».",
      "explanation": "The narrator, while judging the writings to be unoriginal, reports the effective social elevation of Bloch, recognized as a great man by the youth and the reconstituted world."
    }
  ],
  "status_effects": [
    {
      "character": "Bloch",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "Appears as a 'great man' in the new worldly realm, his presence being received as that of an established talent."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-46-p-50"
}

### Candidate characters

[
  "Dreyfus",
  "Gilberte",
  "M. de Marsantes",
  "Mme de Cambremer",
  "Robert de Saint-Loup",
  "Swann",
  "comte de Forcheville",
  "duchesse de Guermantes",
  "le directeur",
  "le grand-père du narrateur",
  "le narrateur",
  "princesse de Guermantes",
  "princesse de Parme"
]

### Prior local context (optional)

Mais il y avait aussi des personnes que je ne pouvais pas reconnaître pour la raison que je ne les avais pas connues, car, aussi bien que sur les êtres eux-mêmes, le temps avait aussi, dans ce salon, exercé sa chimie sur la société. Ce milieu, en la nature spécifique duquel, définie par certaines affinités qui lui attiraient tous les grands noms princiers de l'Europe et par la répulsion qui éloignait d'elle tout élément non aristocratique, j'avais trouvé un refuge matériel pour ce nom de Guermantes auquel il prêtait sa dernière réalité, ce milieu avait lui-même subi, dans sa constitution intime et que j'avais crue stable, une altération profonde. La présence de gens que j'avais vus dans de tout autres sociétés et qui me semblaient ne devoir jamais pénétrer dans celle-là m'étonna moins encore que l'intime familiarité avec laquelle ils y étaient reçus, appelés par leur prénom ; un certain ensemble de préjugés aristocratiques, de snobisme, qui jadis écartait automatiquement du nom de Guermantes tout ce qui ne s'harmonisait pas avec lui, avait cessé de fonctionner.

### Passage

Certains étrangers qui, quand j'avais débuté dans le monde, donnaient de grands dîners où ils ne recevaient que la princesse de Guermantes, la Mme de Guermantes, la princesse de Parme et étaient chez ces dames à la place d'honneur, passaient pour ce qu'il y a de mieux assis dans la société d'alors et l'étaient peut-être, avaient passé sans laisser aucune trace. Étaient-ce des étrangers en mission diplomatique repartis pour leur pays ? Peut-être un scandale, un suicide, un enlèvement les avait-il empêchés de reparaître dans le monde, ou bien étaient-ils allemands ? Mais leur nom ne devait son lustre qu'à leur situation d'alors et n'était plus porté par personne : on ne savait même pas qui je voulais dire ; si je parlais d'eux en essayant d'épeler le nom, on croyait à des rastaquouères.

Les personnes qui n'auraient pas dû, selon l'ancien code social, se trouver là avaient, à mon grand étonnement, pour meilleures amies, des personnes admirablement nées, lesquelles n'étaient venues s'embêter chez la princesse de Guermantes qu'à cause de leurs nouvelles amies. Car ce qui caractérisait le plus cette société, c'était sa prodigieuse aptitude au déclassement.

Détendus ou brisés, les ressorts de la machine refoulante ne fonctionnaient plus, mille corps étrangers y pénétraient, lui ôtaient toute homogénéité, toute tenue, toute couleur. Le faubourg Saint-Germain, comme une douairière gâteuse, ne répondait que par des sourires timides à des domestiques insolents qui envahissaient ses salons, buvaient son orangeade et lui présentaient leurs maîtresses.

Encore la sensation du temps écoulé et de l'anéantissement d'une partie de mon passé disparu m'était-elle donnée moins vivement encore par la destruction de cet ensemble cohérent (qu'avait été le salon Guermantes) d'éléments dont mille nuances, mille raisons expliquaient la présence, la fréquence, la coordination, qu'expliquée par l'anéantissement même de la connaissance des mille raisons, des mille nuances qui faisaient que tel qui s'y trouvait encore maintenant y était tout naturellement indiqué et à sa place, tandis que tel autre qui l'y coudoyait y présentait une nouveauté suspecte. Cette ignorance n'était pas que du monde, mais de la politique, de tout. Car la mémoire dure moins que la vie chez les individus, et, d'ailleurs, de très jeunes, qui n'avaient jamais eu les souvenirs abolis chez les autres, faisant maintenant partie du monde, et très légitimement, même au sens nobiliaire, les débuts étant oubliés ou ignorés, on prenait les gens – au point d'élévation ou de chute – où ils se trouvaient, croyant qu'il en avait toujours été ainsi, et que la princesse de Guermantes et Bloch avaient toujours eu la plus grande situation, que Clemenceau et Viviani avaient toujours été conservateurs. Et comme certains faits ont plus de durée, le souvenir exécré de l'Affaire Dreyfus persistant vaguement chez eux, grâce à ce que leur avaient dit leurs pères, si on leur disait que Clemenceau avait été dreyfusard, ils disaient : « Pas possible, vous confondez, il est juste de l'autre côté. » Des ministres tarés et d'anciennes filles publiques étaient tenus pour des parangons de vertu. Quelqu'un ayant demandé à un jeune homme de la plus grande famille s'il n'y avait pas eu quelque chose à dire sur la mère de Gilberte, le jeune seigneur répondit qu'en effet, dans la première partie de son existence, elle avait épousé un aventurier du nom de Swann, mais qu'ensuite elle avait épousé un des hommes les plus en vue de la société, le comte de Forcheville. Sans doute quelques personnes encore dans ce salon, la Mme de Guermantes par exemple, eussent souri de cette assertion (qui, niant l'élégance de Swann, me paraissait monstrueuse, alors que moi-même jadis, à Combray, j'avais cru avec ma grand'tante que Swann ne pouvait connaître des « princesses ») et aussi des femmes qui eussent pu se trouver là mais qui ne sortaient plus guère, les duchesses de Montmorency, de Mouchy, de Sagan, qui avaient été les amies intimes de Swann et n'avaient jamais aperçu ce Forcheville, non reçu dans le monde au temps où elles y allaient encore. Mais précisément c'est que la société d'alors, de même que les visages aujourd'hui modifiés et les cheveux blonds remplacés par des cheveux blancs, n'existait plus que dans la mémoire d'êtres dont le nombre diminuait tous les jours. Bloch, pendant la guerre, avait cessé de « sortir », de fréquenter ses anciens milieux d'autrefois où il faisait piètre figure. En revanche, il n'avait cessé de publier de ces ouvrages dont je m'efforçais aujourd'hui, pour ne pas être entravé par elle, de détruire l'absurde sophistique, ouvrages sans originalité, mais qui donnaient aux jeunes gens et à beaucoup de femmes du monde l'impression d'une hauteur intellectuelle peu commune, d'une sorte de génie. Ce fut donc après une scission complète entre son ancienne mondanité et la nouvelle que, dans une société reconstituée, il avait fait, pour une phase nouvelle de sa vie, honorée, glorieuse, une apparition de grand homme. Les jeunes gens ignoraient naturellement qu'il fît à cet âge-là des débuts dans la société, d'autant que le peu de noms qu'il avait retenus dans la fréquentation de Saint-Loup lui permettaient de donner à son prestige actuel une sorte de recul indéfini. En tout cas il paraissait un de ces hommes de talent qui à toute époque ont fleuri dans le grand monde et on ne pensait pas qu'il eût jamais vécu ailleurs.

Dès que j'eus fini de parler au prince de Guermantes, Bloch se saisit de moi et me présenta à une jeune femme qui avait beaucoup entendu parler de moi par la Mme de Guermantes. Si les gens des nouvelles générations tenaient la Mme de Guermantes pour peu de chose parce qu'elle connaissait des actrices, etc., les dames – aujourd'hui vieilles – de la famille la considéraient toujours comme un personnage extraordinaire, d'une part parce qu'elles savaient exactement sa naissance, sa primauté héraldique, ses intimités avec ce que Mme de Forcheville eût appelé des « royalties », mais encore parce qu'elle dédaignait de venir dans la famille, s'y ennuyait et qu'on savait qu'on n'y pouvait jamais compter sur elle. Ses relations théâtrales et politiques, d'ailleurs mal sues, ne faisaient qu'augmenter sa rareté, donc son prestige. De sorte que, tandis que dans le monde politique et artistique on la tenait pour une créature mal définie, une sorte de défroquée du faubourg Saint-Germain qui fréquente les sous-secrétaires d'État et les étoiles, dans ce même faubourg Saint-Germain, si on donnait une belle soirée, on disait : « Est-ce même la peine d'inviter Marie Sosthènes ? elle ne viendra pas. Enfin pour la forme, mais il ne faut pas se faire d'illusions. » Et si, vers 10 h. ½, dans une toilette éclatante, paraissant, de ses yeux durs pour elles, mépriser toutes ses cousines, entrait Marie Sosthènes qui s'arrêtait sur le seuil avec une sorte de majestueux dédain, et si elle restait une heure, c'était une plus grande fête pour la vieille grande dame qui donnait la soirée qu'autrefois pour un directeur de théâtre que Sarah Bernhardt, qui avait vaguement promis un concours sur lequel on ne comptait pas, fût venue et eût, avec une complaisance et une simplicité infinies, récité, au lieu du morceau promis, vingt autres. La présence de Marie Sosthènes, à laquelle les chefs de cabinet parlaient de haut en bas et qui n'en continuait pas moins (l'esprit mène ainsi le monde) à chercher à en connaître de plus en plus, venait de classer la soirée de la douairière, où il n'y avait pourtant que des femmes excessivement chic, en dehors et au-dessus de toutes les autres soirées de douairières de la même « season » (comme aurait encore dit Mme de Forcheville), mais pour lesquelles soirées ne s'était pas dérangée Marie Sosthènes qui était une des femmes les plus élégantes du jour. Le nom de la jeune femme à laquelle Bloch m'avait présenté m'était entièrement inconnu, et celui des différents Guermantes ne devait pas lui être très familier, car elle demanda à une Américaine à quel titre Mme de Saint-Loup avait l'air si intime avec toute la plus brillante société qui se trouvait là. Or, cette Américaine était mariée au comte de Furcy, parent obscur des Forcheville et pour lequel ils représentaient ce qu'il y a de plus brillant au monde. Aussi répondit-elle tout naturellement : « Quand ce ne serait que parce qu'elle est née Forcheville. C'est ce qu'il y a de plus grand. » Encore Mme de Furcy, tout en croyant naïvement le nom de Forcheville supérieur à celui de Saint-Loup, savait-elle du moins ce qu'était ce dernier. Mais la charmante amie de Bloch et de la Mme de Guermantes l'ignorait absolument et, étant assez étourdie, répondit de bonne foi à une jeune fille qui lui demandait comment Mme de Saint-Loup était parente du maître de la maison, le prince de Guermantes : « Par les Forcheville », renseignement que la jeune fille communiqua, comme si elle l'avait possédé de tout temps, à une de ses amies, laquelle, ayant mauvais caractère et étant nerveuse, devint rouge comme un coq la première fois qu'un monsieur lui dit que ce n'était pas par les Forcheville que Gilberte tenait aux Guermantes, de sorte que le monsieur crut qu'il s'était trompé, adopta l'erreur et ne tarda pas à la propager. Les dîners, les fêtes mondaines, étaient pour l'Américaine une sorte d'École Berlitz. Elle entendait les noms et les répétait sans avoir connu préalablement leur valeur, leur portée exacte. On expliqua à quelqu'un qui demandait si Tansonville venait à Gilberte de son père M. de Forcheville, que cela ne venait pas du tout par là, que c'était une terre de la famille de son mari, que Tansonville était voisin de Guermantes, appartenait à Mme de Marsantes, mais étant très hypothéqué, avait été racheté, en dot, par Gilberte. Enfin un vieux de la vieille, ayant évoqué Swann ami des Sagan et des Mouchy, et l'Américaine amie de Bloch ayant demandé comment je l'avais connu, déclara que je l'avais connu chez Mme de Guermantes, ne se doutant pas du voisin de campagne, jeune ami de mon grand-père, qu'il représentait pour moi. Des méprises de ce genre ont été commises par les hommes les plus fameux et passent pour particulièrement graves dans toute société conservatrice. Saint-Simon, voulant montrer que Louis XIV était d'une ignorance qui « le fit tomber quelquefois, en public, dans les absurdités les plus grossières », ne donne de cette ignorance que deux exemples, à savoir que le Roi, ne sachant pas que Rénel était de la famille de Clermont-Gallerande ni Saint-Hérem de celle de Montmorin, les traita en hommes de peu. Du moins, en ce qui concerne Saint-Hérem, avons-nous la consolation de savoir que le Roi ne mourut pas dans l'erreur, car il fut détrompé « fort tard » par M. de la Rochefoucauld. « Encore, ajoute Saint-Simon avec un peu de pitié, lui fallut-il expliquer quelles étaient ces maisons que leur nom ne lui apprenait pas. » Cet oubli si vivace qui recouvre si rapidement le passé le plus récent, cette ignorance si envahissante, créent par contre-coup une valeur d'érudition à un petit savoir d'autant plus précieux qu'il est peu répandu, s'appliquant à la généalogie des gens, à leurs vraies situations, à la raison d'amour, d'argent ou autre pour quoi ils se sont alliés à telle famille, ou mésalliés, savoir prisé dans toutes les sociétés où règne un esprit conservateur, savoir que mon grand-père possédait au plus haut degré, concernant la bourgeoisie de Combray et de Paris, savoir que Saint-Simon prisait tant que, au moment où il célèbre la merveilleuse intelligence du prince de Conti, avant même de parler des sciences, ou plutôt comme si c'était la première des sciences, il le loue d'avoir été « un très bel esprit, lumineux, juste, exact, étendu, d'une lecture infinie, qui n'oubliait rien, qui connaissait les généalogies, leurs chimères et leurs réalités, d'une politesse distinguée selon le rang, le mérite, rendant tout ce que les princes du sang doivent et qu'ils ne rendent plus. Il s'en expliquait même et, sur leurs usurpations, l'histoire des livres et des conversations lui fournissait de quoi placer ce qu'il trouvait de plus obligeant sur la naissance, les emplois, etc. » Moins brillant, pour tout ce qui avait trait à la bourgeoisie de Combray et de Paris, mon grand-père ne le savait pas avec moins d'exactitude et ne le savourait pas avec moins de gourmandise. Ces gourmets-là, ces amateurs-là étaient déjà devenus peu nombreux qui savaient que Gilberte n'était pas Forcheville, ni Mme de Cambremer Méséglise, ni la plus jeune une Valintonais. Peu nombreux, peut-être même pas recrutés dans la plus haute aristocratie (ce ne sont pas forcément les dévots, ni même les catholiques, qui sont le plus savants concernant la Légende Dorée ou les vitraux du XIIIe siècle), mais souvent dans une aristocratie secondaire, plus friande de ce qu'elle n'approche guère et qu'elle a d'autant plus le loisir d'étudier qu'elle le fréquente moins, se retrouvant avec plaisir, faisant la connaissance les uns des autres, donnant de succulents dîners de corps, comme la société des bibliophiles ou des amis de Reims, dîners où on déguste des généalogies. Les femmes n'y sont pas admises, mais les maris rentrent en disant à la leur : « J'ai fait un dîner intéressant. Il y avait un M. de la Raspelière qui nous a tenus sous le charme en nous expliquant que cette Mme de Saint-Loup qui a cette jolie fille n'est pas du tout née Forcheville. C'est tout un roman. »
