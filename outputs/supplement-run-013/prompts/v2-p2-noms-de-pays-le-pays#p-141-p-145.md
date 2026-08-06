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
      "canonical_name": "Mme de Villeparisis",
      "surface_forms": [
        "Mme de Villeparisis",
        "Madame de Villeparisis"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme de Villeparisis",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "Mme de Villeparisis qui avait un certain sens de la nature, plus froid que celui de la grand-mère, mais qui sait reconnaître... la beauté simple et majestueuse de certaines choses anciennes, disait au cocher de prendre la vieille route de Balbec...",
      "explanation": "The narrator credits Mme de Villeparisis with aesthetic discernment in nature and follows her choice of the old road, a mild but clear elevation of her taste despite the qualifier that it is 'colder' than the grandmother’s."
    }
  ],
  "status_effects": [
    {
      "character": "Mme de Villeparisis",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "She is locally valued for recognizing natural beauty and guiding the route accordingly."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-141-p-145"
}

### Candidate characters

[
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

C'était cela que je voulais qu'elle sût pour prendre une grande idée de moi. Mais quand j'eus prononcé les mots de « marquise » et « deux chevaux », soudain j'éprouvai un grand apaisement. Je sentis que la pêcheuse se souviendrait de moi et se dissiper, avec mon effroi de ne pouvoir la retrouver, une partie de mon désir de la retrouver. Il me semblait que je venais de toucher sa personne avec des lèvres invisibles et que je lui avais plu. Et cette prise de force de son esprit, cette possession immatérielle, lui avait ôté de son mystère autant que fait la possession physique.

### Passage

Nous descendîmes sur Hudimesnil ; tout d'un coup je fus rempli de ce bonheur profond que je n'avais pas souvent ressenti depuis Combray, un bonheur analogue à celui que m'avaient donné, entre autres, les clochers de Martainville. Mais cette fois il resta incomplet. Je venais d'apercevoir, en retrait de la route en dos d'âne que nous suivions, trois arbres qui devaient servir d'entrée à une allée couverte et formaient un dessin que je ne voyais pas pour la première fois, je ne pouvais arriver à reconnaître le lieu dont ils étaient comme détachés, mais je sentais qu'il m'avait été familier autrefois ; de sorte que mon esprit ayant trébuché entre quelque année lointaine et le moment présent, les environs de Balbec vacillèrent et je me demandai si toute cette promenade n'était pas une fiction, Balbec un endroit où je n'étais jamais allé que par l'imagination, Mme de Villeparisis un personnage de roman et les trois vieux arbres la réalité qu'on retrouve en levant les yeux de dessus le livre qu'on était en train de lire et qui vous décrivait un milieu dans lequel on avait fini par se croire effectivement transporté.

Je regardais les trois arbres, je les voyais bien, mais mon esprit sentait qu'ils recouvraient quelque chose sur quoi ils n'avaient pas prise, comme sur ces objets placés trop loin dont nos doigts allongés au bout de notre bras tendu effleurent seulement par instant l'enveloppe sans arriver à rien saisir. Alors on se repose un moment pour jeter le bras en avant d'un élan plus fort et tâcher d'atteindre plus loin. Mais pour que mon esprit pût ainsi se rassembler, prendre son élan, il m'eût fallu être seul. Que j'aurais voulu pouvoir m'écarter comme je faisais dans les promenades du côté de Guermantes quand je m'isolais de mes parents ! Il me semblait même que j'aurais dû le faire. Je reconnaissais ce genre de plaisir qui requiert, il est vrai, un certain travail de la pensée sur elle-même, mais à côté duquel les agréments de la nonchalance qui vous fait renoncer à lui, semblent bien médiocres. Ce plaisir, dont l'objet n'était que pressenti, que j'avais à créer moi-même, je ne l'éprouvais que de rares fois, mais à chacune d'elles il me semblait que les choses qui s'étaient passées dans l'intervalle n'avaient guère d'importance et qu'en m'attachant à la seule réalité je pourrais commencer enfin une vraie vie. Je mis un instant ma main devant mes yeux pour pouvoir les fermer sans que Mme de Villeparisis s'en aperçût. Je restai sans penser à rien, puis de ma pensée ramassée, ressaisie avec plus de force, je bondis plus avant dans la direction des arbres, ou plutôt dans cette direction intérieure au bout de laquelle je les voyais en moi-même. Je sentis de nouveau derrière eux le même objet connu mais vague et que je pus ramener à moi. Cependant tous trois, au fur et à mesure que la voiture avançait, je les voyais s'approcher. Où les avais-je déjà regardés ? Il n'y avait aucun lieu autour de Combray où une allée s'ouvrît ainsi. Le site qu'ils me rappelaient il n'y avait pas de place pour lui davantage dans la campagne allemande où j'étais allé une année avec ma grand'mère prendre les eaux. Fallait-il croire qu'ils venaient d'années déjà si lointaines de ma vie que le paysage qui les entourait avait été entièrement aboli dans ma mémoire et que, comme ces pages qu'on est tout d'un coup ému de retrouver dans un ouvrage qu'on s'imaginait n'avoir jamais lu, ils surnageaient seuls du livre oublié de ma première enfance. N'appartenaient-ils au contraire qu'à ces paysages du rêve, toujours les mêmes, du moins pour moi chez qui leur aspect étrange n'était que l'objectivation dans mon sommeil de l'effort que je faisais pendant la veille, soit pour atteindre le mystère dans un lieu derrière l'apparence duquel je le pressentais, comme cela m'était arrivé si souvent du côté de Guermantes, soit pour essayer de le réintroduire dans un lieu que j'avais désiré connaître et qui du jour où je l'avais connu m'avait paru tout superficiel, comme Balbec ? N'étaient-ils qu'une image toute nouvelle détachée d'un rêve de la nuit précédente, mais déjà si effacée qu'elle me semblait venir de beaucoup plus loin ? Ou bien ne les avais-je jamais vus et cachaient-ils derrière eux, comme tels arbres, telle touffe d'herbes que j'avais vus du côté de Guermantes, un sens aussi obscur, aussi difficile à saisir qu'un passé lointain, de sorte que, sollicité par eux d'approfondir une pensée, je croyais avoir à reconnaître un souvenir. Ou encore ne cachaient-ils même pas de pensées et était-ce une fatigue de ma vision qui me les faisait voir doubles dans le temps comme on voit quelquefois double dans l'espace ? Je ne savais. Cependant ils venaient vers moi ; peut-être apparition mythique, ronde de sorcières ou de nornes qui me proposait ses oracles. Je crus plutôt que c'étaient des fantômes du passé, de chers compagnons de mon enfance, des amis disparus qui invoquaient nos communs souvenirs. Comme des ombres ils semblaient me demander de les emmener avec moi, de les rendre à la vie. Dans leur gesticulation naïve et passionnée, je reconnaissais le regret impuissant d'un être aimé qui a perdu l'usage de la parole, sent qu'il ne pourra nous dire ce qu'il veut et que nous ne savons pas deviner. Bientôt à un croisement de routes, la voiture les abandonna. Elle m'entraînait loin de ce que je croyais seul vrai, de ce qui m'eût rendu vraiment heureux, elle ressemblait à ma vie.

Je vis les arbres s'éloigner en agitant leurs bras désespérés, semblant me dire : ce que tu n'apprends pas de nous aujourd'hui, tu ne le sauras jamais. Si tu nous laisses retomber au fond de ce chemin d'où nous cherchions à nous hisser jusqu'à toi, toute une partie de toi-même que nous t'apportions tombera pour jamais au néant. En effet, si dans la suite je retrouvai le genre de plaisir et d'inquiétude que je venais de sentir encore une fois, et si un soir – trop tard, mais pour toujours – je m'attachai à lui, de ces arbres eux-mêmes, en revanche je ne sus jamais ce qu'ils avaient voulu m'apporter ni où je les avais vus. Et quand, la voiture ayant bifurqué, je leur tournai le dos et cessai de les voir, tandis que Mme de Villeparisis me demandait pourquoi j'avais l'air rêveur, j'étais triste comme si je venais de perdre un ami, de mourir moi-même, de renier un mort ou de méconnaître un Dieu.

Il fallait songer au retour. Mme de Villeparisis qui avait un certain sens de la nature, plus froid que celui de ma grand'mère, mais qui sait reconnaître, même en dehors des musées et des demeures aristocratiques, la beauté simple et majestueuse de certaines choses anciennes, disait au cocher de prendre la vieille route de Balbec, peu fréquentée, mais plantée de vieux ormes qui nous semblaient admirables.

Une fois que nous connûmes cette vieille route, pour changer, nous revînmes, à moins que nous ne l'eussions prise à l'aller, par une autre qui traversait les bois de Chantereine et de Canteloup. L'invisibilité des innombrables oiseaux qui s'y répondaient tout à côté de nous dans les arbres donnait la même impression de repos qu'on a les yeux fermés. Enchaîné à mon strapontin comme Prométhée sur son rocher, j'écoutais mes Océanides. Et, quand, par hasard, j'apercevais l'un de ces oiseaux qui passait d'une feuille sous une autre, il y avait si peu de lien apparent entre lui et ces chants que je ne croyais pas voir la cause de ceux-ci dans ce petit corps sautillant, étonné et sans regard.
