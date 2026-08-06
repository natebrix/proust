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
        "Albertine"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.96
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
      "confidence": 0.82,
      "evidence": "la croyance ... que j'allais connaître Albertine, l'avait, à quelques secondes d'intervalle, rendue presque insignifiante puis infiniment précieuse à mes yeux",
      "explanation": "The narrator reports a sharp swing in his valuation of Albertine: when he expects imminent introduction her importance contracts; when that belief vanishes, she becomes immensely precious. This locally elevates Albertine in the narrator’s appraisal."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Albertine's local standing in the narrator’s eyes rises when the expected introduction fails and desire re-expands."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-381-p-385"
}

### Candidate characters

[
  "Elstir",
  "Robert de Saint-Loup",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Le rebord de la fenêtre fut bientôt rose. Notre sortie serait en pure perte. Il n'y avait aucune chance de voir les jeunes filles, par conséquent plus aucune importance à ce que Mme Elstir nous quittât plus ou moins vite. Elle ne resta, d'ailleurs, pas très longtemps. Je la trouvai très ennuyeuse ; elle aurait pu être belle, si elle avait eu vingt ans, conduisant un boeuf dans la campagne romaine ; mais ses cheveux noirs blanchissaient ; et elle était commune sans être simple, parce qu'elle croyait que la solennité des manières et la majesté de l'attitude étaient requises par sa beauté sculpturale à laquelle, d'ailleurs, l'âge avait enlevé toutes ses séductions. Elle était mise avec la plus grande simplicité. Et on était touché mais surpris d'entendre Elstir dire à tout propos et avec une douceur respectueuse, comme si rien que prononcer ces mots lui causait de l'attendrissement et de la vénération : « Ma belle Gabrielle ! » Plus tard, quand je connus la peinture mythologique d'Elstir, Mme Elstir prit pour moi aussi de la beauté. Je compris qu'à certain type idéal résumé en certaines lignes, en certaines arabesques qui se retrouvaient sans cesse dans son oeuvre, à un certain canon, il avait attribué en fait un caractère presque divin, puisque tout son temps, tout l'effort de pensée dont il était capable, en un mot toute sa vie, il l'avait consacrée à la tâche de distinguer mieux ces lignes, de les reproduire plus fidèlement. Ce qu'un tel idéal inspirait à Elstir, c'était vraiment un culte si grave, si exigeant, qu'il ne lui permettait jamais d'être content, c'était la partie la plus intime de lui-même, aussi n'avait-il pu le considérer avec détachement, en tirer des émotions, jusqu'au jour où il le rencontra, réalisé au dehors, dans le corps d'une femme, le corps de celle qui était par la suite devenue Mme Elstir et chez qui il avait pu – comme cela ne nous est possible que pour ce qui n'est pas nous-même – le trouver méritoire, attendrissant, divin. Quel repos, d'ailleurs, de poser ses lèvres sur ce Beau que jusqu'ici il fallait avec tant de peine extraire de soi, et qui maintenant mystérieusement incarné, s'offrait à lui pour une suite de communions efficaces ! Elstir à cette époque n'était plus dans la première jeunesse où l'on attend que de la puissance de la pensée la réalisation de son idéal. Il approchait de l'âge où l'on compte sur les satisfactions du corps pour stimuler la force de l'esprit, où la fatigue de celui-ci, en nous inclinant au matérialisme, et la diminution de l'activité à la possibilité d'influences passivement reçues, commencent à nous faire admettre qu'il y a peut-être bien certains corps, certains métiers, certains rythmes privilégiés, réalisant si naturellement notre idéal, que même sans génie, rien qu'en copiant le mouvement d'une épaule, la tension d'un cou, nous ferions un chef-d'oeuvre ; c'est l'âge où nous aimons à caresser la Beauté du regard, hors de nous, près de nous, dans une tapisserie, dans une belle esquisse de Titien découverte chez un brocanteur, dans une maîtresse aussi belle que l'esquisse de Titien. Quand j'eus compris cela, je ne pus plus voir sans plaisir Mme Elstir, et son corps perdit de sa lourdeur, car je le remplis d'une idée, l'idée qu'elle était une créature immatérielle, un portrait d'Elstir. Elle en était un pour moi et pour lui aussi sans doute. Les données de la vie ne comptent pas pour l'artiste, elles ne sont pour lui qu'une occasion de mettre à nu son génie. On sent bien, à voir les uns à côté des autres dix portraits de personnes différentes peintes par Elstir, que ce sont avant tout des Elstir. Seulement, après cette marée montante du génie qui recouvre la vie, quand le cerveau se fatigue, peu à peu l'équilibre se rompt et comme un fleuve qui reprend son cours après le contreflux d'une grande marée, c'est la vie qui reprend le dessus. Or, pendant que durait la première période, l'artiste a, peu à peu, dégagé la loi, la formule de son inconscient. Il sait quelles situations s'il est romancier, quels paysages s'il est le peintre, lui fournissent la matière, indifférente en soi, mais nécessaire à ses recherches comme serait un laboratoire ou un atelier. Il sait qu'il a fait ses chefs d'oeuvre avec des effets de lumière atténuée, avec des remords modifiant l'idée d'une faute, avec des femmes posées sous les arbres ou à demi plongées dans l'eau, comme des statues. Un jour viendra où, par l'usure de son cerveau, il n'aura plus, devant ces matériaux dont se servait son génie, la force de faire l'effort intellectuel qui seul peut produire son oeuvre, et continuera pourtant à les rechercher, heureux de se trouver près d'eux à cause du plaisir spirituel, amorce du travail, qu'ils éveillent en lui ; et les entourant d'ailleurs d'une sorte de superstition comme s'ils étaient supérieurs à autre chose, si en eux résidait déjà une bonne part de l'oeuvre d'art qu'ils porteraient en quelque sorte toute faite, il n'ira pas plus loin que la fréquentation, l'adoration des modèles. Il causera indéfiniment avec des criminels repentis, dont le remords, la régénération a fait l'objet de ses romans ; il achètera une maison de campagne dans un pays où la brume atténue la lumière ; il passera de longues heures à regarder des femmes se baigner ; il collectionnera les belles étoffes. Et ainsi la beauté de la vie, mot en quelque sorte dépourvu de signification, stade situé en deçà de l'art et auquel j'avais vu s'arrêter Swann, était celui où par ralentissement du génie créateur, idolâtrie des formes qui l'avaient favorisé, désir du moindre effort, devait un jour rétrograder peu à peu un Elstir.

### Passage

Il venait enfin de donner un dernier coup de pinceau à ses fleurs ; je perdis un instant à les regarder ; je n'avais pas de mérite à le faire, puisque je savais que les jeunes filles ne se trouveraient plus sur la plage ; mais j'aurais cru qu'elles y étaient encore et que ces minutes perdues me les faisaient manquer que j'aurais regardé tout de même, car je me serais dit qu'Elstir s'intéressait plus à ses fleurs qu'à ma rencontre avec les jeunes filles. La nature de ma grand-mère, nature qui était tout juste l'opposé de mon total égoïsme, se reflétait pourtant dans la mienne. Dans une circonstance où quelqu'un qui m'était indifférent, pour qui j'avais toujours feint de l'affection ou du respect, ne risquait qu'un désagrément tandis que je courais un danger, je n'aurais pas pu faire autrement que de le plaindre de son ennui comme d'une chose considérable et de traiter mon danger comme un rien, parce qu'il me semblait que c'était avec ces proportions que les choses devaient lui apparaître. Pour dire les choses telles qu'elles sont, c'est même un peu plus que cela, et pas seulement ne pas déplorer le danger que je courais moi-même, mais aller au devant de ce danger-là, et pour celui qui concernait les autres, tâcher au contraire, dussé-je avoir plus de chances d'être atteint moi-même, de le leur éviter. Cela tient à plusieurs raisons qui ne sont point à mon honneur. L'une est que si, tant que je ne faisais que raisonner, je croyais surtout tenir à la vie, chaque fois qu'au cours de mon existence, je me suis trouvé obsédé par des soucis moraux ou seulement par des inquiétudes nerveuses, quelquefois si puériles que je n'oserais pas les rapporter, si une circonstance imprévue survenait alors, amenant pour moi le risque d'être tué, cette nouvelle préoccupation était si légère, relativement aux autres, que je l'accueillais avec un sentiment de détente qui allait jusqu'à l'allégresse. Je me trouve ainsi avoir connu, quoique étant l'homme le moins brave du monde, cette chose qui me semblait, quand je résonnais, si étrangère à ma nature, si inconcevable, l'ivresse du danger. Mais même fussé-je, quand il y en a un, et mortel, qui se présente, dans une période entièrement calme et heureuse, je ne pourrais pas, si je suis avec une autre personne, ne pas la mettre à l'abri et choisir pour moi la place dangereuse. Quand un assez grand nombre d'expériences m'eurent appris que j'agissais toujours ainsi, et avec plaisir, je découvris et à ma grande honte, que contrairement à ce que j'avais toujours cru et affirmé, j'étais très sensible à l'opinion des autres. Cette sorte d'amour-propre inavoué n'a pourtant aucun rapport avec la vanité ni avec l'orgueil. Car ce qui peut contenter l'une ou l'autre ne me causerait aucun plaisir et je m'en suis toujours abstenu. Mais les gens devant qui j'ai réussi à cacher le plus complètement les petits avantages qui auraient pu leur donner une moins piètre idée de moi, je n'ai jamais pu me refuser le plaisir de leur montrer que je mets plus de soin à écarter la mort de leur route que de la mienne. Comme son mobile est alors l'amour-propre et non la vertu, je trouve bien naturel qu'en toute circonstance ils agissent autrement. Je suis bien loin de les en blâmer, ce que je ferais, peut-être, si j'avais été mû par l'idée d'un devoir qui me semblerait dans ce cas être obligatoire pour eux aussi bien que pour moi. Au contraire, je les trouve fort sages de préserver leur vie, tout en ne pouvant m'empêcher de faire passer au second plan la mienne, ce qui est particulièrement absurde et coupable, depuis que j'ai cru reconnaître que celle de beaucoup de gens devant qui je me place, quand éclate une bombe, est plus dénuée de prix. D'ailleurs le jour de cette visite à Elstir les temps étaient encore loin où je devais prendre conscience de cette différence de valeur, et il ne s'agissait d'aucun danger, mais simplement, signe avant-coureur du pernicieux amour-propre, de ne pas avoir l'air d'attacher au plaisir que je désirais si ardemment plus d'importance qu'à la besogne d'aquarelliste qu'il n'avait pas achevée. Elle le fut enfin. Et, une fois dehors, je m'aperçus que – tant les jours étaient longs dans cette saison-là – il était moins tard que je ne croyais ; nous allâmes sur la digue. Que de ruses j'employais pour faire demeurer Elstir à l'endroit où je croyais que ces jeunes filles pouvaient encore passer. Lui montrant les falaises qui s'élevaient à côté de nous, je ne cessais de lui demander de me parler d'elles, afin de lui faire oublier l'heure et de le faire rester. Il me semblait que nous avions plus de chance de cerner la petite bande en allant vers l'extrémité de la plage. « J'aurais voulu voir d'un tout petit peu plus près avec vous ces falaises », dis-je à Elstir, ayant remarqué qu'une de ces jeunes filles s'en allait souvent de ce côté. « Et pendant ce temps-là, parlez-moi de Carquethuit. Ah ! que j'aimerais aller à Carquethuit ! » ajoutai-je sans penser que le caractère si nouveau qui se manifestait avec tant de puissance dans le « Port de Carquethuit » d'Elstir tenait peut-être plus à la vision du peintre qu'à un mérite spécial de cette plage. « Depuis que j'ai vu ce tableau, c'est peut-être ce que je désire le plus connaître avec la Pointe du Raz qui serait, d'ailleurs, d'ici, tout un voyage. – Et puis même si ce n'était pas plus près, je vous conseillerais peut-être tout de même davantage Carquethuit, me répondit Elstir. La Pointe du Raz est admirable, mais enfin c'est toujours la grande falaise normande ou bretonne que vous connaissez. Carquethuit c'est tout autre chose avec ces roches sur une plage basse. Je ne connais rien en France d'analogue, cela me rappelle plutôt certains aspects de la Floride. C'est très curieux, et du reste extrêmement sauvage aussi. C'est entre Clitourps et Nehomme et vous savez combien ces parages sont désolés ; la ligne des plages est ravissante. Ici, la ligne de la plage est quelconque ; mais là-bas, je ne peux vous dire quelle grâce elle a, quelle douceur. »

Le soir tombait ; il fallut revenir ; je ramenais Elstir vers sa villa, quand tout d'un coup, tel Méphistophélès surgissant devant Faust, apparurent au bout de l'avenue – comme une simple objectivation irréelle et diabolique du tempérament opposé au mien, de la vitalité quasi barbare et cruelle dont étaient si dépourvue ma faiblesse, mon excès de sensibilité douloureuse et d'intellectualité – quelques taches de l'essence impossible à confondre avec rien d'autre, quelques sporades de la bande zoophytique des jeunes filles, lesquelles avaient l'air de ne pas me voir, mais sans aucun doute n'en étaient pas moins en train de porter sur moi un jugement ironique. Sentant qu'il était inévitable que la rencontre entre elles et nous se produisît, et qu'Elstir allait m'appeler, je tournai le dos comme un baigneur qui va recevoir la lame ; je m'arrêtai net et laissant mon illustre compagnon poursuivre son chemin, je restai en arrière, penché, comme si j'étais subitement intéressé par elle, vers la vitrine du marchand d'antiquités devant lequel nous passions en ce moment ; je n'étais pas fâché d'avoir l'air de pouvoir penser à autre chose qu'à ces jeunes filles, et je savais déjà obscurément que quand Elstir m'appellerait pour me présenter, j'aurais la sorte de regard interrogateur qui décèle non la surprise, mais le désir d'avoir l'air surpris – tant chacun est un mauvais acteur ou le prochain un bon physiognomoniste, – que j'irais même jusqu'à indiquer ma poitrine avec mon doigt pour demander : « C'est bien moi que vous appelez » et accourir vite, la tête courbée par l'obéissance et la docilité, le visage dissimulant froidement l'ennui d'être arraché à la contemplation de vieilles faïences pour être présenté à des personnes que je ne souhaitais pas de connaître. Cependant je considérais la devanture en attendant le moment où mon nom crié par Elstir viendrait me frapper comme une balle attendue et inoffensive. La certitude de la présentation à ces jeunes filles avait eu pour résultat, non seulement de me faire à leur égard jouer, mais éprouver, l'indifférence. Désormais inévitable, le plaisir de les connaître fut comprimé, réduit, me parut plus petit que celui de causer avec Saint-Loup, de dîner avec ma grand-mère, de faire dans les environs des excursions que je regretterais d'être probablement, par le fait de relations avec des personnes qui devaient peu s'intéresser aux monuments historiques, contraint de négliger. D'ailleurs, ce qui diminuait le plaisir que j'allais avoir, ce n'était pas seulement l'imminence mais l'incohérence de sa réalisation. Des lois aussi précises que celles de l'hydrostatique, maintiennent la superposition des images que nous formons dans un ordre fixe que la proximité de l'événement bouleverse. Elstir allait m'appeler. Ce n'était pas du tout de cette façon que je m'étais souvent, sur la plage, dans ma chambre, figuré que je connaîtrais ces jeunes filles. Ce qui allait avoir lieu, c'était un autre événement auquel je n'étais pas préparé. Je ne reconnaissais ni mon désir, ni son objet ; je regrettais presque d'être sorti avec Elstir. Mais, surtout, la contraction du plaisir que j'avais auparavant cru avoir était due à la certitude que rien ne pouvait plus me l'enlever. Et il reprit, comme en vertu d'une force élastique, toute sa hauteur, quand il cessa de subir l'étreinte de cette certitude, au moment où m'étant décidé à tourner la tête, je vis Elstir, arrêté quelques pas plus loin avec les jeunes filles, leur dire au revoir. La figure de celle qui était le plus près de lui, grosse et éclairée par ses regards, avait l'air d'un gâteau où on eût réservé de la place pour un peu de ciel. Ses yeux, même fixes, donnaient l'impression de la mobilité comme il arrive par ces jours de grand vent où l'air, quoique invisible, laisse percevoir la vitesse avec laquelle il passe sur le fond de l'azur. Un instant ses regards croisèrent les miens, comme ces ciels voyageurs des jours d'orage qui approchent d'une nuée moins rapide, la côtoient, la touchent, la dépassent. Mais ils ne se connaissent pas et s'en vont loin l'un de l'autre. Tels nos regards furent un instant face à face, ignorant chacun ce que le continent céleste qui était devant lui contenait de promesses et de menaces pour l'avenir. Au moment seulement où son regard passa exactement sous le mien sans ralentir sa marche, il se voila légèrement. Ainsi, par une nuit claire, la lune emportée par le vent passe sous un nuage et voile un instant son éclat, puis reparaît bien vite. Mais déjà Elstir avait quitté les jeunes filles sans m'avoir appelé. Elles prirent une rue de traverse, il vint vers moi. Tout était manqué.

J'ai dit qu'Albertine ne m'était pas apparue ce jour-là la même que les précédents, et que chaque fois elle devait me sembler différente. Mais je sentis à ce moment que certaines modifications dans l'aspect, l'importance, la grandeur d'un être peuvent tenir aussi à la variabilité de certains états interposés entre cet être et nous. L'un de ceux qui jouent à cet égard le rôle le plus considérable est la croyance (ce soir-là, la croyance, puis l'évanouissement de la croyance que j'allais connaître Albertine, l'avait, à quelques secondes d'intervalle, rendue presque insignifiante puis infiniment précieuse à mes yeux ; quelques années plus tard, la croyance, puis la disparition de la croyance qu'Albertine m'était fidèle, amena des changements analogues).

Certes, à Combray déjà j'avais vu diminuer ou grandir selon les heures, selon que j'entrais dans l'un ou l'autre des deux grands modes qui se partageaient ma sensibilité, le chagrin de n'être pas près de ma mère, aussi imperceptible tout l'après-midi que la lumière de la lune tant que brille le soleil et, la nuit venue, régnant seul dans mon âme anxieuse à la place de souvenirs effacés et récents. Mais ce jour-là, en voyant qu'Elstir quittait les jeunes filles sans m'avoir appelé, j'appris que les variations de l'importance qu'ont à nos yeux un plaisir ou un chagrin peuvent ne pas tenir seulement à cette alternance de deux états, mais au déplacement de croyances invisibles, lesquelles par exemple nous font paraître indifférente la mort parce qu'elles répandent sur celle-ci une lumière d'irréalité, et nous permettent ainsi d'attacher de l'importance à nous rendre à une soirée musicale qui perdrait de son charme si, à l'annonce que nous allons être guillotinés, la croyance qui baigne cette soirée se dissipait tout à coup ; ce rôle des croyances, il est vrai que quelque chose en moi le savait, c'était la volonté, mais elle le sait en vain si l'intelligence, la sensibilité continuent à l'ignorer ; celles-ci sont de bonne foi quand elles croient que nous avons envie de quitter une maîtresse à laquelle seule notre volonté sait que nous tenons. C'est qu'elles sont obscurcies par la croyance que nous la retrouverons dans un instant. Mais que cette croyance se dissipe, qu'elles apprennent tout d'un coup que cette maîtresse est partie pour toujours, alors l'intelligence et la sensibilité ayant perdu leur mise au point sont comme folles, le plaisir infime s'agrandit à l'infini.

Variation d'une croyance, néant de l'amour aussi, lequel, préexistant et mobile, s'arrête à l'image d'une femme simplement parce que cette femme sera presque impossible à atteindre. Dès lors on pense moins à la femme, qu'on se représente difficilement, qu'aux moyens de la connaître. Tout un processus d'angoisses se développe et suffit pour fixer notre amour sur celle qui en est l'objet à peine connu de nous. L'amour devient immense, nous ne songeons pas combien la femme réelle y tient peu de place. Et si tout d'un coup, comme au moment où j'avais vu Elstir s'arrêter avec les jeunes filles, nous cessons d'être inquiets, d'avoir de l'angoisse, comme c'est elle qui est tout notre amour, il semble brusquement qu'il se soit évanoui au moment où nous tenons enfin la proie à la valeur de laquelle nous n'avons pas assez pensé. Que connaissais-je d'Albertine ? Un ou deux profils sur la mer, moins beaux assurément que ceux des femmes de Véronèse que j'aurais dû, si j'avais obéi à des raisons purement esthétiques, lui préférer. Or, pouvais-je en d'autres raisons, puisque, l'anxiété tombée, je ne pouvais retrouver que ces profils muets, je ne possédais rien d'autre ? Depuis que j'avais vu Albertine, j'avais fait chaque jour à son sujet des milliers de réflexions, j'avais poursuivi, avec ce que j'appelais elle, tout un entretien intérieur, où je la faisais questionner, répondre, penser, agir, et dans la série indéfinie d'Albertines imaginées qui se succédaient en moi heure par heure, l'Albertine réelle, aperçue sur la plage, ne figurait qu'en tête, comme la créatrice d'un rôle, l'étoile, ne paraît, dans une longue série de représentations, que dans toutes les premières. Cette Albertine-là n'était guère qu'une silhouette, tout ce qui était superposé était de mon cru, tant dans l'amour les apports qui viennent de nous l'emportent – à ne se placer même qu'au point de vue quantité – sur ceux qui nous viennent de l'être aimé. Et cela est vrai des amours les plus effectifs. Il en est qui peuvent non seulement se former mais subsister autour de bien peu de chose – et même parmi ceux qui ont reçu leur exaucement charnel. Un ancien professeur de dessin de ma grand'mère avait eu d'une maîtresse obscure une fille. La mère mourut peu de temps après la naissance de l'enfant et le professeur de dessin en eut un chagrin tel qu'il ne survécut pas longtemps. Dans les derniers mois de sa vie, ma grand'mère et quelques dames de Combray, qui n'avaient jamais voulu faire même allusion devant leur professeur à cette femme, avec laquelle d'ailleurs il n'avait pas officiellement vécu et n'avait eu que peu de relations, songèrent à assurer le sort de la petite fille en se cotisant pour lui faire une rente viagère. Ce fut ma grand'mère qui le proposa, certaines amies se firent tirer l'oreille ; cette petite-fille était-elle vraiment si intéressante, était-elle seulement la fille de celui qui s'en croyait le père ; avec des femmes comme était la mère, on n'est jamais sûr. Enfin on se décida. La petite fille vint remercier. Elle était laide et d'une ressemblance avec le vieux maître de dessin qui ôta tous les doutes ; comme ses cheveux étaient tout ce qu'elle avait de bien, une dame dit au père qui l'avait conduite : « Comme elle a de beaux cheveux. » Et pensant que maintenant, la femme coupable étant morte et le professeur à demi mort, une allusion à ce passé qu'on avait toujours feint d'ignorer n'avait plus de conséquence, ma grand-mère ajouta : « Ça doit être de famille. Est-ce que sa mère avait ces beaux cheveux-là ? – Je ne sais pas, répondit naïvement le père. Je ne l'ai jamais vue qu'en chapeau. »
