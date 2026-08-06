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
      "canonical_name": "M. Vinteuil",
      "surface_forms": [
        "M. Vinteuil",
        "Vinteuil"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "M. Vinteuil",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« les dernières oeuvres de M. Vinteuil m'avaient paru synthétiser »; la “petite phrase” rendait à Swann ses jours tels qu’il les avait sentis.",
      "explanation": "The narrator credits Vinteuil’s music with uniquely synthesizing and reviving lived sensations, presenting it as a powerful, truth-bearing art."
    }
  ],
  "status_effects": [
    {
      "character": "M. Vinteuil",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Vinteuil’s standing is locally elevated as his work is treated as the privileged vehicle of authentic, profound experience."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-11-p-15"
}

### Candidate characters

[
  "Bergotte",
  "Swann",
  "duchesse de Guermantes",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

Ce plaisir me semblait aujourd'hui un plaisir purement frivole, celui d'aller à une matinée chez duchesse de Guermantes. Mais puisque je savais maintenant que je ne pouvais rien atteindre de plus que des plaisirs frivoles, à quoi bon me les refuser ? Je me redisais que je n'avais éprouvé en essayant cette description rien de cet enthousiasme qui n'est pas le seul mais qui est un premier critérium du talent. J'essayais maintenant de tirer de ma mémoire d'autres « instantanés », notamment des instantanés qu'elle avait pris à Venise, mais rien que ce mot me la rendait ennuyeuse comme une exposition de photographies, et je ne me sentais pas plus de goût, plus de talent, pour décrire maintenant ce que j'avais vu autrefois qu'hier ce que j'observais d'un oeil minutieux et morne, au moment même. Dans un instant tant d'amis que je n'avais pas vus depuis si longtemps allaient sans doute me demander de ne plus m'isoler ainsi, de leur consacrer mes journées. Je n'aurais aucune raison de le leur refuser, puisque j'avais maintenant la preuve que je n'étais plus bon à rien, que la littérature ne pouvait plus me causer aucune joie, soit par ma faute, étant trop peu doué, soit par la sienne, si elle était, en effet, moins chargée de réalité que je n'avais cru.

### Passage

Quand je pensais à ce que Bergotte m'avait dit : « Vous êtes malade, mais on ne peut vous plaindre car vous avez les joies de l'esprit », je voyais combien il s'était trompé sur moi. Comme il y avait peu de joie dans cette lucidité stérile ! J'ajoute même que si quelquefois j'avais peut-être des plaisirs – non de l'intelligence – je les dépensais toujours pour une femme différente ; de sorte que le Destin, m'eût-il accordé cent ans de vie de plus, et sans infirmités, n'eût fait qu'ajouter des rallonges successives à une existence toute en longueur, dont on ne voyait même pas l'intérêt qu'elle se prolongeât davantage, à plus forte raison longtemps encore.

Quant aux « joies de l'intelligence », pouvais-je ainsi appeler ces froides constatations que mon oeil clairvoyant ou mon raisonnement juste relevaient sans aucun plaisir et qui restaient infécondes.

Mais c'est quelquefois au moment où tout nous semble perdu que l'avertissement arrive qui peut nous sauver : on a frappé à toutes les portes qui ne donnent sur rien, et la seule par où on peut entrer et qu'on aurait cherchée en vain pendant cent ans, on y heurte sans le savoir et elle s'ouvre.

En roulant les tristes pensées que je disais il y a un instant j'étais entré dans la cour de l'hôtel de Guermantes, et dans ma distraction je n'avais pas vu une voiture qui s'avançait ; au cri du wattman je n'eus que le temps de me ranger vivement de côté, et je reculai assez pour buter malgré moi contre des pavés assez mal équarris derrière lesquels était une remise. Mais au moment où, me remettant d'aplomb, je posai mon pied sur un pavé qui était un peu moins élevé que le précédent, tout mon découragement s'évanouit devant la même félicité qu'à diverses époques de ma vie m'avaient donnée la vue d'arbres que j'avais cru reconnaître dans une promenade en voiture autour de Balbec, la vue des clochers de Martinville, la saveur d'une madeleine trempée dans une infusion, tant d'autres sensations dont j'ai parlé et que les dernières oeuvres de Vinteuil m'avaient paru synthétiser. Comme au moment où je goûtais la madeleine, toute inquiétude sur l'avenir, tout doute intellectuel étaient dissipés. Ceux qui m'assaillaient tout à l'heure au sujet de la réalité de mes dons littéraires, et même de la réalité de la littérature, se trouvaient levés comme par enchantement. Cette fois je me promettais bien de ne pas me résigner à ignorer pourquoi, sans que j'eusse fait aucun raisonnement nouveau, trouvé aucun argument décisif, les difficultés, insolubles tout à l'heure, avaient perdu toute importance, comme je l'avais fait le jour où j'avais goûté d'une madeleine trempée dans une infusion. La félicité que je venais d'éprouver était bien, en effet, la même que celle que j'avais éprouvée en mangeant la madeleine et dont j'avais alors ajourné de rechercher les causes profondes. La différence, purement matérielle, était dans les images évoquées. Un azur profond enivrait mes yeux, des impressions de fraîcheur, d'éblouissante lumière tournoyaient près de moi et, dans mon désir de les saisir, sans oser plus bouger que quand je goûtais la saveur de la madeleine en tâchant de faire parvenir jusqu'à moi ce qu'elle me rappelait, je restais, quitte à faire rire la foule innombrable des wattmen, à tituber comme j'avais fait tout à l'heure, un pied sur le pavé plus élevé, l'autre pied sur le pavé le plus bas. Chaque fois que je refaisais, rien que matériellement, ce même pas, il me restait inutile ; mais si je réussissais, oubliant la matinée Guermantes, à retrouver ce que j'avais senti en posant ainsi mes pieds, de nouveau la vision éblouissante et indistincte me frôlait comme si elle m'avait dit : « Saisis-moi au passage si tu en as la force et tâche à résoudre l'énigme du bonheur que je te propose. » Et presque tout de suite, je le reconnus, c'était Venise, dont mes efforts pour la décrire et les prétendus instantanés pris par ma mémoire ne m'avaient jamais rien dit et que la sensation que j'avais ressentie jadis sur deux dalles inégales du baptistère de Saint-Marc m'avait rendue avec toutes les autres sensations jointes ce jour-là à cette sensation-là, et qui étaient restées dans l'attente, à leur rang, d'où un brusque hasard les avait impérieusement fait sortir, dans la série des jours oubliés. De même le goût de la petite madeleine m'avait rappelé Combray. Mais pourquoi les images de Combray et de Venise m'avaient-elles, à l'un et à l'autre moment, donné une joie pareille à une certitude et suffisante sans autres preuves à me rendre la mort indifférente ? Tout en me le demandant et en étant résolu aujourd'hui à trouver la réponse, j'entrai dans l'hôtel de Guermantes, parce que nous faisons toujours passer avant la besogne intérieure que nous avons à faire le rôle apparent que nous jouons et qui, ce jour-là, était celui d'un invité. Mais arrivé au premier étage, un maître d'hôtel me demanda d'entrer un instant dans un petit salon-bibliothèque attenant au buffet, jusqu'à ce que le morceau qu'on jouait fût achevé, la princesse ayant défendu qu'on ouvrît les portes pendant son exécution. Or, à ce moment même, un second avertissement vint renforcer celui que m'avaient donné les pavés inégaux et m'exhorter à persévérer dans ma tâche. Un domestique, en effet, venait, dans ses efforts infructueux pour ne pas faire de bruit, de cogner une cuiller contre une assiette. Le même genre de félicité que m'avaient donné les dalles inégales m'envahit ; les sensations étaient de grande chaleur encore, mais toutes différentes, mêlées d'une odeur de fumée apaisée par la fraîche odeur d'un cadre forestier ; et je reconnus que ce qui me paraissait si agréable était la même rangée d'arbres que j'avais trouvée ennuyeuse à observer et à décrire, et devant laquelle, débouchant la canette de bière que j'avais dans le wagon, je venais de croire un instant, dans une sorte d'étourdissement, que je me trouvais, tant le bruit identique de la cuiller contre l'assiette m'avait donné, avant que j'eusse eu le temps de me ressaisir, l'illusion du bruit du marteau d'un employé qui avait arrangé quelque chose à une roue de train pendant que nous étions arrêtés devant ce petit bois. Alors on eût dit que les signes qui devaient, ce jour-là, me tirer de mon découragement et me rendre la foi dans les lettres avaient à coeur de se multiplier, car un maître d'hôtel depuis longtemps au service du prince de Guermantes m'ayant reconnu, et m'ayant apporté dans la bibliothèque où j'étais, pour m'éviter d'aller au buffet, un choix de petits fours, un verre d'orangeade, je m'essuyai la bouche avec la serviette qu'il m'avait donnée ; mais aussitôt, comme le personnage des Mille et une Nuits qui, sans le savoir, accomplit précisément le rite qui fait apparaître, visible pour lui seul, un docile génie prêt à le transporter au loin, une nouvelle vision d'azur passa devant mes yeux ; mais il était pur et salin, il se gonfla en mamelles bleuâtres ; l'impression fut si forte que le moment que je vivais me sembla être le moment actuel, plus hébété que le jour où je me demandais si j'allais vraiment être accueilli par la princesse de Guermantes ou si tout n'allait pas s'effondrer, je croyais que le domestique venait d'ouvrir la fenêtre sur la plage et que tout m'invitait à descendre me promener le long de la digue à marée haute ; la serviette que j'avais prise pour m'essuyer la bouche avait précisément le genre de raideur et d'empesé de celle avec laquelle j'avais eu tant de peine à me sécher devant la fenêtre, le premier jour de mon arrivée à Balbec, et maintenant, devant cette bibliothèque de l'hôtel de Guermantes, elle déployait, réparti dans ses plis et dans ses cassures, le plumage d'un océan vert et bleu comme la queue d'un paon. Et je ne jouissais pas que de ces couleurs, mais de tout un instant de ma vie qui les soulevait, qui avait été sans doute aspiration vers elles, dont quelque sentiment de fatigue ou de tristesse m'avait peut-être empêché de jouir à Balbec, et qui maintenant, débarrassé de ce qu'il y a d'imparfait dans la perception extérieure, pur et désincarné, me gonflait d'allégresse. Le morceau qu'on jouait pouvait finir d'un moment à l'autre et je pouvais être obligé d'entrer au salon. Aussi je m'efforçais de tâcher de voir clair le plus vite possible dans la nature des plaisirs identiques que je venais, par trois fois en quelques minutes, de ressentir, et ensuite de dégager l'enseignement que je devais en tirer. Sur l'extrême différence qu'il y a entre l'impression vraie que nous avons eue d'une chose et l'impression factice que nous nous en donnons quand volontairement nous essayons de nous la représenter, je ne m'arrêtais pas ; me rappelant trop avec quelle indifférence relative Swann avait pu parler autrefois des jours où il était aimé, parce que sous cette phrase il voyait autre chose qu'eux, et de la douleur subite que lui avait causée la petite phrase de Vinteuil en lui rendant ces jours eux-mêmes tels qu'il les avait jadis sentis, je comprenais trop que ce que la sensation des dalles inégales, la raideur de la serviette, le goût de la madeleine avaient réveillé en moi, n'avait aucun rapport avec ce que je cherchais souvent à me rappeler de Venise, de Balbec, de Combray, à l'aide d'une mémoire uniforme ; et je comprenais que la vie pût être jugée médiocre, bien qu'à certains moments elle parût si belle, parce que dans le premier cas c'est sur tout autre chose qu'elle-même, sur des images qui ne gardent rien d'elle qu'on la juge et qu'on la déprécie. Tout au plus notais-je accessoirement que la différence qu'il y a entre chacune des impressions réelles – différences qui expliquent qu'une peinture uniforme de la vie ne puisse être ressemblante – tenait probablement à cette cause : que la moindre parole que nous avons dite à une époque de notre vie, le geste le plus insignifiant que nous avons fait était entouré, portait sur lui le reflet des choses qui logiquement ne tenaient pas à lui, en ont été séparées par l'intelligence, qui n'avait rien à faire d'elles pour les besoins du raisonnement, mais au milieu desquelles – ici reflet rose du soir sur le mur fleuri d'un restaurant champêtre, sensation de faim, désir des femmes, plaisir du luxe ; là volutes bleues de la mer matinale enveloppant des phrases musicales qui en émergent partiellement comme les épaules des ondines – le geste, l'acte le plus simple reste enfermé comme dans mille vases clos dont chacun serait rempli de choses d'une couleur, d'une odeur, d'une température absolument différentes ; sans compter que ces vases, disposés sur toute la hauteur de nos années pendant lesquelles nous n'avons cessé de changer, fût-ce seulement de rêve et de pensée, sont situés à des altitudes bien diverses, et nous donnent la sensation d'atmosphères singulièrement variées. Il est vrai que, ces changements, nous les avons accomplis insensiblement ; mais entre le souvenir qui nous revient brusquement et notre état actuel, de même qu'entre deux souvenirs d'années, de lieux, d'heures différentes, la distance est telle que cela suffirait, en dehors même d'une originalité spécifique, à les rendre incomparables les uns aux autres. Oui, si le souvenir, grâce à l'oubli, n'a pu contracter aucun lien, jeter aucun chaînon entre lui et la minute présente, s'il est resté à sa place, à sa date, s'il a gardé ses distances, son isolement dans le creux d'une vallée ou à la pointe d'un sommet ; il nous fait tout à coup respirer un air nouveau, précisément parce que c'est un air qu'on a respiré autrefois, cet air plus pur que les poètes ont vainement essayé de faire régner dans le Paradis et qui ne pourrait donner cette sensation profonde de renouvellement que s'il avait été respiré déjà, car les vrais paradis sont les paradis qu'on a perdus. Et, au passage, je remarquais qu'il y aurait dans l'oeuvre d'art que je me sentais prêt déjà, sans m'y être consciemment résolu, à entreprendre, de grandes difficultés. Car j'en devrais exécuter les parties successives dans une matière en quelque sorte différente. Elle serait bien différente, celle qui conviendrait aux souvenirs de matins au bord de la mer, de celle d'après-midi à Venise, une matière distincte, nouvelle, d'une transparence, d'une sonorité spéciale, compacte, fraîchissante et rose, et différente encore si je voulais décrire les soirs de Rivebelle où, dans la salle à manger ouverte sur le jardin, la chaleur commençait à se décomposer, à retomber, à se déposer, où une dernière lueur éclairait encore les roses sur les murs du restaurant tandis que les dernières aquarelles du jour étaient encore visibles au ciel. Je glissais rapidement sur tout cela, plus impérieusement sollicité que j'étais de chercher la cause de cette félicité, du caractère de certitude avec lequel elle s'imposait, recherche ajournée autrefois. Or, cette cause, je la devinais en comparant entre elles ces diverses impressions bienheureuses et qui avaient entre elles ceci de commun que je les éprouvais à la fois dans le moment actuel et dans un moment éloigné où le bruit de la cuiller sur l'assiette, l'inégalité des dalles, le goût de la madeleine allaient jusqu'à faire empiéter le passé sur le présent, à me faire hésiter à savoir dans lequel des deux je me trouvais ; au vrai, l'être qui alors goûtait en moi cette impression la goûtait en ce qu'elle avait de commun dans un jour ancien et maintenant, dans ce qu'elle avait d'extra-temporel, un être qui n'apparaissait que quand, par une de ces identités entre le présent et le passé, il pouvait se trouver dans le seul milieu où il pût vivre, jouir de l'essence des choses, c'est-à-dire en dehors du temps. Cela expliquait que mes inquiétudes au sujet de ma mort eussent cessé au moment où j'avais reconnu, inconsciemment, le goût de la petite madeleine, puisqu'à ce moment-là l'être que j'avais été était un être extra-temporel, par conséquent insoucieux des vicissitudes de l'avenir. Cet être-là n'était jamais venu à moi, ne s'était jamais manifesté qu'en dehors de l'action, de la jouissance immédiate, chaque fois que le miracle d'une analogie m'avait fait échapper au présent. Seul il avait le pouvoir de me faire retrouver les jours anciens, le Temps Perdu, devant quoi les efforts de ma mémoire et de mon intelligence échouaient toujours.

Et peut-être, si tout à l'heure je trouvais que Bergotte avait jadis dit faux en parlant des joies de la vie spirituelle, c'était parce que j'appelais vie spirituelle, à ce moment-là, des raisonnements logiques qui étaient sans rapport avec elle, avec ce qui existait en moi à ce moment – exactement comme j'avais pu trouver le monde et la vie ennuyeux parce que je les jugeais d'après des souvenirs sans vérité, alors que j'avais un tel appétit de vivre, maintenant que venait de renaître en moi, à trois reprises, un véritable moment du passé.
