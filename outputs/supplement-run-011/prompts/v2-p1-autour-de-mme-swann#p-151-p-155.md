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
        "Les Swann",
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
      "confidence": 0.9,
      "evidence": "« Les Swann participaient à ce travers des gens chez qui peu de monde va… On en parlait aux amis, on les faisait passer de mains en mains. Le salon des Swann ressemblait… à ces hôtels… où on affiche les dépêches. » / « l'ancien Swann avait cessé d'être non seulement discret… mais difficile quand il s'agissait de les choisir. Comment Mme Bontemps, si commune, si méchante, ne l'exaspérait-elle pas ? » / « Il en usait maintenant à l'égard des gens qu'il recevait… il mettait en valeur les mérites de Mme Bontemps… »",
      "explanation": "The narrator portrays Swann as having lost discretion and discriminating taste, publicizing flattering signs of regard and praising socially mediocre figures like Mme Bontemps. This ironic framing lowers Swann’s local standing."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Swann is locally diminished as vain and less discerning than he once was, shifting from discreet selectiveness to indiscriminate approval and publicity."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-151-p-155"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Bontemps",
  "Odette",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "princesse de Parme",
  "le narrateur"
]

### Prior local context (optional)

Ce n'est pas ainsi que Swann parlait autrefois ; mais qui n'a vu des princesses royales fort simples, si dix ans plus tard elles se sont fait enlever par un valet de chambre, et qu'elles cherchent à revoir du monde et sentent qu'on ne vient pas volontiers chez elles, prendre spontanément le langage des vieilles raseuses, et quand on cite une duchesse à la mode, ne les a entendues dire : « Elle était hier chez moi », et : « Je vis très à l'écart » ? Aussi est-il inutile d'observer les moeurs, puisqu'on peut les déduire des lois psychologiques.

### Passage

Les Swann participaient à ce travers des gens chez qui peu de monde va ; la visite, l'invitation, une simple parole aimable de personnes un peu marquantes étaient pour eux un événement auquel ils souhaitaient de donner de la publicité. Si la mauvaise chance voulait que les Verdurin fussent à Londres quand Odette avait eu un dîner un peu brillant, on s'arrangeait pour que par quelque ami commun la nouvelle leur en fût câblée outre-Manche. Il n'est pas jusqu'aux lettres, aux télégrammes flatteurs reçus par Odette, que les Swann ne fussent incapables de garder pour eux. On en parlait aux amis, on les faisait passer de mains en mains. Le salon des Swann ressemblait ainsi à ces hôtels de villes d'eaux où on affiche les dépêches.

Du reste, les personnes qui n'avaient pas seulement connu l'ancien Swann en dehors du monde, comme j'avais fait, mais dans le monde, dans ce milieu Guermantes, où, en exceptant les Altesses et les Duchesses, on était d'une exigence infinie pour l'esprit et le charme, où on prononçait l'exclusive pour des hommes éminents qu'on trouvait ennuyeux ou vulgaires, ces personnes-là auraient pu s'étonner en constatant que l'ancien Swann avait cessé d'être non seulement discret quand il parlait de ses relations mais difficile quand il s'agissait de les choisir. Comment Mme Bontemps, si commune, si méchante, ne l'exaspérait-elle pas ? Comment pouvait-il la déclarer agréable ? Le souvenir du milieu Guermantes aurait dû l'en empêcher, semblait-il ; en réalité il l'y aidait. Il y avait certes chez les Guermantes, à l'encontre des trois quarts des milieux mondains, du goût, un goût raffiné même, mais aussi du snobisme, d'où possibilité d'une interruption momentanée dans l'exercice du goût. S'il s'agissait de quelqu'un qui n'était pas indispensable à cette coterie, d'un ministre des Affaires étrangères, républicain un peu solennel, d'un académicien bavard, le goût s'exerçait à fond contre lui, Swann plaignait Mme de Guermantes d'avoir dîné à côté de pareils convives dans une ambassade et on leur préférait mille fois un homme élégant, c'est-à-dire un homme du milieu Guermantes, bon à rien, mais possédant l'esprit des Guermantes, quelqu'un qui était de la même chapelle. Seulement, une grande-duchesse, une princesse du sang dînait-elle souvent chez Mme de Guermantes, elle se trouvait alors faire partie de cette chapelle elle aussi, sans y avoir aucun droit, sans en posséder en rien l'esprit. Mais avec la naïveté des gens du monde, du moment qu'on la recevait, on s'ingéniait à la trouver agréable, faute de pouvoir se dire que c'est parce qu'on l'avait trouvée agréable qu'on la recevait. Swann venant au secours de Mme de Guermantes lui disait quand l'Altesse était partie : « Au fond elle est bonne femme, elle a même un certain sens du comique. Mon Dieu je ne pense pas qu'elle ait approfondi la Critique de la Raison pure, mais elle n'est pas déplaisante. – Je suis absolument de votre avis, répondait la duchesse. Et encore elle était intimidée, mais vous verrez qu'elle peut être charmante. – Elle est bien moins embêtante que Mme X (la femme de l'académicien bavard, laquelle était remarquable) qui vous cite vingt volumes. – Mais il n'y a même pas de comparaison possible. »

La faculté de dire de telles choses, de les dire sincèrement, Swann l'avait acquise chez la duchesse, et conservée. Il en usait maintenant à l'égard des gens qu'il recevait. Il s'efforçait à discerner, à aimer en eux les qualités que tout être humain révèle, si on l'examine avec une prévention favorable et non avec le dégoût des délicats ; il mettait en valeur les mérites de Mme Bontemps comme autrefois ceux de la princesse de Parme, laquelle eût dû être exclue du milieu Guermantes, s'il n'y avait pas eu entrée de faveur pour certaines Altesses et si même quand il s'agissait d'elles on n'eût vraiment considéré que l'esprit et un certain charme. On a vu d'ailleurs autrefois que Swann avait le goût (dont il faisait maintenant une application seulement plus durable) d'échanger sa situation mondaine contre une autre qui dans certaines circonstances lui convenait mieux. Il n'y a que les gens incapables de décomposer, dans leur perception, ce qui au premier abord paraît indivisible, qui croient que la situation fait corps avec la personne. Un même être, pris à des moments successifs de sa vie, baigne à différents degrés de l'échelle sociale dans des milieux qui ne sont pas forcément de plus en plus élevés ; et chaque fois que dans une période autre de l'existence, nous nouons, ou renouons, des liens avec un certain milieu, que nous nous y sentons choyés, nous commençons tout naturellement à nous y attacher en y poussant d'humaines racines.

Pour ce qui concerne Mme Bontemps, je crois aussi que Swann en parlant d'elle avec cette insistance n'était pas fâché de penser que mes parents apprendraient qu'elle venait voir sa femme. À vrai dire, à la maison, le nom des personnes que celle-ci arrivait peu à peu à connaître piquait plus la curiosité qu'il n'excitait d'admiration. Au nom de Mme Trombert, ma mère disait :

– Ah ! mais voilà une nouvelle recrue et qui lui en amènera d'autres.
