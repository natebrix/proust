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
      "evidence": "« En somme il mentait autant qu'Odette ... il n'était pas moins égoïste. »; « Son esprit se voila ... il continua à serrer la main à tous ces amis qu'il avait soupçonnés. »; Swann interprète le signe d'Odette comme possiblement coupable.",
      "explanation": "The narrator exposes Swann's moral weakness and instability: generalized suspicions, fluctuating criteria, manipulation to obtain the 'truth', and admission of being 'odious'. This diminishes him locally both as a lover and as a judge of others."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Swann appears hypocritical, jealous, and devoid of stable criteria, which significantly lowers his local esteem in the eyes of the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-541-p-545"
}

### Candidate characters

[
  "Bergotte",
  "M. Verdurin",
  "M. d'Orsan",
  "Mme Verdurin",
  "Odette",
  "Rémi",
  "baron de Charlus",
  "comte de Forcheville",
  "duchesse de Guermantes",
  "le grand-père du narrateur",
  "le narrateur",
  "le peintre",
  "marquis de Bréauté",
  "prince des Laumes"
]

### Prior local context (optional)

Ne pouvant se séparer d'elle sans retour, du moins, s'il l'avait vue sans séparations, sa douleur aurait fini par s'apaiser et peut-être son amour par s'éteindre. Et du moment qu'elle ne voulait pas quitter Paris à jamais, il eût souhaité qu'elle ne le quittât jamais. Du moins comme il savait que la seule grande absence qu'elle faisait était tous les ans celle d'août et septembre, il avait le loisir plusieurs mois d'avance d'en dissoudre l'idée amère dans tout le Temps à venir qu'il portait en lui par anticipation et qui, composé de jours homogènes aux jours actuels, circulait transparent et froid en son esprit où il entretenait la tristesse, mais sans lui causer de trop vives souffrances. Mais cet avenir intérieur, ce fleuve, incolore et libre, voici qu'une seule parole d'Odette venait l'atteindre jusqu'en Swann et, comme un morceau de glace, l'immobilisait, durcissait sa fluidité, le faisait geler tout entier ; et Swann s'était senti soudain rempli d'une masse énorme et infrangible qui pesait sur les parois intérieures de son être jusqu'à le faire éclater : c'est qu'Odette lui avait dit, avec un regard souriant et sournois qui l'observait : « comte de Forcheville va faire un beau voyage, à la Pentecôte. Il va en Égypte », et Swann avait aussitôt compris que cela signifiait : « Je vais aller en Égypte à la Pentecôte avec comte de Forcheville. » Et en effet, si quelques jours après, Swann lui disait : « Voyons, à propos de ce voyage que tu m'as dit que tu ferais avec comte de Forcheville », elle répondait étourdiment : « Oui, mon petit, nous partons le 19, on t'enverra une vue des Pyramides. » Alors il voulait apprendre si elle était la maîtresse de comte de Forcheville, le lui demander à elle-même. Il savait que, superstitieuse comme elle était, il y avait certains parjures qu'elle ne ferait pas et puis la crainte, qui l'avait retenu jusqu'ici, d'irriter Odette en l'interrogeant, de se faire détester d'elle, n'existait plus maintenant qu'il avait perdu tout espoir d'en être jamais aimé.

### Passage

Un jour il reçut une lettre anonyme, qui lui disait qu'Odette avait été la maîtresse d'innombrables hommes (dont on lui citait quelques-uns parmi lesquels Forcheville, M. de Bréauté et le peintre), de femmes, et qu'elle fréquentait les maisons de passe. Il fut tourmenté de penser qu'il y avait parmi ses amis un être capable de lui avoir adressé cette lettre (car par certains détails elle révélait chez celui qui l'avait écrite une connaissance familière de la vie de Swann). Il chercha qui cela pouvait être. Mais il n'avait jamais eu aucun soupçon des actions inconnues des êtres, de celles qui sont sans liens visibles avec leurs propos. Et quand il voulut savoir si c'était plutôt sous le caractère apparent de Charlus, de M. des Laumes, de M. d'Orsan, qu'il devait situer la région inconnue où cet acte ignoble avait dû naître, comme aucun de ces hommes n'avait jamais approuvé devant lui les lettres anonymes et que tout ce qu'ils lui avaient dit impliquait qu'ils les réprouvaient, il ne vit pas plus de raisons pour relier cette infamie plutôt à la nature de l'un que de l'autre. Celle de Charlus était un peu d'un détraqué mais foncièrement bonne et tendre ; celle de M. des Laumes un peu sèche, mais saine et droite. Quant à M. d'Orsan, Swann n'avait jamais rencontré personne qui dans les circonstances même les plus tristes vînt à lui avec une parole plus sentie, un geste plus discret et plus juste. C'était au point qu'il ne pouvait comprendre le rôle peu délicat qu'on prêtait à M. d'Orsan dans la liaison qu'il avait avec une femme riche, et que chaque fois que Swann pensait à lui, il était obligé de laisser de côté cette mauvaise réputation inconciliable avec tant de témoignages certains de délicatesse. Un instant Swann sentit que son esprit s'obscurcissait, et il pensa à autre chose pour retrouver un peu de lumière. Puis il eut le courage de revenir vers ces réflexions. Mais alors, après n'avoir pu soupçonner personne, il lui fallut soupçonner tout le monde. Après tout Charlus l'aimait, avait bon coeur. Mais c'était un névropathe, peut-être demain pleurerait-il de le savoir malade, et aujourd'hui par jalousie, par colère, sur quelque idée subite qui s'était emparée de lui, avait-il désiré lui faire du mal. Au fond, cette race d'hommes est la pire de toutes. Certes, le prince des Laumes était bien loin d'aimer Swann autant que Charlus. Mais à cause de cela même, il n'avait pas avec lui les mêmes susceptibilités ; et puis c'était une nature froide sans doute, mais aussi incapable de vilenies que de grandes actions ; Swann se repentait de ne s'être pas attaché, dans la vie, qu'à de tels êtres. Puis il songeait que ce qui empêche les hommes de faire du mal à leur prochain, c'est la bonté, qu'il ne pouvait au fond répondre que de natures analogues à la sienne, comme était, à l'égard du coeur, celle de Charlus. La seule pensée de faire cette peine à Swann eût révolté celui-ci. Mais avec un homme insensible, d'une autre humanité, comme était le prince des Laumes, comment prévoir à quels actes pouvaient le conduire des mobiles d'une essence différente. Avoir du coeur, c'est tout, et Charlus en avait. M. d'Orsan n'en manquait pas non plus et ses relations cordiales mais peu intimes avec Swann, nées de l'agrément que, pensant de même sur tout, ils avaient à causer ensemble, étaient de plus de repos que l'affection exaltée de Charlus, capable de se porter à des actes de passion, bons ou mauvais. S'il y avait quelqu'un par qui Swann s'était toujours senti compris et délicatement aimé, c'était par M. d'Orsan. Oui, mais cette vie peu honorable qu'il menait ? Swann regrettait de n'en avoir pas tenu compte, d'avoir souvent avoué en plaisantant qu'il n'avait jamais éprouvé si vivement des sentiments de sympathie et d'estime que dans la société d'une canaille. « Ce n'est pas pour rien, se disait-il maintenant, que depuis que les hommes jugent leur prochain, c'est sur les actes. Il n'y a que cela qui signifie quelque chose, et nullement ce que nous disons, ce que nous pensons. Charlus et des Laumes peuvent avoir tels ou tels défauts, ce sont d'honnêtes gens. Orsan n'en a peut-être pas, mais ce n'est pas un honnête homme. Il a pu mal agir une fois de plus. » Puis Swann soupçonna Rémi, qui, il est vrai, n'aurait pu qu'inspirer la lettre, mais cette piste lui parut un instant la bonne. D'abord Lorédan avait des raisons d'en vouloir à Odette. Et puis comment ne pas supposer que nos domestiques, vivant dans une situation inférieure à la nôtre, ajoutant à notre fortune et à nos défauts des richesses et des vices imaginaires pour lesquels ils nous envient et nous méprisent, se trouveront fatalement amenés à agir autrement que des gens de notre monde ? Il soupçonna aussi mon grand-père. Chaque fois que Swann lui avait demandé un service, ne le lui avait-il pas toujours refusé ? puis avec ses idées bourgeoises il avait pu croire agir pour le bien de Swann. Celui-ci soupçonna encore Bergotte, le peintre, les Verdurin, admira une fois de plus au passage la sagesse des gens du monde de ne pas vouloir frayer avec ces milieux artistes où de telles choses sont possibles, peut-être même avouées sous le nom de bonnes farces ; mais il se rappelait des traits de droiture de ces bohèmes, et les rapprocha de la vie d'expédients, presque d'escroqueries, où le manque d'argent, le besoin de luxe, la corruption des plaisirs conduisent souvent l'aristocratie. Bref cette lettre anonyme prouvait qu'il connaissait un être capable de scélératesse, mais il ne voyait pas plus de raison pour que cette scélératesse fût cachée dans le tuf – inexploré d'autrui – du caractère de l'homme tendre que de l'homme froid, de l'artiste que du bourgeois, du grand seigneur que du valet. Quel critérium adopter pour juger les hommes ? au fond il n'y avait pas une seule des personnes qu'il connaissait qui ne pût être capable d'une infamie. Fallait-il cesser de les voir toutes ? Son esprit se voila ; il passa deux ou trois fois ses mains sur son front, essuya les verres de son lorgnon avec son mouchoir, et, songeant qu'après tout, des gens qui le valaient fréquentaient Charlus, le prince des Laumes, et les autres, il se dit que cela signifiait, sinon qu'ils fussent incapables d'infamie, du moins que c'est une nécessité de la vie à laquelle chacun se soumet de fréquenter des gens qui n'en sont peut-être pas incapables. Et il continua à serrer la main à tous ces amis qu'il avait soupçonnés, avec cette réserve de pur style qu'ils avaient peut-être cherché à le désespérer. Quant au fond même de la lettre, il ne s'en inquiéta pas, car pas une des accusations formulées contre Odette n'avait l'ombre de vraisemblance. Swann comme beaucoup de gens avait l'esprit paresseux et manquait d'invention. Il savait bien comme une vérité générale que la vie des êtres est pleine de contrastes, mais pour chaque être en particulier il imaginait toute la partie de sa vie qu'il ne connaissait pas comme identique à la partie qu'il connaissait. Il imaginait ce qu'on lui taisait à l'aide de ce qu'on lui disait. Dans les moments où Odette était auprès de lui, s'ils parlaient ensemble d'une action indélicate commise, ou d'un sentiment indélicat éprouvé par un autre, elle les flétrissait en vertu des mêmes principes que Swann avait toujours entendu professer par ses parents et auxquels il était resté fidèle ; et puis elle arrangeait ses fleurs, elle buvait une tasse de thé, elle s'inquiétait des travaux de Swann. Donc Swann étendait ces habitudes au reste de la vie d'Odette, il répétait ces gestes quand il voulait se représenter les moments où elle était loin de lui. Si on la lui avait dépeinte telle qu'elle était, ou plutôt qu'elle avait été si longtemps avec lui, mais auprès d'un autre homme, il eût souffert, car cette image lui eût paru vraisemblable. Mais qu'elle allât chez des maquerelles, se livrât à des orgies avec des femmes, qu'elle menât la vie crapuleuse de créatures abjectes, quelle divagation insensée à la réalisation de laquelle, Dieu merci, les chrysanthèmes imaginés, les thés successifs, les indignations vertueuses ne laissaient aucune place. Seulement de temps à autre, il laissait entendre à Odette que, par méchanceté, on lui racontait tout ce qu'elle faisait ; et, se servant à propos d'un détail insignifiant mais vrai, qu'il avait appris par hasard, comme s'il était le seul petit bout qu'il laissât passer malgré lui, entre tant d'autres, d'une reconstitution complète de la vie d'Odette qu'il tenait cachée en lui, il l'amenait à supposer qu'il était renseigné sur des choses qu'en réalité il ne savait ni même ne soupçonnait, car si bien souvent il adjurait Odette de ne pas altérer la vérité, c'était seulement, qu'il s'en rendît compte ou non, pour qu'Odette lui dît tout ce qu'elle faisait. Sans doute, comme il le disait à Odette, il aimait la sincérité, mais il l'aimait comme une proxénète pouvant le tenir au courant de la vie de sa maîtresse. Aussi son amour de la sincérité n'étant pas désintéressé, ne l'avait pas rendu meilleur. La vérité qu'il chérissait c'était celle que lui dirait Odette ; mais lui-même, pour obtenir cette vérité, ne craignait pas de recourir au mensonge, le mensonge qu'il ne cessait de peindre à Odette comme conduisant à la dégradation toute créature humaine. En somme il mentait autant qu'Odette parce que, plus malheureux qu'elle, il n'était pas moins égoïste. Et elle, entendant Swann lui raconter ainsi à elle-même des choses qu'elle avait faites, le regardait d'un air méfiant, et, à toute aventure, fâché, pour ne pas avoir l'air de s'humilier et de rougir de ses actes.

Un jour, étant dans la période de calme la plus longue qu'il eût encore pu traverser sans être repris d'accès de jalousie, il avait accepté d'aller le soir au théâtre avec la princesse des Laumes. Ayant ouvert le journal, pour chercher ce qu'on jouait, la vue du titre : Les Filles de Marbre de Théodore Barrière le frappa si cruellement qu'il eut un mouvement de recul et détourna la tête. Éclairé comme par la lumière de la rampe, à la place nouvelle où il figurait, ce mot de « marbre » qu'il avait perdu la faculté de distinguer tant il avait l'habitude de l'avoir souvent sous les yeux, lui était soudain redevenu visible et l'avait aussitôt fait souvenir de cette histoire qu'Odette lui avait racontée autrefois, d'une visite qu'elle avait faite au Salon du Palais de l'Industrie avec Mme Verdurin et où celle-ci lui avait dit : « Prends garde, je saurai bien te dégeler, tu n'es pas de marbre. » Odette lui avait affirmé que ce n'était qu'une plaisanterie, et il n'y avait attaché aucune importance. Mais il avait alors plus de confiance en elle qu'aujourd'hui. Et justement la lettre anonyme parlait d'amour de ce genre. Sans oser lever les yeux vers le journal, il le déplia, tourna une feuille pour ne plus voir ce mot : « Les Filles de Marbre » et commença à lire machinalement les nouvelles des départements. Il y avait eu une tempête dans la Manche, on signalait des dégâts à Dieppe, à Cabourg, à Beuzeval. Aussitôt il fit un nouveau mouvement en arrière.

Le nom de Beuzeval l'avait fait penser à celui d'une autre localité de cette région, Beuzeville, qui porte uni à celui-là par un trait d'union un autre nom, celui de Bréauté, qu'il avait vu souvent sur les cartes, mais dont pour la première fois il remarquait que c'était le même que celui de son ami M. de Bréauté, dont la lettre anonyme disait qu'il avait été l'amant d'Odette. Après tout, pour M. de Bréauté, l'accusation n'était pas invraisemblable ; mais en ce qui concernait Mme Verdurin, il y avait impossibilité. De ce qu'Odette mentait quelquefois, on ne pouvait conclure qu'elle ne disait jamais la vérité et, dans ces propos qu'elle avait échangés avec Mme Verdurin et qu'elle avait racontés elle-même à Swann, il avait reconnu ces plaisanteries inutiles et dangereuses que, par inexpérience de la vie et ignorance du vice, tiennent des femmes dont ils révèlent l'innocence, et qui – comme par exemple Odette – sont plus éloignées qu'aucune d'éprouver une tendresse exaltée pour une autre femme. Tandis qu'au contraire, l'indignation avec laquelle elle avait repoussé les soupçons qu'elle avait involontairement fait naître un instant en lui par son récit, cadrait avec tout ce qu'il savait des goûts, du tempérament de sa maîtresse. Mais à ce moment, par une de ces inspirations de jaloux, analogues à celle qui apporte au poète ou au savant, qui n'a encore qu'une rime ou qu'une observation, l'idée ou la loi qui leur donnera toute leur puissance, Swann se rappela pour la première fois une phrase qu'Odette lui avait dite, il y avait déjà deux ans : « Oh ! Mme Verdurin, en ce moment il n'y en a que pour moi, je suis un amour, elle m'embrasse, elle veut que je fasse des courses avec elle, elle veut que je la tutoie. » Loin de voir alors dans cette phrase un rapport quelconque avec les absurdes propos destinés à simuler le vice que lui avait racontés Odette, il l'avait accueillie comme la preuve d'une chaleureuse amitié. Maintenant voilà que le souvenir de cette tendresse de Mme Verdurin était venu brusquement rejoindre le souvenir de sa conversation de mauvais goût. Il ne pouvait plus les séparer dans son esprit, et les vit mêlées aussi dans la réalité, la tendresse donnant quelque chose de sérieux et d'important à ces plaisanteries qui en retour lui faisaient perdre de son innocence. Il alla chez Odette. Il s'assit loin d'elle. Il n'osait l'embrasser, ne sachant si en elle, si en lui, c'était l'affection ou la colère qu'un baiser réveillerait. Il se taisait, il regardait mourir leur amour. Tout à coup il prit une résolution.

– Odette, lui dit-il, mon chéri, je sais bien que je suis odieux, mais il faut que je te demande des choses. Tu te souviens de l'idée que j'avais eue à propos de toi et de Mme Verdurin ? Dis-moi si c'était vrai, avec elle ou avec une autre.

Elle secoua la tête en fronçant la bouche, signe fréquemment employé par les gens pour répondre qu'ils n'iront pas, que cela les ennuie, à quelqu'un qui leur a demandé : « Viendrez-vous voir passer la cavalcade, assisterez-vous à la Revue ? » Mais ce hochement de tête affecté ainsi d'habitude à un événement à venir mêle à cause de cela de quelque incertitude la dénégation d'un événement passé. De plus il n'évoque que des raisons de convenance personnelle plutôt que la réprobation, qu'une impossibilité morale. En voyant Odette lui faire ainsi le signe que c'était faux, Swann comprit que c'était peut-être vrai.
