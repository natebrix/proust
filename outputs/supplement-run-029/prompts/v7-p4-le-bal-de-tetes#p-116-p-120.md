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
      "canonical_name": "Françoise",
      "surface_forms": [
        "Françoise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Françoise",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« je travaillerais à mon oeuvre, regardé par Françoise » ; « elle s'était fait du travail littéraire une sorte de compréhension instinctive, plus juste que celle de bien des gens intelligents » ; « Françoise, au contraire, devinait mon bonheur et respectait mon travail »",
      "explanation": "The narrator strongly values Françoise as a witness and aid to his work, attributing to her an instinctive understanding of literary labor superior to that of many intelligent people. This praise elevates her locally."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Françoise gains local esteem thanks to the recognition of her discernment and respect for the writer's work."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-116-p-120"
}

### Candidate characters

[
  "Albertine",
  "Bloch",
  "Mme de Villeparisis",
  "Norpois",
  "Robert de Saint-Loup",
  "Swann",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Je vis Gilberte s'avancer. Moi, pour qui le mariage de Robert de Saint-Loup – les pensées qui m'occupaient alors et qui étaient les mêmes ce matin – était d'hier, je fus étonné de voir à côté d'elle une jeune fille d'environ seize ans, dont la taille élevée mesurait cette distance que je n'avais pas voulu voir.

### Passage

Le temps incolore et insaisissable s'était, afin que, pour ainsi dire, je puisse le voir et le toucher, matérialisé en elle et l'avait pétrie comme un chef-d'oeuvre, tandis que parallèlement sur moi, hélas ! il n'avait fait que son oeuvre. Cependant Mlle de Saint-Loup était devant moi. Elle avait les yeux profonds, nets, forés et perçants. Je fus frappé que son nez, fait comme sur le patron de celui de sa mère et de sa grand'mère, s'arrêtât juste par cette ligne tout à fait horizontale sous le nez, sublime quoique pas assez courte. Un trait aussi particulier eût fait reconnaître une statue entre des milliers, n'eût-on vu que ce trait-là, et j'admirais que la nature fût revenue à point nommé pour la petite fille, comme pour la mère, comme pour la grand'mère, donner, en grand et original sculpteur, ce puissant et décisif coup de ciseau. Ce nez charmant, légèrement avancé en forme de bec, avait la courbe, non point de celui de Swann mais de celui de Saint-Loup. L'âme de ce Guermantes s'était évanouie ; mais la charmante tête aux yeux perçants de l'oiseau envolé était venue se poser sur les épaules de Mlle de Saint-Loup, ce qui faisait longuement rêver ceux qui avaient connu son père. Je la trouvais bien belle, pleine encore d'espérances. Riante, formée des années mêmes que j'avais perdues, elle ressemblait à ma jeunesse.

Enfin cette idée de temps avait un dernier prix pour moi, elle était un aiguillon, elle me disait qu'il était temps de commencer si je voulais atteindre ce que j'avais quelquefois senti au cours de ma vie, dans de brefs éclairs, du côté de Guermantes, dans mes promenades en voiture avec Mme de Villeparisis et qui m'avait fait considérer la vie comme digne d'être vécue. Combien me le semblait-elle davantage, maintenant qu'elle me semblait pouvoir être éclaircie, elle qu'on vit dans les ténèbres ; ramenée au vrai de ce qu'elle était, elle qu'on fausse sans cesse, en somme réalisée dans un livre. Que celui qui pourrait écrire un tel livre serait heureux, pensais-je ; quel labeur devant lui ! Pour en donner une idée, c'est aux arts les plus élevés et les plus différents qu'il faudrait emprunter des comparaisons ; car cet écrivain, qui, d'ailleurs, pour chaque caractère, aurait à en faire apparaître les faces les plus opposées, pour faire sentir son volume comme celui d'un solide devrait préparer son livre minutieusement, avec de perpétuels regroupements de forces, comme pour une offensive, le supporter comme une fatigue, l'accepter comme une règle, le construire comme une église, le suivre comme un régime, le vaincre comme un obstacle, le conquérir comme une amitié, le suralimenter comme un enfant, le créer comme un monde, sans laisser de côté ces mystères qui n'ont probablement leur explication que dans d'autres mondes et dont le pressentiment est ce qui nous émeut le plus dans la vie et dans l'art. Et dans ces grands livres-là, il y a des parties qui n'ont eu le temps que d'être esquissées, et qui ne seront sans doute jamais finies, à cause de l'ampleur même du plan de l'architecte. Combien de grandes cathédrales restent inachevées. Longtemps, un tel livre, on le nourrit, on fortifie ses parties faibles, on le préserve, mais ensuite c'est lui qui grandit, qui désigne notre tombe, la protège contre les rumeurs et quelque peu contre l'oubli. Mais, pour en revenir à moi-même, je pensais plus modestement à mon livre, et ce serait même inexact que de dire en pensant à ceux qui le liraient, à mes lecteurs. Car ils ne seraient pas, comme je l'ai déjà montré, mes lecteurs, mais les propres lecteurs d'eux-mêmes, mon livre n'étant qu'une sorte de ces verres grossissants comme ceux que tendait à un acheteur l'opticien de Combray, mon livre, grâce auquel je leur fournirais le moyen de lire en eux-mêmes. De sorte que je ne leur demanderais pas de me louer ou de me dénigrer, mais seulement de me dire si c'est bien cela, si les mots qu'ils lisent en eux-mêmes sont bien ceux que j'ai écrits (les divergences possibles à cet égard ne devant pas, du reste, provenir toujours de ce que je me serais trompé, mais quelquefois de ce que les yeux du lecteur ne seraient pas de ceux à qui mon livre conviendrait pour bien lire en soi-même). Et changeant à chaque instant de comparaison, selon que je me représentais mieux, et plus matériellement, la besogne à laquelle je me livrerais, je pensais que sur ma grande table de bois blanc je travaillerais à mon oeuvre, regardé par Françoise. Comme tous les êtres sans prétention qui vivent à côté de nous ont une certaine intuition de nos tâches et comme j'avais assez oublié Albertine pour avoir pardonné à Françoise ce qu'elle avait pu faire contre elle, je travaillerais auprès d'elle, et presque comme elle (du moins comme elle faisait autrefois : si vieille maintenant, elle n'y voyait plus goutte), car, épinglant de-ci de-là un feuillet supplémentaire, je bâtirais mon livre, je n'ose pas dire ambitieusement comme une cathédrale, mais tout simplement comme une robe. Quand je n'aurais pas auprès de moi tous mes papiers, toutes mes paperoles, comme disait Françoise, et que me manquerait juste celui dont j'aurais eu besoin, Françoise comprendrait bien mon énervement, elle qui disait toujours qu'elle ne pouvait pas coudre si elle n'avait pas le numéro du fil et les boutons qu'il fallait, et puis, parce que, à force de vivre ma vie, elle s'était fait du travail littéraire une sorte de compréhension instinctive, plus juste que celle de bien des gens intelligents, à plus forte raison que celle des gens bêtes. Ainsi quand j'avais autrefois fait mon article pour le Figaro, pendant que le vieux maître d'hôtel, avec une figure de commisération qui exagère toujours un peu ce qu'a de pénible un labeur qu'on ne pratique pas, qu'on ne conçoit même pas, et même une habitude qu'on n'a pas, comme les gens qui vous disent : « Comme ça doit vous fatiguer d'éternuer comme ça », plaignait sincèrement les écrivains en disant : « Quel casse-tête ça doit être », Françoise, au contraire, devinait mon bonheur et respectait mon travail. Elle se fâchait seulement que je contasse d'avance mes articles à Bloch, craignant qu'il me devançât, et disant : « Tous ces gens-là, vous n'avez pas assez de méfiance, c'est des copiateurs. » Et Bloch se donnait, en effet, un alibi rétrospectif en me disant, chaque fois que je lui avais esquissé quelque chose qu'il trouvait bien : « Tiens, c'est curieux, j'ai fait quelque chose de presque pareil, il faudra que je te lise cela. » (Il n'aurait pas pu me le lire encore, mais allait l'écrire le soir même.)

À force de coller les uns aux autres ces papiers, que Françoise appelait mes paperoles, ils se déchiraient çà et là. Au besoin Françoise pourrait m'aider à les consolider, de la même façon qu'elle mettait des pièces aux parties usées de ses robes ou qu'à la fenêtre de la cuisine, en attendant le vitrier comme moi l'imprimeur, elle collait un morceau de journal à la place d'un carreau cassé.

Elle me disait, en me montrant mes cahiers rongés comme le bois où l'insecte s'est mis : « C'est tout mité, regardez, c'est malheureux, voilà un bout de page qui n'est plus qu'une dentelle, et – l'examinant comme un tailleur – je ne crois pas que je pourrai la refaire, c'est perdu. C'est dommage, c'est peut-être vos plus belles idées. Comme on dit à Combray, il n'y a pas de fourreurs qui s'y connaissent aussi bien comme les mites. Elles se mettent toujours dans les meilleures étoffes. »

D'ailleurs, comme les individualités (humaines ou non) seraient dans ce livre faites d'impressions nombreuses, qui, prises de bien des jeunes filles, de bien des églises, de bien des sonates, serviraient à faire une seule sonate, une seule église, une seule jeune fille, ne ferais-je pas mon livre de la façon que Françoise faisait ce boeuf mode, apprécié par Norpois, et dont tant de morceaux de viande ajoutés et choisis enrichissaient la gelée. Et je réaliserais ce que j'avais tant désiré dans mes promenades du côté de Guermantes et cru impossible, comme j'avais cru impossible, en rentrant, de m'habituer jamais à me coucher sans embrasser ma mère ou, plus tard, à l'idée qu'Albertine aimât les femmes, idée avec laquelle j'avais fini par vivre sans même m'apercevoir de sa présence, car nos plus grandes craintes, comme nos plus grandes espérances, ne sont pas au-dessus de nos forces, et nous pouvons finir par dominer les unes et réaliser les autres. – Oui, à cette oeuvre, cette idée du temps, que je venais de former, disait qu'il était temps de me mettre. Il était grand temps, cela justifiait l'anxiété qui s'était emparée de moi dès mon entrée dans le salon, quand les visages grimés m'avaient donné la notion du temps perdu ; mais était-il temps encore ? L'esprit a ses paysages dont la contemplation ne lui est laissée qu'un temps. J'avais vécu comme un peintre montant un chemin qui surplombe un lac dont un rideau de rochers et d'arbres lui cache la vue. Par une brèche il l'aperçoit, il l'a tout entier devant lui, il prend ses pinceaux. Mais déjà vient la nuit, où l'on ne peut plus peindre, et sur laquelle le jour ne se relèvera plus !
