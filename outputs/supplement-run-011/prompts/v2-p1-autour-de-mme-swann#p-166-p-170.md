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
        "Swann",
        "M."
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
      "confidence": 0.8,
      "evidence": "« j'eus le soupçon que … [nous avions commis] une autre [erreur], celle de le croire un des hommes les plus élégants de Paris » ; « il goûtait un divertissement assez vulgaire à faire comme des bouquets sociaux » ; projet d’« inviter ensemble les docteur Cottard et duchesse de Guermantes de Vendôme » ; « Être l'ami du comte de Paris ne signifie rien. »",
      "explanation": "The narrator locally belittles Swann's worldly image: his taste for 'declassed great ladies' and his eclectic mixtures are described as vulgar, which contradicts the idea that he would be one of the most elegant in Paris."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "His worldly prestige is put into perspective and tinged with vulgarity by the narrator, despite his visible connections."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-166-p-170"
}

### Candidate characters

[
  "Gilberte",
  "Mme Bontemps",
  "Mme Cottard",
  "Mme Verdurin",
  "Odette",
  "comte de Forcheville",
  "docteur Cottard",
  "duchesse de Guermantes",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Swann était du reste aveugle, en ce qui concernait Odette, non seulement devant ces lacunes de son éducation, mais aussi devant la médiocrité de son intelligence. Bien plus, chaque fois qu'Odette racontait une histoire bête, Swann écoutait sa femme avec une complaisance, une gaieté, presque une admiration où il devait entrer des restes de volupté ; tandis que, dans la même conversation, ce que lui-même pouvait dire de fin, même de profond, était écouté par Odette, habituellement sans intérêt, assez vite, avec impatience et quelquefois contredit avec sévérité. Et on conclura que cet asservissement de l'élite à la vulgarité est de règle dans bien des ménages, si l'on pense, inversement, à tant de femmes supérieures qui se laissent charmer par un butor, censeur impitoyable de leurs plus délicates paroles, tandis qu'elles s'extasient, avec l'indulgence infinie de la tendresse, devant ses facéties les plus plates. Pour revenir aux raisons qui empêchèrent à cette époque Odette de pénétrer dans le faubourg Saint-Germain, il faut dire que le plus récent tour du kaléidoscope mondain avait été provoqué par une série de scandales. Des femmes chez qui on allait en toute confiance avaient été reconnues être des filles publiques, des espionnes anglaises. On allait pendant quelque temps demander aux gens, on le croyait du moins, d'être avant tout, bien posés, bien assis... Odette représentait exactement tout ce avec quoi on venait de rompre et d'ailleurs immédiatement de renouer (car les hommes, ne changeant pas du jour au lendemain, cherchent dans un nouveau régime la continuation de l'ancien, mais en le cherchant sous une forme différente qui permît d'être dupe et de croire que ce n'était plus la société d'avant la crise). Or, aux dames « brûlées » de cette société Odette ressemblait trop. Les gens du monde sont fort myopes ; au moment où ils cessent toutes relations avec des dames israélites qu'ils connaissaient, pendant qu'ils se demandent comment remplacer ce vide, ils aperçoivent, poussée là comme à la faveur d'une nuit d'orage, une dame nouvelle, israélite aussi ; mais grâce à sa nouveauté, elle n'est pas associée dans leur esprit, comme les précédentes, avec ce qu'ils croient devoir détester. Elle ne demande pas qu'on respecte son Dieu. On l'adopte. Il ne s'agissait pas d'antisémitisme à l'époque où je commençai d'aller chez Odette. Mais elle était pareille à ce qu'on voulait fuir pour un temps.

### Passage

Swann, lui, allait souvent faire visite à quelques-unes de ses relations d'autrefois et par conséquent appartenant toutes au plus grand monde. Pourtant, quand il nous parlait des gens qu'il venait d'aller voir, je remarquai qu'entre celles qu'il avait connues jadis le choix qu'il faisait était guidé par cette même sorte de goût, mi-artistique, mi-historique, qui inspirait chez lui le collectionneur. Et remarquant que c'était souvent telle ou telle grande dame déclassée qui l'intéressait parce qu'elle avait été la maîtresse de Liszt ou qu'un roman de Balzac avait été dédié à sa grand'mère (comme il achetait un dessin si Chateaubriand l'avait décrit), j'eus le soupçon que nous avions remplacé à Combray l'erreur de croire Swann un bourgeois n'allant pas dans le monde, par une autre, celle de le croire un des hommes les plus élégants de Paris. Être l'ami du comte de Paris ne signifie rien. Combien y en a-t-il de ces « amis des princes » qui ne seraient pas reçus dans un salon un peu fermé. Les princes se savent princes, ne sont pas snobs et se croient d'ailleurs tellement au-dessus de ce qui n'est pas de leur sang que grands seigneurs et bourgeois leur apparaissent, au-dessous d'eux, presque au même niveau.

Au reste, Swann ne se contentait pas de chercher dans la société telle qu'elle existe et en s'attachant aux noms que le passé y a inscrits et qu'on peut encore y lire, un simple plaisir de lettré et d'artiste, il goûtait un divertissement assez vulgaire à faire comme des bouquets sociaux en groupant des éléments hétérogènes, en réunissant des personnes prises ici et là. Ces expériences de sociologie amusante (ou que Swann trouvait telle) n'avaient pas sur toutes les amies de sa femme – du moins d'une façon constante – une répercussion identique. « J'ai l'intention d'inviter ensemble les Cottard et la duchesse de Vendôme », disait-il en riant à Mme Bontemps, de l'air friand d'un gourmet qui a l'intention et veut faire l'essai de remplacer dans une sauce les clous de girofle par du poivre de Cayenne. Or ce projet qui allait paraître en effet plaisant, dans le sens ancien du mot, aux Cottard, avait le don d'exaspérer Mme Bontemps. Elle avait été récemment présentée par les Swann à la duchesse de Vendôme et avait trouvé cela aussi agréable que naturel. En tirer gloire auprès des Cottard, en le leur racontant, n'avait pas été la partie la moins savoureuse de son plaisir. Mais comme les nouveaux décorés qui, dès qu'ils le sont, voudraient voir se fermer aussitôt le robinet des croix, Mme Bontemps eût souhaité qu'après elle personne de son monde à elle ne fût présenté à la princesse. Elle maudissait intérieurement le goût dépravé de Swann qui lui faisait, pour réaliser une misérable bizarrerie esthétique, dissiper d'un seul coup toute la poudre qu'elle avait jetée aux yeux des Cottard en leur parlant de la duchesse de Vendôme. Comment allait-elle même oser annoncer à son mari que le professeur et sa femme allaient à leur tour avoir leur part de ce plaisir qu'elle lui avait vanté comme unique ? Encore si les Cottard avaient pu savoir qu'ils n'étaient pas invités pour de bon, mais pour l'amusement. Il est vrai que les Bontemps l'avaient été de même, mais Swann ayant pris à l'aristocratie cet éternel donjuanisme qui entre deux femmes de rien fait croire à chacune que ce n'est qu'elle qu'on aime sérieusement, avait parlé à Mme Bontemps de la duchesse de Vendôme comme d'une personne avec qui il était tout indiqué qu'elle dînât. « Oui, nous comptons inviter la princesse avec les Cottard, dit, quelques semaines plus tard Odette, mon mari croit que cette conjonction pourra donner quelque chose d'amusant » car si elle avait gardé du « petit noyau » certaines habitudes chères à Mme Verdurin, comme de crier très fort pour être entendue de tous les fidèles, en revanche, elle employait certaines expressions – comme « conjonction » – chères au milieu Guermantes duquel elle subissait ainsi à distance et à son insu, comme la mer le fait pour la lune, l'attraction, sans pourtant se rapprocher sensiblement de lui. « Oui, les Cottard et la duchesse de Vendôme, est-ce que vous ne trouvez pas que cela sera drôle ? » demanda Swann. « Je crois que ça marchera très mal et que ça ne vous attirera que des ennuis, il ne faut pas jouer avec le feu », répondit Mme Bontemps, furieuse. Elle et son mari furent, d'ailleurs, ainsi que le prince d'Agrigente, invités à ce dîner, que Mme Bontemps et Cottard eurent deux manières de raconter, selon les personnes à qui ils s'adressaient. Aux uns, Mme Bontemps de son côté, Cottard du sien, disaient négligemment quand on leur demandait qui il y avait d'autre au dîner : « Il n'y avait que le prince d'Agrigente, c'était tout à fait intime. » Mais d'autres, risquaient d'être mieux informés (même une fois quelqu'un avait dit à Cottard : « Mais est-ce qu'il n'y avait pas aussi les Bontemps ? – Je les oubliais », avait en rougissant répondu Cottard au maladroit qu'il classa désormais dans la catégorie des mauvaises langues). Pour ceux-là les Bontemps et les Cottard adoptèrent chacun sans s'être consultés une version dont le cadre était identique et où seuls leurs noms respectifs étaient interchangés. Cottard disait : « Eh bien, il y avait seulement les maîtres de maison, le duc et la duchesse de Vendôme – (en souriant avantageusement) le professeur et Mme Cottard, et, ma foi, du diable si on a jamais su pourquoi, car ils allaient là comme des cheveux sur la soupe, M. et Mme Bontemps. » Mme Bontemps récitait exactement le même morceau, seulement c'était M. et Mme Bontemps qui étaient nommés avec une emphase satisfaite, entre la duchesse de Vendôme et le prince d'Agrigente, et les pelés qu'à la fin elle accusait de s'être invités eux-mêmes et qui faisaient tache, c'était les Cottard.

De ses visites Swann rentrait souvent assez peu de temps avant le dîner. À ce moment de six heures du soir où jadis il se sentait si malheureux, il ne se demandait plus ce qu'Odette pouvait être en train de faire et s'inquiétait peu qu'elle eût du monde chez elle, ou fût sortie. Il se rappelait parfois qu'il avait, bien des années auparavant, essayé un jour de lire à travers l'enveloppe une lettre adressée par Odette à Forcheville. Mais ce souvenir ne lui était pas agréable et, plutôt que d'approfondir la honte qu'il ressentait, il préférait se livrer à une petite grimace du coin de la bouche complétée au besoin d'un hochement de tête qui signifiait : « Qu'est-ce que ça peut me faire ? » Certes, il estimait maintenant que l'hypothèse à laquelle il s'était souvent arrêté jadis et d'après quoi c'étaient les imaginations de sa jalousie qui seules noircissaient la vie, en réalité innocente d'Odette, que cette hypothèse (en somme bienfaisante puisque tant qu'avait duré sa maladie amoureuse elle avait diminué ses souffrances en les faisant paraître imaginaires) n'était pas la vraie, que c'était sa jalousie qui avait vu juste, et que si Odette l'avait aimé plus qu'il n'avait cru, elle l'avait aussi trompé davantage. Autrefois pendant qu'il souffrait tant, il s'était juré que, dès qu'il n'aimerait plus Odette et ne craindrait plus de la fâcher ou de lui faire croire qu'il l'aimait trop, il se donnerait la satisfaction d'élucider avec elle, par simple amour de la vérité et comme un point d'histoire, si oui ou non Forcheville était couché avec elle le jour où il avait sonné et frappé au carreau sans qu'on lui ouvrît, et où elle avait écrit à Forcheville que c'était un oncle à elle qui était venu. Mais le problème si intéressant qu'il attendait seulement la fin de sa jalousie pour tirer au clair avait précisément perdu tout intérêt aux yeux de Swann, quand il avait cessé d'être jaloux. Pas immédiatement pourtant. Il n'éprouvait déjà plus de jalousie à l'égard d'Odette, que le jour des coups frappés en vain par lui l'après-midi à la porte du petit hôtel de la rue Lapérouse, avait continué à en exciter chez lui. C'était comme si la jalousie, pareille un peu en cela à ces maladies qui semblent avoir leur siège, leur source de contagionnement, moins dans certaines personnes que dans certains lieux, dans certaines maisons, n'avait pas eu tant pour objet Odette elle-même que ce jour, cette heure du passé perdu où Swann avait frappé à toutes les entrées de l'hôtel d'Odette. On aurait dit que ce jour, cette heure avaient seuls fixé quelques dernières parcelles de la personnalité amoureuse que Swann avait eue autrefois et qu'il ne les retrouvait plus que là. Il était depuis longtemps insoucieux qu'Odette l'eût trompé et le trompât encore. Et pourtant il avait continué pendant quelques années à rechercher d'anciens domestiques d'Odette, tant avait persisté chez lui la douloureuse curiosité de savoir si ce jour-là, tellement ancien, à six heures, Odette était couchée avec Forcheville. Puis cette curiosité elle-même avait disparu, sans pourtant que ses investigations cessassent. Il continuait à tâcher d'apprendre ce qui ne l'intéressait plus, parce que son moi ancien, parvenu à l'extrême décrépitude, agissait encore machinalement, selon des préoccupations abolies au point que Swann ne réussissait même plus à se représenter cette angoisse, si forte pourtant autrefois qu'il ne pouvait se figurer alors qu'il s'en délivrât jamais et que seule la mort de celle qu'il aimait (la mort qui, comme le montrera plus loin, dans ce livre, une cruelle contre-épreuve, ne diminue en rien les souffrances de la jalousie) lui semblait capable d'aplanir pour lui la route, entièrement barrée, de sa vie.

Mais éclaircir un jour les faits de la vie d'Odette auxquels il avait dû ces souffrances n'avait pas été le seul souhait de Swann ; il avait mis en réserve aussi celui de se venger d'elles, quand n'aimant plus Odette il ne la craindrait plus ; or, d'exaucer ce second souhait, l'occasion se présentait justement car Swann aimait une autre femme, une femme qui ne lui donnait pas de motifs de jalousie mais pourtant de la jalousie parce qu'il n'était plus capable de renouveler sa façon d'aimer, et que c'était celle dont il avait usé pour Odette qui lui servait encore pour une autre. Pour que la jalousie de Swann renaquît, il n'était pas nécessaire que cette femme fût infidèle, il suffisait que pour une raison quelconque elle fût loin de lui, à une soirée par exemple, et eût paru s'y amuser. C'était assez pour réveiller en lui l'ancienne angoisse, lamentable et contradictoire excroissance de son amour, et qui éloignait Swann de ce qu'elle était comme un besoin d'atteindre (le sentiment réel que cette jeune femme avait pour lui, le désir caché de ses journées, le secret de son coeur), car entre Swann et celle qu'il aimait cette angoisse interposait un amas réfractaire de soupçons antérieurs, ayant leur cause en Odette, ou en telle autre peut-être qui avait précédé Odette, et qui ne permettait plus à l'amant vieilli de connaître sa maîtresse d'aujourd'hui qu'à travers le fantôme ancien et collectif de la « femme qui excitait sa jalousie » dans lequel il avait arbitrairement incarné son nouvel amour. Souvent pourtant Swann l'accusait, cette jalousie, de le faire croire à des trahisons imaginaires ; mais alors il se rappelait qu'il avait fait bénéficier Odette du même raisonnement et à tort. Aussi tout ce que la jeune femme qu'il aimait faisait aux heures où il n'était pas avec elle cessait de lui paraître innocent. Mais alors qu'autrefois, il avait fait le serment, si jamais il cessait d'aimer celle qu'il ne devinait pas devoir être un jour sa femme, de lui manifester implacablement son indifférence, enfin sincère, pour venger son orgueil longtemps humilié, ces représailles qu'il pouvait exercer maintenant sans risques (car que pouvait lui faire d'être pris au mot et privé de ces tête-à-tête avec Odette qui lui étaient jadis si nécessaires), ces représailles il n'y tenait plus ; avec l'amour avait disparu le désir de montrer qu'il n'avait plus d'amour. Et lui qui, quand il souffrait par Odette, eût tant désiré de lui laisser voir un jour qu'il était épris d'une autre, maintenant qu'il l'aurait pu, il prenait mille précautions pour que sa femme ne soupçonnât pas ce nouvel amour.

Ce ne fut pas seulement à ces goûters, à cause desquels j'avais eu autrefois la tristesse de voir Gilberte me quitter et rentrer plus tôt, que désormais je pris part, mais les sorties qu'elle faisait avec sa mère, soit pour aller en promenade ou à une matinée, et qui en l'empêchant de venir aux Champs-Élysées m'avaient privé d'elle, les jours où je restais seul le long de la pelouse ou devant les chevaux de bois, ces sorties maintenant M. et Odette m'y admettaient, j'avais une place dans leur landau et même c'était à moi qu'on demandait si j'aimais mieux aller au théâtre, à une leçon de danse chez une camarade de Gilberte, à une réunion mondaine chez des amies des Swann (ce que celle-ci appelait « un petit meeting ») ou visiter les Tombeaux de Saint-Denis.
