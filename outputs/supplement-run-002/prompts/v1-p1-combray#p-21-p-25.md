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
      "source": "collective_social_voice",
      "target": "Swann",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.9,
      "evidence": "« On ne se gênait guère pour l'envoyer quérir ... pour de grands dîners où on ne l'invitait pas »; « elle lui faisait pousser le piano et tourner les pages »; « en usait-elle cavalièrement avec lui ».",
      "explanation": "The family treats Swann as lacking prestige, using him for errands and service while excluding him from grand dinners. The narrator reports this with mild irony."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "He is locally excluded from valued social occasions and made to serve, signaling clear social diminishment within the household."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-21-p-25"
}

### Candidate characters

[
  "Mme de Villeparisis",
  "la grand-mère",
  "la mère du narrateur",
  "le grand-père du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Un jour qu'il était venu nous voir à Paris, après dîner, en s'excusant d'être en habit, Françoise ayant, après son départ, dit tenir du cocher qu'il avait dîné « chez une princesse », – « Oui, chez une princesse du demi-monde ! » avait répondu ma tante en haussant les épaules sans lever les yeux de sur son tricot, avec une ironie sereine.

### Passage

Aussi, ma grand'tante en usait-elle cavalièrement avec lui. Comme elle croyait qu'il devait être flatté par nos invitations, elle trouvait tout naturel qu'il ne vînt pas nous voir l'été sans avoir à la main un panier de pêches ou de framboises de son jardin, et que de chacun de ses voyages d'Italie il m'eût rapporté des photographies de chefs-d'oeuvre.

On ne se gênait guère pour l'envoyer quérir dès qu'on avait besoin d'une recette de sauce gribiche ou de salade à l'ananas pour de grands dîners où on ne l'invitait pas, ne lui trouvant pas un prestige suffisant pour qu'on pût le servir à des étrangers qui venaient pour la première fois. Si la conversation tombait sur les princes de la Maison de France : « des gens que nous ne connaîtrons jamais ni vous ni moi et nous nous en passons, n'est-ce pas », disait ma grand'tante à Swann qui avait peut-être dans sa poche une lettre de Twickenham ; elle lui faisait pousser le piano et tourner les pages les soirs où la soeur de ma grand'mère chantait, ayant, pour manier cet être ailleurs si recherché, la naïve brusquerie d'un enfant qui joue avec un bibelot de collection sans plus de précautions qu'avec un objet bon marché. Sans doute le Swann que connurent à la même époque tant de clubmen était bien différent de celui que créait ma grand'tante, quand le soir, dans le petit jardin de Combray, après qu'avaient retenti les deux coups hésitants de la clochette, elle injectait et vivifiait de tout ce qu'elle savait sur la famille Swann l'obscur et incertain personnage qui se détachait, suivi de ma grand'mère, sur un fond de ténèbres, et qu'on reconnaissait à la voix. Mais même au point de vue des plus insignifiantes choses de la vie, nous ne sommes pas un tout matériellement constitué, identique pour tout le monde et dont chacun n'a qu'à aller prendre connaissance comme d'un cahier des charges ou d'un testament ; notre personnalité sociale est une création de la pensée des autres. Même l'acte si simple que nous appelons « voir une personne que nous connaissons » est en partie un acte intellectuel. Nous remplissons l'apparence physique de l'être que nous voyons de toutes les notions que nous avons sur lui, et dans l'aspect total que nous nous représentons, ces notions ont certainement la plus grande part. Elles finissent par gonfler si parfaitement les joues, par suivre en une adhérence si exacte la ligne du nez, elles se mêlent si bien de nuancer la sonorité de la voix comme si celle-ci n'était qu'une transparente enveloppe, que chaque fois que nous voyons ce visage et que nous entendons cette voix, ce sont ces notions que nous retrouvons, que nous écoutons. Sans doute, dans le Swann qu'ils s'étaient constitué, mes parents avaient omis par ignorance de faire entrer une foule de particularités de sa vie mondaine qui étaient cause que d'autres personnes, quand elles étaient en sa présence, voyaient les élégances régner dans son visage et s'arrêter à son nez busqué comme à leur frontière naturelle ; mais aussi ils avaient pu entasser dans ce visage désaffecté de son prestige, vacant et spacieux, au fond de ces yeux dépréciés, le vague et doux résidu – mi-mémoire, mi-oubli – des heures oisives passées ensemble après nos dîners hebdomadaires, autour de la table de jeu ou au jardin, durant notre vie de bon voisinage campagnard. L'enveloppe corporelle de notre ami en avait été si bien bourrée, ainsi que de quelques souvenirs relatifs à ses parents, que ce Swann-là était devenu un être complet et vivant, et que j'ai l'impression de quitter une personne pour aller vers une autre qui en est distincte, quand, dans ma mémoire, du Swann que j'ai connu plus tard avec exactitude, je passe à ce premier Swann – à ce premier Swann dans lequel je retrouve les erreurs charmantes de ma jeunesse, et qui d'ailleurs ressemble moins à l'autre qu'aux personnes que j'ai connues à la même époque, comme s'il en était de notre vie ainsi que d'un musée où tous les portraits d'un même temps ont un air de famille, une même tonalité – à ce premier Swann rempli de loisir, parfumé par l'odeur du grand marronnier, des paniers de framboises et d'un brin d'estragon.

Pourtant un jour que ma grand'mère était allée demander un service à une dame qu'elle avait connue au Sacré-Coeur (et avec laquelle, à cause de notre conception des castes, elle n'avait pas voulu rester en relations, malgré une sympathie réciproque), la Mme de Villeparisis, de la célèbre famille de Bouillon, celle-ci lui avait dit : « Je crois que vous connaissez beaucoup Swann qui est un grand ami de mes neveux des Laumes ». Ma grand'mère était revenue de sa visite enthousiasmée par la maison qui donnait sur des jardins et où Mme de Villeparisis lui conseillait de louer, et aussi par un giletier et sa fille, qui avaient leur boutique dans la cour et chez qui elle était entrée demander qu'on fît un point à sa jupe qu'elle avait déchirée dans l'escalier. Ma grand'mère avait trouvé ces gens parfaits, elle déclarait que la petite était une perle et que le giletier était l'homme le plus distingué, le mieux qu'elle eût jamais vu. Car pour elle, la distinction était quelque chose d'absolument indépendant du rang social. Elle s'extasiait sur une réponse que le giletier lui avait faite, disant à maman : « Sévigné n'aurait pas mieux dit ! » et, en revanche, d'un neveu de Mme de Villeparisis qu'elle avait rencontré chez elle : « Ah ! ma fille, comme il est commun ! »

Or le propos relatif à Swann avait eu pour effet, non pas de relever celui-ci dans l'esprit de ma grand'tante, mais d'y abaisser Mme de Villeparisis. Il semblait que la considération que, sur la foi de ma grand'mère, nous accordions à Mme de Villeparisis, lui créât un devoir de ne rien faire qui l'en rendît moins digne et auquel elle avait manqué en apprenant l'existence de Swann, en permettant à des parents à elle de le fréquenter. « Comment ! elle connaît Swann ? Pour une personne que tu prétendais parente du maréchal de Mac-Mahon ! » Cette opinion de mes parents sur les relations de Swann leur parut ensuite confirmée par son mariage avec une femme de la pire société, presque une cocotte que, d'ailleurs, il ne chercha jamais à présenter, continuant à venir seul chez nous, quoique de moins en moins, mais d'après laquelle ils crurent pouvoir juger – supposant que c'était là qu'il l'avait prise – le milieu, inconnu d'eux, qu'il fréquentait habituellement.

Mais une fois, mon grand-père lut dans son journal que Swann était un des plus fidèles habitués des déjeuners du dimanche chez le duc de X..., dont le père et l'oncle avaient été les hommes d'État les plus en vue du règne de Louis-Philippe. Or mon grand-père était curieux de tous les petits faits qui pouvaient l'aider à entrer par la pensée dans la vie privée d'hommes comme Molé, comme le duc Pasquier, comme le duc de Broglie. Il fut enchanté d'apprendre que Swann fréquentait des gens qui les avaient connus. Ma grand'tante au contraire interpréta cette nouvelle dans un sens défavorable à Swann : quelqu'un qui choisissait ses fréquentations en dehors de la caste où il était né, en dehors de sa « classe » sociale, subissait à ses yeux un fâcheux déclassement. Il lui semblait qu'on renonçât d'un coup au fruit de toutes les belles relations avec des gens bien posés, qu'avaient honorablement entretenues et engrangées pour leurs enfants les familles prévoyantes (ma grand'tante avait même cessé de voir le fils d'un notaire de nos amis parce qu'il avait épousé une altesse et était par là descendu pour elle du rang respecté de fils de notaire à celui d'un de ces aventuriers, anciens valets de chambre ou garçons d'écurie, pour qui on raconte que les reines eurent parfois des bontés). Elle blâma le projet qu'avait mon grand-père d'interroger Swann, le soir prochain où il devait venir dîner, sur ces amis que nous lui découvrions. D'autre part les deux soeurs de ma grand'mère, vieilles filles qui avaient sa noble nature, mais non son esprit, déclarèrent ne pas comprendre le plaisir que leur beau-frère pouvait trouver à parler de niaiseries pareilles. C'étaient des personnes d'aspirations élevées et qui à cause de cela même étaient incapables de s'intéresser à ce qu'on appelle un potin, eût-il même un intérêt historique, et d'une façon générale à tout ce qui ne se rattachait pas directement à un objet esthétique ou vertueux. Le désintéressement de leur pensée était tel, à l'égard de tout ce qui, de près ou de loin semblait se rattacher à la vie mondaine, que leur sens auditif, – ayant fini par comprendre son inutilité momentanée dès qu'à dîner la conversation prenait un ton frivole ou seulement terre à terre sans que ces deux vieilles demoiselles aient pu la ramener aux sujets qui leur étaient chers, – mettait alors au repos ses organes récepteurs et leur laissait subir un véritable commencement d'atrophie. Si alors mon grand-père avait besoin d'attirer l'attention des deux soeurs, il fallait qu'il eût recours à ces avertissements physiques dont usent les médecins aliénistes à l'égard de certains maniaques de la distraction : coups frappés à plusieurs reprises sur un verre avec la lame d'un couteau, coïncidant avec une brusque interpellation de la voix et du regard, moyens violents que ces psychiatres transportent souvent dans les rapports courants avec des gens bien portants, soit par habitude professionnelle, soit qu'ils croient tout le monde un peu fou.
