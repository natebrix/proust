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
      "canonical_name": "Elstir",
      "surface_forms": [
        "Elstir"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Elstir",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« Je ne pus contenir mon admiration. » … « on sentait qu'Elstir … s'était au contraire attaché à ces traits d'ambiguïté comme à un élément esthétique qui valait d'être mis en relief » … « c'était vraiment un culte si grave, si exigeant, qu'il ne lui permettait jamais d'être content »",
      "explanation": "The narrator strongly elevates Elstir by admiring the watercolor and describing the rigor of his aesthetic ideal, almost religious, to which he devoted his entire life."
    }
  ],
  "status_effects": [
    {
      "character": "Elstir",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "In this passage, Elstir clearly gains esteem through the narrator's explicit admiration for his art and his cult of Beauty."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-376-p-380"
}

### Candidate characters

[
  "Albertine",
  "Gilberte",
  "Swann",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Elstir et moi nous étions allés jusqu'au fond de l'atelier, devant la fenêtre qui donnait derrière le jardin sur une étroite avenue de traverse, presque un petit chemin rustique. Nous étions venus là pour respirer l'air rafraîchi de l'après-midi avancé. Je me croyais bien loin des jeunes filles de la petite bande, et c'est en sacrifiant pour une fois l'espérance de les voir, que j'avais fini par obéir à la prière de la grand-mère et aller voir Elstir. Car où se trouve ce qu'on cherche on ne le sait pas, et on fuit souvent pendant bien longtemps le lieu où, pour d'autres raisons, chacun nous invite. Mais nous ne soupçonnons pas que nous y verrions justement l'être auquel nous pensons. Je regardais vaguement le chemin campagnard qui, extérieur à l'atelier, passait tout près de lui mais n'appartenait pas à Elstir. Tout à coup y apparut, le suivant à pas rapides, la jeune cycliste de la petite bande avec, sur ses cheveux noirs, son polo abaissé vers ses grosses joues, ses yeux gais et un peu insistants ; et dans ce sentier fortuné miraculeusement rempli de douces promesses, je la vis sous les arbres adresser à Elstir un salut souriant d'amie, arc-en-ciel qui unit pour moi notre monde terraqué à des régions que j'avais jugées jusque-là inaccessibles. Elle s'approcha même pour tendre la main au le peintre, sans s'arrêter, et je vis qu'elle avait un petit grain de beauté au menton. « Vous connaissez cette jeune fille, monsieur ? » dis-je à Elstir, comprenant qu'il pourrait me présenter à elle, l'inviter chez lui. Et cet atelier paisible avec son horizon rural s'était rempli d'un surcroît délicieux, comme il arrive d'une maison où un enfant se plaisait déjà et où il apprend que, en plus, de par la générosité qu'ont les belles choses et les nobles gens à accroître indéfiniment leurs dons, se prépare pour lui un magnifique goûter. Elstir me dit qu'elle s'appelait Albertine Simonet et me nomma aussi ses autres amies que je lui décrivis avec assez d'exactitude pour qu'il n'eût guère d'hésitation. J'avais commis à l'égard de leur situation sociale une erreur, mais pas dans le même sens que d'habitude à Balbec. J'y prenais facilement pour des princes des fils de boutiquiers montant à cheval. Cette fois j'avais situé dans un milieu interlope des filles d'une petite bourgeoisie fort riche, du monde de l'industrie et des affaires. C'était celui qui de prime abord m'intéressait le moins, n'ayant pour moi le mystère ni du peuple, ni d'une société comme celle des Guermantes. Et sans doute si un prestige préalable qu'elles ne perdraient plus ne leur avait été conféré, devant mes yeux éblouis, par la vacuité éclatante de la vie de plage, je ne serais peut-être pas arrivé à lutter victorieusement contre l'idée qu'elles étaient les filles de gros négociants. Je ne pus qu'admirer combien la bourgeoisie française était un atelier merveilleux de sculpture la plus généreuse et la plus variée. Que de types imprévus, quelle invention dans le caractère des visages, quelle décision, quelle fraîcheur, quelle naïveté dans les traits ! Les vieux bourgeois avares d'où étaient issues ces Dianes et ces nymphes me semblaient les plus grands des statuaires. Avant que j'eusse eu le temps de m'apercevoir de la métamorphose sociale de ces jeunes filles, et tant ces découvertes d'une erreur, ces modifications de la notion qu'on a d'une personne ont l'instantanéité d'une réaction chimique, s'était déjà installée derrière le visage d'un genre si voyou de ces jeunes filles que j'avais prises pour des maîtresses de coureurs cyclistes, de champions de boxe, l'idée qu'elles pouvaient très bien être liées avec la famille de tel notaire que nous connaissions. Je ne savais guère ce qu'était Albertine Simonet. Elle ignorait certes ce qu'elle devait être un jour pour moi. Même ce nom de Simonet que j'avais déjà entendu sur la plage, si on m'avait demandé de l'écrire je l'aurais orthographié avec deux n. ne me doutant pas de l'importance que cette famille attachait à n'en posséder qu'un seul. Au fur et à mesure que l'on descend dans l'échelle sociale, le snobisme s'accroche à des riens qui ne sont peut-être pas plus nuls que les distinctions de l'aristocratie, mais qui plus obscurs, plus particuliers à chacun, surprennent davantage. Peut-être y avait-il eu des Simonet qui avaient fait de mauvaises affaires ou pis encore. Toujours est-il que les Simonet s'étaient, paraît-il, toujours irrités comme d'une calomnie quand on doublait leur n. Ils avaient l'air d'être les seuls Simonet avec un n au lieu de deux, avec autant de fierté peut-être que les Montmorency d'être les premiers barons de France. Je demandai à Elstir si ces jeunes filles habitaient Balbec, il me répondit oui pour certaines d'entre elles. La villa de l'une était précisément située tout au bout de la plage, là où commencent les falaises du Canapville. Comme cette jeune fille était une grande amie d'Albertine Simonet, ce me fut une raison de plus de croire que c'était bien cette dernière que j'avais rencontrée, quand j'étais avec la grand-mère. Certes il y avait tant de ces petites rues perpendiculaires à la plage où elles faisaient un angle pareil, que je n'aurais pu spécifier exactement laquelle c'était. On voudrait avoir un souvenir exact mais au moment même la vision a été trouble. Pourtant qu'Albertine et cette jeune fille entrant chez son amie fussent une seule et même personne, c'était pratiquement une certitude. Malgré cela, tandis que les innombrables images que m'a présentées dans la suite la brune joueuse de golf, si différentes qu'elles soient les unes des autres, se superposent (parce que je sais qu'elles lui appartiennent toutes), et que si je remonte le fil de mes souvenirs, je peux, sous le couvert de cette identité et comme dans un chemin de communication intérieure, repasser par toutes ces images sans sortir d'une même personne, en revanche, si je veux remonter jusqu'à la jeune fille que je croisai le jour où j'étais avec la grand-mère, il me faut ressortir à l'air libre. Je suis persuadé que c'est Albertine que je retrouve, la même que celle qui s'arrêtait souvent, au milieu de ses amies, dans sa promenade, dépassant l'horizon de la mer ; mais toutes ces images restent séparées de cette autre parce que je ne peux pas lui conférer rétrospectivement une identité qu'elle n'avait pas pour moi au moment où elle a frappé mes yeux ; quoi que puisse m'assurer le calcul des probabilités, cette jeune fille aux grosses joues qui me regarda si hardiment au coin de la petite rue et de la plage et par qui je crois que j'aurais pu être aimé, au sens strict du mot revoir, je ne l'ai jamais revue.

### Passage

Mon hésitation entre les diverses jeunes filles de la petite bande, lesquelles gardaient toutes un peu du charme collectif qui m'avait d'abord troublé, s'ajouta-t-il aussi à ces causes pour me laisser plus tard, même au temps de mon plus grand – de mon second – amour pour Albertine, une sorte de liberté intermittente, et bien brève, de ne l'aimer pas ? Pour avoir erré entre toutes ses amies avant de se porter définitivement sur elle, mon amour garda parfois entre lui et l'image d'Albertine certain « jeu » qui lui permettait, comme un éclairage mal adapté, de se poser sur d'autres avant de revenir s'appliquer à elles ; le rapport entre le mal que je ressentais au coeur et le souvenir d'Albertine ne me semblait pas nécessaire, j'aurais peut-être pu le coordonner avec l'image d'une autre personne. Ce qui me permettait, l'éclair d'un instant, de faire évanouir la réalité, non pas seulement la réalité extérieure comme dans mon amour pour Gilberte (que j'avais reconnu pour un état intérieur où je tirais de moi seul la qualité particulière, le caractère spécial de l'être que j'aimais, tout ce qui le rendait indispensable à mon bonheur), mais même la réalité intérieure et purement subjective.

« Il n'y a pas de jour qu'une ou l'autre d'entre elles ne passe devant l'atelier et n'entre me faire un bout de visite », me dit Elstir, me désespérant aussi par la pensée que si j'avais été le voir aussitôt que ma grand-mère m'avait demandé de le faire, j'eusse probablement, depuis longtemps déjà, fait la connaissance d'Albertine.

Elle s'était éloignée ; de l'atelier on ne la voyait plus. Je pensai qu'elle était allée rejoindre ses amies sur la digue. Si j'avais pu m'y trouver avec Elstir, j'eusse fait leur connaissance. J'inventai mille prétextes pour qu'il consentît à venir faire un tour de plage avec moi. Je n'avais plus le même calme qu'avant l'apparition de la jeune fille dans le cadre de la petite fenêtre si charmante jusque-là sous ses chèvrefeuilles et maintenant bien vide. Elstir me causa une joie mêlée de torture en me disant qu'il ferait quelques pas avec moi, mais qu'il était obligé de terminer d'abord le morceau qu'il était en train de peindre. C'était des fleurs, mais pas de celles dont j'eusse mieux aimé lui commander le portrait que celui d'une personne, afin d'apprendre par la révélation de son génie ce que j'avais si souvent cherché en vain devant elles – aubépines, épines roses, bluets, fleurs de pommier. Elstir tout en peignant me parlait de botanique, mais je ne l'écoutais guère ; il ne se suffisait plus à lui-même, il n'était plus que l'intermédiaire nécessaire entre ces jeunes filles et moi ; le prestige que, quelques instants encore auparavant, lui donnait pour moi son talent, ne valait plus qu'en tant qu'il m'en conférait un peu à moi-même aux yeux de la petite bande à qui je serais présenté par lui.

J'allais et venais, impatient qu'il eût fini de travailler ; je saisissais pour les regarder des études dont beaucoup, tournées contre le mur, étaient empilées les unes sur les autres. Je me trouvais ainsi mettre au jour une aquarelle qui devait être d'un temps bien plus ancien de la vie d'Elstir et me causa cette sorte particulière d'enchantement que dispensent des oeuvres, non seulement d'une exécution délicieuse mais aussi d'un sujet si singulier et si séduisant, que c'est à lui que nous attribuons une partie de leur charme, comme si, ce charme, le peintre n'avait eu qu'à le découvrir, qu'à l'observer, matériellement réalisé déjà dans la nature et à le reproduire. Que de tels objets puissent exister, beaux en dehors même de l'interprétation du peintre, cela contente en nous un matérialisme inné, combattu par la raison, et sert de contrepoids aux abstractions de l'esthétique. C'était – cette aquarelle – le portrait d'une jeune femme pas jolie, mais d'un type curieux, que coiffait un serre-tête assez semblable à un chapeau melon bordé d'un ruban de soie cerise ; une de ses mains gantées de mitaines tenait une cigarette allumée, tandis que l'autre élevait à la hauteur du genou une sorte de grand chapeau de jardin, simple écran de paille contre le soleil. À côté d'elle, un porte-bouquet plein de roses sur une table. Souvent et c'était le cas ici, la singularité de ces oeuvres tient surtout à ce qu'elles ont été exécutées dans des conditions particulières dont nous ne nous rendons pas clairement compte d'abord, par exemple si la toilette étrange d'un modèle féminin est un déguisement de bal costumé, ou si au contraire le manteau rouge d'un vieillard, qui a l'air de l'avoir revêtu pour se prêter à une fantaisie du peintre, est sa robe de professeur ou de conseiller, ou son camail de cardinal. Le caractère ambigu de l'être dont j'avais le portrait sous les yeux tenait sans que je le comprisse à ce que c'était une jeune actrice d'autrefois en demi-travesti. Mais son melon, sous lequel ses cheveux étaient bouffants, mais courts, son veston de velours sans revers ouvrant sur un plastron blanc me firent hésiter sur la date de la mode et le sexe du modèle, de façon que je ne savais pas exactement ce que j'avais sous les yeux, sinon le plus clair des morceaux de peinture. Et le plaisir qu'il me donnait était troublé seulement par la peur qu'Elstir en s'attardant encore me fît manquer les jeunes filles, car le soleil était déjà oblique et bas dans la petite fenêtre. Aucune chose dans cette aquarelle n'était simplement constatée en fait et peinte à cause de son utilité dans la scène, le costume parce qu'il fallait que la femme fût habillée, le porte-bouquet pour les fleurs. Le verre du porte-bouquet, aimé pour lui-même, avait l'air d'enfermer l'eau où trempaient les tiges des oeillets dans quelque chose d'aussi limpide, presque d'aussi liquide qu'elle ; l'habillement de la femme l'entourait d'une manière qui avait un charme indépendant, fraternel, et comme si les oeuvres de l'industrie pouvaient rivaliser de charme avec les merveilles de la nature, aussi délicates, aussi savoureuses au toucher du regard, aussi fraîchement peintes que la fourrure d'une chatte, les pétales d'un oeillet, les plumes d'une colombe. La blancheur du plastron, d'une finesse de grésil et dont le frivole plissage avait des clochettes comme celles du muguet, s'étoilait des clairs reflets de la chambre, aigus eux-mêmes et finement nuancés comme des bouquets de fleurs qui auraient broché le linge. Et le velours du veston, brillant et nacré, avait çà et là quelque chose de hérissé, de déchiqueté et de velu qui faisait penser à l'ébouriffage des oeillets dans le vase. Mais surtout on sentait qu'Elstir, insoucieux de ce que pouvait présenter d'immoral ce travesti d'une jeune actrice, pour qui le talent avec lequel elle jouerait son rôle avait sans doute moins d'importance que l'attrait irritant qu'elle allait offrir aux sens blasés ou dépravés de certains spectateurs, s'était au contraire attaché à ces traits d'ambiguïté comme à un élément esthétique qui valait d'être mis en relief et qu'il avait tout fait pour souligner. Le long des lignes du visage, le sexe avait l'air d'être sur le point d'avouer qu'il était celui d'une fille un peu garçonnière, s'évanouissait, et plus loin se retrouvait, suggérant plutôt l'idée d'un jeune efféminé vicieux et songeur, puis fuyait encore, restait insaisissable. Le caractère de tristesse rêveuse du regard, par son contraste même avec les accessoires appartenant au monde de la noce et du théâtre, n'était pas ce qui était le moins troublant. On pensait du reste qu'il devait être factice et que le jeune être qui semblait s'offrir aux caresses dans ce provocant costume avait probablement trouvé piquant d'y ajouter l'expression romanesque d'un sentiment secret, d'un chagrin inavoué. Au bas du portrait était écrit : Miss Sacripant , octobre 1872. Je ne pus contenir mon admiration. « Oh ! ce n'est rien, c'est une pochade de jeunesse, c'était un costume pour une revue des Variétés. Tout cela est bien loin. – Et qu'est devenu le modèle ? » Un étonnement provoqué par mes paroles précéda sur la figure d'Elstir l'air indifférent et distrait qu'au bout d'une seconde il y étendit. « Tenez, passez-moi vite cette toile, me dit-il, j'entends Madame Elstir qui arrive et bien que la jeune personne au melon n'ait joué, je vous assure, aucun rôle dans ma vie, il est inutile que ma femme ait cette aquarelle sous les yeux. Je n'ai gardé cela que comme un document amusant sur le théâtre de cette époque. » Et avant de cacher l'aquarelle derrière lui, Elstir qui peut-être ne l'avait pas vue depuis longtemps y attacha un regard attentif. « Il faudra que je ne garde que la tête, murmura-t-il, le bas est vraiment trop mal peint, les mains sont d'un commençant. » J'étais désolé de l'arrivée de Mme Elstir qui allait encore nous retarder.

Le rebord de la fenêtre fut bientôt rose. Notre sortie serait en pure perte. Il n'y avait aucune chance de voir les jeunes filles, par conséquent plus aucune importance à ce que Mme Elstir nous quittât plus ou moins vite. Elle ne resta, d'ailleurs, pas très longtemps. Je la trouvai très ennuyeuse ; elle aurait pu être belle, si elle avait eu vingt ans, conduisant un boeuf dans la campagne romaine ; mais ses cheveux noirs blanchissaient ; et elle était commune sans être simple, parce qu'elle croyait que la solennité des manières et la majesté de l'attitude étaient requises par sa beauté sculpturale à laquelle, d'ailleurs, l'âge avait enlevé toutes ses séductions. Elle était mise avec la plus grande simplicité. Et on était touché mais surpris d'entendre Elstir dire à tout propos et avec une douceur respectueuse, comme si rien que prononcer ces mots lui causait de l'attendrissement et de la vénération : « Ma belle Gabrielle ! » Plus tard, quand je connus la peinture mythologique d'Elstir, Mme Elstir prit pour moi aussi de la beauté. Je compris qu'à certain type idéal résumé en certaines lignes, en certaines arabesques qui se retrouvaient sans cesse dans son oeuvre, à un certain canon, il avait attribué en fait un caractère presque divin, puisque tout son temps, tout l'effort de pensée dont il était capable, en un mot toute sa vie, il l'avait consacrée à la tâche de distinguer mieux ces lignes, de les reproduire plus fidèlement. Ce qu'un tel idéal inspirait à Elstir, c'était vraiment un culte si grave, si exigeant, qu'il ne lui permettait jamais d'être content, c'était la partie la plus intime de lui-même, aussi n'avait-il pu le considérer avec détachement, en tirer des émotions, jusqu'au jour où il le rencontra, réalisé au dehors, dans le corps d'une femme, le corps de celle qui était par la suite devenue Mme Elstir et chez qui il avait pu – comme cela ne nous est possible que pour ce qui n'est pas nous-même – le trouver méritoire, attendrissant, divin. Quel repos, d'ailleurs, de poser ses lèvres sur ce Beau que jusqu'ici il fallait avec tant de peine extraire de soi, et qui maintenant mystérieusement incarné, s'offrait à lui pour une suite de communions efficaces ! Elstir à cette époque n'était plus dans la première jeunesse où l'on attend que de la puissance de la pensée la réalisation de son idéal. Il approchait de l'âge où l'on compte sur les satisfactions du corps pour stimuler la force de l'esprit, où la fatigue de celui-ci, en nous inclinant au matérialisme, et la diminution de l'activité à la possibilité d'influences passivement reçues, commencent à nous faire admettre qu'il y a peut-être bien certains corps, certains métiers, certains rythmes privilégiés, réalisant si naturellement notre idéal, que même sans génie, rien qu'en copiant le mouvement d'une épaule, la tension d'un cou, nous ferions un chef-d'oeuvre ; c'est l'âge où nous aimons à caresser la Beauté du regard, hors de nous, près de nous, dans une tapisserie, dans une belle esquisse de Titien découverte chez un brocanteur, dans une maîtresse aussi belle que l'esquisse de Titien. Quand j'eus compris cela, je ne pus plus voir sans plaisir Mme Elstir, et son corps perdit de sa lourdeur, car je le remplis d'une idée, l'idée qu'elle était une créature immatérielle, un portrait d'Elstir. Elle en était un pour moi et pour lui aussi sans doute. Les données de la vie ne comptent pas pour l'artiste, elles ne sont pour lui qu'une occasion de mettre à nu son génie. On sent bien, à voir les uns à côté des autres dix portraits de personnes différentes peintes par Elstir, que ce sont avant tout des Elstir. Seulement, après cette marée montante du génie qui recouvre la vie, quand le cerveau se fatigue, peu à peu l'équilibre se rompt et comme un fleuve qui reprend son cours après le contreflux d'une grande marée, c'est la vie qui reprend le dessus. Or, pendant que durait la première période, l'artiste a, peu à peu, dégagé la loi, la formule de son inconscient. Il sait quelles situations s'il est romancier, quels paysages s'il est peintre, lui fournissent la matière, indifférente en soi, mais nécessaire à ses recherches comme serait un laboratoire ou un atelier. Il sait qu'il a fait ses chefs d'oeuvre avec des effets de lumière atténuée, avec des remords modifiant l'idée d'une faute, avec des femmes posées sous les arbres ou à demi plongées dans l'eau, comme des statues. Un jour viendra où, par l'usure de son cerveau, il n'aura plus, devant ces matériaux dont se servait son génie, la force de faire l'effort intellectuel qui seul peut produire son oeuvre, et continuera pourtant à les rechercher, heureux de se trouver près d'eux à cause du plaisir spirituel, amorce du travail, qu'ils éveillent en lui ; et les entourant d'ailleurs d'une sorte de superstition comme s'ils étaient supérieurs à autre chose, si en eux résidait déjà une bonne part de l'oeuvre d'art qu'ils porteraient en quelque sorte toute faite, il n'ira pas plus loin que la fréquentation, l'adoration des modèles. Il causera indéfiniment avec des criminels repentis, dont le remords, la régénération a fait l'objet de ses romans ; il achètera une maison de campagne dans un pays où la brume atténue la lumière ; il passera de longues heures à regarder des femmes se baigner ; il collectionnera les belles étoffes. Et ainsi la beauté de la vie, mot en quelque sorte dépourvu de signification, stade situé en deçà de l'art et auquel j'avais vu s'arrêter Swann, était celui où par ralentissement du génie créateur, idolâtrie des formes qui l'avaient favorisé, désir du moindre effort, devait un jour rétrograder peu à peu un Elstir.
