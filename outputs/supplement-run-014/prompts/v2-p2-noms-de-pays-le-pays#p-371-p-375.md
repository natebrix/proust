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
        "Elstir",
        "le peintre"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Elstir",
      "type": "prestige_association",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "Sous les arbres [Albertine] adress[a] à Elstir un salut souriant d’amie… Elle s’approcha même pour tendre la main au le peintre, sans s’arrêter… arc‑en‑ciel qui unit pour moi notre monde terraqué à des régions… inaccessibles. — « Vous connaissez cette jeune fille, monsieur ? » dis‑je à Elstir, comprenant qu’il pourrait me présenter à elle.",
      "explanation": "Albertine’s friendly recognition of Elstir and the narrator’s immediate appeal for an introduction confer social and desirability prestige on Elstir, positioning him as a gateway to a coveted circle."
    }
  ],
  "status_effects": [
    {
      "character": "Elstir",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Being publicly saluted by Albertine and seen as able to introduce her raises Elstir’s local standing in the narrator’s eyes through association with the desired young women."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-371-p-375"
}

### Candidate characters

[
  "Albertine",
  "Legrandin",
  "la grand-mère",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Cette vaste vision céleste dont il me parlait, ce gigantesque poème théologique que je comprenais avoir été écrit là, pourtant quand mes yeux pleins de désirs s'étaient ouverts devant la façade, ce n'est pas eux que j'avais vus. Je lui parlais de ces grandes statues de saints qui montées sur des échasses forment une sorte d'avenue.

### Passage

– Elle part des fonds des âges pour aboutir à Jésus-Christ, me dit-il. Ce sont d'un côté ses ancêtres selon l'esprit, de l'autre, les Rois de Juda, ses ancêtres selon la chair. Tous les siècles sont là. Et si vous aviez mieux regardé ce qui vous a paru des échasses, vous auriez pu nommer ceux qui y étaient perchés. Car sous les pieds de Moïse, vous auriez reconnu le veau d'or, sous les pieds d'Abraham le bélier, sous ceux de Joseph le démon conseillant la femme de Putiphar.

Je lui dis aussi que je m'étais attendu à trouver un monument presque persan et que ç'avait sans doute été là une des causes de mon mécompte. « Mais non, me répondit-il, il y a beaucoup de vrai. Certaines parties sont tout orientales ; un chapiteau reproduit si exactement un sujet persan que la persistance des traditions orientales ne suffit pas à l'expliquer. Le sculpteur a dû copier quelque coffret apporté par des navigateurs. » Et en effet il devait me montrer plus tard la photographie d'un chapiteau où je vis des dragons quasi chinois qui se dévoraient, mais à Balbec ce petit morceau de sculpture avait passé pour moi inaperçu dans l'ensemble du monument qui ne ressemblait pas à ce que m'avaient montré ces mots : « église presque persane ».

Les joies intellectuelles que je goûtais dans cet atelier ne m'empêchaient nullement de sentir, quoiqu'ils nous entourassent comme malgré nous, les tièdes glacis, la pénombre étincelante de la pièce, et au bout de la petite fenêtre encadrée de chèvrefeuilles, dans l'avenue toute rustique, la résistante sécheresse de la terre brûlée de soleil que voilait seulement la transparence de l'éloignement et de l'ombre des arbres. Peut-être l'inconscient bien-être que me causait ce jour d'été venait-il agrandir comme un affluent la joie que me causait la vue du « Port de Carquethuit ».

J'avais cru Elstir modeste, mais je compris que je m'étais trompé, en voyant son visage se nuancer de tristesse quand dans une phrase de remerciements je prononçai le mot de gloire. Ceux qui croient leurs oeuvres durables – et c'était le cas pour Elstir – prennent l'habitude de les situer dans une époque où eux-mêmes ne seront plus que poussière. Et ainsi en les forçant à réfléchir au néant, l'idée de la gloire les attriste parce qu'elle est inséparable de l'idée de la mort. Je changeai de conversation pour dissiper ce nuage d'orgueilleuse mélancolie dont j'avais sans le vouloir chargé le front d'Elstir. « On m'avait conseillé, lui dis-je en pensant à la conversation que nous avions eue avec Legrandin à Combray et sur laquelle j'étais content d'avoir son avis, de ne pas aller en Bretagne, parce que c'était malsain pour un esprit déjà porté au rêve. – Mais non, me répondit-il, quand un esprit est porté au rêve, il ne faut pas l'en tenir écarté, le lui rationner. Tant que vous détournerez votre esprit de ses rêves, il ne les connaîtra pas ; vous serez le jouet de mille apparences parce que vous n'en aurez pas compris la nature. Si un peu de rêve est dangereux, ce qui en guérit, ce n'est pas moins de rêve, mais plus de rêve, mais tout le rêve. Il importe qu'on connaisse entièrement ses rêves pour n'en plus souffrir ; il y a une certaine séparation du rêve et de la vie qu'il est si souvent utile de faire que je me demande si on ne devrait pas à tout hasard la pratiquer préventivement, comme certains chirurgiens prétendent qu'il faudrait, pour éviter la possibilité d'une appendicite future, enlever l'appendice chez tous les enfants. »

Elstir et moi nous étions allés jusqu'au fond de l'atelier, devant la fenêtre qui donnait derrière le jardin sur une étroite avenue de traverse, presque un petit chemin rustique. Nous étions venus là pour respirer l'air rafraîchi de l'après-midi avancé. Je me croyais bien loin des jeunes filles de la petite bande, et c'est en sacrifiant pour une fois l'espérance de les voir, que j'avais fini par obéir à la prière de ma grand-mère et aller voir Elstir. Car où se trouve ce qu'on cherche on ne le sait pas, et on fuit souvent pendant bien longtemps le lieu où, pour d'autres raisons, chacun nous invite. Mais nous ne soupçonnons pas que nous y verrions justement l'être auquel nous pensons. Je regardais vaguement le chemin campagnard qui, extérieur à l'atelier, passait tout près de lui mais n'appartenait pas à Elstir. Tout à coup y apparut, le suivant à pas rapides, la jeune cycliste de la petite bande avec, sur ses cheveux noirs, son polo abaissé vers ses grosses joues, ses yeux gais et un peu insistants ; et dans ce sentier fortuné miraculeusement rempli de douces promesses, je la vis sous les arbres adresser à Elstir un salut souriant d'amie, arc-en-ciel qui unit pour moi notre monde terraqué à des régions que j'avais jugées jusque-là inaccessibles. Elle s'approcha même pour tendre la main au peintre, sans s'arrêter, et je vis qu'elle avait un petit grain de beauté au menton. « Vous connaissez cette jeune fille, monsieur ? » dis-je à Elstir, comprenant qu'il pourrait me présenter à elle, l'inviter chez lui. Et cet atelier paisible avec son horizon rural s'était rempli d'un surcroît délicieux, comme il arrive d'une maison où un enfant se plaisait déjà et où il apprend que, en plus, de par la générosité qu'ont les belles choses et les nobles gens à accroître indéfiniment leurs dons, se prépare pour lui un magnifique goûter. Elstir me dit qu'elle s'appelait Albertine Simonet et me nomma aussi ses autres amies que je lui décrivis avec assez d'exactitude pour qu'il n'eût guère d'hésitation. J'avais commis à l'égard de leur situation sociale une erreur, mais pas dans le même sens que d'habitude à Balbec. J'y prenais facilement pour des princes des fils de boutiquiers montant à cheval. Cette fois j'avais situé dans un milieu interlope des filles d'une petite bourgeoisie fort riche, du monde de l'industrie et des affaires. C'était celui qui de prime abord m'intéressait le moins, n'ayant pour moi le mystère ni du peuple, ni d'une société comme celle des Guermantes. Et sans doute si un prestige préalable qu'elles ne perdraient plus ne leur avait été conféré, devant mes yeux éblouis, par la vacuité éclatante de la vie de plage, je ne serais peut-être pas arrivé à lutter victorieusement contre l'idée qu'elles étaient les filles de gros négociants. Je ne pus qu'admirer combien la bourgeoisie française était un atelier merveilleux de sculpture la plus généreuse et la plus variée. Que de types imprévus, quelle invention dans le caractère des visages, quelle décision, quelle fraîcheur, quelle naïveté dans les traits ! Les vieux bourgeois avares d'où étaient issues ces Dianes et ces nymphes me semblaient les plus grands des statuaires. Avant que j'eusse eu le temps de m'apercevoir de la métamorphose sociale de ces jeunes filles, et tant ces découvertes d'une erreur, ces modifications de la notion qu'on a d'une personne ont l'instantanéité d'une réaction chimique, s'était déjà installée derrière le visage d'un genre si voyou de ces jeunes filles que j'avais prises pour des maîtresses de coureurs cyclistes, de champions de boxe, l'idée qu'elles pouvaient très bien être liées avec la famille de tel notaire que nous connaissions. Je ne savais guère ce qu'était Albertine Simonet. Elle ignorait certes ce qu'elle devait être un jour pour moi. Même ce nom de Simonet que j'avais déjà entendu sur la plage, si on m'avait demandé de l'écrire je l'aurais orthographié avec deux n. ne me doutant pas de l'importance que cette famille attachait à n'en posséder qu'un seul. Au fur et à mesure que l'on descend dans l'échelle sociale, le snobisme s'accroche à des riens qui ne sont peut-être pas plus nuls que les distinctions de l'aristocratie, mais qui plus obscurs, plus particuliers à chacun, surprennent davantage. Peut-être y avait-il eu des Simonet qui avaient fait de mauvaises affaires ou pis encore. Toujours est-il que les Simonet s'étaient, paraît-il, toujours irrités comme d'une calomnie quand on doublait leur n. Ils avaient l'air d'être les seuls Simonet avec un n au lieu de deux, avec autant de fierté peut-être que les Montmorency d'être les premiers barons de France. Je demandai à Elstir si ces jeunes filles habitaient Balbec, il me répondit oui pour certaines d'entre elles. La villa de l'une était précisément située tout au bout de la plage, là où commencent les falaises du Canapville. Comme cette jeune fille était une grande amie d'Albertine Simonet, ce me fut une raison de plus de croire que c'était bien cette dernière que j'avais rencontrée, quand j'étais avec ma grand-mère. Certes il y avait tant de ces petites rues perpendiculaires à la plage où elles faisaient un angle pareil, que je n'aurais pu spécifier exactement laquelle c'était. On voudrait avoir un souvenir exact mais au moment même la vision a été trouble. Pourtant qu'Albertine et cette jeune fille entrant chez son amie fussent une seule et même personne, c'était pratiquement une certitude. Malgré cela, tandis que les innombrables images que m'a présentées dans la suite la brune joueuse de golf, si différentes qu'elles soient les unes des autres, se superposent (parce que je sais qu'elles lui appartiennent toutes), et que si je remonte le fil de mes souvenirs, je peux, sous le couvert de cette identité et comme dans un chemin de communication intérieure, repasser par toutes ces images sans sortir d'une même personne, en revanche, si je veux remonter jusqu'à la jeune fille que je croisai le jour où j'étais avec ma grand-mère, il me faut ressortir à l'air libre. Je suis persuadé que c'est Albertine que je retrouve, la même que celle qui s'arrêtait souvent, au milieu de ses amies, dans sa promenade, dépassant l'horizon de la mer ; mais toutes ces images restent séparées de cette autre parce que je ne peux pas lui conférer rétrospectivement une identité qu'elle n'avait pas pour moi au moment où elle a frappé mes yeux ; quoi que puisse m'assurer le calcul des probabilités, cette jeune fille aux grosses joues qui me regarda si hardiment au coin de la petite rue et de la plage et par qui je crois que j'aurais pu être aimé, au sens strict du mot revoir, je ne l'ai jamais revue.
