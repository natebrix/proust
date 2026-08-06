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
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "son père",
        "petit papa"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Swann",
      "target": "Gilberte",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.95,
      "evidence": "« Tu es une bonne fille » (dit avec tendresse pendant qu’elle se blottit contre lui).",
      "explanation": "Swann addresses his daughter with an affectionate praise that positively confirms her in the immediate family space."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "emotional_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "Swann's praise and tenderness reinforce her emotional position and her inclusion with her father."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-226-p-230"
}

### Candidate characters

[
  "Odette",
  "la Berma",
  "le peintre",
  "le narrateur"
]

### Prior local context (optional)

Je revins à Norpois. « Ne vous y fiez pas, il est au contraire très mauvaise langue », dit Odette avec un accent qui me parut d'autant plus signifier que Norpois avait mal parlé d'elle, que Swann regarda sa femme d'un air de réprimande et comme pour l'empêcher d'en dire davantage.

### Passage

Cependant Gilberte qu'on avait déjà priée deux fois d'aller se préparer pour sortir, restait à nous écouter, entre sa mère et son père, à l'épaule duquel elle était câlinement appuyée. Rien, au premier aspect, ne faisait plus contraste avec Odette qui était brune que cette jeune fille à la chevelure rousse, à la peau dorée. Mais au bout d'un instant on reconnaissait en Gilberte bien des traits – par exemple le nez arrêté avec une brusque et infaillible décision par le sculpteur invisible qui travaille de son ciseau pour plusieurs générations – l'expression, les mouvements de sa mère ; pour prendre une comparaison dans un autre art, elle avait l'air d'un portrait peu ressemblant encore de Odette que le peintre, par un caprice de coloriste, eût fait poser à demi-déguisée, prête à se rendre à un dîner de « têtes », en Vénitienne. Et comme elle n'avait pas qu'une perruque blonde, mais que tout atome sombre avait été expulsé de sa chair laquelle dévêtue de ses voiles bruns semblait plus nue, recouverte seulement des rayons dégagés par un soleil intérieur, le grimage n'était pas que superficiel, mais incarné ; Gilberte avait l'air de figurer quelque animal fabuleux, ou de porter un travesti mythologique. Cette peau rousse c'était celle de son père au point que la nature semblait avoir eu, quand Gilberte avait été créée, à résoudre le problème de refaire peu à peu Odette, en n'ayant à sa disposition comme matière que la peau de Swann. Et la nature l'avait utilisée parfaitement, comme un maître huchier qui tient à laisser apparents le grain, les noeuds du bois. Dans la figure de Gilberte, au coin du nez d'Odette parfaitement reproduit, la peau se soulevait pour garder intacts les deux grains de beauté de Swann. C'était une nouvelle variété de Odette qui était obtenue là, à côté d'elle, comme un lilas blanc près d'un lilas violet. Il ne faudrait pourtant pas se représenter la ligne de démarcation entre les deux ressemblances comme absolument nette. Par moments, quand Gilberte riait, on distinguait l'ovale de la joue de son père dans la figure de sa mère comme si on les avait mis ensemble pour voir ce que donnerait le mélange ; cet ovale se précisait comme un embryon se forme, il s'allongeait obliquement, se gonflait, au bout d'un instant il avait disparu. Dans les yeux de Gilberte il y avait le bon regard franc de son père ; c'est celui qu'elle avait eu quand elle m'avait donné la bille d'agate et m'avait dit : « Gardez-la en souvenir de notre amitié. » Mais, posait-on à Gilberte une question sur ce qu'elle avait fait, alors on voyait dans ces mêmes yeux l'embarras, l'incertitude, la dissimulation, la tristesse qu'avait autrefois Odette quand Swann lui demandait où elle était allée, et qu'elle lui faisait une de ces réponses mensongères qui désespéraient l'amant et maintenant lui faisaient brusquement changer la conversation en mari incurieux et prudent. Souvent, aux Champs-Élysées, j'étais inquiet en voyant ce regard chez Gilberte. Mais, la plupart du temps, c'était à tort. Car chez elle, survivance toute physique de sa mère, ce regard – celui-là du moins – ne correspondait plus à rien. C'est quand elle était allée à son cours, quand elle devait rentrer pour une leçon que les pupilles de Gilberte exécutaient ce mouvement qui jadis en les yeux d'Odette était causé par la peur de révéler qu'elle avait reçu dans la journée un de ses amants ou qu'elle était pressée de se rendre à un rendez-vous. Telles on voyait ces deux natures de M. et de Odette onduler, refluer, empiéter tour à tour l'une sur l'autre, dans le corps de cette Mélusine.

Sans doute on sait bien qu'un enfant tient de son père et de sa mère. Encore la distribution des qualités et des défauts dont il hérite se fait-elle si étrangement que, de deux qualités qui semblaient inséparables chez l'un des parents, on ne trouve plus que l'une chez l'enfant, et alliée à celui des défauts de l'autre parent qui semblait inconciliable avec elle. Même l'incarnation d'une qualité morale dans un défaut physique incompatible est souvent une des lois de la ressemblance filiale. De deux soeurs, l'une aura, avec la fière stature de son père, l'esprit mesquin de sa mère ; l'autre, toute remplie de l'intelligence paternelle, la présentera au monde sous l'aspect qu'a sa mère ; le gros nez, le ventre noueux, et jusqu'à la voix sont devenus les vêtements de dons qu'on connaissait sous une apparence superbe. De sorte que de chacune des deux soeurs on peut dire avec autant de raison que c'est elle qui tient le plus de tel de ses parents. Il est vrai que Gilberte était fille unique, mais il y avait au moins deux Gilberte. Les deux natures, de son père et de sa mère, ne faisaient pas que se mêler en elle ; elles se la disputaient, et encore ce serait parler inexactement et donnerait à supposer qu'une troisième Gilberte souffrait pendant ce temps-là d'être la proie des deux autres. Or, Gilberte était tour à tour l'une et puis l'autre, et à chaque moment rien de plus que l'une, c'est-à-dire incapable, quand elle était moins bonne, d'en souffrir, la meilleure Gilberte ne pouvant alors, du fait de son absence momentanée, constater cette déchéance. Aussi la moins bonne des deux était-elle libre de se réjouir de plaisirs peu nobles. Quand l'autre parlait avec le coeur de son père, elle avait des vues larges, on aurait voulu conduire avec elle une belle et bienfaisante entreprise, on le lui disait, mais au moment où l'on allait conclure, le coeur de sa mère avait déjà repris son tour ; et c'est lui qui vous répondait ; et on était déçu et irrité – presque intrigué comme devant une substitution de personne – par une réflexion mesquine, un ricanement fourbe, où Gilberte se complaisait, car ils sortaient de ce qu'elle-même était à ce moment-là. L'écart était même parfois tellement grand entre les deux Gilberte qu'on se demandait, vainement du reste, ce qu'on avait pu lui faire pour la retrouver si différente. Le rendez-vous qu'elle vous avait proposé, non seulement elle n'y était pas venue et ne s'excusait pas ensuite, mais quelle que fût l'influence qui eût pu faire changer sa détermination, elle se montrait si différente ensuite, qu'on aurait cru que, victime d'une ressemblance comme celle qui fait le fond des Ménechmes, on n'était pas devant la personne qui vous avait si gentiment demandé à vous voir, si elle ne nous eût témoigné une mauvaise humeur qui décelait qu'elle se sentait en faute et désirait éviter les explications.

– Allons, va, tu vas nous faire attendre, lui dit sa mère.

– Je suis si bien près de mon petit papa, je veux rester encore un moment, répondit Gilberte en cachant sa tête sous le bras de son père qui passa tendrement les doigts dans la chevelure blonde.

Swann était un de ces hommes qui, ayant vécu longtemps dans les illusions de l'amour, ont vu le bien-être qu'ils ont donné à nombre de femmes accroître le bonheur de celles-ci sans créer de leur part aucune reconnaissance, aucune tendresse envers eux ; mais dans leur enfant ils croient sentir une affection qui, incarnée dans leur nom même, les fera durer après leur mort. Quand il n'y aurait plus de Swann Swann, il y aurait encore une Gilberte, ou une Mme X., née Swann, qui continuerait à aimer le père disparu. Même à l'aimer trop peut-être, pensait sans doute Swann, car il répondit à Gilberte : « Tu es une bonne fille » de ce ton attendri par l'inquiétude que nous inspire, pour l'avenir, la tendresse trop passionnée d'un être destiné à nous survivre. Pour dissimuler son émotion, il se mêla à notre conversation sur la Berma. Il me fit remarquer, mais d'un ton détaché, ennuyé, comme s'il voulait rester en quelque sorte en dehors de ce qu'il disait, avec quelle intelligence, quelle justesse imprévue l'actrice disait à Œnone : « Tu le savais ! » Il avait raison : cette intonation-là du moins, avait une valeur vraiment intelligible et aurait pu par là satisfaire à mon désir de trouver des raisons irréfutables d'admirer la Berma. Mais c'est à cause de sa clarté même qu'elle ne le contentait point. L'intonation était si ingénieuse, d'une intention, d'un sens si définis, qu'elle semblait exister en elle-même et que toute artiste intelligente eût pu l'acquérir. C'était une belle idée ; mais quiconque la concevrait aussi pleinement la posséderait de même. Il restait à la Berma qu'elle l'avait trouvée, mais peut-on employer ce mot de « trouver » quand il s'agit de quelque chose qui ne serait pas différent si on l'avait reçu, quelque chose qui ne tient pas essentiellement à votre être, puisqu'un autre peut ensuite le reproduire ?
