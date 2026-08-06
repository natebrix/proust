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
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "« ...d’un seul mot qui faisait pâlir Françoise... »; « Françoise attachait de plus en plus aux moindres paroles... une attention extraordinaire »; comparaison aux courtisans guettant l’humeur du Roi à Versailles.",
      "explanation": "The narrator presents Françoise in a subordinate and fearful position, forced to anticipate the slightest signs of mood, which symbolically places her in the rank of a dependent courtier."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "She is locally diminished by the dynamics of dependence and fear, monitoring and suffering the moods of authority."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-251-p-255"
}

### Candidate characters

[
  "Octave",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Nous causions un moment avec M. Vinteuil devant le porche en sortant de l'église. Il intervenait entre les gamins qui se chamaillaient sur la place, prenait la défense des petits, faisait des sermons aux grands. Si sa fille nous disait de sa grosse voix combien elle avait été contente de nous voir, aussitôt il semblait qu'en elle-même une soeur plus sensible rougissait de ce propos de bon garçon étourdi qui avait pu nous faire croire qu'elle sollicitait d'être invitée chez nous. Son père lui jetait un manteau sur les épaules, ils montaient dans un petit buggy qu'elle conduisait elle-même et tous deux retournaient à Montjouvain. Quant à nous, comme c'était le lendemain dimanche et qu'on ne se lèverait que pour la grand'messe, s'il faisait clair de lune et que l'air fût chaud, au lieu de nous faire rentrer directement, le père du narrateur, par amour de la gloire, nous faisait faire par le calvaire une longue promenade, que le peu d'aptitude de la mère du narrateur à s'orienter et à se reconnaître dans son chemin, lui faisait considérer comme la prouesse d'un génie stratégique. Parfois nous allions jusqu'au viaduc, dont les enjambées de pierre commençaient à la gare et me représentaient l'exil et la détresse hors du monde civilisé, parce que chaque année en venant de Paris, on nous recommandait de faire bien attention, quand ce serait Combray, de ne pas laisser passer la station, d'être prêts d'avance, car le train repartait au bout de deux minutes et s'engageait sur le viaduc au delà des pays chrétiens dont Combray marquait pour moi l'extrême limite. Nous revenions par le boulevard de la gare, où étaient les plus agréables villas de la commune. Dans chaque jardin le clair de lune, comme Hubert Robert de Saint-Loup, semait ses degrés rompus de marbre blanc, ses jets d'eau, ses grilles entr'ouvertes. Sa lumière avait détruit le bureau du télégraphe. Il n'en subsistait plus qu'une colonne à demi brisée, mais qui gardait la beauté d'une ruine immortelle. Je traînais la jambe, je tombais de sommeil, l'odeur des tilleuls qui embaumait m'apparaissait comme une récompense qu'on ne pouvait obtenir qu'au prix des plus grandes fatigues et qui n'en valait pas la peine. De grilles fort éloignées les unes des autres, des chiens réveillés par nos pas solitaires faisaient alterner des aboiements comme il m'arrive encore quelquefois d'en entendre le soir, et entre lesquels dut venir (quand sur son emplacement on créa le jardin public de Combray) se réfugier le boulevard de la gare, car, où que je me trouve, dès qu'ils commencent à retentir et à se répondre, je l'aperçois, avec ses tilleuls et son trottoir éclairé par la lune.

### Passage

Tout d'un coup mon père nous arrêtait et demandait à ma mère : « Où sommes-nous ? » Épuisée par la marche, mais fière de lui, elle lui avouait tendrement qu'elle n'en savait absolument rien. Il haussait les épaules et riait. Alors, comme s'il l'avait sortie de la poche de son veston avec sa clef, il nous montrait debout devant nous la petite porte de derrière de notre jardin qui était venue avec le coin de la rue du Saint-Esprit nous attendre au bout de ces chemins inconnus. Ma mère lui disait avec admiration : « Tu es extraordinaire ! » Et à partir de cet instant, je n'avais plus un seul pas à faire, le sol marchait pour moi dans ce jardin où depuis si longtemps mes actes avaient cessé d'être accompagnés d'attention volontaire : l'Habitude venait de me prendre dans ses bras et me portait jusqu'à mon lit comme un petit enfant.

Si la journée du samedi, qui commençait une heure plus tôt, et où elle était privée de Françoise, passait plus lentement qu'une autre pour ma tante, elle en attendait pourtant le retour avec impatience depuis le commencement de la semaine, comme contenant toute la nouveauté et la distraction que fût encore capable de supporter son corps affaibli et maniaque. Et ce n'est pas cependant qu'elle n'aspirât parfois à quelque plus grand changement, qu'elle n'eût de ces heures d'exception où l'on a soif de quelque chose d'autre que ce qui est, et où ceux que le manque d'énergie ou d'imagination empêche de tirer d'eux-mêmes un principe de rénovation demandent à la minute qui vient, au facteur qui sonne, de leur apporter du nouveau, fût-ce du pire, une émotion, une douleur ; où la sensibilité, que le bonheur a fait taire comme une harpe oisive, veut résonner sous une main, même brutale, et dût-elle en être brisée ; où la volonté, qui a si difficilement conquis le droit d'être livrée sans obstacle à ses désirs, à ses peines, voudrait jeter les rênes entre les mains d'événements impérieux, fussent-ils cruels. Sans doute, comme les forces de ma tante, taries à la moindre fatigue, ne lui revenaient que goutte à goutte au sein de son repos, le réservoir était très long à remplir, et il se passait des mois avant qu'elle eût ce léger trop-plein que d'autres dérivent dans l'activité et dont elle était incapable de savoir et de décider comment user. Je ne doute pas qu'alors – comme le désir de la remplacer par des pommes de terre béchamel finissait au bout de quelque temps par naître du plaisir même que lui causait le retour quotidien de la purée dont elle ne se « fatiguait » pas – elle ne tirât de l'accumulation de ces jours monotones auxquels elle tenait tant l'attente d'un cataclysme domestique, limité à la durée d'un moment, mais qui la forcerait d'accomplir une fois pour toutes un de ces changements dont elle reconnaissait qu'ils lui seraient salutaires et auxquels elle ne pouvait d'elle-même se décider. Elle nous aimait véritablement, elle aurait eu plaisir à nous pleurer ; survenant à un moment où elle se sentait bien et n'était pas en sueur, la nouvelle que la maison était la proie d'un incendie où nous avions déjà tous péri et qui n'allait plus bientôt laisser subsister une seule pierre des murs, mais auquel elle aurait eu tout le temps d'échapper sans se presser, à condition de se lever tout de suite, a dû souvent hanter ses espérances comme unissant aux avantages secondaires de lui faire savourer dans un long regret toute sa tendresse pour nous, et d'être la stupéfaction du village en conduisant notre deuil, courageuse et accablée, moribonde debout, celui bien plus précieux de la forcer au bon moment, sans temps à perdre, sans possibilité d'hésitation énervante, à aller passer l'été dans sa jolie ferme de Mirougrain, où il y avait une chute d'eau. Comme n'était jamais survenu aucun événement de ce genre, dont elle méditait certainement la réussite quand elle était seule absorbée dans ses innombrables jeux de patience (et qui l'eût désespérée au premier commencement de réalisation, au premier de ces petits faits imprévus, de cette parole annonçant une mauvaise nouvelle et dont on ne peut plus jamais oublier l'accent, de tout ce qui porte l'empreinte de la mort réelle, bien différente de sa possibilité logique et abstraite), elle se rabattait pour rendre de temps en temps sa vie plus intéressante, à y introduire des péripéties imaginaires qu'elle suivait avec passion. Elle se plaisait à supposer tout d'un coup que Françoise la volait, qu'elle recourait à la ruse pour s'en assurer, la prenait sur le fait ; habituée, quand elle faisait seule des parties de cartes, à jouer à la fois son jeu et le jeu de son adversaire, elle se prononçait à elle-même les excuses embarrassées de Françoise et y répondait avec tant de feu et d'indignation que l'un de nous, entrant à ces moments-là, la trouvait en nage, les yeux étincelants, ses faux cheveux déplacés laissant voir son front chauve. Françoise entendit peut-être parfois dans la chambre voisine de mordants sarcasmes qui s'adressaient à elle et dont l'invention n'eût pas soulagé suffisamment ma tante s'ils étaient restés à l'état purement immatériel, et si en les murmurant à mi-voix elle ne leur eût donné plus de réalité. Quelquefois, ce « spectacle dans un lit » ne suffisait même pas à ma tante, elle voulait faire jouer ses pièces. Alors, un dimanche, toutes portes mystérieusement fermées, elle confiait à Eulalie ses doutes sur la probité de Françoise, son intention de se défaire d'elle, et une autre fois, à Françoise ses soupçons de l'infidélité d'Eulalie, à qui la porte serait bientôt fermée ; quelques jours après elle était dégoûtée de sa confidente de la veille et racoquinée avec le traître, lesquels d'ailleurs, pour la prochaine représentation, échangeraient leurs emplois. Mais les soupçons que pouvait parfois lui inspirer Eulalie n'étaient qu'un feu de paille et tombaient vite, faute d'aliment, Eulalie n'habitant pas la maison. Il n'en était pas de même de ceux qui concernaient Françoise, que ma tante sentait perpétuellement sous le même toit qu'elle, sans que, par crainte de prendre froid si elle sortait de son lit, elle osât descendre à la cuisine se rendre compte s'ils étaient fondés. Peu à peu son esprit n'eut plus d'autre occupation que de chercher à deviner ce qu'à chaque moment pouvait faire, et chercher à lui cacher, Françoise. Elle remarquait les plus furtifs mouvements de physionomie de celle-ci, une contradiction dans ses paroles, un désir qu'elle semblait dissimuler. Et elle lui montrait qu'elle l'avait démasquée, d'un seul mot qui faisait pâlir Françoise et que ma tante semblait trouver, à enfoncer au coeur de la malheureuse, un divertissement cruel. Et le dimanche suivant, une révélation d'Eulalie – comme ces découvertes qui ouvrent tout d'un coup un champ insoupçonné à une science naissante et qui se traînait dans l'ornière – prouvait à ma tante qu'elle était dans ses suppositions bien au-dessous de la vérité. « Mais Françoise doit le savoir maintenant que vous y avez donné une voiture. » – « Que je lui ai donné une voiture ! » s'écriait ma tante. – « Ah ! mais je ne sais pas, moi, je croyais, je l'avais vue qui passait maintenant en calèche, fière comme Artaban, pour aller au marché de Roussainville. J'avais cru que c'était Octave qui lui avait donné. » Peu à peu Françoise et ma tante, comme la bête et le chasseur, ne cessaient plus de tâcher de prévenir les ruses l'une de l'autre. Ma mère craignait qu'il ne se développât chez Françoise une véritable haine pour ma tante qui l'offensait le plus durement qu'elle le pouvait. En tous cas Françoise attachait de plus en plus aux moindres paroles, aux moindres gestes de ma tante une attention extraordinaire. Quand elle avait quelque chose à lui demander, elle hésitait longtemps sur la manière dont elle devait s'y prendre. Et quand elle avait proféré sa requête, elle observait ma tante à la dérobée, tâchant de deviner dans l'aspect de sa figure ce que celle-ci avait pensé et déciderait. Et ainsi – tandis que quelque artiste lisant les Mémoires du XVIIe siècle, et désirant de se rapprocher du grand Roi, croit marcher dans cette voie en se fabriquant une généalogie qui le fait descendre d'une famille historique ou en entretenant une correspondance avec un des souverains actuels de l'Europe, tourne précisément le dos à ce qu'il a le tort de chercher sous des formes identiques et par conséquent mortes – une vieille dame de province qui ne faisait qu'obéir sincèrement à d'irrésistibles manies et à une méchanceté née de l'oisiveté, voyait sans avoir jamais pensé à Louis XIV les occupations les plus insignifiantes de sa journée, concernant son lever, son déjeuner, son repos, prendre par leur singularité despotique un peu de l'intérêt de ce que Saint-Simon appelait la « mécanique » de la vie à Versailles, et pouvait croire aussi que ses silences, une nuance de bonne humeur ou de hauteur dans sa physionomie, étaient de la part de Françoise l'objet d'un commentaire aussi passionné, aussi craintif que l'étaient le silence, la bonne humeur, la hauteur du Roi quand un courtisan, ou même les plus grands seigneurs, lui avaient remis une supplique, au détour d'une allée, à Versailles.

Un dimanche, où ma tante avait eu la visite simultanée du curé et d'Eulalie, et s'était ensuite reposée, nous étions tous montés lui dire bonsoir, et maman lui adressait ses condoléances sur la mauvaise chance qui amenait toujours ses visiteurs à la même heure :

– Je sais que les choses se sont encore mal arrangées tantôt, Léonie, lui dit-elle avec douceur, vous avez eu tout votre monde à la fois.

Ce que ma grand'tante interrompit par : « Abondance de biens... » car depuis que sa fille était malade elle croyait devoir la remonter en lui présentant toujours tout par le bon côté. Mais mon père prenant la parole :
