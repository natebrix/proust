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
        "il",
        "lui"
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
      "evidence": "« Il lui sourit avec la lâcheté soudaine de l'être sans forces qu'avaient fait de lui ces accablantes paroles. » … « ébranlant pierre à pierre tout son passé. »",
      "explanation": "Odette’s blunt revelation about having been with Forcheville (and the implied pattern of past lies) is narrated as shattering Swann’s confidence and corroding his happiest memories, leaving him weakened and consumed by suspicion."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Swann suffers a sharp local collapse—his past is reinterpreted as deceitful, his jealousy multiplies, and he is depicted as powerless and distressed."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-571-p-575"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Cottard",
  "Mme Verdurin",
  "Odette",
  "comte de Forcheville",
  "docteur Cottard",
  "duchesse de Guermantes",
  "le narrateur",
  "le peintre",
  "marquise de Saint-Euverte"
]

### Prior local context (optional)

D'ailleurs ses aveux même, quand elle lui en faisait, de fautes qu'elle le supposait avoir découvertes, servaient plutôt pour Swann de point de départ à de nouveaux doutes qu'ils ne mettaient un terme aux anciens. Car ils n'étaient jamais exactement proportionnés à ceux-ci. Odette avait eu beau retrancher de sa confession tout l'essentiel, il restait dans l'accessoire quelque chose que Swann n'avait jamais imaginé, qui l'accablait de sa nouveauté et allait lui permettre de changer les termes du problème de sa jalousie. Et ces aveux il ne pouvait plus les oublier. Son âme les charriait, les rejetait, les berçait, comme des cadavres. Et elle en était empoisonnée.

### Passage

Une fois elle lui parla d'une visite que Forcheville lui avait faite le jour de la Fête de Paris-Murcie. « Comment, tu le connaissais déjà ? Ah ! oui, c'est vrai », dit-il en se reprenant pour ne pas paraître l'avoir ignoré. Et tout d'un coup il se mit à trembler à la pensée que le jour de cette fête de Paris-Murcie où il avait reçu d'elle la lettre qu'il avait si précieusement gardée, elle déjeunait peut-être avec Forcheville à la Maison d'Or. Elle lui jura que non. « Pourtant la Maison d'Or me rappelle je ne sais quoi que j'ai su ne pas être vrai », lui dit-il pour l'effrayer. – « Oui, que je n'y étais pas allée le soir où je t'ai dit que j'en sortais quand tu m'avais cherchée chez Prévost », lui répondit-elle (croyant à son air qu'il le savait), avec une décision où il y avait, beaucoup plus que du cynisme, de la timidité, une peur de contrarier Swann et que par amour-propre elle voulait cacher, puis le désir de lui montrer qu'elle pouvait être franche. Aussi frappa-t-elle avec une netteté et une vigueur de bourreau et qui étaient exemptes de cruauté, car Odette n'avait pas conscience du mal qu'elle faisait à Swann ; et même elle se mit à rire, peut-être il est vrai, surtout pour ne pas avoir l'air humilié, confus. « C'est vrai que je n'avais pas été à la Maison Dorée, que je sortais de chez Forcheville. J'avais vraiment été chez Prévost, ça c'était pas de la blague, il m'y avait rencontrée et m'avait demandé d'entrer regarder ses gravures. Mais il était venu quelqu'un pour le voir. Je t'ai dit que je venais de la Maison d'Or parce que j'avais peur que cela ne t'ennuie. Tu vois, c'était plutôt gentil de ma part. Mettons que j'aie eu tort, au moins je te le dis carrément. Quel intérêt aurais-je à ne pas te dire aussi bien que j'avais déjeuné avec lui le jour de la Fête Paris-Murcie, si c'était vrai ? D'autant plus qu'à ce moment-là on ne se connaissait pas encore beaucoup tous les deux, dis, chéri. » Il lui sourit avec la lâcheté soudaine de l'être sans forces qu'avaient fait de lui ces accablantes paroles. Ainsi, même dans les mois auxquels il n'avait jamais plus osé repenser parce qu'ils avaient été trop heureux, dans ces mois où elle l'avait aimé, elle lui mentait déjà ! Aussi bien que ce moment (le premier soir qu'ils avaient « fait catleya ») où elle lui avait dit sortir de la Maison Dorée, combien devait-il y en avoir eu d'autres, receleurs eux aussi d'un mensonge que Swann n'avait pas soupçonné. Il se rappela qu'elle lui avait dit un jour : « Je n'aurais qu'à dire à Mme Verdurin que ma robe n'a pas été prête, que mon cab est venu en retard. Il y a toujours moyen de s'arranger. » À lui aussi probablement bien des fois où elle lui avait glissé de ces mots qui expliquent un retard, justifient un changement d'heure dans un rendez-vous, ils avaient dû cacher, sans qu'il s'en fût douté alors, quelque chose qu'elle avait à faire avec un autre à qui elle avait dit : « Je n'aurai qu'à dire à Swann que ma robe n'a pas été prête, que mon cab est arrivé en retard, il y a toujours moyen de s'arranger. » Et sous tous les souvenirs les plus doux de Swann, sous les paroles les plus simples que lui avait dites autrefois Odette, qu'il avait crues comme paroles d'évangile, sous les actions quotidiennes qu'elle lui avait racontées, sous les lieux les plus accoutumés, la maison de sa couturière, l'avenue du Bois, l'Hippodrome, il sentait (dissimulée à la faveur de cet excédent de temps qui dans les journées les plus détaillées laisse encore du jeu, de la place, et peut servir de cachette à certaines actions), il sentait s'insinuer la présence possible et souterraine de mensonges qui lui rendaient ignoble tout ce qui lui était resté le plus cher, ses meilleurs soirs, la rue La Pérouse elle-même, qu'Odette avait toujours dû quitter à d'autres heures que celles qu'elle lui avait dites, faisant circuler partout un peu de la ténébreuse horreur qu'il avait ressentie en entendant l'aveu relatif à la Maison Dorée, et, comme les bêtes immondes dans la Désolation de Ninive, ébranlant pierre à pierre tout son passé. Si maintenant il se détournait chaque fois que sa mémoire lui disait le nom cruel de la Maison Dorée, ce n'était plus, comme tout récemment encore à la soirée de Mme de Saint-Euverte, parce qu'il lui rappelait un bonheur qu'il avait perdu depuis longtemps, mais un malheur qu'il venait seulement d'apprendre. Puis il en fut du nom de la Maison Dorée comme de celui de l'île du Bois, il cessa peu à peu de faire souffrir Swann. Car ce que nous croyons notre amour, notre jalousie, n'est pas une même passion continue, indivisible. Ils se composent d'une infinité d'amours successifs, de jalousies différentes et qui sont éphémères, mais par leur multitude ininterrompue donnent l'impression de la continuité, l'illusion de l'unité. La vie de l'amour de Swann, la fidélité de sa jalousie, étaient faites de la mort, de l'infidélité, d'innombrables désirs, d'innombrables doutes, qui avaient tous Odette pour objet. S'il était resté longtemps sans la voir, ceux qui mouraient n'auraient pas été remplacés par d'autres. Mais la présence d'Odette continuait d'ensemencer le coeur de Swann de tendresse et de soupçons alternés.

Certains soirs elle redevenait tout d'un coup avec lui d'une gentillesse dont elle l'avertissait durement qu'il devait profiter tout de suite, sous peine de ne pas la voir se renouveler avant des années ; il fallait rentrer immédiatement chez elle « faire catleya » et ce désir qu'elle prétendait avoir de lui était si soudain, si inexplicable, si impérieux, les caresses qu'elle lui prodiguait ensuite si démonstratives et si insolites, que cette tendresse brutale et sans vraisemblance faisait autant de chagrin à Swann qu'un mensonge et qu'une méchanceté. Un soir qu'il était ainsi, sur l'ordre qu'elle lui en avait donné, rentré avec elle, et qu'elle entremêlait ses baisers de paroles passionnées qui contrastaient avec sa sécheresse ordinaire, il crut tout d'un coup entendre du bruit ; il se leva, chercha partout, ne trouva personne, mais n'eut pas le courage de reprendre sa place auprès d'elle qui alors, au comble de la rage, brisa un vase et dit à Swann : « On ne peut jamais rien faire avec toi ! » Et il resta incertain si elle n'avait pas caché quelqu'un dont elle avait voulu faire souffrir la jalousie ou allumer les sens.

Quelquefois il allait dans des maisons de rendez-vous, espérant apprendre quelque chose d'elle, sans oser la nommer cependant. « J'ai une petite qui va vous plaire », disait l'entremetteuse. » Et il restait une heure à causer tristement avec quelque pauvre fille étonnée qu'il ne fît rien de plus. Une toute jeune et ravissante lui dit un jour : « Ce que je voudrais, c'est trouver un ami, alors il pourrait être sûr, je n'irais plus jamais avec personne. » – « Vraiment, crois-tu que ce soit possible qu'une femme soit touchée qu'on l'aime, ne vous trompe jamais ? » lui demanda Swann anxieusement. – « Pour sûr ! ça dépend des caractères ! » Swann ne pouvait s'empêcher de dire à ces filles les mêmes choses qui auraient plu à la princesse des Laumes. À celle qui cherchait un ami, il dit en souriant : « C'est gentil, tu as mis des yeux bleus de la couleur de ta ceinture. » – « Vous aussi, vous avez des manchettes bleues. » – « Comme nous avons une belle conversation, pour un endroit de ce genre ! Je ne t'ennuie pas, tu as peut-être à faire ? » – « Non, j'ai tout mon temps. Si vous m'aviez ennuyée, je vous l'aurais dit. Au contraire j'aime bien vous entendre causer. » – « Je suis très flatté. N'est-ce pas que nous causons gentiment ? » dit-il à l'entremetteuse qui venait d'entrer. – « Mais oui, c'est justement ce que je me disais. Comme ils sont sages ! Voilà ! on vient maintenant pour causer chez moi. Le Prince le disait, l'autre jour, c'est bien mieux ici que chez sa femme. Il paraît que maintenant dans le monde elles ont toutes un genre, c'est un vrai scandale ! Je vous quitte, je suis discrète. » Et elle laissa Swann avec la fille qui avait les yeux bleus. Mais bientôt il se leva et lui dit adieu, elle lui était indifférente, elle ne connaissait pas Odette.

Le peintre ayant été malade, le docteur Cottard lui conseilla un voyage en mer ; plusieurs fidèles parlèrent de partir avec lui ; les Verdurin ne purent se résoudre à rester seuls, louèrent un yacht, puis s'en rendirent acquéreurs et ainsi Odette fit de fréquentes croisières. Chaque fois qu'elle était partie depuis un peu de temps, Swann sentait qu'il commençait à se détacher d'elle, mais comme si cette distance morale était proportionnée à la distance matérielle, dès qu'il savait Odette de retour, il ne pouvait pas rester sans la voir. Une fois, partis pour un mois seulement, croyaient-ils, soit qu'ils eussent été tentés en route, soit que M. Verdurin eût sournoisement arrangé les choses d'avance pour faire plaisir à sa femme et n'eût averti les fidèles qu'au fur et à mesure, d'Alger, ils allèrent à Tunis, puis en Italie, puis en Grèce, à Constantinople, en Asie Mineure. Le voyage durait depuis près d'un an. Swann se sentait absolument tranquille, presque heureux. Bien que M. Verdurin eût cherché à persuader au pianiste et au docteur Cottard que la tante de l'un et les malades de l'autre n'avaient aucun besoin d'eux, et, qu'en tous cas il était imprudent de laisser Mme Cottard rentrer à Paris que Mme Verdurin assurait être en révolution, il fut obligé de leur rendre leur liberté à Constantinople. Et le peintre partit avec eux. Un jour, peu après le retour de ces trois voyageurs, Swann voyant passer un omnibus pour le Luxembourg où il avait à faire, avait sauté dedans, et s'y était trouvé assis en face de Mme Cottard qui faisait sa tournée de visites « de jours » en grande tenue, plumet au chapeau, robe de soie, manchon, en-tout-cas, porte-cartes, et gants blancs nettoyés. Revêtue de ces insignes, quand il faisait sec elle allait à pied d'une maison à l'autre, dans un même quartier, mais pour passer ensuite dans un quartier différent usait de l'omnibus avec correspondance. Pendant les premiers instants, avant que la gentillesse native de la femme eût pu percer l'empesé de la petite bourgeoise, et ne sachant trop d'ailleurs si elle devait parler des Verdurin à Swann, elle tint tout naturellement, de sa voix lente, gauche et douce que par moments l'omnibus couvrait complètement de son tonnerre, des propos choisis parmi ceux qu'elle entendait et répétait dans les vingt-cinq maisons dont elle montait les étages dans une journée :

– Je ne vous demande pas, monsieur, si un homme dans le mouvement comme vous, a vu, aux Mirlitons, le portrait de Machard qui fait courir tout Paris. Eh bien ! qu'en dites-vous ? Êtes-vous dans le camp de ceux qui approuvent ou dans le camp de ceux qui blâment ? Dans tous les salons on ne parle que du portrait de Machard ; on n'est pas chic, on n'est pas pur, on n'est pas dans le train, si on ne donne pas son opinion sur le portrait de Machard.
