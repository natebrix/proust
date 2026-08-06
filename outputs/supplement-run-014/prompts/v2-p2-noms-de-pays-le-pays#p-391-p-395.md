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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« le coup d’oeil du grand le peintre la détruit en une seconde »; « Maintenant déchue… elle n’est plus qu’une femme quelconque »; « la première manière d’Elstir était l’extrait de naissance le plus accablant pour Odette »",
      "explanation": "The narrator asserts that a great painter like Elstir dismantles a woman's carefully composed 'type,' leaving Odette locally 'déchue' and ordinary in the spectator’s eyes; Elstir’s early manner is said to be especially damning for her."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "Locally, Odette’s crafted identity is disassembled by the portrait; she is depicted as fallen from a unique, exalted type to ‘une femme quelconque’ in the viewer’s perception."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-391-p-395"
}

### Candidate characters

[
  "Elstir",
  "Françoise",
  "M. Verdurin",
  "Robert de Saint-Loup",
  "Swann",
  "la grand-mère",
  "le directeur",
  "le narrateur",
  "le peintre",
  "princesse de Luxembourg"
]

### Prior local context (optional)

Elstir se tut. « Ce n'est pourtant pas Odette avant son mariage », dis-je par une de ces brusques rencontres fortuites de la vérité, qui sont somme toute assez rares, mais qui suffisent après coup à donner un certain fondement à la théorie des pressentiments si on prend soin d'oublier toutes les erreurs qui l'infirmeraient. Elstir ne me répondit pas. C'était bien un portrait d'Odette. Elle n'avait pas voulu le garder pour beaucoup de raisons dont quelques-unes sont trop évidentes. Il y en avait d'autres. Le portrait était antérieur au moment où Odette disciplinant ses traits avait fait de son visage et de sa taille cette création dont, à travers les années, ses coiffeurs, ses couturiers, elle-même – dans sa façon de se tenir, de parler, de sourire, de poser ses mains, ses regards, de penser – devaient respecter les grandes lignes. Il fallait la dépravation d'un amant rassasié pour que Swann préférât aux nombreuses photographies de l'Odette ne varietur qu'était sa ravissante femme, la petite photographie qu'il avait dans sa chambre, et où sous un chapeau de paille orné de pensées on voyait une maigre jeune femme assez laide, aux cheveux bouffants, aux traits tirés.

### Passage

Mais d'ailleurs le portrait eût-il été, non pas antérieur, comme la photographie préférée de Swann, à la systématisation des traits d'Odette en un type nouveau, majestueux et charmant, mais postérieur, qu'il eût suffi de la vision d'Elstir pour désorganiser ce type. Le génie artistique agit à la façon de ces températures extrêmement élevées qui ont le pouvoir de dissocier les combinaisons d'atomes et de grouper ceux-ci suivant un ordre absolument contraire, répondant à un autre type. Toute cette harmonie factice que la femme a imposée à ses traits et dont chaque jour avant de sortir elle surveille la persistance dans sa glace, chargeant l'inclinaison du chapeau, le lissage des cheveux, l'enjouement du regard, afin d'en assurer la continuité, cette harmonie, le coup d'oeil du grand peintre la détruit en une seconde, et à sa place il fait un regroupement des traits de la femme, de manière à donner satisfaction à un certain idéal féminin et pictural qu'il porte en lui. De même, il arrive souvent qu'à partir d'un certain âge, l'oeil d'un grand chercheur trouve partout les éléments nécessaires à établir les rapports qui seuls l'intéressent. Comme ces ouvriers et ces joueurs qui ne font pas d'embarras et se contentent de ce qui leur tombe sous la main, ils pourraient dire de n'importe quoi : cela fera l'affaire. Ainsi une cousine de la princesse de Luxembourg, beauté des plus altières, s'étant éprise autrefois d'un art qui était nouveau à cette époque, avait demandé au plus grand des peintres naturalistes de faire son portrait. Aussitôt l'oeil de l'artiste avait trouvé ce qu'il cherchait partout. Et sur la toile il y avait à la place de la grande dame un trottin, et derrière lui un vaste décor incliné et violet qui faisait penser à la place Pigalle. Mais même sans aller jusque-là, non seulement le portrait d'une femme par un grand artiste ne cherchera aucunement à donner satisfaction à quelques-unes des exigences de la femme – comme celles qui, par exemple, quand elle commence à vieillir, la font se faire photographier dans des tenues presque de fillette qui font valoir sa taille restée jeune et la font paraître comme la soeur ou même la fille de sa fille, celle-ci au besoin « fagotée » pour la circonstance, à côté d'elle – et mettra au contraire en relief les désavantages qu'elle cherche à cacher et qui, comme un teint fiévreux, voire verdâtre, le tentent d'autant plus parce qu'ils ont du « caractère » ; mais ils suffisent à désenchanter le spectateur vulgaire et réduisent pour lui en miettes l'idéal dont la femme soutenait si fièrement l'armature et qui la plaçait dans sa forme unique, irréductible, si en dehors, si au-dessus du reste de l'humanité. Maintenant déchue, située hors de son propre type où elle trônait invulnérable, elle n'est plus qu'une femme quelconque en la supériorité de qui nous avons perdu toute foi. Ce type, nous faisions tellement consister en lui, non seulement la beauté d'une Odette, mais sa personnalité, son identité, que devant le portrait qui l'a dépouillée de lui, nous sommes tentés de nous écrier non pas seulement : « Comme c'est enlaidi », mais : « Comme c'est peu ressemblant. » Nous avons peine à croire que ce soit elle. Nous ne la reconnaissons pas. Et pourtant il y a là un être que nous sentons bien que nous avons déjà vu. Mais cet être-là ce n'est pas Odette ; le visage de cet être, son corps, son aspect, nous sont bien connus. Ils nous rappellent, non pas la femme, qui ne se tenait jamais ainsi, dont la pose habituelle ne dessine nullement une telle étrange et provocante arabesque, mais d'autres femmes, toutes celles qu'a peintes Elstir et que toujours, si différentes qu'elles puissent être, il a aimé à camper ainsi de face, le pied cambré dépassant de la jupe, le large chapeau rond tenu à la main, répondant symétriquement, à la hauteur du genou qu'il couvre, à cet autre disque vu de face, le visage. Et enfin non seulement un portrait génial disloque le type d'une femme, tel que l'ont défini sa coquetterie et sa conception égoïste de la beauté, mais s'il est ancien, il ne se contente pas de vieillir l'original de la même manière que la photographie, en le montrant dans des atours démodés. Dans le portrait, ce n'est pas seulement la manière que la femme avait de s'habiller qui date, c'est aussi la manière que l'artiste avait de peindre. Cette manière, la première manière d'Elstir, était l'extrait de naissance le plus accablant pour Odette, parce qu'il faisait d'elle non pas seulement comme ses photographies d'alors une cadette de cocottes connues, mais parce qu'il faisait de son portrait le contemporain d'un des nombreux portraits que Manet ou Whistler ont peints d'après tant de modèles disparus qui appartiennent déjà à l'oubli ou à l'histoire.

C'est dans ces pensées silencieusement ruminées à côté d'Elstir, tandis que je le conduisais chez lui, que m'entraînait la découverte que je venais de faire relativement à l'identité de son modèle, quand cette première découverte m'en fit faire une seconde, plus troublante encore pour moi, concernant l'identité de l'artiste. Il avait fait le portrait d'Odette de Crécy. Serait-il possible que cet homme de génie, ce sage, ce solitaire, ce philosophe à la conversation magnifique et qui dominait toutes choses, fût le peintre ridicule et pervers, adopté jadis par les Verdurin ? Je lui demandai s'il les avait connus, si par hasard ils ne le surnommaient pas alors M. Biche. Il me répondit que si, sans embarras, comme s'il s'agissait d'une partie déjà un peu ancienne de son existence et s'il ne se doutait pas de la déception extraordinaire qu'il éveillait en moi, mais levant les yeux, il la lut sur mon visage. Le sien eut une expression de mécontentement. Et comme nous étions déjà presque arrivés chez lui, un homme moins éminent par l'intelligence et par le coeur m'eût peut-être simplement dit au revoir un peu sèchement et après cela eût évité de me revoir. Mais ce ne fut pas ainsi qu'Elstir agit avec moi ; en vrai maître – et c'était peut-être au point de vue de la création pure son seul défaut d'en être un, dans ce sens du mot maître, car un artiste pour être tout à fait dans la vérité de la vie spirituelle doit être seul, et ne pas prodiguer de son moi, même à des disciples, – de toute circonstance, qu'elle fût relative à lui ou à d'autres, il cherchait à extraire pour le meilleur enseignement des jeunes gens la part de vérité qu'elle contenait. Il préféra donc aux paroles qui auraient pu venger son amour-propre celles qui pouvaient m'instruire. « Il n'y a pas d'homme si sage qu'il soit, me dit-il, qui n'ait à telle époque de sa jeunesse prononcé des paroles, ou même mené une vie, dont le souvenir lui soit désagréable et qu'il souhaiterait être aboli. Mais il ne doit pas absolument le regretter, parce qu'il ne peut être assuré d'être devenu un sage, dans la mesure où cela est possible, que s'il a passé par toutes les incarnations ridicules ou odieuses qui doivent précéder cette dernière incarnation-là. Je sais qu'il y a des jeunes gens, fils et petits-fils d'hommes distingués, à qui leurs précepteurs ont enseigné la noblesse de l'esprit et l'élégance morale dès le collège. Ils n'ont peut-être rien à retrancher de leur vie, ils pourraient publier et signer tout ce qu'ils ont dit, mais ce sont de pauvres esprits, descendants sans force de doctrinaires, et de qui la sagesse est négative et stérile. On ne reçoit pas la sagesse, il faut la découvrir soi-même après un trajet que personne ne peut faire pour nous, ne peut nous épargner, car elle est un point de vue sur les choses. Les vies que vous admirez, les attitudes que vous trouvez nobles n'ont pas été disposées par le père de famille ou par le précepteur, elles ont été précédées de débuts bien différents, ayant été influencées par ce qui régnait autour d'elles de mal ou de banalité. Elles représentent un combat et une victoire. Je comprends que l'image de ce que nous avons été dans une période première ne soit plus reconnaissable et soit en tous cas déplaisante. Elle ne doit pas être reniée pourtant, car elle est un témoignage que nous avons vraiment vécu, que c'est selon les lois de la vie et de l'esprit que nous avons, des éléments communs de la vie, de la vie des ateliers, des coteries artistiques s'il s'agit d'un peintre, extrait quelque chose qui les dépasse. » Nous étions arrivés devant sa porte. J'étais déçu de ne pas avoir connu ces jeunes filles. Mais enfin maintenant il y aurait une possibilité de les retrouver dans la vie ; elles avaient cessé de ne faire que passer à un horizon où j'avais pu croire que je ne les verrais plus jamais apparaître. Autour d'elles ne flottait plus comme ce grand remous qui nous séparait et qui n'était que la traduction du désir en perpétuelle activité, mobile, urgent, alimenté d'inquiétudes, qu'éveillaient en moi leur inaccessibilité, leur fuite peut-être pour toujours. Mon désir d'elles, je pouvais maintenant le mettre au repos, le garder en réserve, à côté de tant d'autres dont, une fois que je la savais possible, j'ajournais la réalisation. Je quittai Elstir, je me retrouvai seul. Alors tout d'un coup, malgré ma déception, je vis dans mon esprit tous ces hasards que je n'eusse pas soupçonné pouvoir se produire, qu'Elstir fût justement lié avec ces jeunes filles, que celles qui le matin encore étaient pour moi des figures dans un tableau ayant pour fond la mer, m'eussent vu, m'eussent vu lié avec un grand peintre, lequel savait maintenant mon désir de les connaître et le seconderait sans doute. Tout cela avait causé pour moi du plaisir, mais ce plaisir m'était resté caché ; il était de ces visiteurs qui attendent, pour nous faire savoir qu'ils sont là, que les autres nous aient quitté, que nous soyons seuls. Alors nous les apercevons, nous pouvons leur dire : je suis tout à vous, et les écouter. Quelquefois entre le moment où ces plaisirs sont entrés en nous et le moment où nous pouvons y rentrer nous-même, il s'est écoulé tant d'heures, nous avons vu tant de gens dans l'intervalle que nous craignons qu'ils ne nous aient pas attendus. Mais ils sont patients, ils ne se lassent pas et dès que tout le monde est parti nous les trouvons en face de nous. Quelquefois c'est nous alors qui sommes si fatigués qu'il nous semble que nous n'aurons plus dans notre pensée défaillante assez de force pour retenir ces souvenirs, ces impressions, pour qui notre moi fragile est le seul lieu habitable, l'unique mode de réalisation. Et nous le regretterions, car l'existence n'a guère d'intérêt que dans les journées où la poussière des réalités est mêlée de sable magique, où quelque vulgaire incident de la vie devient un ressort romanesque. Tout un promontoire du monde inaccessible surgit alors de l'éclairage du songe et entre dans notre vie, dans notre vie où comme le dormeur éveillé nous voyons les personnes dont nous avions si ardemment rêvé que nous avions cru que nous ne les verrions jamais qu'en rêve.

L'apaisement apporté par la probabilité de connaître maintenant ces jeunes filles quand je le voudrais me fut d'autant plus précieux que je n'aurais pu continuer à les guetter les jours suivants, lesquels furent pris par les préparatifs du départ de Saint-Loup. Ma grand'mère était désireuse de témoigner à mon ami sa reconnaissance de tant de gentillesses qu'il avait eues pour elle et pour moi. Je lui dis qu'il était grand admirateur de Proudhon et je lui donnai l'idée de faire venir de nombreuses lettres autographes de ce philosophe qu'elle avait achetées ; Saint-Loup vint les voir à l'hôtel, le jour où elles arrivèrent qui était la veille de son départ. Il les lut avidement, maniant chaque feuille avec respect, tâchant de retenir les phrases, puis s'étant levé, s'excusait déjà auprès de ma grand'mère d'être resté aussi longtemps, quand il l'entendit lui répondre :

– Mais non, emportez-les, c'est à vous, c'est pour vous les donner que je les ai fait venir.

Il fut pris d'une joie dont il ne fut pas plus le maître que d'un état physique qui se produit sans intervention de la volonté, il devint écarlate comme un enfant qu'on vient de punir, et ma grand'mère fut beaucoup plus touchée de voir tous les efforts qu'il avait faits (sans y réussir) pour contenir la joie qui le secouait, que par tous les remerciements qu'il aurait pu proférer. Mais lui, craignant d'avoir mal témoigné sa reconnaissance, me priait encore de l'en excuser, le lendemain, penché à la fenêtre du petit chemin de fer d'intérêt local qu'il prit pour rejoindre sa garnison. Celle-ci était, en effet, très peu éloignée. Il avait pensé s'y rendre, comme il faisait souvent, quand il devait revenir le soir et qu'il ne s'agissait pas d'un départ définitif, en voiture. Mais il eût fallu cette fois-ci qu'il mît ses nombreux bagages dans le train. Et il trouva plus simple d'y monter aussi lui-même, suivant en cela l'avis du directeur qui, consulté, répondit que, voiture ou petit chemin de fer, « ce serait à peu près équivoque ». Il entendait signifier par là que ce serait équivalent (en somme, à peu près ce que Françoise eût exprimé en disant que « cela reviendrait du pareil au même »).
