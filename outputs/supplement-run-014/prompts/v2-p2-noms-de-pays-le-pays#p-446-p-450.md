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
      "canonical_name": "Andrée",
      "surface_forms": [
        "Andrée"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Andrée",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« Andrée, beaucoup plus forte qu’elles toutes et qui pouvait lui donner de bons tuyaux. »; Albertine « désirait beaucoup avoir l’avis d’Andrée »",
      "explanation": "The narrator frames Andrée as the most capable among the girls, and Albertine’s deference (seeking her advice) confirms her superior standing in the group."
    }
  ],
  "status_effects": [
    {
      "character": "Andrée",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "She is singled out as the most competent and is deferred to for advice, raising her micro-status within the group."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-446-p-450"
}

### Candidate characters

[
  "Albertine",
  "Mme de Villeparisis",
  "Robert de Saint-Loup",
  "le narrateur"
]

### Prior local context (optional)

Ce n'était pas seulement une matinée mondaine, une promenade avec Mme de Villeparisis que j'eusse sacrifiées au « furet » ou aux « devinettes » de mes amies. À plusieurs reprises M. de Marsantes me fit dire que puisque je n'allais pas le voir à Doncières, il avait demandé une permission de vingt-quatre heures et la passerait à Balbec. Chaque fois je lui écrivis de n'en rien faire, en invoquant l'excuse d'être obligé de m'absenter justement ce jour-là pour aller remplir dans le voisinage un devoir de famille avec la grand-mère. Sans doute me jugea-t-il mal en apprenant par sa tante en quoi consistait le devoir de famille et quelles personnes tenaient en l'espèce le rôle de la grand-mère. Et pourtant je n'avais peut-être pas tort de sacrifier les plaisirs non seulement de la mondanité, mais de l'amitié, à celui de passer tout le jour dans ce jardin. Les êtres qui en ont la possibilité – il est vrai que ce sont les artistes et j'étais convaincu depuis longtemps que je ne le serais jamais – ont aussi le devoir de vivre pour eux-mêmes ; or l'amitié leur est une dispense de ce devoir, une abdication de soi. La conversation même qui est le mode d'expression de l'amitié est une divagation superficielle, qui ne nous donne rien à acquérir. Nous pouvons causer pendant toute une vie sans rien faire que répéter indéfiniment le vide d'une minute, tandis que la marche de la pensée dans le travail solitaire de la création artistique se fait dans le sens de la profondeur, la seule direction qui ne nous soit pas fermée, où nous puissions progresser, avec plus de peine il est vrai, pour un résultat de vérité. Et l'amitié n'est pas seulement dénuée de vertu comme la conversation, elle est de plus funeste. Car l'impression d'ennui que ne peuvent pas ne pas éprouver auprès de leur ami, c'est-à-dire à rester à la surface de soi-même, au lieu de poursuivre leur voyage de découvertes dans les profondeurs, ceux d'entre nous dont la loi de développement est purement interne, cette impression d'ennui, l'amitié nous persuade de la rectifier quand nous nous retrouvons seuls, de nous rappeler avec émotion les paroles que notre ami nous a dites, de les considérer comme un précieux apport, alors que nous ne sommes pas comme des bâtiments à qui on peut ajouter des pierres du dehors, mais comme des arbres qui tirent de leur propre sève le noeud suivant de leur tige, l'étage supérieur de leur frondaison. Je me mentais à moi-même, j'interrompais la croissance dans le sens selon lequel je pouvais en effet véritablement grandir et être heureux, quand je me félicitais d'être aimé, admiré, par un être aussi bon, aussi intelligent, aussi recherché que Robert de Saint-Loup, quand j'adaptais mon intelligence, non à mes propres obscures impressions que c'eût été mon devoir de démêler, mais aux paroles de mon ami à qui en me les redisant – en me les faisant redire, par cet autre que soi-même qui vit en nous et sur qui on est toujours si content de se décharger du fardeau de penser – je m'efforçais de trouver une beauté, bien différente de celle que je poursuivais silencieusement quand j'étais vraiment seul, mais qui donnerait plus de mérite à Robert de Saint-Loup, à moi-même, à ma vie. Dans celle qu'un tel ami me faisait, je m'apparaissais comme douillettement préservé de la solitude, noblement désireux de me sacrifier moi-même pour lui, en somme incapable de me réaliser. Près de ces jeunes filles au contraire si le plaisir que je goûtais était égoïste, du moins n'était-il pas basé sur le mensonge qui cherche à nous faire croire que nous ne sommes pas irrémédiablement seuls et qui, quand nous causons avec un autre, nous empêche de nous avouer que ce n'est plus nous qui parlons, que nous nous modelons alors à la ressemblance des étrangers et non d'un moi qui diffère d'eux. Les paroles qui s'échangeaient entre les jeunes filles de la petite bande et moi étaient peu intéressantes, rares d'ailleurs, coupées de ma part de longs silences. Cela ne m'empêchait pas de prendre à les écouter quand elles me parlaient autant de plaisir qu'à les regarder, à découvrir dans la voix de chacune d'elles un tableau vivement coloré. C'est avec délices que j'écoutais leur pépiement. Aimer aide à discerner, à différencier. Dans un bois l'amateur d'oiseaux distingue aussitôt ces gazouillis particuliers à chaque oiseau, que le vulgaire confond. L'amateur de jeunes filles sait que les voix humaines sont encore bien plus variées. Chacune possède plus de notes que le plus riche instrument. Et les combinaisons selon lesquelles elle les groupe sont aussi inépuisables que l'infinie variété des personnalités. Quand je causais avec une de mes amies, je m'apercevais que le tableau original, unique de son individualité, m'était ingénieusement dessiné, tyranniquement imposé, aussi bien par les inflexions de sa voix que par celles de son visage et que c'était deux spectacles qui traduisaient, chacun dans son plan, la même réalité singulière. Sans doute les lignes de la voix, comme celles du visage, n'étaient pas encore définitivement fixées ; la première muerait encore, comme le second changerait. Comme les enfants possèdent une glande dont la liqueur les aide à digérer le lait et qui n'existe plus chez les grandes personnes, il y avait dans le gazouillis de ces jeunes filles des notes que les femmes n'ont plus. Et de cet instrument plus varié, elles jouaient avec leurs lèvres, avec cette application, cette ardeur des petits anges musiciens de Bellini, lesquelles sont aussi un apanage exclusif de la jeunesse. Plus tard ces jeunes filles perdraient cet accent de conviction enthousiaste qui donnait du charme aux choses les plus simples, soit qu'Albertine sur un ton d'autorité débitât des calembours que les plus jeunes écoutaient avec admiration jusqu'à ce que le fou rire se saisît d'elles avec la violence irrésistible d'un éternuement, soit qu'Andrée mît à parler de leurs travaux scolaires, plus enfantins encore que leurs jeux, une gravité essentiellement puérile ; et leurs paroles détonnaient, pareilles à ces strophes des temps antiques où la poésie encore peu différenciée de la musique se déclamait sur des notes différentes. Malgré tout, la voix de ces jeunes filles accusait déjà nettement le parti pris que chacune de ces petites personnes avait sur la vie, parti pris si individuel que c'est user d'un mot bien trop général que de dire pour l'une : « elle prend tout en plaisantant » ; pour l'autre : « elle va d'affirmation en affirmation » ; pour la troisième : « elle s'arrête à une hésitation expectante ». Les traits de notre visage ne sont guère que des gestes devenus, par l'habitude, définitifs. La nature, comme la catastrophe de Pompéi, comme une métamorphose de nymphe, nous a immobilisés dans le mouvement accoutumé. De même nos intonations contiennent notre philosophie de la vie, ce que la personne se dit à tout moment sur les choses. Sans doute ces traits n'étaient pas qu'à ces jeunes filles. Ils étaient à leurs parents. L'individu baigne dans quelque chose de plus général que lui. À ce compte, les parents ne fournissent pas que ce geste habituel que sont les traits du visage et de la voix, mais aussi certaines manières de parler, certaines phrases consacrées, qui presque aussi inconscientes qu'une intonation, presque aussi profondes, indiquent, comme elle, un point de vue sur la vie. Il est vrai que pour les jeunes filles, il y a certaines de ces expressions que leurs parents ne leur donnent pas avant un certain âge, généralement pas avant qu'elles soient des femmes. On les garde en réserve. Ainsi par exemple si on parlait des tableaux d'un ami d'Elstir, Andrée, qui avait encore les cheveux dans le dos, ne pouvait encore faire personnellement usage de l'expression dont usaient sa mère et sa soeur mariée : « Il paraît que l'homme est charmant. » Mais cela viendrait avec la permission d'aller au Palais-Royal. Et déjà depuis sa première communion, Albertine disait comme une amie de sa tante : « Je trouverais cela assez terrible. » On lui avait aussi donné en présent l'habitude de faire répéter ce qu'on disait pour avoir l'air de s'intéresser et de chercher à se former une opinion personnelle. Si on disait que la peinture d'un le peintre était bien, ou sa maison jolie : « Ah ! c'est bien, sa peinture ? Ah ! c'est joli, sa maison ? » Enfin plus générale encore que n'est le legs familial était la savoureuse matière imposée par la province originelle d'où elles tiraient leur voix et à même laquelle mordaient leurs intonations. Quand Andrée pinçait sèchement une note grave, elle ne pouvait faire que la corde périgourdine de son instrument vocal ne rendît un son chantant, fort en harmonie d'ailleurs avec la pureté méridionale de ses traits ; et aux perpétuelles gamineries de Rosemonde, la matière de son visage et de sa voix du Nord répondaient, quoi qu'elle en eût, avec l'accent de sa province. Entre cette province et le tempérament de la jeune fille qui dictait les inflexions je percevais un beau dialogue. Dialogue, non pas discorde. Aucune ne saurait diviser la jeune fille et son pays natal. Elle, c'est lui encore. Du reste cette réaction des matériaux locaux sur le génie qui les utilise et à qui elle donne plus de verdeur ne rend pas l'oeuvre moins individuelle, et que ce soit celle d'un architecte, d'un ébéniste, ou d'un musicien, elle ne reflète pas moins minutieusement les traits les plus subtils de la personnalité de l'artiste, parce qu'il a été forcé de travailler dans la pierre meulière de Senlis ou le grès rouge de Strasbourg, qu'il a respecté les noeuds particuliers au frêne, qu'il a tenu compte dans son écriture des ressources et des limites, de la sonorité, des possibilités, de la flûte ou de l'alto.

### Passage

Je m'en rendais compte et pourtant nous causions si peu. Tandis qu'avec Mme de Villeparisis ou Saint-Loup, j'eusse démontré par mes paroles beaucoup plus de plaisir que je n'en eusse ressenti, car je les quittais avec fatigue, au contraire couché entre ces jeunes filles, la plénitude de ce que j'éprouvais l'emportait infiniment sur la pauvreté, la rareté de nos propos et débordait de mon immobilité et de mon silence, en flots de bonheur dont le clapotis venait mourir au pied de ces jeunes roses.

Pour un convalescent qui se repose tout le jour dans un jardin fleuri ou dans un verger, une odeur de fleurs et de fruits n'imprègne pas plus profondément les mille riens dont se compose son farniente que pour moi cette couleur, cet arôme que mes regards allaient chercher sur ces jeunes filles et dont la douceur finissait par s'incorporer à moi. Ainsi les raisins se sucrent-ils au soleil. Et par leur lente continuité, ces jeux si simples avaient aussi amené en moi, comme chez ceux qui ne font autre chose que rester étendus au bord de la mer, à respirer le sel, à se hâler, une détente, un sourire béat, un éblouissement vague qui avait gagné jusqu'à mes yeux.

Parfois une gentille attention de telle ou telle éveillait en moi d'amples vibrations qui éloignaient pour un temps le désir des autres. Ainsi un jour Albertine avait dit : « Qu'est-ce qui a un crayon ? » Andrée l'avait fourni, Rosemonde le papier, Albertine leur avait dit : « Mes petites bonnes femmes, je vous défends de regarder ce que j'écris. » Après s'être appliquée à bien tracer chaque lettre, le papier appuyé à ses genoux, elle me l'avait passé en me disant : « Faites attention qu'on ne voie pas. » Alors je l'avais déplié et j'avais lu ces mots qu'elle m'avait écrits : « Je vous aime bien. »

« Mais au lieu d'écrire des bêtises, cria-t-elle en se tournant d'un air impétueux et grave vers Andrée et Rosemonde, il faut que je vous montre la lettre que Gisèle m'a écrite ce matin. Je suis folle, je l'ai dans ma poche, et dire que cela peut nous être si utile ! » Gisèle avait cru devoir adresser à son amie, afin qu'elle la communiquât aux autres, la composition qu'elle avait faite pour son certificat d'études. Les craintes d'Albertine sur la difficulté des sujets proposés avaient encore été dépassées par les deux entre lesquels Gisèle avait eu à opter. L'un était : « Sophocle écrit des Enfers à Racine pour le consoler de l'insuccès d'Athalie » ; l'autre : « Vous supposerez qu'après la première représentation d'Esther, Mme de Sévigné écrit à Mme de la Fayette pour lui dire combien elle a regretté son absence. » Or Gisèle, par un excès de zèle qui avait dû toucher les examinateurs, avait choisi le premier, le plus difficile de ces deux sujets, et l'avait traité si remarquablement qu'elle avait eu quatorze et avait été félicitée par le jury. Elle aurait obtenu la mention « très bien » si elle n'avait « séché » dans son examen d'espagnol. La composition dont Gisèle avait envoyé la copie à Albertine nous fut immédiatement lue par celle-ci, car, devant elle-même passer le même examen, elle désirait beaucoup avoir l'avis d'Andrée, beaucoup plus forte qu'elles toutes et qui pouvait lui donner de bons tuyaux. « Elle en a eu une veine, dit Albertine. C'est justement un sujet que lui avait fait piocher ici sa maîtresse de français. » La lettre de Sophocle à Racine, rédigée par Gisèle, commençait ainsi : « Mon cher ami, excusez-moi de vous écrire sans avoir l'honneur d'être personnellement connu de vous, mais votre nouvelle tragédie d'Athalie ne montre-t-elle pas que vous avez parfaitement étudié mes modestes ouvrages ? Vous n'avez pas mis de vers que dans la bouche des protagonistes, ou personnages principaux du drame, mais vous en avez écrit, et de charmants, permettez-moi de vous le dire sans cajolerie, pour les choeurs qui ne faisaient pas trop mal à ce qu'on dit dans la tragédie grecque, mais qui sont en France une véritable nouveauté. De plus, votre talent, si délié, si fignolé, si charmeur, si fin, si délicat, a atteint à une énergie dont je vous félicite. Athalie, Joad, voilà des personnages que votre rival, Corneille, n'eût pas su mieux charpenter. Les caractères sont virils, l'intrigue est simple et forte. Voilà une tragédie dont l'amour n'est pas le ressort et je vous en fais mes compliments les plus sincères. Les préceptes les plus fameux ne sont pas toujours les plus vrais. Je vous citerai comme exemple : « De cette passion la sensible peinture est pour aller au coeur la route la plus sûre. » Vous avez montré que le sentiment religieux dont débordent vos choeurs n'est pas moins capable d'attendrir. Le grand public a pu être dérouté, mais les vrais connaisseurs vous rendent justice. J'ai tenu à vous envoyer toutes mes congratulations auxquelles je joins, mon cher confrère, l'expression de mes sentiments les plus distingués. »

Les yeux d'Albertine n'avaient cessé d'étinceler pendant qu'elle faisait cette lecture.
