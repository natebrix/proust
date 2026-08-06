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
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "« une crème au chocolat, inspiration, attention personnelle de Françoise, nous était offerte, fugitive et légère comme une oeuvre de circonstance où elle avait mis tout son talent »; refuser d’en goûter « se serait immédiatement ravalé au rang de ces goujats … alors que n’y valent que l’intention et la signature »",
      "explanation": "The narrator frames Françoise’s cooking as a work of art bearing her 'signature' and deserving respectful reception, elevating her from servant to artist within the household scene."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Within this passage, Françoise’s standing rises clearly through narrator-endorsed praise of her culinary artistry and the etiquette it commands."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-106-p-110"
}

### Candidate characters

[
  "Octave",
  "Swann",
  "la mère du narrateur",
  "le père du narrateur",
  "le narrateur"
]

### Prior local context (optional)

– Ah ! soupirait ma tante, je voudrais que ce soit déjà l'heure d'Eulalie. Il n'y a vraiment qu'elle qui pourra me dire cela.

### Passage

Eulalie était une fille boiteuse, active et sourde qui s'était « retirée » après la mort de Mme de la Bretonnerie où elle avait été en place depuis son enfance, et qui avait pris à côté de l'église une chambre, d'où elle descendait tout le temps soit aux offices, soit, en dehors des offices, dire une petite prière ou donner un coup de main à Théodore ; le reste du temps elle allait voir des personnes malades comme ma tante Léonie à qui elle racontait ce qui s'était passé à la messe ou aux vêpres. Elle ne dédaignait pas d'ajouter quelque casuel à la petite rente que lui servait la famille de ses anciens maîtres en allant de temps en temps visiter le linge du curé ou de quelque autre personnalité marquante du monde clérical de Combray. Elle portait au-dessus d'une mante de drap noir un petit béguin blanc, presque de religieuse, et une maladie de peau donnait à une partie de ses joues et à son nez recourbé, les tons rose vif de la balsamine. Ses visites étaient la grande distraction de ma tante Léonie qui ne recevait plus guère personne d'autre, en dehors de M. le Curé. Ma tante avait peu à peu évincé tous les autres visiteurs parce qu'ils avaient le tort à ses yeux de rentrer tous dans l'une ou l'autre des deux catégories de gens qu'elle détestait. Les uns, les pires et dont elle s'était débarrassée les premiers, étaient ceux qui lui conseillaient de ne pas « s'écouter » et professaient, fût-ce négativement et en ne la manifestant que par certains silences de désapprobation ou par certains sourires de doute, la doctrine subversive qu'une petite promenade au soleil et un bon bifteck saignant (quand elle gardait quatorze heures sur l'estomac deux méchantes gorgées d'eau de Vichy !) lui feraient plus de bien que son lit et ses médecines. L'autre catégorie se composait des personnes qui avaient l'air de croire qu'elle était plus gravement malade qu'elle ne pensait, qu'elle était aussi gravement malade qu'elle le disait. Aussi, ceux qu'elle avait laissé monter après quelques hésitations et sur les officieuses instances de Françoise et qui, au cours de leur visite, avaient montré combien ils étaient indignes de la faveur qu'on leur faisait en risquant timidement un : « Ne croyez-vous pas que si vous vous secouiez un peu par un beau temps », ou qui, au contraire, quand elle leur avait dit : « Je suis bien bas, bien bas, c'est la fin, mes pauvres amis », lui avaient répondu : « Ah ! quand on n'a pas la santé ! Mais vous pouvez durer encore comme ça », ceux-là, les uns comme les autres, étaient sûrs de ne plus jamais être reçus. Et si Françoise s'amusait de l'air épouvanté de ma tante quand de son lit elle avait aperçu dans la rue du Saint-Esprit une de ces personnes qui avait l'air de venir chez elle ou quand elle avait entendu un coup de sonnette, elle riait encore bien plus, et comme d'un bon tour, des ruses toujours victorieuses de ma tante pour arriver à les faire congédier et de leur mine déconfite en s'en retournant sans l'avoir vue, et, au fond, admirait sa maîtresse qu'elle jugeait supérieure à tous ces gens puisqu'elle ne voulait pas les recevoir. En somme, ma tante exigeait à la fois qu'on l'approuvât dans son régime, qu'on la plaignît pour ses souffrances et qu'on la rassurât sur son avenir.

C'est à quoi Eulalie excellait. Ma tante pouvait lui dire vingt fois en une minute : « C'est la fin, ma pauvre Eulalie », vingt fois Eulalie répondait : « Connaissant votre maladie comme vous la connaissez, madame Octave, vous irez à cent ans, comme me disait hier encore Mme Sazerin. » (Une des plus fermes croyances d'Eulalie, et que le nombre imposant des démentis apportés par l'expérience n'avait pas suffi à entamer, était que Mme Sazerat s'appelait Mme Sazerin.)

– Je ne demande pas à aller à cent ans, répondait ma tante, qui préférait ne pas voir assigner à ses jours un terme précis.

Et comme Eulalie savait avec cela comme personne distraire ma tante sans la fatiguer, ses visites qui avaient lieu régulièrement tous les dimanches sauf empêchement inopiné, étaient pour ma tante un plaisir dont la perspective l'entretenait ces jours-là dans un état agréable d'abord, mais bien vite douloureux comme une faim excessive, pour peu qu'Eulalie fût en retard. Trop prolongée, cette volupté d'attendre Eulalie tournait en supplice, ma tante ne cessait de regarder l'heure, bâillait, se sentait des faiblesses. Le coup de sonnette d'Eulalie, s'il arrivait tout à la fin de la journée, quand elle ne l'espérait plus, la faisait presque se trouver mal. En réalité, le dimanche, elle ne pensait qu'à cette visite et sitôt le déjeuner fini, Françoise avait hâte que nous quittions la salle à manger pour qu'elle pût monter « occuper » ma tante. Mais (surtout à partir du moment où les beaux jours s'installaient à Combray) il y avait bien longtemps que l'heure altière de midi, descendue de la tour de Saint-Hilaire qu'elle armoriait des douze fleurons momentanés de sa couronne sonore, avait retenti autour de notre table, auprès du pain bénit venu lui aussi familièrement en sortant de l'église, quand nous étions encore assis devant les assiettes des Mille et une Nuits, appesantis par la chaleur et surtout par le repas. Car, au fond permanent d'oeufs, de côtelettes, de pommes de terre, de confitures, de biscuits, qu'elle ne nous annonçait même plus, Françoise ajoutait – selon les travaux des champs et des vergers, le fruit de la marée, les hasards du commerce, les politesses des voisins et son propre génie, et si bien que notre menu, comme ces quatre-feuilles qu'on sculptait au XIIIe siècle au portail des cathédrales, reflétait un peu le rythme des saisons et des épisodes de la vie – : une barbue parce que la marchande lui en avait garanti la fraîcheur, une dinde parce qu'elle en avait vu une belle au marché de Roussainville-le-Pin, des cardons à la moelle parce qu'elle ne nous en avait pas encore fait de cette manière-là, un gigot rôti parce que le grand air creuse et qu'il avait bien le temps de descendre d'ici sept heures, des épinards pour changer, des abricots parce que c'était encore une rareté, des groseilles parce que dans quinze jours il n'y en aurait plus, des framboises que Swann avait apportées exprès, des cerises, les premières qui vinssent du cerisier du jardin après deux ans qu'il n'en donnait plus, du fromage à la crème que j'aimais bien autrefois, un gâteau aux amandes parce qu'elle l'avait commandé la veille, une brioche parce que c'était notre tour de l'offrir. Quand tout cela était fini, composée expressément pour nous, mais dédiée plus spécialement à mon père qui était amateur, une crème au chocolat, inspiration, attention personnelle de Françoise, nous était offerte, fugitive et légère comme une oeuvre de circonstance où elle avait mis tout son talent. Celui qui eût refusé d'en goûter en disant : « J'ai fini, je n'ai plus faim », se serait immédiatement ravalé au rang de ces goujats qui, même dans le présent qu'un artiste leur fait d'une de ses oeuvres, regardent au poids et à la matière alors que n'y valent que l'intention et la signature. Même en laisser une seule goutte dans le plat eût témoigné de la même impolitesse que se lever avant la fin du morceau au nez du compositeur.

Enfin ma mère me disait : « Voyons, ne reste pas ici indéfiniment, monte dans ta chambre si tu as trop chaud dehors, mais va d'abord prendre l'air un instant pour ne pas lire en sortant de table. » J'allais m'asseoir près de la pompe et de son auge, souvent ornée, comme un fond gothique, d'une salamandre, qui sculptait sur la pierre fruste le relief mobile de son corps allégorique et fuselé, sur le banc sans dossier ombragé d'un lilas, dans ce petit coin du jardin qui s'ouvrait par une porte de service sur la rue du Saint-Esprit et de la terre peu soignée duquel s'élevait par deux degrés, en saillie de la maison, et comme une construction indépendante, l'arrière-cuisine. On apercevait son dallage rouge et luisant comme du porphyre. Elle avait moins l'air de l'antre de Françoise que d'un petit temple de Vénus. Elle regorgeait des offrandes du crémier, du fruitier, de la marchande de légumes, venus parfois de hameaux assez lointains pour lui dédier les prémices de leurs champs. Et son faîte était toujours couronné du roucoulement d'une colombe.
