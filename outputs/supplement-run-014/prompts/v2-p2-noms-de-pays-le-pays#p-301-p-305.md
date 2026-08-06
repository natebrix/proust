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
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "prestige_association",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.68,
      "evidence": "ne l'avais-je pas aimée surtout parce qu'elle m'était apparue nimbée par cette auréole d'être l'amie de Bergotte, d'aller visiter avec lui les cathédrales",
      "explanation": "The narrator states he chiefly loved Gilberte because of the aura conferred by her association with Bergotte and shared cultured outings, locally crediting her with borrowed prestige."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.67,
      "explanation": "Within this passage, Gilberte is locally elevated as someone haloed by her connection with Bergotte, which enhances her perceived standing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-301-p-305"
}

### Candidate characters

[
  "Bergotte",
  "le narrateur"
]

### Prior local context (optional)

Maintenant, leurs traits charmants n'étaient plus indistincts et mêlés. Je les avais répartis et agglomérés (à défaut du nom de chacune, que j'ignorais) autour de la grande qui avait sauté par dessus le vieux banquier ; de la petite qui détachait sur l'horizon de la mer ses joues bouffies et roses, ses yeux verts ; de celle au teint bruni, au nez droit, qui tranchait au milieu des autres ; d'une autre, au visage blanc comme un oeuf dans lequel un petit nez faisait un arc de cercle comme un bec de poussin, visage comme en ont certains très jeunes gens ; d'une autre encore, grande, couverte d'une pèlerine (qui lui donnait un aspect si pauvre et démentait tellement sa tournure élégante que l'explication qui se présentait à l'esprit était que cette jeune fille devait avoir des parents assez brillants et plaçant leur amour-propre assez au-dessus des baigneurs de Balbec et de l'élégance vestimentaire de leurs propres enfants pour qu'il leur fût absolument égal de la laisser se promener sur la digue dans une tenue que de petites gens eussent jugée trop modeste) ; d'une fille aux yeux brillants, rieurs, aux grosses joues mates, sous un « polo » noir, enfoncé sur sa tête, qui poussait une bicyclette avec un dandinement de hanches si dégingandé, en employant des termes d'argot si voyous et criés si fort, quand je passai auprès d'elle (parmi lesquels je distinguai cependant la phrase fâcheuse de « vivre sa vie ») qu'abandonnant l'hypothèse que la pèlerine de sa camarade m'avait fait échafauder, je conclus plutôt que toutes ces filles appartenaient à la population qui fréquente les vélodromes, et devaient être les très jeunes maîtresses de coureurs cyclistes. En tous cas, dans aucune de mes suppositions, ne figurait celle qu'elles eussent pu être vertueuses. À première vue – dans la manière dont elles se regardaient en riant, dans le regard insistant de celle aux joues mates – j'avais compris qu'elles ne l'étaient pas. D'ailleurs, la grand-mère avait toujours veillé sur moi avec une délicatesse trop timorée pour que je ne crusse pas que l'ensemble des choses qu'on ne doit pas faire est indivisible et que des jeunes filles qui manquent de respect à la vieillesse fussent tout d'un coup arrêtées par des scrupules quand il s'agit de plaisirs plus tentateurs que de sauter par-dessus un octogénaire.

### Passage

Individualisées maintenant pourtant, la réplique que se donnaient les uns aux autres leurs regards animés de suffisance et d'esprit de camaraderie, et dans lesquels se rallumaient d'instant en instant tantôt l'intérêt, tantôt l'insolente indifférence dont brillait chacune, selon qu'il s'agissait de l'une de ses amies ou des passants, cette conscience aussi de se connaître entre elles assez intimement pour se promener toujours ensemble, en faisant « bande à part », mettaient entre leurs corps indépendants et séparés, tandis qu'ils s'avançaient lentement, une liaison invisible, mais harmonieuse comme une même ombre chaude, une même atmosphère, faisant d'eux un tout aussi homogène en ses parties qu'il était différent de la foule au milieu de laquelle se déroulait lentement leur cortège.

Un instant, tandis que je passais à côté de la brune aux grosses joues qui poussait une bicyclette, je croisai ses regards obliques et rieurs, dirigés du fond de ce monde inhumain qui enfermait la vie de cette petite tribu, inaccessible inconnu où l'idée de ce que j'étais ne pouvait certainement ni parvenir ni trouver place. Toute occupée à ce que disaient ses camarades, cette jeune fille coiffée d'un polo qui descendait très bas sur son front m'avait-elle vu au moment où le rayon noir émané de ses yeux m'avait rencontré. Si elle m'avait vu, qu'avais-je pu lui représenter ? Du sein de quel univers me distinguait-elle ? Il m'eût été aussi difficile de le dire que, lorsque certaines particularités nous apparaissent grâce au télescope, dans un astre voisin, il est malaisé de conclure d'elles que des humains y habitent, qu'ils nous voient, et quelles idées cette vue a pu éveiller en eux.

Si nous pensions que les yeux d'une telle fille ne sont qu'une brillante rondelle de mica, nous ne serions pas avides de connaître et d'unir à nous sa vie. Mais nous sentons que ce qui luit dans ce disque réfléchissant n'est pas dû uniquement à sa composition matérielle ; que ce sont, inconnues de nous, les noires ombres des idées que cet être se fait, relativement aux gens et aux lieux qu'il connaît – pelouses des hippodromes, sable des chemins où, pédalant à travers champs et bois, m'eût entraîné cette petite péri, plus séduisante pour moi que celle du paradis persan, – les ombres aussi de la maison où elle va rentrer, des projets qu'elle forme ou qu'on a formés pour elle ; et surtout que c'est elle, avec ses désirs, ses sympathies, ses répulsions, son obscure et incessante volonté. Je savais que je ne posséderais pas cette jeune cycliste si je ne possédais aussi ce qu'il y avait dans ses yeux. Et c'était par conséquent toute sa vie qui m'inspirait du désir ; désir douloureux, parce que je le sentais irréalisable, mais enivrant, parce que ce qui avait été jusque-là ma vie ayant brusquement cessé d'être ma vie totale, n'étant plus qu'une petite partie de l'espace étendu devant moi que je brûlais de couvrir, et qui était fait de la vie de ces jeunes filles, m'offrait ce prolongement, cette multiplication possible de soi-même, qui est le bonheur. Et, sans doute, qu'il n'y eût entre nous aucune habitude – comme aucune idée – communes, devait me rendre plus difficile de me lier avec elles et de leur plaire. Mais peut-être aussi c'était grâce à ces différences, à la conscience qu'il n'entrait pas, dans la composition de la nature et des actions de ces filles, un seul élément que je connusse ou possédasse, que venait en moi de succéder à la satiété, la soif – pareille à celle dont brûle une terre altérée – d'une vie que mon âme, parce qu'elle n'en avait jamais reçu jusqu'ici une seule goutte, absorberait d'autant plus avidement, à longs traits, dans une plus parfaite imbibition.

J'avais tant regardé cette cycliste aux yeux brillants qu'elle parut s'en apercevoir et dit à la plus grande un mot que je n'entendis pas, mais qui fit rire celle-ci. À vrai dire, cette brune n'était pas celle qui me plaisait le plus, justement parce qu'elle était brune, et que (depuis le jour où dans le petit raidillon de Tansonville, j'avais vu Gilberte) une jeune fille rousse à la peau dorée était restée pour moi l'idéal inaccessible. Mais Gilberte elle-même, ne l'avais-je pas aimée surtout parce qu'elle m'était apparue nimbée par cette auréole d'être l'amie de Bergotte, d'aller visiter avec lui les cathédrales. Et de la même façon ne pouvais-je me réjouir d'avoir vu cette brune me regarder (ce qui me faisait espérer qu'il me serait plus facile d'entrer en relations avec elle d'abord), car elle me présenterait aux autres, à l'impitoyable qui avait sauté par-dessus le vieillard, à la cruelle qui avait dit : « Il me fait de la peine, ce pauvre vieux » ; à toutes successivement, desquelles elle avait d'ailleurs le prestige d'être l'inséparable compagne. Et cependant, la supposition que je pourrais un jour être l'ami de telle ou telle de ces jeunes filles, que ces yeux, dont les regards inconnus me frappaient parfois en jouant sur moi sans le savoir comme un effet de soleil sur un mur, pourraient jamais par une alchimie miraculeuse laisser transpénétrer entre leurs parcelles ineffables l'idée de mon existence, quelque amitié pour ma personne, que moi-même je pourrais un jour prendre place entre elles, dans la théorie qu'elles déroulaient le long de la mer – cette supposition me paraissait enfermer en elle une contradiction aussi insoluble que si, devant quelque frise attique ou quelque fresque figurant un cortège, j'avais cru possible, moi spectateur, de prendre place, aimé d'elles, entre les divines processionnaires.

Le bonheur de connaître ces jeunes filles était-il donc irréalisable ? Certes ce n'eût pas été le premier de ce genre auquel j'eusse renoncé. Je n'avais qu'à me rappeler tant d'inconnues que, même à Balbec, la voiture s'éloignant à toute vitesse m'avait fait à jamais abandonner. Et même le plaisir que me donnait la petite bande, noble comme si elle était composée de vierges helléniques, venait de ce qu'elle avait quelque chose de la fuite des passantes sur la route. Cette fugacité des êtres qui ne sont pas connus de nous, qui nous forcent à démarrer de la vie habituelle où les femmes que nous fréquentons finissent par dévoiler leurs tares, nous met dans cet état de poursuite où rien n'arrête plus l'imagination. Or dépouiller d'elle nos plaisirs, c'est les réduire à eux-mêmes, à rien. Offertes chez une de ces entremetteuses que, par ailleurs, on a vu que je ne méprisais pas, retirées de l'élément qui leur donnait tant de nuances et de vague, ces jeunes filles m'eussent moins enchanté. Il faut que l'imagination, éveillée par l'incertitude de pouvoir atteindre son objet, crée un but qui nous cache l'autre, et en substituant au plaisir sensuel l'idée de pénétrer dans une vie, nous empêche de reconnaître ce plaisir, d'éprouver son goût véritable, de le restreindre à sa portée.
