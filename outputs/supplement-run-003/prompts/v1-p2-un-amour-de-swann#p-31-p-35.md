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
    },
    {
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Odette",
      "target": "Swann",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.9,
      "evidence": "« Quel rêve ce serait d’être mêlée à vos travaux ! »; « Vous êtes un être si à part »; « Je suis toujours libre, je le serai toujours pour vous... faites-moi chercher »; dit « d’une voix si naturelle, si convaincue, qu’il en avait été remué »",
      "explanation": "Odette openly flatters and defers to Swann’s intellect and person, pledging availability; the narrator notes it moves him, giving Swann local emotional leverage."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "By pledging constant availability and admiring Swann, she places herself in a supplicant, yielding posture."
    },
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Odette’s deference and praise affect him and give him emotional advantage in the budding relation."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-31-p-35"
}

### Candidate characters

[
  "Mme Verdurin",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Quelques mois après, si mon grand-père demandait au nouvel ami de Swann : « Et Swann, le voyez-vous toujours beaucoup ? » la figure de l'interlocuteur s'allongeait : « Ne prononcez jamais son nom devant moi ! » – « Mais je croyais que vous étiez si liés... » Il avait été ainsi pendant quelques mois le familier de cousins de la grand-mère, dînant presque chaque jour chez eux. Brusquement il cessa de venir, sans avoir prévenu. On le crut malade, et la cousine de la grand-mère allait envoyer demander de ses nouvelles, quand à l'office elle trouva une lettre de lui qui traînait par mégarde dans le livre de comptes de la cuisinière. Il y annonçait à cette femme qu'il allait quitter Paris, qu'il ne pourrait plus venir. Elle était sa maîtresse, et au moment de rompre, c'était elle seule qu'il avait jugé utile d'avertir.

### Passage

Quand sa maîtresse du moment était au contraire une personne mondaine ou du moins une personne qu'une extraction trop humble ou une situation trop irrégulière n'empêchait pas qu'il fît recevoir dans le monde, alors pour elle il y retournait, mais seulement dans l'orbite particulier où elle se mouvait ou bien où il l'avait entraînée. « Inutile de compter sur Swann ce soir, disait-on, vous savez bien que c'est le jour d'Opéra de son Américaine. » Il la faisait inviter dans les salons particulièrement fermés où il avait ses habitudes, ses dîners hebdomadaires, son poker ; chaque soir, après qu'un léger crépelage ajouté à la brosse de ses cheveux roux avait tempéré de quelque douceur la vivacité de ses yeux verts, il choisissait une fleur pour sa boutonnière et partait pour retrouver sa maîtresse à dîner chez l'une ou l'autre des femmes de sa coterie ; et alors, pensant à l'admiration et à l'amitié que les gens à la mode, pour qui il faisait la pluie et le beau temps et qu'il allait retrouver là, lui prodigueraient devant la femme qu'il aimait, il retrouvait du charme à cette vie mondaine sur laquelle il s'était blasé, mais dont la matière, pénétrée et colorée chaudement d'une flamme insinuée qui s'y jouait, lui semblait précieuse et belle depuis qu'il y avait incorporé un nouvel amour.

Mais tandis que chacune de ces liaisons, ou chacun de ces flirts, avait été la réalisation plus ou moins complète d'un rêve né de la vue d'un visage ou d'un corps que Swann avait, spontanément, sans s'y efforcer, trouvés charmants, en revanche, quand un jour au théâtre il fut présenté à Odette de Crécy par un de ses amis d'autrefois, qui lui avait parlé d'elle comme d'une femme ravissante avec qui il pourrait peut-être arriver à quelque chose, mais en la lui donnant pour plus difficile qu'elle n'était en réalité afin de paraître lui-même avoir fait quelque chose de plus aimable en la lui faisant connaître, elle était apparue à Swann non pas certes sans beauté, mais d'un genre de beauté qui lui était indifférent, qui ne lui inspirait aucun désir, lui causait même une sorte de répulsion physique, de ces femmes comme tout le monde a les siennes, différentes pour chacun, et qui sont l'opposé du type que nos sens réclament. Pour lui plaire elle avait un profil trop accusé, la peau trop fragile, les pommettes trop saillantes, les traits trop tirés. Ses yeux étaient beaux, mais si grands qu'ils fléchissaient sous leur propre masse, fatiguaient le reste de son visage et lui donnaient toujours l'air d'avoir mauvaise mine ou d'être de mauvaise humeur. Quelque temps après cette présentation au théâtre, elle lui avait écrit pour lui demander à voir ses collections qui l'intéressaient tant, « elle, ignorante qui avait le goût des jolies choses », disant qu'il lui semblait qu'elle le connaîtrait mieux, quand elle l'aurait vu dans « son home » où elle l'imaginait « si confortable avec son thé et ses livres », quoiqu'elle ne lui eût pas caché sa surprise qu'il habitât ce quartier qui devait être si triste et « qui était si peu smart pour lui qui l'était tant ». Et après qu'il l'eut laissée venir, en le quittant, elle lui avait dit son regret d'être restée si peu dans cette demeure où elle avait été heureuse de pénétrer, parlant de lui comme s'il avait été pour elle quelque chose de plus que les autres êtres qu'elle connaissait, et semblant établir entre leurs deux personnes une sorte de trait d'union romanesque qui l'avait fait sourire. Mais à l'âge déjà un peu désabusé dont approchait Swann, et où l'on sait se contenter d'être amoureux pour le plaisir de l'être sans trop exiger de réciprocité, ce rapprochement des coeurs, s'il n'est plus comme dans la première jeunesse le but vers lequel tend nécessairement l'amour, lui reste uni en revanche par une association d'idées si forte, qu'il peut en devenir la cause, s'il se présente avant lui. Autrefois on rêvait de posséder le coeur de la femme dont on était amoureux ; plus tard sentir qu'on possède le coeur d'une femme peut suffire à vous en rendre amoureux. Ainsi, à l'âge où il semblerait, comme on cherche surtout dans l'amour un plaisir subjectif, que la part du goût pour la beauté d'une femme devrait y être la plus grande, l'amour peut naître – l'amour le plus physique – sans qu'il y ait eu, à sa base, un désir préalable. À cette époque de la vie, on a déjà été atteint plusieurs fois par l'amour ; il n'évolue plus seul suivant ses propres lois inconnues et fatales, devant notre coeur étonné et passif. Nous venons à son aide, nous le faussons par la mémoire, par la suggestion. En reconnaissant un de ses symptômes, nous nous rappelons, nous faisons renaître les autres. Comme nous possédons sa chanson, gravée en nous tout entière, nous n'avons pas besoin qu'une femme nous en dise le début – rempli par l'admiration qu'inspire la beauté – pour en trouver la suite. Et si elle commence au milieu – là où les coeurs se rapprochent, où l'on parle de n'exister plus que l'un pour l'autre – nous avons assez l'habitude de cette musique pour rejoindre tout de suite notre partenaire au passage où elle nous attend.

Odette de Crécy retourna voir Swann, puis rapprocha ses visites ; et sans doute chacune d'elles renouvelait pour lui la déception qu'il éprouvait à se retrouver devant ce visage dont il avait un peu oublié les particularités dans l'intervalle, et qu'il ne s'était rappelé ni si expressif ni, malgré sa jeunesse, si fané ; il regrettait, pendant qu'elle causait avec lui, que la grande beauté qu'elle avait ne fût pas du genre de celles qu'il aurait spontanément préférées. Il faut d'ailleurs dire que le visage d'Odette paraissait plus maigre et plus proéminent parce que le front et le haut des joues, cette surface unie et plus plane était recouverte par la masse de cheveux qu'on portait, alors, prolongés en « devants », soulevés en « crêpés », répandus en mèches folles le long des oreilles ; et quant à son corps qui était admirablement fait, il était difficile d'en apercevoir la continuité (à cause des modes de l'époque et quoiqu'elle fût une des femmes de Paris qui s'habillaient le mieux), tant le corsage, s'avançant en saillie comme sur un ventre imaginaire et finissant brusquement en pointe pendant que par en dessous commençait à s'enfler le ballon des doubles jupes, donnait à la femme l'air d'être composée de pièces différentes mal emmanchées les unes dans les autres ; tant les ruchés, les volants, le gilet suivaient en toute indépendance, selon la fantaisie de leur dessin ou la consistance de leur étoffe, la ligne qui les conduisait aux noeuds, aux bouillons de dentelle, aux effilés de jais perpendiculaires, ou qui les dirigeait le long du busc, mais ne s'attachaient nullement à l'être vivant, qui selon que l'architecture de ces fanfreluches se rapprochait ou s'écartait trop de la sienne, s'y trouvait engoncé ou perdu.

Mais, quand Odette était partie, Swann souriait en pensant qu'elle lui avait dit combien le temps lui durerait jusqu'à ce qu'il lui permît de revenir ; il se rappelait l'air inquiet, timide, avec lequel elle l'avait une fois prié que ce ne fût pas dans trop longtemps, et les regards qu'elle avait eus à ce moment-là, fixés sur lui en une imploration craintive, et qui la faisaient touchante sous le bouquet de fleurs de pensées artificielles fixé devant son chapeau rond de paille blanche, à brides de velours noir. « Et vous, avait-elle dit, vous ne viendriez pas une fois chez moi prendre le thé ? » Il avait allégué des travaux en train, une étude – en réalité abandonnée depuis des années – sur Ver Meer de Delft. « Je comprends que je ne peux rien faire, moi chétive, à côté de grands savants comme vous autres, lui avait-elle répondu. Je serais comme la grenouille devant l'aréopage. Et pourtant j'aimerais tant m'instruire, savoir, être initiée. Comme cela doit être amusant de bouquiner, de fourrer son nez dans de vieux papiers », avait-elle ajouté avec l'air de contentement de soi-même que prend une femme élégante pour affirmer que sa joie est de se livrer sans crainte de se salir à une besogne malpropre, comme de faire la cuisine en « mettant elle-même les mains à la pâte ». « Vous allez vous moquer de moi, ce peintre qui vous empêche de me voir (elle voulait parler de Ver Meer), je n'avais jamais entendu parler de lui ; vit-il encore ? Est-ce qu'on peut voir de ses oeuvres à Paris, pour que je puisse me représenter ce que vous aimez, deviner un peu ce qu'il y a sous ce grand front qui travaille tant, dans cette tête qu'on sent toujours en train de réfléchir, me dire : voilà, c'est à cela qu'il est en train de penser. Quel rêve ce serait d'être mêlée à vos travaux ! » Il s'était excusé sur sa peur des amitiés nouvelles, ce qu'il avait appelé, par galanterie, sa peur d'être malheureux. « Vous avez peur d'une affection ? comme c'est drôle, moi qui ne cherche que cela, qui donnerais ma vie pour en trouver une, avait-elle dit d'une voix si naturelle, si convaincue, qu'il en avait été remué. Vous avez dû souffrir par une femme. Et vous croyez que les autres sont comme elle. Elle n'a pas su vous comprendre ; vous êtes un être si à part. C'est cela que j'ai aimé d'abord en vous, j'ai bien senti que vous n'étiez pas comme tout le monde. » – « Et puis d'ailleurs vous aussi, lui avait-il dit, je sais bien ce que c'est que les femmes, vous devez avoir des tas d'occupations, être peu libre. »

– « Moi, je n'ai jamais rien à faire ! Je suis toujours libre, je le serai toujours pour vous. À n'importe quelle heure du jour ou de la nuit où il pourrait vous être commode de me voir, faites-moi chercher, et je serai trop heureuse d'accourir. Le ferez-vous ? Savez-vous ce qui serait gentil, ce serait de vous faire présenter à Mme Verdurin chez qui je vais tous les soirs. Croyez-vous ! si on s'y retrouvait et si je pensais que c'est un peu pour moi que vous y êtes ! »
