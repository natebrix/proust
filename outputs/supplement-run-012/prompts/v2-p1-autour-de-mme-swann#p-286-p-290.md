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
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.78,
      "evidence": "Ses invitations au « five o'clock tea » sont reçues « comme s'ils avaient été quelque chose d'important et de singulier qui commandât la déférence »; Mme Bontemps dit: « On ne peut pas s'en aller de cette maison »; des Messieurs du Jockey se confondent en saluts quand Odette les présente. Odette se voit « une espèce de Lespinasse » et s'imagine avoir fondé un salon.",
      "explanation": "The passage showcases the social charm and mastery of Odette as a hostess: her words command deference and her introductions confer honor. This elevation is tinged with irony by the narrator who points out Odette's self-staging as a salonnière."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "Locally, Odette appears as a hostess to whom one defers and who retains her guests; her introductions and her 'tea' have the value of distinction."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-286-p-290"
}

### Candidate characters

[
  "Albertine",
  "Gilberte",
  "Mme Bontemps",
  "Mme Cottard",
  "Mme Verdurin",
  "Swann",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Depuis bien longtemps et fort avant ma brouille avec sa fille, Odette m'avait dit : « C'est très bien de venir voir Gilberte, mais j'aimerais aussi que vous veniez quelquefois pour moi, pas à mon Choufleury, où vous vous ennuieriez parce que j'ai trop de monde, mais les autres jours où vous me trouverez toujours un peu tard. » J'avais donc l'air, en allant la voir, de n'obéir que longtemps après à un désir anciennement exprimé par elle. Et très tard, déjà dans la nuit, presque au moment où mes parents se mettaient à table, je partais faire à Odette une visite pendant laquelle je savais que je ne verrais pas Gilberte et où pourtant je ne penserais qu'à elle. Dans ce quartier, considéré alors comme éloigné, d'un Paris plus sombre qu'aujourd'hui, et qui, même dans le centre, n'avait pas d'électricité sur la voie publique et bien peu dans les maisons, les lampes d'un salon situé au rez-de-chaussée ou à un entresol très bas (tel qu'était celui de ses appartements où recevait habituellement Odette), suffisaient à illuminer la rue et à faire lever les yeux au passant qui rattachait à leur clarté comme à sa cause apparente et voilée la présence devant la porte de quelques coupés bien attelés. Le passant croyait, et non sans un certain émoi, à une modification survenue dans cette cause mystérieuse, quand il voyait l'un de ces coupés se mettre en mouvement ; mais c'était seulement un cocher qui, craignant que ses bêtes prissent froid, leur faisait faire de temps à autre des allées et venues d'autant plus impressionnantes que les roues caoutchoutées donnaient au pas des chevaux un fond de silence sur lequel il se détachait plus distinct et plus explicite.

### Passage

Le « jardin d'hiver », que dans ces années-là le passant apercevait d'ordinaire, quelle que fût la rue, si l'appartement n'était pas à un niveau trop élevé au-dessus du trottoir, ne se voit plus que dans les héliogravures des livres d'étrennes de P.-J. Stahl où, en contraste avec les rares ornements floraux des salons Louis XVI d'aujourd'hui – une rose ou un iris du Japon dans un vase de cristal à long col qui ne pourrait pas contenir une fleur de plus – il semble, à cause de la profusion des plantes d'appartement qu'on avait alors, et du manque absolu de stylisation dans leur arrangement, avoir dû, chez les maîtresses de maison, répondre plutôt à quelque vivante et délicieuse passion pour la botanique qu'à un froid souci de morte décoration. Il faisait penser en plus grand, dans les hôtels d'alors, à ces serres minuscules et portatives posées au matin du 1er janvier sous la lampe allumée – les enfants n'ayant pas eu la patience d'attendre qu'il fît jour – parmi les autres cadeaux du jour de l'an, mais le plus beau d'entre eux, consolant, avec les plantes qu'on va pouvoir cultiver, de la nudité de l'hiver ; plus encore qu'à ces serres-là elles-mêmes, ces jardins d'hiver ressemblaient à celle qu'on voyait tout auprès d'elles, figurée dans un beau livre, autre cadeau du jour de l'an, et qui bien qu'elle fût donnée non aux enfants, mais à Mlle Lili, l'héroïne de l'ouvrage, les enchantait à tel point que, devenus maintenant presque vieillards, ils se demandaient si dans ces années fortunées l'hiver n'était pas la plus belle des saisons. Enfin, au fond de ce jardin d'hiver, à travers les arborescences d'espèces variées qui de la rue faisaient ressembler la fenêtre éclairée au vitrage de ces serres d'enfants, dessinées ou réelles, le passant, se hissant sur ses pointes, apercevait généralement un homme en redingote, un gardénia ou un oeillet à la boutonnière, debout devant une femme assise, tous deux vagues, comme deux intailles dans une topaze, au fond de l'atmosphère du salon, ambrée par le samovar – importation récente alors – de vapeurs qui s'en échappent peut-être encore aujourd'hui, mais qu'à cause de l'habitude personne ne voit plus. Odette tenait beaucoup à ce « thé » ; elle croyait montrer de l'originalité et dégager du charme en disant à un homme : « Vous me trouverez tous les jours un peu tard, venez prendre le thé », de sorte qu'elle accompagnait d'un sourire fin et doux ces mots prononcés par elle avec un accent anglais momentané et desquels son interlocuteur prenait bonne note en saluant d'un air grave, comme s'ils avaient été quelque chose d'important et de singulier qui commandât la déférence et exigeât de l'attention. Il y avait une autre raison que celles données plus haut et pour laquelle les fleurs n'avaient pas qu'un caractère d'ornement dans le salon de Odette, et cette raison-là ne tenait pas à l'époque, mais en partie à l'existence qu'avait menée jadis Odette. Une grande cocotte, comme elle avait été, vit beaucoup pour ses amants, c'est-à-dire chez elle, ce qui peut la conduire à vivre pour elle. Les choses que chez une honnête femme on voit et qui certes peuvent lui paraître, à elle aussi, avoir de l'importance, sont celles, en tous cas, qui pour la cocotte en ont le plus. Le point culminant de sa journée est celui non pas où elle s'habille pour le monde, mais où elle se déshabille pour un homme. Il lui faut être aussi élégante en robe de chambre, en chemise de nuit, qu'en toilette de ville. D'autres femmes montrent leurs bijoux, elle, elle vit dans l'intimité de ses perles. Ce genre d'existence impose l'obligation et finit par donner le goût d'un luxe secret, c'est-à-dire bien près d'être désintéressé. Odette l'étendait aux fleurs. Il y avait toujours près de son fauteuil une immense coupe de cristal remplie entièrement de violettes de Parme ou de marguerites effeuillées dans l'eau, et qui semblait témoigner aux yeux de l'arrivant de quelque occupation préférée et interrompue, comme eût été la tasse de thé que Odette eût bue seule, pour son plaisir ; d'une occupation plus intime même et plus mystérieuse, si bien qu'on avait envie de s'excuser en voyant les fleurs étalées là, comme on l'eût fait de regarder le titre du volume encore ouvert qui eût révélé la lecture récente, donc peut-être la pensée actuelle d'Odette. Et plus que le livre, les fleurs vivaient ; on était gêné si on entrait faire une visite à Odette, de s'apercevoir qu'elle n'était pas seule, ou, si on rentrait avec elle, de ne pas trouver le salon vide, tant y tenaient une place énigmatique et se rapportant à des heures de la vie de la maîtresse de maison, qu'on ne connaissait pas, ces fleurs qui n'avaient pas été préparées pour les visiteurs d'Odette, mais comme oubliées là par elle, avaient eu et auraient encore avec elle des entretiens particuliers qu'on avait peur de déranger, et dont on essayait en vain de lire le secret, en fixant des yeux la couleur délavée, liquide, mauve et dissolue des violettes de Parme.

Dès la fin d'octobre Odette rentrait le plus régulièrement qu'elle pouvait pour le thé, qu'on appelait encore dans ce temps-là le « five o'clock tea », ayant entendu dire (et aimant à répéter) que si Mme Verdurin s'était fait un salon c'était parce qu'on était toujours sûr de pouvoir la rencontrer chez elle à la même heure. Elle s'imaginait elle-même en avoir un, du même genre, mais plus libre, « senza rigore », aimait-elle à dire. Elle se voyait ainsi comme une espèce de Lespinasse et croyait avoir fondé un salon rival en enlevant à la du Deffant du petit groupe ses hommes les plus agréables, en particulier Swann, qui l'avait suivie dans sa sécession et sa retraite, selon une version qu'on comprend qu'elle eût réussi à accréditer auprès de nouveaux venus, ignorants du passé, mais non auprès d'elle-même. Mais certains rôles favoris sont par nous joués tant de fois devant le monde, et ressassés en nous-mêmes, que nous nous référons plus aisément à leur témoignage fictif qu'à celui d'une réalité presque complètement oubliée. Les jours où Odette n'était pas sortie du tout, on la trouvait dans une robe de chambre de crêpe de Chine, blanche comme une première neige, parfois aussi dans un de ces longs tuyautages de mousseline de soie, qui ne semblent qu'une jonchée de pétales roses ou blancs et qu'on trouverait aujourd'hui peu appropriés à l'hiver, et bien à tort. Car ces étoffes légères et ces couleurs tendres donnaient à la femme – dans la grande chaleur des salons d'alors fermés de portières et desquels ce que les romanciers mondains de l'époque trouvaient à dire de plus élégant, c'est qu'ils étaient « douillettement capitonnés » – le même air frileux qu'aux roses, qui pouvaient y rester à côté d'elle, malgré l'hiver, dans l'incarnat de leur nudité, comme au printemps. À cause de cet étouffement des sons par les tapis et de sa retraite dans des enfoncements, la maîtresse de la maison n'étant pas avertie de votre entrée comme aujourd'hui continuait à lire pendant que vous étiez déjà presque devant elle, ce qui ajoutait encore à cette impression de romanesque, à ce charme d'une sorte de secret surpris, que nous retrouvons aujourd'hui dans le souvenir de ces robes déjà démodées alors, que Odette était peut-être la seule à ne pas avoir encore abandonnées et qui nous donnent l'idée que la femme qui les portait devait être une héroïne de roman parce que nous, pour la plupart, ne les avons guère vues que dans certains romans d'Henry Gréville. Odette avait maintenant, dans son salon, au commencement de l'hiver, des chrysanthèmes énormes et d'une variété de couleurs comme Swann jadis n'eût pu en voir chez elle. Mon admiration pour eux – quand j'allais faire à Odette une de ces tristes visites où, lui ayant, de par mon chagrin, retrouvé toute sa mystérieuse poésie de mère de cette Gilberte à qui elle dirait le lendemain : « Ton ami m'a fait une visite » – venait sans doute de ce que, rose pâle comme la soie Louis XIV de ses fauteuils, blancs de neige comme sa robe de chambre en crêpe de Chine, ou d'un rouge métallique comme son samovar, ils superposaient à celle du salon une décoration supplémentaire, d'un coloris aussi riche, aussi raffiné, mais vivante et qui ne durerait que quelques jours. Mais j'étais touché, moins par ce que ces chrysanthèmes avaient d'éphémère, que de relativement durable par rapport à ces tons aussi roses ou aussi cuivrés, que le soleil couché exalte si somptueusement dans la brume des fins d'après-midi de novembre, et qu'après les avoir aperçus avant que j'entrasse chez Odette, s'éteignant dans le ciel, je retrouvais prolongés, transposés dans la palette enflammée des fleurs. Comme des feux arrachés par un grand coloriste à l'instabilité de l'atmosphère et du soleil, afin qu'ils vinssent orner une demeure humaine, ils m'invitaient, ces chrysanthèmes, et malgré toute ma tristesse, à goûter avidement pendant cette heure du thé les plaisirs si courts de novembre dont ils faisaient flamber près de moi la splendeur intime et mystérieuse. Hélas, ce n'était pas dans les conversations que j'entendais que je pouvais l'atteindre ; elles lui ressemblaient bien peu. Même avec Mme Cottard et quoique l'heure fût avancée, Odette se faisait caressante pour dire : « Mais non, il n'est pas tard, ne regardez pas la pendule, ce n'est pas l'heure, elle ne va pas ; qu'est-ce que vous pouvez avoir de si pressé à faire » ; et elle offrait une tartelette de plus à la femme du professeur qui gardait son porte-cartes à la main.

– On ne peut pas s'en aller de cette maison, disait Mme Bontemps à Odette tandis que Mme Cottard, dans sa surprise d'entendre exprimer sa propre impression, s'écriait : « C'est ce que je me dis toujours, avec ma petite jugeotte, dans mon for intérieur ! » approuvée par des Messieurs du Jockey qui s'étaient confondus en saluts, et comme comblés par tant d'honneur, quand Odette les avait présentés à cette petite bourgeoise peu aimable, qui restait devant les brillants amis d'Odette sur la réserve sinon sur ce qu'elle appelait la « défensive », car elle employait toujours un langage noble pour les choses les plus simples. « On ne le dirait pas, voilà trois mercredis que vous me faites faux-bond », disait Odette à Mme Cottard. « C'est vrai, Odette, il y a des siècles, des éternités que je ne vous ai vue. Vous voyez que je plaide coupable, mais il faut vous dire, ajoutait-elle d'un air pudibond et vague, car quoique femme de médecin elle n'aurait pas oser parler sans périphrases de rhumatismes ou de coliques néphrétiques, que j'ai eu bien des petites misères. Chacun a les siennes. Et puis j'ai eu une crise dans ma domesticité mâle. Sans être plus qu'une autre très imbue de mon autorité, j'ai dû, pour faire un exemple, renvoyer mon Vatel qui, je crois, cherchait d'ailleurs une place plus lucrative. Mais son départ a failli entraîner la démission de tout le ministère. Ma femme de chambre ne voulait pas rester non plus, il y a eu des scènes homériques. Malgré tout, j'ai tenu ferme le gouvernail, et c'est une véritable leçon de choses qui n'aura pas été perdue pour moi. Je vous ennuie avec ces histoires de serviteurs, mais vous savez comme moi quel tracas c'est d'être obligée de procéder à des remaniements dans son personnel. »

– Et nous ne verrons pas votre délicieuse fille, demandait-elle. – Non, ma délicieuse fille, dîne chez une amie », répondait Odette, et elle ajoutait en se tournant vers moi : « Je crois qu'elle vous a écrit pour que vous veniez la voir demain... Et nos babys », demandait-elle à la femme du Professeur. Je respirai largement. Ces mots de Odette, qui me prouvaient que je pourrais voir Gilberte quand je voudrais, me faisaient justement le bien que j'étais venu chercher et qui me rendait à cette époque-là les visites à Odette si nécessaires. « Non, je lui écrirai un mot ce soir. Du reste, Gilberte et moi nous ne pouvons plus nous voir », ajoutais-je, ayant l'air d'attribuer notre séparation à une cause mystérieuse, ce qui me donnait encore une illusion d'amour, entretenue aussi par la manière tendre dont je parlais de Gilberte et dont elle parlait de moi. « Vous savez qu'elle vous aime infiniment, me disait Odette. Vraiment vous ne voulez pas demain ? » Tout d'un coup une allégresse me soulevait, je venais de me dire : « Mais après tout pourquoi pas, puisque c'est sa mère elle-même qui me le propose. » Mais aussitôt je retombais dans ma tristesse. Je craignais qu'en me revoyant Gilberte pensât que mon indifférence de ces derniers temps avait été simulée et j'aimais mieux prolonger la séparation. Pendant ces apartés Mme Bontemps se plaignait de l'ennui que lui causaient les femmes des hommes politiques, car elle affectait de trouver tout le monde assommant et ridicule, et d'être désolée de la position de son mari. « Alors vous pouvez comme ça recevoir cinquante femmes de médecins de suite, disait-elle à Mme Cottard qui elle, au contraire, était pleine de bienveillance pour chacun et de respect pour toutes les obligations. Ah, vous avez de la vertu ! Moi, au ministère, n'est-ce pas, je suis obligée, naturellement. Eh bien ! c'est plus fort que moi, vous savez, ces femmes de fonctionnaires, je ne peux pas m'empêcher de leur tirer la langue. Et ma nièce Albertine est comme moi. Vous ne savez pas ce qu'elle est effrontée cette petite. La semaine dernière il y avait à mon jour la femme du sous-secrétaire d'État aux Finances qui disait qu'elle ne s'y connaissait pas en cuisine. « Mais, Madame, lui a répondu ma nièce avec son plus gracieux sourire, vous devriez pourtant savoir ce que c'est puisque votre père était marmiton. » « Oh ! j'aime beaucoup cette histoire, je trouve cela exquis, disait Odette. Mais au moins pour les jours de consultation du docteur vous devriez avoir un petit home, avec vos fleurs, vos livres, les choses que vous aimez », conseillait-elle à Mme Cottard. « Comme ça, v'lan dans la figure, v'lan, elle ne lui a pas envoyé dire. Et elle ne m'avait prévenue de rien cette petite masque, elle est rusée comme un singe. Vous avez de la chance de pouvoir vous retenir ; j'envie les gens qui savent déguiser leur pensée. » « Mais je n'en ai pas besoin, Madame : je ne suis pas si difficile, répondait avec douceur Mme Cottard. D'abord, je n'y ai pas les mêmes droits que vous, ajoutait-elle d'une voix un peu plus forte qu'elle prenait, afin de les souligner, chaque fois qu'elle glissait dans la conversation quelqu'une de ces amabilités délicates, de ces ingénieuses flatteries qui faisaient l'admiration et aidaient à la carrière de son mari. Et puis je fais avec plaisir tout ce qui peut être utile au professeur.

– Mais, Madame, il faut pouvoir. Probablement vous n'êtes pas nerveuse. Moi quand je vois la femme du ministre de la Guerre faire des grimaces, immédiatement je me mets à l'imiter. C'est terrible d'avoir un tempérament comme ça.
