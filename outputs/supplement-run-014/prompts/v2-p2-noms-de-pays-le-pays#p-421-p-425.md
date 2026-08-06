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
      "canonical_name": "Albertine",
      "surface_forms": [
        "Albertine"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Albertine",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.78,
      "evidence": "« obstinément placée entre nous deux », elle répond « de plus en plus brièvement », puis « cessa de répondre » jusqu’à ce que l’amie « abandonnât la place »; « Je reprochai à Albertine d’avoir été si désagréable »; « Ce n’est pas une mauvaise fille mais elle est barbante… Pourquoi se colle-t-elle à nous… je déteste qu’elle ait ses cheveux comme ça, ça donne mauvais genre »; « Les jeunes filles… ne sont pas censées avoir pour amis des messieurs »; « Gisèle ne pourrait s’en tirer qu’avec un bon coup de piston ».",
      "explanation": "The narrator portrays Albertine as jealous and coarse: she blocks the conversation, forces the friend to withdraw, and disparages her. He explicitly blames her, and his 'gatekeeper' discourse reinforces an image of exclusion and smallness."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "Locally, Albertine loses esteem: her exclusionary behavior and petty remarks are noted and criticized by the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-421-p-425"
}

### Candidate characters

[
  "Andrée",
  "Remi",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Un des matins qui suivirent celui où Andrée m'avait dit qu'elle était obligée de rester auprès de sa mère, je faisais quelques pas avec Albertine que j'avais aperçue, élevant au bout d'un cordonnet un attribut bizarre qui la faisait ressembler à l'« Idolâtrie » de Giotto ; il s'appelle d'ailleurs un « diabolo » et est tellement tombé en désuétude que devant le portrait d'une jeune fille en tenant un, les commentateurs de l'avenir pourront disserter comme devant telle figure allégorique de l'Arêna, sur ce qu'elle a dans la main. Au bout d'un moment, leur amie à l'air pauvre et dur, qui avait ricané le premier jour d'un air si méchant : « Il me fait de la peine ce pauvre vieux » en parlant du vieux monsieur effleuré par les pieds légers d'Andrée, vint dire à Albertine : « Bonjour, je vous dérange ? » Elle avait ôté son chapeau qui la gênait, et ses cheveux comme une variété végétale ravissante et inconnue reposaient sur son front dans la minutieuse délicatesse de leur foliation. Albertine, peut-être irritée de la voir tête nue, ne répondit rien, garda un silence glacial malgré lequel l'autre resta, tenue à distance de moi par Albertine qui s'arrangeait à certains instants pour être seule avec elle, à d'autres pour marcher avec moi, en la laissant derrière. Je fus obligé pour qu'elle me présentât de le lui demander devant l'autre. Alors au moment où Albertine me nomma, sur la figure et dans les yeux bleus de cette jeune fille à qui j'avais trouvé un air si cruel quand elle avait dit : « Ce pauvre vieux, y m'fait d'la peine », je vis passer et briller un sourire cordial, aimant, et elle me tendit la main. Ses cheveux étaient dorés, et ne l'étaient pas seuls ; car si ses joues étaient roses et ses yeux bleus, c'était comme le ciel encore empourpré du matin où partout pointe et brille l'or.

### Passage

Prenant feu aussitôt, je me dis que c'était une enfant timide quand elle aimait et que c'était pour moi, par amour pour moi, qu'elle était restée avec nous malgré les rebuffades d'Albertine, et qu'elle avait dû être heureuse de pouvoir m'avouer enfin, par ce regard souriant et bon, qu'elle serait aussi douce avec moi que terrible aux autres. Sans doute m'avait-elle remarqué sur la plage même quand je ne la connaissais pas encore et pensa-t-elle à moi depuis ; peut-être était-ce pour se faire admirer de moi qu'elle s'était moquée du vieux monsieur et parce qu'elle ne parvenait pas à me connaître qu'elle avait eu les jours suivants l'air morose. De l'hôtel, je l'avais souvent aperçue le soir se promenant sur la plage. C'était probablement avec l'espoir de me rencontrer. Et maintenant, gênée par la présence d'Albertine autant qu'elle l'eût été par celle de toute la bande, elle ne s'attachait évidemment à nos pas, malgré l'attitude de plus en plus froide de son amie, que dans l'espoir de rester la dernière, de prendre rendez-vous avec moi pour un moment où elle trouverait moyen de s'échapper sans que sa famille et ses amies le sussent et me donner rendez-vous dans un lieu sûr avant la messe ou après le golf. Il était d'autant plus difficile de la voir qu'Andrée était mal avec elle et la détestait.

– J'ai supporté longtemps sa terrible fausseté, me dit-elle, sa bassesse, les innombrables crasses qu'elle m'a faites. J'ai tout supporté à cause des autres. Mais le dernier trait a tout fait déborder. Et elle me raconta un potin qu'avait fait cette jeune fille et qui, en effet, pouvait nuire à Andrée.

Mais les paroles à moi promises par le regard de Gisèle pour le moment où Albertine nous aurait laissés ensemble ne purent m'être dites, parce qu'Albertine, obstinément placée entre nous deux, ayant continué de répondre de plus en plus brièvement, puis ayant cessé de répondre du tout aux propos de son amie, celle-ci finit par abandonner la place. Je reprochai à Albertine d'avoir été si désagréable. « Cela lui apprendra à être plus discrète. Ce n'est pas une mauvaise fille mais elle est barbante. Elle n'a pas besoin de venir fourrer son nez partout. Pourquoi se colle-t-elle à nous sans qu'on lui demande ? Il était moins cinq que je l'envoie paître. D'ailleurs, je déteste qu'elle ait ses cheveux comme ça, ça donne mauvais genre. » Je regardais les joues d'Albertine pendant qu'elle me parlait et je me demandais quel parfum, quel goût elles pouvaient avoir : ce jour-là elle était non pas fraîche, mais lisse, d'un rose uni, violacé, crémeux, comme certaines roses qui ont un vernis de cire. J'étais passionné pour elles comme on l'est parfois pour une espèce de fleurs. « Je ne l'avais pas remarquée, lui répondis-je. – Vous l'avez pourtant assez regardée, on aurait dit que vous vouliez faire son portrait, me dit-elle sans être radoucie par le fait qu'en ce moment ce fût elle-même que je regardais tant. Je ne crois pourtant pas qu'elle vous plairait. Elle n'est pas flirt du tout. Vous devez aimer les jeunes filles flirt, vous. En tous cas, elle n'aura plus l'occasion d'être collante et de se faire semer, parce qu'elle repart tantôt pour Paris. – Vos autres amies s'en vont avec elle ? – Non, elle seulement, elle et miss, parce qu'elle a à repasser ses examens, elle va potasser, la pauvre gosse. Ce n'est pas gai, je vous assure. Il peut arriver qu'on tombe sur un bon sujet. Le hasard est si grand. Ainsi une de nos amies a eu : « Racontez un accident auquel vous avez assisté ». Ça c'est une veine. Mais je connais une jeune fille qui a eu à traiter (et à l'écrit encore) : « D'Alceste ou de Philinte, qui préféreriez-vous avoir comme ami ? » Ce que j'aurais séché là-dessus ! D'abord en dehors de tout, ce n'est pas une question à poser à des jeunes filles. Les jeunes filles sont liées avec d'autres jeunes filles et ne sont pas censées avoir pour amis des messieurs. (Cette phrase, en me montrant que j'avais peu de chance d'être admis dans la petite bande, me fit trembler.) Mais en tous cas, même si la question était posée à des jeunes gens, qu'est-ce que vous voulez qu'on puisse trouver à dire là-dessus ? Plusieurs familles ont écrit au Gaulois pour se plaindre de la difficulté de questions pareilles. Le plus fort est que dans un recueil des meilleurs devoirs d'élèves couronnées, le sujet a été traité deux fois d'une façon absolument opposée. Tout dépend de l'examinateur. L'un voulait qu'on dise que Philinte était un homme flatteur et fourbe, l'autre qu'on ne pouvait pas refuser son admiration à Alceste, mais qu'il était par trop acariâtre et que comme ami il fallait lui préférer Philinte. Comment voulez-vous que les malheureuses élèves s'y reconnaissent quand les professeurs ne sont pas d'accord entre eux ? Et encore ce n'est rien, chaque année ça devient plus difficile. Gisèle ne pourrait s'en tirer qu'avec un bon coup de piston. »

Je rentrai à l'hôtel, ma grand'mère n'y était pas, je l'attendis longtemps ; enfin, quand elle rentra, je la suppliai de me laisser aller faire dans des conditions inespérées une excursion qui durerait peut-être quarante-huit heures, je déjeunai avec elle, commandai une voiture et me fis conduire à la gare. Gisèle ne serait pas étonnée de m'y voir ; une fois que nous aurions changé à Doncières, dans le train de Paris, il y avait un wagon couloir où tandis que miss sommeillerait je pourrais emmener Gisèle dans des coins obscurs, prendre rendez-vous avec elle pour ma rentrée à Paris que je tâcherais de rapprocher le plus possible. Selon la volonté qu'elle m'exprimerait, je l'accompagnerais jusqu'à Caen ou jusqu'à Évreux, et reprendrais le train suivant. Tout de même, qu'eût-elle pensé si elle avait su que j'avais hésité longtemps entre elle et ses amies, que tout autant que d'elle j'avais voulu être amoureux d'Albertine, de la jeune fille aux yeux clairs, et de Rosemonde ! J'éprouvais des remords, maintenant qu'un amour réciproque allait m'unir à Gisèle. J'aurais pu du reste lui assurer très véridiquement qu'Albertine ne me plaisait plus. Je l'avais vue ce matin s'éloigner en me tournant presque le dos, pour parler à Gisèle. Sur sa tête inclinée d'un air boudeur, ses cheveux qu'elle avait derrière, différents et plus noirs encore, luisaient comme si elle venait de sortir de l'eau. J'avais pensé à une poule mouillée et ces cheveux m'avaient fait incarner en Albertine une autre âme que jusque-là la figure violette et le regard mystérieux. Ces cheveux luisants derrière la tête, c'est tout ce que j'avais pu apercevoir d'elle pendant un moment, et c'est cela seulement que je continuais à voir. Notre mémoire ressemble à ces magasins, qui, à leurs devantures, exposent d'une certaine personne, une fois une photographie, une fois une autre. Et d'habitude la plus récente reste quelque temps seule en vue. Tandis que le cocher pressait son cheval, j'écoutais les paroles de reconnaissance et de tendresse que Gisèle me disait, toutes nées de son bon sourire, et de sa main tendue : c'est que dans les périodes de ma vie où je n'étais pas amoureux et où je désirais l'être, je ne portais pas seulement en moi un idéal physique de beauté qu'on a vu que je reconnaissais de loin dans chaque passante assez éloignée pour que ses traits confus ne s'opposassent pas à cette identification, mais encore le fantôme moral – toujours prêt à être incarné – de la femme qui allait être éprise de moi, me donner la réplique dans la comédie amoureuse que j'avais tout écrite dans ma tête depuis mon enfance et que toute jeune fille aimable me semblait avoir la même envie de jouer, pourvu qu'elle eût aussi un peu le physique de l'emploi. De cette pièce, quelle que fût la nouvelle « étoile » que j'appelais à créer ou à reprendre le rôle, le scénario, les péripéties, le texte même, gardaient une forme ne varietur.

Quelques jours plus tard, malgré le peu d'empressement qu'Albertine avait mis à nous présenter, je connaissais toute la petite bande du premier jour, restée au complet à Balbec (sauf Gisèle, qu'à cause d'un arrêt prolongé devant la barrière de la gare, et un changement dans l'horaire, je n'avais pu rejoindre au train, parti cinq minutes avant mon arrivée, et à laquelle d'ailleurs je ne pensais plus) et en plus deux ou trois de leurs amies qu'à ma demande elles me firent connaître. Et ainsi l'espoir du plaisir que je retrouverais avec une jeune fille nouvelle venant d'une autre jeune fille par qui je l'avais connue, la plus récente était alors comme une de ces variétés de roses qu'on obtient grâce à une rose d'une autre espèce. Et remontant de corolle en corolle dans cette chaîne de fleurs, le plaisir d'en connaître une différente me faisait retourner vers celle à qui je la devais, avec une reconnaissance mêlée d'autant de désir que mon espoir nouveau. Bientôt je passai toutes mes journées avec ces jeunes filles.
