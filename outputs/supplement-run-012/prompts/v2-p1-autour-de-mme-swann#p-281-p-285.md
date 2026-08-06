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
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.86,
      "evidence": "« quand, dans la suite, elle me fixa des rendez-vous, je les acceptais souvent et, au dernier moment, je lui écrivais que je ne pouvais pas venir »; il veut que « ces expressions de regret qu'on réserve d'ordinaire aux indifférents » la persuadent de son indifférence, tout en reconnaissant: « Hélas! ce serait en vain. »",
      "explanation": "The narrator adopts a last-minute refusal behavior to feign indifference. This is a tactical affront to Gilberte, presented with critical distance since it highlights her probable vanity."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "She is sidelined by last-minute cancellations, a local form of exclusion/snub that leaves her in a diminished position in the exchange."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-281-p-285"
}

### Candidate characters

[
  "Odette",
  "Swann",
  "le narrateur"
]

### Prior local context (optional)

Je venais d'écrire à Gilberte une lettre où je laissais tonner ma fureur, non sans pourtant jeter la bouée de quelques mots placés comme au hasard, et où mon amie pourrait accrocher une réconciliation ; un instant après, le vent ayant tourné, c'était des phrases tendres que je lui adressais pour la douceur de certaines expressions désolées, de tels « jamais plus », si attendrissants pour ceux qui les emploient, si fastidieux pour celle qui les lira, soit qu'elle les croie mensongers et traduise « jamais plus » par « ce soir même, si vous voulez bien de moi » ou qu'elle les croie vrais et lui annonçant alors une de ces séparations définitives qui nous sont si parfaitement égales dans la vie quand il s'agit d'êtres dont nous ne sommes pas épris. Mais puisque nous sommes incapables tandis que nous aimons d'agir en dignes prédécesseurs de l'être prochain que nous serons et qui n'aimera plus, comment pourrions-nous tout à fait imaginer l'état d'esprit d'une femme à qui, même si nous savions que nous lui sommes indifférents, nous avons perpétuellement fait tenir dans nos rêveries, pour nous bercer d'un beau songe ou nous consoler d'un gros chagrin, les mêmes propos que si elle nous aimait. Devant les pensées, les actions d'une femme que nous aimons, nous sommes aussi désorientés que le pouvaient être devant les phénomènes de la nature, les premiers physiciens (avant que la science fût constituée et eût mis un peu de lumière dans l'inconnu). Ou pis encore, comme un être pour l'esprit de qui le principe de causalité existerait à peine, un être qui ne serait pas capable d'établir un lien entre un phénomène et un autre et devant qui le spectacle du monde serait incertain comme un rêve. Certes je m'efforçais de sortir de cette incohérence, de trouver des causes. Je tâchais même d'être « objectif » et pour cela de bien tenir compte de la disproportion qui existait entre l'importance qu'avait pour moi Gilberte et celle non seulement que j'avais pour elle, mais qu'elle-même avait pour les autres êtres que moi, disproportion qui, si je l'eusse omise, eût risqué de me faire prendre une simple amabilité de mon amie pour un aveu passionné, une démarche grotesque et avilissante de ma part pour le simple et gracieux mouvement qui vous dirige vers de beaux yeux. Mais je craignais aussi de tomber dans l'excès contraire, où j'aurais vu dans l'arrivée inexacte de Gilberte à un rendez-vous un mouvement de mauvaise humeur, une hostilité irrémédiable. Je tâchais de trouver entre ces deux optiques également déformantes celle qui me donnerait la vision juste des choses ; les calculs qu'il me fallait faire pour cela me distrayaient un peu de ma souffrance ; et soit par obéissance à la réponse des nombres, soit que je leur eusse fait dire ce que je désirais, je me décidai le lendemain à aller chez les Swann, heureux, mais de la même façon que ceux qui, s'étant tourmentés longtemps à cause d'un voyage qu'ils ne voulaient pas faire, ne vont pas plus loin que la gare, et rentrent chez eux défaire leur malle. Et comme, pendant qu'on hésite, la seule idée d'une résolution possible (à moins d'avoir rendu cette idée inerte en décidant qu'on ne prendra pas la résolution) développe, comme une graine vivace, les linéaments, tout le détail des émotions qui naîtraient de l'acte exécuté, je me dis que j'avais été bien absurde de me faire, en projetant de ne plus voir Gilberte, autant de mal que si j'eusse dû réaliser ce projet et que, puisque au contraire c'était pour finir par retourner chez elle, j'aurais pu faire l'économie de tant de velléités et d'acceptations douloureuses. Mais cette reprise des relations d'amitié ne dura que le temps d'aller jusqu'à chez les Swann, non pas parce que leur maître d'hôtel, lequel m'aimait beaucoup, me dit que Gilberte était sortie (je sus en effet, dès le soir même, que c'était vrai, par des gens qui l'avaient rencontrée), mais à cause de la façon dont il me le dit : « Monsieur, Mademoiselle est sortie, je peux affirmer à Monsieur que je ne mens pas. Si Monsieur veut se renseigner, je peux faire venir la femme de chambre. Monsieur pense bien que je ferais tout ce que je pourrais pour lui faire plaisir et que si Mademoiselle était là, je mènerais tout de suite Monsieur auprès d'elle. » Ces paroles, de la sorte qui est la seule importante, involontaires, nous donnant la radiographie au moins sommaire de la réalité insoupçonnable que cacherait un discours étudié, prouvaient que dans l'entourage de Gilberte on avait l'impression que je lui étais importun ; aussi, à peine le maître d'hôtel les eut-il prononcées, qu'elles engendrèrent chez moi de la haine à laquelle je préférai donner comme objet, au lieu de Gilberte, le maître d'hôtel ; il concentra sur lui tous les sentiments de colère que j'avais pu avoir pour mon amie ; débarrassé d'eux grâce à ces paroles, mon amour subsista seul ; mais elles m'avaient montré en même temps que je devais pendant quelque temps ne pas chercher à voir Gilberte. Elle allait certainement m'écrire pour s'excuser. Malgré cela, je ne retournerais pas tout de suite la voir, afin de lui prouver que je pouvais vivre sans elle. D'ailleurs, une fois que j'aurais reçu sa lettre, fréquenter Gilberte serait une chose dont je pourrais plus aisément me priver pendant quelque temps, parce que je serais sûr de la retrouver dès que je le voudrais. Ce qu'il me fallait pour supporter moins tristement l'absence volontaire, c'était sentir mon coeur débarrassé de la terrible incertitude de savoir si nous n'étions pas brouillés pour toujours, si elle n'était pas fiancée, partie, enlevée. Les jours qui suivirent ressemblèrent à ceux de cette ancienne semaine du jour de l'an que j'avais dû passer sans Gilberte. Mais cette semaine-là finie, jadis, d'une part mon amie reviendrait aux Champs-Élysées, je la reverrais comme auparavant, j'en étais sûr ; et, d'autre part, je savais avec non moins de certitude que tant que dureraient les vacances du jour de l'an, ce n'était pas la peine d'aller aux Champs-Élysées. De sorte que, durant cette triste semaine déjà lointaine, j'avais supporté ma tristesse avec calme parce qu'elle n'était mêlée ni de crainte ni d'espérance. Maintenant, au contraire, c'était ce dernier sentiment qui presque autant que la crainte rendait ma souffrance intolérable. N'ayant pas eu de lettre de Gilberte le soir même, j'avais fait la part de sa négligence, de ses occupations, je ne doutais pas d'en trouver une d'elle dans le courrier du matin. Il fut attendu par moi, chaque jour, avec des palpitations de coeur auxquelles succédait un état d'abattement quand je n'y avais trouvé que des lettres de personnes qui n'étaient pas Gilberte ou bien rien, ce qui n'était pas pire, les preuves d'amitié d'une autre me rendant plus cruelles celles de son indifférence. Je me remettais à espérer pour le courrier de l'après-midi. Même entre les heures des levées des lettres je n'osais pas sortir, car elle eût pu faire porter la sienne. Puis le moment finissait par arriver où, ni facteur ni valet de pied des Swann ne pouvant plus venir, il fallait remettre au lendemain matin l'espoir d'être rassuré, et ainsi, parce que je croyais que ma souffrance ne durerait pas, j'étais obligé pour ainsi dire de la renouveler sans cesse. Le chagrin était peut-être le même, mais au lieu de ne faire, comme autrefois, que prolonger uniformément une émotion initiale, recommençait plusieurs fois par jour en débutant par une émotion si fréquemment renouvelée qu'elle finissait – elle, état tout physique, si momentané – par se stabiliser, si bien que les troubles causés par l'attente ayant à peine le temps de se calmer avant qu'une nouvelle raison d'attendre survînt, il n'y avait plus une seule minute par jour où je ne fusse dans cette anxiété qu'il est pourtant si difficile de supporter pendant une heure. Ainsi ma souffrance était infiniment plus cruelle qu'au temps de cet ancien 1er janvier, parce que cette fois il y avait en moi, au lieu de l'acceptation pure et simple de cette souffrance, l'espoir, à chaque instant, de la voir cesser.

### Passage

À cette acceptation, je finis pourtant par arriver, alors je compris qu'elle devait être définitive et je renonçai pour toujours à Gilberte, dans l'intérêt même de mon amour, et parce que je souhaitais avant tout qu'elle ne conservât pas de moi un souvenir dédaigneux. Même, à partir de ce moment-là, et pour qu'elle ne pût former la supposition d'une sorte de dépit amoureux de ma part, quand, dans la suite, elle me fixa des rendez-vous, je les acceptais souvent et, au dernier moment, je lui écrivais que je ne pouvais pas venir, mais en protestant que j'en étais désolé comme j'aurais fait avec quelqu'un que je n'aurais pas désiré voir. Ces expressions de regret qu'on réserve d'ordinaire aux indifférents persuaderaient mieux Gilberte de mon indifférence, me semblait-il, que ne ferait le ton d'indifférence qu'on affecte seulement envers celle qu'on aime. Quand mieux qu'avec des paroles, par des actions indéfiniment répétées, je lui aurais prouvé que je n'avais pas de goût à la voir, peut-être en retrouverait-elle pour moi. Hélas ! ce serait en vain : chercher en ne la voyant plus à ranimer en elle ce goût de me voir, c'était la perdre pour toujours ; d'abord, parce que quand il commencerait à renaître, si je voulais qu'il durât, il ne faudrait pas y céder tout de suite ; d'ailleurs, les heures les plus cruelles seraient passées ; c'était en ce moment qu'elle m'était indispensable et j'aurais voulu pouvoir l'avertir que bientôt elle ne calmerait, en me revoyant, qu'une douleur tellement diminuée qu'elle ne serait plus, comme elle l'eût été encore en ce moment même, et pour y mettre fin, un motif de capitulation, de se réconcilier, de se revoir. Et enfin plus tard quand je pourrais enfin avouer sans péril à Gilberte, tant son goût pour moi aurait repris de force, le mien pour elle, celui-ci n'aurait pu résister à une si longue absence et n'existerait plus ; Gilberte me serait devenue indifférente. Je le savais, mais je ne pouvais pas le lui dire ; elle aurait cru que si je prétendais que je cesserais de l'aimer en restant trop longtemps sans la voir, c'était à seule fin qu'elle me dît de revenir vite auprès d'elle.

En attendant, ce qui me rendait plus aisé de me condamner à cette séparation, c'est que (afin qu'elle se rendît bien compte que, malgré mes affirmations contraires, c'était ma volonté, et non un empêchement, non mon état de santé, qui me privaient de la voir) toutes les fois où je savais d'avance que Gilberte ne serait pas chez ses parents, devait sortir avec une amie, et ne rentrerait pas dîner, j'allais voir Odette (laquelle était redevenue pour moi ce qu'elle était au temps où je voyais si difficilement sa fille et où, les jours où celle-ci ne venait pas aux Champs-Élysées, j'allais me promener avenue des Acacias). De cette façon j'entendrais parler de Gilberte et j'étais sûr qu'elle entendrait ensuite parler de moi et d'une façon qui lui montrerait que je ne tenais pas à elle. Et je trouvais, comme tous ceux qui souffrent, que ma triste situation aurait pu être pire. Car ayant libre entrée dans la demeure où habitait Gilberte, je me disais toujours, bien que décidé à ne pas user de cette faculté, que si jamais ma douleur était trop vive je pourrais la faire cesser. Je n'étais malheureux qu'au jour le jour. Et c'est trop dire encore. Combien de fois par heure (mais maintenant sans l'anxieuse attente qui m'avait étreint les premières semaines après notre brouille, avant d'être retourné chez les Swann) ne me récitais-je pas la lettre que Gilberte m'enverrait bien un jour, m'apporterait peut-être elle-même. La constante vision de ce bonheur imaginaire m'aidait à supporter la destruction du bonheur réel. Pour les femmes qui ne nous aiment pas, comme pour les « disparus », savoir qu'on n'a plus rien à espérer n'empêche pas de continuer à attendre. On vit aux aguets, aux écoutes ; des mères dont le fils est parti en mer pour une exploration dangereuse se figurent à toute minute, et alors que la certitude qu'il a péri est acquise depuis longtemps, qu'il va entrer miraculeusement sauvé et bien portant. Et cette attente, selon la force du souvenir et la résistance des organes, ou bien les aide à traverser les années au bout desquelles elles supporteront que leur fils ne soit plus, d'oublier peu à peu et de survivre – ou bien les fait mourir.

D'autre part, mon chagrin était un peu consolé par l'idée qu'il profitait à mon amour. Chaque visite que je faisais à Odette sans voir Gilberte m'était cruelle, mais je sentais qu'elle améliorait d'autant l'idée que Gilberte avait de moi.

D'ailleurs si je m'arrangeais toujours, avant d'aller chez Odette, à être certain de l'absence de sa fille, cela tenait peut-être autant qu'à ma résolution d'être brouillé avec elle, à cet espoir de réconciliation qui se superposait à ma volonté de renoncement (bien peu sont absolus, au moins d'une façon continue, dans cette âme humaine dont une des lois, fortifiée par les afflux inopinés de souvenirs différents, est l'intermittence) et me masquait ce qu'elle avait de trop cruel. Cet espoir je savais bien ce qu'il avait de chimérique. J'étais comme un pauvre qui mêle moins de larmes à son pain sec s'il se dit que tout à l'heure peut-être un étranger va lui laisser toute sa fortune. Nous sommes tous obligés, pour rendre la réalité supportable, d'entretenir en nous quelques petites folies. Or mon espérance restait plus intacte – tout en même temps que la séparation s'effectuait mieux – si je ne rencontrais pas Gilberte. Si je m'étais trouvé face à face avec elle chez sa mère nous aurions peut-être échangé des paroles irréparables qui eussent rendu définitive notre brouille, tué mon espérance et d'autre part, en créant une anxiété nouvelle, réveillé mon amour et rendu plus difficile ma résignation.

Depuis bien longtemps et fort avant ma brouille avec sa fille, Odette m'avait dit : « C'est très bien de venir voir Gilberte, mais j'aimerais aussi que vous veniez quelquefois pour moi, pas à mon Choufleury, où vous vous ennuieriez parce que j'ai trop de monde, mais les autres jours où vous me trouverez toujours un peu tard. » J'avais donc l'air, en allant la voir, de n'obéir que longtemps après à un désir anciennement exprimé par elle. Et très tard, déjà dans la nuit, presque au moment où mes parents se mettaient à table, je partais faire à Odette une visite pendant laquelle je savais que je ne verrais pas Gilberte et où pourtant je ne penserais qu'à elle. Dans ce quartier, considéré alors comme éloigné, d'un Paris plus sombre qu'aujourd'hui, et qui, même dans le centre, n'avait pas d'électricité sur la voie publique et bien peu dans les maisons, les lampes d'un salon situé au rez-de-chaussée ou à un entresol très bas (tel qu'était celui de ses appartements où recevait habituellement Odette), suffisaient à illuminer la rue et à faire lever les yeux au passant qui rattachait à leur clarté comme à sa cause apparente et voilée la présence devant la porte de quelques coupés bien attelés. Le passant croyait, et non sans un certain émoi, à une modification survenue dans cette cause mystérieuse, quand il voyait l'un de ces coupés se mettre en mouvement ; mais c'était seulement un cocher qui, craignant que ses bêtes prissent froid, leur faisait faire de temps à autre des allées et venues d'autant plus impressionnantes que les roues caoutchoutées donnaient au pas des chevaux un fond de silence sur lequel il se détachait plus distinct et plus explicite.
