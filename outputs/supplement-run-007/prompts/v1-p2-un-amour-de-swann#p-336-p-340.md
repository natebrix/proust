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
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Quel mensonge déprimant était-elle en train de faire à Swann... »; description of her habitual lies and visible distress; the letter reading: « J’ai eu raison d’ouvrir, c’était mon oncle. » leading to the inference that Forcheville had been there and was sent away.",
      "explanation": "The narrator reconstructs and confirms Odette’s current deception (with Forcheville present and concealed), framing her as a habitual, fearful liar whose present behavior masks the truth."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Her standing is locally lowered by the narrator’s exposure of a manipulative lie and of her pattern of deceit."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-336-p-340"
}

### Candidate characters

[
  "Mme Verdurin",
  "Swann",
  "comte de Forcheville",
  "le peintre"
]

### Prior local context (optional)

Mais il ne lui fit pas remarquer cette contradiction, car il pensait que, livrée à elle-même, Odette produirait peut-être quelque mensonge qui serait un faible indice de la vérité ; elle parlait ; il ne l'interrompait pas, il recueillait avec une piété avide et douloureuse ces mots qu'elle lui disait et qu'il sentait (justement, parce qu'elle la cachait derrière eux tout en lui parlant) garder vaguement, comme le voile sacré, l'empreinte, dessiner l'incertain modelé, de cette réalité infiniment précieuse et hélas introuvable : – ce qu'elle faisait tantôt à trois heures, quand il était venu – de laquelle il ne posséderait jamais que ces mensonges, illisibles et divins vestiges, et qui n'existait plus que dans le souvenir receleur de cet être qui la contemplait sans savoir l'apprécier, mais ne la lui livrerait pas. Certes il se doutait bien par moments qu'en elles-mêmes les actions quotidiennes d'Odette n'étaient pas passionnément intéressantes, et que les relations qu'elle pouvait avoir avec d'autres hommes n'exhalaient pas naturellement d'une façon universelle et pour tout être pensant une tristesse morbide, capable de donner la fièvre du suicide. Il se rendait compte alors que cet intérêt, cette tristesse n'existaient qu'en lui comme une maladie, et que quand celle-ci serait guérie, les actes d'Odette, les baisers qu'elle aurait pu donner redeviendraient inoffensifs comme ceux de tant d'autres femmes. Mais que la curiosité douloureuse que Swann y portait maintenant n'eût sa cause qu'en lui n'était pas pour lui faire trouver déraisonnable de considérer cette curiosité comme importante et de mettre tout en oeuvre pour lui donner satisfaction. C'est que Swann arrivait à un âge dont la philosophie – favorisée par celle de l'époque, par celle aussi du milieu où Swann avait beaucoup vécu, de cette coterie de la princesse des Laumes où il était convenu qu'on est intelligent dans la mesure où on doute de tout et où on ne trouvait de réel et d'incontestable que les goûts de chacun – n'est déjà plus celle de la jeunesse, mais une philosophie positive, presque médicale, d'hommes qui au lieu d'extérioriser les objets de leurs aspirations, essayent de dégager de leurs années déjà écoulées un résidu fixe d'habitudes, de passions qu'ils puissent considérer en eux comme caractéristiques et permanentes et auxquelles, délibérément, ils veilleront d'abord que le genre d'existence qu'ils adoptent puisse donner satisfaction. Swann trouvait sage de faire dans sa vie la part de la souffrance qu'il éprouvait à ignorer ce qu'avait fait Odette, aussi bien que la part de la recrudescence qu'un climat humide causait à son eczéma ; de prévoir dans son budget une disponibilité importante pour obtenir sur l'emploi des journées d'Odette des renseignements sans lesquels il se sentirait malheureux, aussi bien qu'il en réservait pour d'autres goûts dont il savait qu'il pouvait attendre du plaisir, au moins avant qu'il fût amoureux, comme celui des collections et de la bonne cuisine.

### Passage

Quand il voulut dire adieu à Odette pour rentrer, elle lui demanda de rester encore et le retint même vivement, en lui prenant le bras, au moment où il allait ouvrir la porte pour sortir. Mais il n'y prit pas garde, car, dans la multitude des gestes, des propos, des petits incidents qui remplissent une conversation, il est inévitable que nous passions, sans y rien remarquer qui éveille notre attention, près de ceux qui cachent une vérité que nos soupçons cherchent au hasard, et que nous nous arrêtions au contraire à ceux sous lesquels il n'y a rien. Elle lui redisait tout le temps : « Quel malheur que toi, qui ne viens jamais l'après-midi, pour une fois que cela t'arrive, je ne t'aie pas vu. » Il savait bien qu'elle n'était pas assez amoureuse de lui pour avoir un regret si vif d'avoir manqué sa visite, mais comme elle était bonne, désireuse de lui faire plaisir, et souvent triste quand elle l'avait contrarié, il trouva tout naturel qu'elle le fût cette fois de l'avoir privé de ce plaisir de passer une heure ensemble qui était très grand, non pour elle, mais pour lui. C'était pourtant une chose assez peu importante pour que l'air douloureux qu'elle continuait d'avoir finît par l'étonner. Elle rappelait ainsi, plus encore qu'il ne le trouvait d'habitude, les figures de femmes du peintre de la Primavera. Elle avait en ce moment leur visage abattu et navré qui semble succomber sous le poids d'une douleur trop lourde pour elles, simplement quand elles laissent l'enfant Jésus jouer avec une grenade ou regardent Moïse verser de l'eau dans une auge. Il lui avait déjà vu une fois une telle tristesse, mais ne savait plus quand. Et tout d'un coup, il se rappela : c'était quand Odette avait menti en parlant à Mme Verdurin le lendemain de ce dîner où elle n'était pas venue sous prétexte qu'elle était malade et en réalité pour rester avec Swann. Certes, eût-elle été la plus scrupuleuse des femmes qu'elle n'aurait pu avoir de remords d'un mensonge aussi innocent. Mais ceux que faisait couramment Odette l'étaient moins et servaient à empêcher des découvertes qui auraient pu lui créer avec les uns ou avec les autres, de terribles difficultés. Aussi quand elle mentait, prise de peur, se sentant peu armée pour se défendre, incertaine du succès, elle avait envie de pleurer, par fatigue, comme certains enfants qui n'ont pas dormi. Puis elle savait que son mensonge lésait d'ordinaire gravement l'homme à qui elle le faisait, et à la merci duquel elle allait peut-être tomber si elle mentait mal. Alors elle se sentait à la fois humble et coupable devant lui. Et quand elle avait à faire un mensonge insignifiant et mondain, par association de sensations et de souvenirs, elle éprouvait le malaise d'un surmenage et le regret d'une méchanceté.

Quel mensonge déprimant était-elle en train de faire à Swann pour qu'elle eût ce regard douloureux, cette voix plaintive qui semblaient fléchir sous l'effort qu'elle s'imposait, et demander grâce ? Il eut l'idée que ce n'était pas seulement la vérité sur l'incident de l'après-midi qu'elle s'efforçait de lui cacher, mais quelque chose de plus actuel, peut-être de non encore survenu et de tout prochain, et qui pourrait l'éclairer sur cette vérité. À ce moment, il entendit un coup de sonnette. Odette ne cessa plus de parler, mais ses paroles n'étaient qu'un gémissement : son regret de ne pas avoir vu Swann dans l'après-midi, de ne pas lui avoir ouvert, était devenu un véritable désespoir.

On entendit la porte d'entrée se refermer et le bruit d'une voiture, comme si repartait une personne – celle probablement que Swann ne devait pas rencontrer – à qui on avait dit qu'Odette était sortie. Alors en songeant que rien qu'en venant à une heure où il n'en avait pas l'habitude, il s'était trouvé déranger tant de choses qu'elle ne voulait pas qu'il sût, il éprouva un sentiment de découragement, presque de détresse. Mais comme il aimait Odette, comme il avait l'habitude de tourner vers elle toutes ses pensées, la pitié qu'il eût pu s'inspirer à lui-même, ce fut pour elle qu'il la ressentit, et il murmura : « Pauvre chérie ! » Quand il la quitta, elle prit plusieurs lettres qu'elle avait sur sa table et lui demanda s'il ne pourrait pas les mettre à la poste. Il les emporta et, une fois rentré, s'aperçut qu'il avait gardé les lettres sur lui. Il retourna jusqu'à la poste, les tira de sa poche et avant de les jeter dans la boîte regarda les adresses. Elles étaient toutes pour des fournisseurs, sauf une pour Forcheville. Il la tenait dans sa main. Il se disait : « Si je voyais ce qu'il y a dedans, je saurais comment elle l'appelle, comment elle lui parle, s'il y a quelque chose entre eux. Peut-être même qu'en ne la regardant pas, je commets une indélicatesse à l'égard d'Odette, car c'est la seule manière de me délivrer d'un soupçon peut-être calomnieux pour elle, destiné en tous cas à la faire souffrir et que rien ne pourrait plus détruire, une fois la lettre partie. »

Il rentra chez lui en quittant la poste, mais il avait gardé sur lui cette dernière lettre. Il alluma une bougie et en approcha l'enveloppe qu'il n'avait pas osé ouvrir. D'abord il ne put rien lire, mais l'enveloppe était mince, et en la faisant adhérer à la carte dure qui y était incluse, il put à travers sa transparence, lire les derniers mots. C'était une formule finale très froide. Si, au lieu que ce fût lui qui regardât une lettre adressée à Forcheville, c'eût été Forcheville qui eût lu une lettre adressée à Swann, il aurait pu voir des mots autrement tendres. Il maintint immobile la carte qui dansait dans l'enveloppe plus grande qu'elle, puis, la faisant glisser avec le pouce, en amena successivement les différentes lignes sous la partie de l'enveloppe qui n'était pas doublée, la seule à travers laquelle on pouvait lire.

Malgré cela il ne distinguait pas bien. D'ailleurs cela ne faisait rien, car il en avait assez vu pour se rendre compte qu'il s'agissait d'un petit événement sans importance et qui ne touchait nullement à des relations amoureuses ; c'était quelque chose qui se rapportait à un oncle d'Odette. Swann avait bien lu au commencement de la ligne : « J'ai eu raison », mais ne comprenait pas ce qu'Odette avait eu raison de faire, quand soudain, un mot qu'il n'avait pas pu déchiffrer d'abord, apparut et éclaira le sens de la phrase tout entière : « J'ai eu raison d'ouvrir, c'était mon oncle. » D'ouvrir ! alors Forcheville était là tantôt quand Swann avait sonné et elle l'avait fait partir, d'où le bruit qu'il avait entendu.
