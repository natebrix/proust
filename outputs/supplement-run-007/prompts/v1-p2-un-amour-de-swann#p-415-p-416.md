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
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.9,
      "evidence": "« elle le voyait peu », elle « invoquait les convenances » et disait qu’il voulait « afficher leur liaison », « la traiter comme une fille »; elle annule ou part « d’un élan irrésistible » pour d’autres sorties, le réprimandant: « Voilà comme tu me remercies… C’est bon à savoir pour une autre fois ! »",
      "explanation": "Odette actively restricts Swann’s access, refuses to be seen with him in public, and rebuffs him when he appears sad; taken together this constitutes keeping him at a distance and a socially coded refusal."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Swann is locally excluded: Odette avoids seeing him, refuses public appearances with him, and favors other invitations, which keeps him at a distance and puts him in a position of dependence."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-415-p-416"
}

### Candidate characters

[
  "baron de Charlus",
  "comte de Forcheville",
  "le narrateur",
  "oncle Adolphe"
]

### Prior local context (optional)

Mais, par les intimités déjà anciennes qu'il avait parmi eux, les gens du monde, dans une certaine mesure, faisaient aussi partie de sa maison, de son domestique et de sa famille. Il se sentait, à considérer ses brillantes amitiés, le même appui hors de lui-même, le même confort, qu'à regarder les belles terres, la belle argenterie, le beau linge de table, qui lui venaient des siens. Et la pensée que s'il tombait chez lui frappé d'une attaque, ce serait tout naturellement le duc de Chartres, le prince de Reuss, le duc de Luxembourg, et le baron de Charlus, que son valet de chambre courrait chercher, lui apportait la même consolation qu'à notre vieille Françoise de savoir qu'elle serait ensevelie dans des draps fins à elle, marqués, non reprisés (ou si finement que cela ne donnait qu'une plus haute idée du soin de l'ouvrière), linceul de l'image fréquente duquel elle tirait une certaine satisfaction, sinon de bien-être, au moins d'amour-propre. Mais surtout, comme dans toutes celles de ses actions et de ses pensées qui se rapportaient à Odette, Swann était constamment dominé et dirigé par le sentiment inavoué qu'il lui était peut-être pas moins cher, mais moins agréable à voir que quiconque, que le plus ennuyeux fidèle des M. Verdurin, quand il se reportait à un monde pour qui il était l'homme exquis par excellence, qu'on faisait tout pour attirer, qu'on se désolait de ne pas voir, il recommençait à croire à l'existence d'une vie plus heureuse, presque à en éprouver l'appétit, comme il arrive à un malade alité depuis des mois, à la diète, et qui aperçoit dans un journal le menu d'un déjeuner officiel ou l'annonce d'une croisière en Sicile.

### Passage

S'il était obligé de donner des excuses aux gens du monde pour ne pas leur faire de visites, c'était de lui en faire qu'il cherchait à s'excuser auprès d'Odette. Encore les payait-il (se demandant à la fin du mois, pour peu qu'il eût un peu abusé de sa patience et fût allé souvent la voir, si c'était assez de lui envoyer quatre mille francs), et pour chacune trouvait un prétexte, un présent à lui apporter, un renseignement dont elle avait besoin, Charlus qu'elle avait rencontré allant chez elle et qui avait exigé qu'il l'accompagnât. Et à défaut d'aucun, il priait Charlus de courir chez elle, de lui dire comme spontanément, au cours de la conversation, qu'il se rappelait avoir à parler à Swann, qu'elle voulût bien lui faire demander de passer tout de suite chez elle ; mais le plus souvent Swann attendait en vain et Charlus lui disait le soir que son moyen n'avait pas réussi. De sorte que si elle faisait maintenant de fréquentes absences, même à Paris, quand elle y restait, elle le voyait peu, et elle qui, quand elle l'aimait, lui disait : « Je suis toujours libre » et « Qu'est-ce que l'opinion des autres peut me faire ? », maintenant, chaque fois qu'il voulait la voir, elle invoquait les convenances ou prétextait des occupations. Quand il parlait d'aller à une fête de charité, à un vernissage, à une première, où elle serait, elle lui disait qu'il voulait afficher leur liaison, qu'il la traitait comme une fille. C'est au point que pour tâcher de n'être pas partout privé de la rencontrer, Swann qui savait qu'elle connaissait et affectionnait beaucoup mon grand-oncle Adolphe dont il avait été lui-même l'ami, alla le voir un jour dans son petit appartement de la rue de Bellechasse afin de lui demander d'user de son influence sur Odette. Comme elle prenait toujours, quand elle parlait à Swann de mon oncle, des airs poétiques, disant : « Ah ! lui, ce n'est pas comme toi, c'est une si belle chose, si grande, si jolie, que son amitié pour moi. Ce n'est pas lui qui me considérerait assez peu pour vouloir se montrer avec moi dans tous les lieux publics », Swann fut embarrassé et ne savait pas à quel ton il devait se hausser pour parler d'elle à mon oncle. Il posa d'abord l'excellence a priori d'Odette, l'axiome de sa supra-humanité séraphique, la révélation de ses vertus indémontrables et dont la notion ne pouvait dériver de l'expérience. « Je veux parler avec vous. Vous, vous savez quelle femme au-dessus de toutes les femmes, quel être adorable, quel ange est Odette. Mais vous savez ce que c'est que la vie de Paris. Tout le monde ne connaît pas Odette sous le jour où nous la connaissons vous et moi. Alors il y a des gens qui trouvent que je joue un rôle un peu ridicule ; elle ne peut même pas admettre que je la rencontre dehors, au théâtre. Vous, en qui elle a tant de confiance, ne pourriez-vous lui dire quelques mots pour moi, lui assurer qu'elle s'exagère le tort qu'un salut de moi lui cause ? »

Mon oncle conseilla à Swann de rester un peu sans voir Odette qui ne l'en aimerait que plus, et à Odette de laisser Swann la retrouver partout où cela lui plairait. Quelques jours après, Odette disait à Swann qu'elle venait d'avoir une déception en voyant que mon oncle était pareil à tous les hommes : il venait d'essayer de la prendre de force. Elle calma Swann qui au premier moment voulait aller provoquer mon oncle, mais il refusa de lui serrer la main quand il le rencontra. Il regretta d'autant plus cette brouille avec mon oncle Adolphe qu'il avait espéré, s'il l'avait revu quelquefois et avait pu causer en toute confiance avec lui, tâcher de tirer au clair certains bruits relatifs à la vie qu'Odette avait menée autrefois à Nice. Or mon oncle Adolphe y passait l'hiver. Et Swann pensait que c'était même peut-être là qu'il avait connu Odette. Le peu qui avait échappé à quelqu'un devant lui, relativement à un homme qui aurait été l'amant d'Odette, avait bouleversé Swann. Mais les choses qu'il aurait, avant de les connaître, trouvé le plus affreux d'apprendre et le plus impossible de croire, une fois qu'il les savait, elles étaient incorporées à tout jamais à sa tristesse, il les admettait, il n'aurait plus pu comprendre qu'elles n'eussent pas été. Seulement chacune opérait sur l'idée qu'il se faisait de sa maîtresse une retouche ineffaçable. Il crut même comprendre, une fois, que cette légèreté des moeurs d'Odette qu'il n'eût pas soupçonnée, était assez connue, et qu'à Bade et à Nice, quand elle y passait jadis plusieurs mois, elle avait eu une sorte de notoriété galante. Il chercha, pour les interroger, à se rapprocher de certains viveurs ; mais ceux-ci savaient qu'il connaissait Odette ; et puis il avait peur de les faire penser de nouveau à elle, de les mettre sur ses traces. Mais lui à qui jusque-là rien n'aurait pu paraître aussi fastidieux que tout ce qui se rapportait à la vie cosmopolite de Bade ou de Nice, apprenant qu'Odette avait peut-être fait autrefois la fête dans ces villes de plaisir, sans qu'il dût jamais arriver à savoir si c'était seulement pour satisfaire à des besoins d'argent que grâce à lui elle n'avait plus, ou à des caprices qui pouvaient renaître, maintenant il se penchait avec une angoisse impuissante, aveugle et vertigineuse vers l'abîme sans fond où étaient allées s'engloutir ces années du début du Septennat pendant lesquelles on passait l'hiver sur la promenade des Anglais, l'été sous les tilleuls de Bade, et il leur trouvait une profondeur douloureuse mais magnifique comme celle que leur eût prêtée un poète ; et il eût mis à reconstituer les petits faits de la chronique de la Côte d'Azur d'alors, si elle avait pu l'aider à comprendre quelque chose du sourire ou des regards – pourtant si honnêtes et si simples – d'Odette, plus de passion que l'esthéticien qui interroge les documents subsistant de la Florence du XVe siècle pour tâcher d'entrer plus avant dans l'âme de la Primavera, de la bella Vanna, ou de la Vénus, de Botticelli. Souvent sans lui rien dire il la regardait, il songeait ; elle lui disait : « Comme tu as l'air triste ! » Il n'y avait pas bien longtemps encore, de l'idée qu'elle était une créature bonne, analogue aux meilleures qu'il eût connues, il avait passé à l'idée qu'elle était une femme entretenue ; inversement il lui était arrivé depuis de revenir de l'Odette de Crécy, peut-être trop connue des fêtards, des hommes à femmes, à ce visage d'une expression parfois si douce, à cette nature si humaine. Il se disait : « Qu'est-ce que cela veut dire qu'à Nice tout le monde sache qui est Odette de Crécy ? Ces réputations-là, même vraies, sont faites avec les idées des autres » ; il pensait que cette légende – fût-elle authentique – était extérieure à Odette, n'était pas en elle comme une personnalité irréductible et malfaisante ; que la créature qui avait pu être amenée à mal faire, c'était une femme aux bons yeux, au coeur plein de pitié pour la souffrance, au corps docile qu'il avait tenu, qu'il avait serré dans ses bras et manié, une femme qu'il pourrait arriver un jour à posséder toute, s'il réussissait à se rendre indispensable à elle. Elle était là, souvent fatiguée, le visage vidé pour un instant de la préoccupation fébrile et joyeuse des choses inconnues qui faisaient souffrir Swann ; elle écartait ses cheveux avec ses mains ; son front, sa figure paraissaient plus larges ; alors, tout d'un coup, quelque pensée simplement humaine, quelque bon sentiment comme il en existe dans toutes les créatures, quand dans un moment de repos ou de repliement elles sont livrées à elles-mêmes, jaillissait de ses yeux comme un rayon jaune. Et aussitôt tout son visage s'éclairait comme une campagne grise, couverte de nuages qui soudain s'écartent, pour sa transfiguration, au moment du soleil couchant. La vie qui était en Odette à ce moment-là, l'avenir même qu'elle semblait rêveusement regarder, Swann aurait pu les partager avec elle ; aucune agitation mauvaise ne semblait y avoir laissé de résidu. Si rares qu'ils devinssent, ces moments-là ne furent pas inutiles. Par le souvenir Swann reliait ces parcelles, abolissait les intervalles, coulait comme en or une Odette de bonté et de calme pour laquelle il fit plus tard (comme on le verra dans la deuxième partie de cet ouvrage), des sacrifices que l'autre Odette n'eût pas obtenus. Mais que ces moments étaient rares, et que maintenant il la voyait peu ! Même pour leur rendez-vous du soir, elle ne lui disait qu'à la dernière minute si elle pourrait le lui accorder car, comptant qu'elle le trouverait toujours libre, elle voulait d'abord être certaine que personne d'autre ne lui proposerait de venir. Elle alléguait qu'elle était obligée d'attendre une réponse de la plus haute importance pour elle, et même si, après qu'elle avait fait venir Swann, des amis demandaient à Odette, quand la soirée était déjà commencée, de les rejoindre au théâtre ou à souper, elle faisait un bond joyeux et s'habillait à la hâte. Au fur et à mesure qu'elle avançait dans sa toilette, chaque mouvement qu'elle faisait rapprochait Swann du moment où il faudrait la quitter, où elle s'enfuirait d'un élan irrésistible ; et quand, enfin prête, plongeant une dernière fois dans son miroir ses regards tendus et éclairés par l'attention, elle remettait un peu de rouge à ses lèvres, fixait une mèche sur son front et demandait son manteau de soirée bleu ciel avec des glands d'or, Swann avait l'air si triste qu'elle ne pouvait réprimer un geste d'impatience et disait : « Voilà comme tu me remercies de t'avoir gardé jusqu'à la dernière minute. Moi qui croyais avoir fait quelque chose de gentil. C'est bon à savoir pour une autre fois ! » Parfois, au risque de la fâcher, il se promettait de chercher à savoir où elle était allée, il rêvait d'une alliance avec Forcheville qui peut-être aurait pu le renseigner. D'ailleurs quand il savait avec qui elle passait la soirée, il était bien rare qu'il ne pût pas découvrir dans toutes ses relations à lui quelqu'un qui connaissait, fût-ce indirectement, l'homme avec qui elle était sortie et pouvait facilement en obtenir tel ou tel renseignement. Et tandis qu'il écrivait à un de ses amis pour lui demander de chercher à éclaircir tel ou tel point, il éprouvait le repos de cesser de se poser ces questions sans réponses et de transférer à un autre la fatigue d'interroger. Il est vrai que Swann n'était guère plus avancé quand il avait certains renseignements. Savoir ne permet pas toujours d'empêcher, mais du moins les choses que nous savons, nous les tenons, sinon entre nos mains, du moins dans notre pensée où nous les disposons à notre gré, ce qui nous donne l'illusion d'une sorte de pouvoir sur elles. Il était heureux toutes les fois où Charlus était avec Odette. Entre Charlus et elle, Swann savait qu'il ne pouvait rien se passer, que quand Charlus sortait avec elle, c'était par amitié pour lui et qu'il ne ferait pas difficulté à lui raconter ce qu'elle avait fait. Quelquefois elle avait déclaré si catégoriquement à Swann qu'il lui était impossible de le voir un certain soir, elle avait l'air de tenir tant à une sortie, que Swann attachait une véritable importance à ce que Charlus fût libre de l'accompagner. Le lendemain, sans oser poser beaucoup de questions à Charlus, il le contraignait, en ayant l'air de ne pas bien comprendre ses premières réponses, à lui en donner de nouvelles, après chacune desquelles il se sentait plus soulagé, car il apprenait bien vite qu'Odette avait occupé sa soirée aux plaisirs les plus innocents. « Mais comment, mon petit Mémé, je ne comprends pas bien..., ce n'est pas en sortant de chez elle que vous êtes allés au musée Grévin ? Vous étiez allés ailleurs d'abord. Non ? Oh ! que c'est drôle ! Vous ne savez pas comme vous m'amusez, mon petit Mémé. Mais quelle drôle d'idée elle a eue d'aller ensuite au Chat Noir, c'est bien une idée d'elle... Non ? c'est vous ? C'est curieux. Après tout ce n'est pas une mauvaise idée, elle devait y connaître beaucoup de monde ? Non ? elle n'a parlé à personne ? C'est extraordinaire. Alors vous êtes restés là comme cela tous les deux tous seuls ? Je vois d'ici cette scène. Vous êtes gentil, mon petit Mémé, je vous aime bien. » Swann se sentait soulagé. Pour lui, à qui il était arrivé en causant avec des indifférents qu'il écoutait à peine, d'entendre quelquefois certaines phrases (celle-ci par exemple : « J'ai vu hier Mme de Crécy, elle était avec un monsieur que je ne connais pas »), phrases qui aussitôt dans le coeur de Swann passaient à l'état solide, s'y durcissaient comme une incrustation, le déchiraient, n'en bougeaient plus, qu'ils étaient doux au contraire ces mots : « Elle ne connaissait personne, elle n'a parlé à personne ! » comme ils circulaient aisément en lui, qu'ils étaient fluides, faciles, respirables ! Et pourtant au bout d'un instant il se disait qu'Odette devait le trouver bien ennuyeux pour que ce fussent là les plaisirs qu'elle préférait à sa compagnie. Et leur insignifiance, si elle le rassurait, lui faisait pourtant de la peine comme une trahison.
