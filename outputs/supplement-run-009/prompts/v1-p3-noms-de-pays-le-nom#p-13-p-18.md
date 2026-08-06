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
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
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
      "source": "collective_social_voice",
      "target": "Gilberte",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.86,
      "evidence": "« Brava ! Brava ! ... c'est chic, c'est crâne ... Vous êtes comme moi, fidèle quand même à nos vieux Champs-Élysées ; nous sommes deux intrépides. »",
      "explanation": "An old lady, speaking « au nom des Champs-Élysées silencieux », thanks and congratulates Gilberte for having come despite the bad weather, calling her intrepid and valorizing her bearing. It is a public act of admiration that locally raises Gilberte."
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
      "confidence": 0.82,
      "explanation": "The public praise, formulated in an aristocratic register, confers on Gilberte an extra measure of prestige and consideration on the spot."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-13-p-18"
}

### Candidate characters

[
  "Françoise",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Retournerait-elle seulement aux Champs-Élysées ? Le lendemain elle n'y était pas ; mais je l'y vis les jours suivants ; je tournais tout le temps autour de l'endroit où elle jouait avec ses amies, si bien qu'une fois où elles ne se trouvèrent pas en nombre pour leur partie de barres elle me fit demander si je voulais compléter leur camp, et je jouai désormais avec elle chaque fois qu'elle était là. Mais ce n'était pas tous les jours ; il y en avait où elle était empêchée de venir par ses cours, le catéchisme, un goûter, toute cette vie séparée de la mienne que par deux fois, condensée dans le nom de Gilberte, j'avais senti passer si douloureusement près de moi, dans le raidillon de Combray et sur la pelouse des Champs-Élysées. Ces jours-là, elle annonçait d'avance qu'on ne la verrait pas ; si c'était à cause de ses études, elle disait : « C'est rasant, je ne pourrai pas venir demain ; vous allez tous vous amuser sans moi », d'un air chagrin qui me consolait un peu ; mais en revanche quand elle était invitée à une matinée, et que, ne le sachant pas je lui demandais si elle viendrait jouer, elle me répondait : « J'espère bien que non ! J'espère bien que la mère du narrateur me laissera aller chez mon amie. » Du moins ces jours-là, je savais que je ne la verrais pas, tandis que d'autres fois, c'était à l'improviste que sa mère l'emmenait faire des courses avec elle, et le lendemain elle disait : « Ah ! oui, je suis sortie avec la mère du narrateur », comme une chose naturelle, et qui n'eût pas été pour quelqu'un le plus grand malheur possible. Il y avait aussi les jours de mauvais temps où son institutrice, qui pour elle-même craignait la pluie, ne voulait pas l'emmener aux Champs-Élysées.

### Passage

Aussi si le ciel était douteux, dès le matin je ne cessais de l'interroger et je tenais compte de tous les présages. Si je voyais la dame d'en face qui, près de la fenêtre, mettait son chapeau, je me disais : « Cette dame va sortir ; donc il fait un temps où l'on peut sortir : pourquoi Gilberte ne ferait-elle pas comme cette dame ? » Mais le temps s'assombrissait, ma mère disait qu'il pouvait se lever encore, qu'il suffirait pour cela d'un rayon de soleil, mais que plus probablement il pleuvrait ; et s'il pleuvait, à quoi bon aller aux Champs-Élysées ? Aussi depuis le déjeuner mes regards anxieux ne quittaient plus le ciel incertain et nuageux. Il restait sombre. Devant la fenêtre, le balcon était gris. Tout d'un coup, sur sa pierre maussade je ne voyais pas une couleur moins terne, mais je sentais comme un effort vers une couleur moins terne, la pulsation d'un rayon hésitant qui voudrait libérer sa lumière. Un instant après, le balcon était pâle et réfléchissant comme une eau matinale, et mille reflets de la ferronnerie de son treillage étaient venus s'y poser. Un souffle de vent les dispersait, la pierre s'était de nouveau assombrie, mais, comme apprivoisés, ils revenaient, elle recommençait imperceptiblement à blanchir et par un de ces crescendos continus comme ceux qui, en musique, à la fin d'une Ouverture, mènent une seule note jusqu'au fortissimo suprême en la faisant passer rapidement par tous les degrés intermédiaires, je la voyais atteindre à cet or inaltérable et fixe des beaux jours, sur lequel l'ombre découpée de l'appui ouvragé de la balustrade se détachait en noir comme une végétation capricieuse, avec une ténuité dans la délinéation des moindres détails qui semblait trahir une conscience appliquée, une satisfaction d'artiste, et avec un tel relief, un tel velours dans le repos de ses masses sombres et heureuses qu'en vérité ces reflets larges et feuillus qui reposaient sur ce lac de soleil semblaient savoir qu'ils étaient des gages de calme et de bonheur.

Lierre instantané, flore pariétaire et fugitive ! la plus incolore, la plus triste, au gré de beaucoup, de celles qui peuvent ramper sur le mur ou décorer la croisée ; pour moi, de toutes la plus chère depuis le jour où elle était apparue sur notre balcon, comme l'ombre même de la présence de Gilberte qui était peut-être déjà aux Champs-Élysées, et dès que j'y arriverais me dirait : « Commençons tout de suite à jouer aux barres, vous êtes dans mon camp » ; fragile, emportée par un souffle, mais aussi en rapport non pas avec la saison, mais avec l'heure ; promesse du bonheur immédiat que la journée refuse ou accomplira, et par là du bonheur immédiat par excellence, le bonheur de l'amour ; plus douce, plus chaude sur la pierre que n'est la mousse même ; vivace, à qui il suffit d'un rayon pour naître et faire éclore de la joie, même au coeur de l'hiver.

Et jusque dans ces jours où toute autre végétation a disparu, où le beau cuir vert qui enveloppe le tronc des vieux arbres est caché sous la neige, quand celle-ci cessait de tomber, mais que le temps restait trop couvert pour espérer que Gilberte sortît, alors tout d'un coup, faisant dire à ma mère : « Tiens voilà justement qu'il fait beau, vous pourriez peut-être essayer tout de même d'aller aux Champs-Élysées », sur le manteau de neige qui couvrait le balcon, le soleil apparu entrelaçait des fils d'or et brodait des reflets noirs. Ce jour-là nous ne trouvions personne, ou une seule fillette prête à partir qui m'assurait que Gilberte ne viendrait pas. Les chaises désertées par l'assemblée imposante mais frileuse des institutrices étaient vides. Seule, près de la pelouse, était assise une dame d'un certain âge qui venait par tous les temps, toujours hanarchée d'une toilette identique, magnifique et sombre, et pour faire la connaissance de laquelle j'aurais à cette époque sacrifié, si l'échange m'avait été permis, tous les plus grands avantages futurs de ma vie. Car Gilberte allait tous les jours la saluer ; elle demandait à Gilberte des nouvelles de « son amour de mère » ; et il me semblait que si je l'avais connue, j'avais été pour Gilberte quelqu'un de tout autre, quelqu'un qui connaissait les relations de ses parents. Pendant que ses petits-enfants jouaient plus loin, elle lisait toujours les Débats qu'elle appelait « mes vieux Débats » et, par genre aristocratique, disait en parlant du sergent de ville ou de la loueuse de chaises : « Mon vieil ami le sergent de ville », « la loueuse de chaises et moi qui sommes de vieux amis ».

Françoise avait trop froid pour rester immobile, nous allâmes jusqu'au pont de la Concorde voir la Seine prise, dont chacun et même les enfants s'approchaient sans peur comme d'une immense baleine échouée, sans défense, et qu'on allait dépecer. Nous revenions aux Champs-Élysées ; je languissais de douleur entre les chevaux de bois immobiles et la pelouse blanche prise dans le réseau noir des allées dont on avait enlevé la neige, et sur laquelle la statue avait à la main un jet de glace ajouté qui semblait l'explication de son geste. La vieille dame elle-même, ayant plié ses Débats, demanda l'heure à une bonne d'enfants qui passait et qu'elle remercia en lui disant : « Comme vous êtes aimable ! » puis, priant le cantonnier de dire à ses petits enfants de revenir, qu'elle avait froid, ajouta : « Vous serez mille fois bon. Vous savez que je suis confuse ! » Tout à coup l'air se déchira : entre le guignol et le cirque, à l'horizon embelli, sur le ciel entr'ouvert, je venais d'apercevoir, comme un signe fabuleux, le plumet bleu de Mademoiselle. Et déjà Gilberte courait à toute vitesse dans ma direction, étincelante et rouge sous un bonnet carré de fourrure, animée par le froid, le retard et le désir du jeu ; un peu avant d'arriver à moi, elle se laissa glisser sur la glace et, soit pour mieux garder son équilibre, soit parce qu'elle trouvait cela plus gracieux, ou par affectation du maintien d'une patineuse, c'est les bras grands ouverts qu'elle avançait en souriant, comme si elle avait voulu m'y recevoir. « Brava ! Brava ! ça c'est très bien, je dirais comme vous que c'est chic, que c'est crâne, si je n'étais pas d'un autre temps, du temps de l'ancien régime, s'écria la vieille dame prenant la parole au nom des Champs-Élysées silencieux pour remercier Gilberte d'être venue sans se laisser intimider par le temps. Vous êtes comme moi, fidèle quand même à nos vieux Champs-Élysées ; nous sommes deux intrépides. Si je vous disais que je les aime, même ainsi. Cette neige, vous allez rire de moi, ça me fait penser à de l'hermine ! » Et la vieille dame se mit à rire.

Le premier de ces jours – auxquels la neige, image des puissances qui pouvaient me priver de voir Gilberte, donnait la tristesse d'un jour de séparation et jusqu'à l'aspect d'un jour de départ, parce qu'il changeait la figure et empêchait presque l'usage du lieu habituel de nos seules entrevues, maintenant changé, tout enveloppé de housses – ce jour fit pourtant faire un progrès à mon amour, car il fut comme un premier chagrin qu'elle eût partagé avec moi. Il n'y avait que nous deux de notre bande, et être ainsi le seul qui fût avec elle, c'était non seulement comme un commencement d'intimité, mais aussi de sa part – comme si elle ne fût venue rien que pour moi par un temps pareil – cela me semblait aussi touchant que si, un de ces jours où elle était invitée à une matinée, elle y avait renoncé pour venir me retrouver aux Champs-Élysées ; je prenais plus de confiance en la vitalité et en l'avenir de notre amitié qui restait vivace au milieu de l'engourdissement, de la solitude et de la ruine des choses environnantes ; et tandis qu'elle me mettait des boules de neige dans le cou, je souriais avec attendrissement à ce qui me semblait à la fois une prédilection qu'elle me marquait, en me tolérant comme compagnon de voyage dans ce pays hivernal et nouveau, et une sorte de fidélité qu'elle me gardait au milieu du malheur. Bientôt l'une après l'autre, comme des moineaux hésitants, ses amies arrivèrent toutes noires sur la neige. Nous commençâmes à jouer et comme ce jour si tristement commencé devait finir dans la joie, comme je m'approchais, avant de jouer aux barres, de l'amie à la voix brève que j'avais entendue le premier jour crier le nom de Gilberte, elle me dit : « Non, non, on sait bien que vous aimez mieux être dans le camp de Gilberte, d'ailleurs, vous voyez elle vous fait signe. » Elle m'appelait en effet pour que je vinsse sur la pelouse de neige, dans son camp, dont le soleil en lui donnant les reflets roses, l'usure métallique des brocarts anciens, faisait un camp du drap d'or.

Ce jour que j'avais tant redouté fut au contraire un des seuls où je ne fus pas trop malheureux.
