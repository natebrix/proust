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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "l'Ambassadeur",
        "le marquis"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Norpois",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.78,
      "evidence": "« tombant comme le marteau du commissaire-priseur, ou comme un oracle de Delphes, la voix de l'Ambassadeur… »; il donne sa carte pour obtenir des conseils, oriente des placements « de tout premier ordre », et sa parole tombe après une immobilité de buste antique.",
      "explanation": "The narrator stages Norpois's authority and rhetorical mastery: he decides, advises, wields authority, and imposes the rhythm of the exchanges. The presentation is admiring but tinged with irony."
    }
  ],
  "status_effects": [
    {
      "character": "Norpois",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "In this passage, Norpois dominates the conversation and sets himself up as a competent arbiter whose word decides and directs the others."
    }
  ],
  "ambiguities": [
    "Le ton du narrateur mêle admiration et ironie (comparaisons à un buste antique, un oracle), rendant incertaine la part d’adhésion à l’élévation de Norpois."
  ],
  "unit_id": "v2-p1-autour-de-mme-swann#p-16-p-24"
}

### Candidate characters

[
  "la Berma",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Enfin éclata mon premier sentiment d'admiration : il fut provoqué par les applaudissements frénétiques des spectateurs. J'y mêlai les miens en tâchant de les prolonger, afin que, par reconnaissance, la Berma se surpassant, je fusse certain de l'avoir entendue dans un de ses meilleurs jours. Ce qui est du reste curieux, c'est que le moment où se déchaîna cet enthousiasme du public, fut, je l'ai su depuis, celui où la Berma a une de ses plus belles trouvailles. Il semble que certaines réalités transcendantes émettent autour d'elles des rayons auxquels la foule est sensible. C'est ainsi que, par exemple, quand un événement se produit, quand à la frontière une armée est en danger, ou battue, ou victorieuse, les nouvelles assez obscures qu'on reçoit et d'où l'homme cultivé ne sait pas tirer grand'chose excitent dans la foule une émotion qui le surprend et dans laquelle, une fois que les experts l'ont mis au courant de la véritable situation militaire, il reconnaît la perception par le peuple de cette « aura » qui entoure les grands événements et qui peut être visible à des centaines de kilomètres. On apprend la victoire, ou après-coup quand la guerre est finie, ou tout de suite par la joie du concierge. On découvre un trait génial du jeu de la Berma huit jours après l'avoir entendue, par la critique, ou sur le coup par les acclamations du parterre. Mais cette connaissance immédiate de la foule étant mêlée à cent autres toutes erronées, les applaudissements tombaient le plus souvent à faux, sans compter qu'ils étaient mécaniquement soulevés par la force des applaudissements antérieurs comme dans une tempête, une fois que la mer a été suffisamment remuée, elle continue à grossir, même si le vent ne s'accroît plus. N'importe, au fur et à mesure que j'applaudissais, il me semblait que la Berma avait mieux joué. « Au moins, disait à côté de moi une femme assez commune, elle se dépense celle-là, elle se frappe à se faire mal, elle court, parlez-moi de ça, c'est jouer. » Et heureux de trouver ces raisons de la supériorité de la Berma, tout en me doutant qu'elles ne l'expliquaient pas plus que celle de la Joconde, ou du Persée de Benvenuto, l'exclamation d'un paysan : « C'est bien fait tout de même ! c'est tout en or, et du beau ! quel travail ! », je partageai avec ivresse le vin grossier de cet enthousiasme populaire. Je n'en sentis pas moins, le rideau tombé, un désappointement que ce plaisir que j'avais tant désiré n'eût pas été plus grand, mais en même temps le besoin de le prolonger, de ne pas quitter pour jamais, en sortant de la salle, cette vie du théâtre qui pendant quelques heures avait été la mienne, et dont je me serais arraché comme en un départ pour l'exil, en rentrant directement à la maison, si je n'avais espéré d'y apprendre beaucoup sur la Berma par son admirateur auquel je devais qu'on m'eût permis d'aller à Phèdre, Norpois. Je lui fus présenté avant le dîner par mon père qui m'appela pour cela dans son cabinet. À mon entrée, l'Ambassadeur se leva, me tendit la main, inclina sa haute taille et fixa attentivement sur moi ses yeux bleus. Comme les étrangers de passage qui lui étaient présentés, au temps où il représentait la France, étaient plus ou moins – jusqu'aux chanteurs connus – des personnes de marque et dont il savait alors qu'il pourrait dire plus tard, quand on prononcerait leur nom à Paris ou à Pétersbourg, qu'il se rappelait parfaitement la soirée qu'il avait passée avec eux à Munich ou à Sofia, il avait pris l'habitude de leur marquer par son affabilité la satisfaction qu'il avait de les connaître : mais de plus, persuadé que dans la vie des capitales, au contact à la fois des individualités intéressantes qui les traversent et des usages du peuple qui les habite, on acquiert une connaissance approfondie, et que les livres ne donnent pas, de l'histoire, de la géographie, des moeurs des différentes nations, du mouvement intellectuel de l'Europe, il exerçait sur chaque nouveau venu ses facultés aiguës d'observateur afin de savoir de suite à quelle espèce d'homme il avait à faire. Le gouvernement ne lui avait plus depuis longtemps confié de poste à l'étranger, mais dès qu'on lui présentait quelqu'un, ses yeux, comme s'ils n'avaient pas reçu notification de sa mise en disponibilité, commençaient à observer avec fruit, cependant que par toute son attitude il cherchait à montrer que le nom de l'étranger ne lui était pas inconnu. Aussi, tout en me parlant avec bonté et de l'air d'importance d'un homme qui sait sa vaste expérience, il ne cessait de m'examiner avec une curiosité sagace et pour son profit, comme si j'eusse été quelque usage exotique, quelque monument instructif, ou quelque étoile en tournée. Et de la sorte il faisait preuve à la fois, à mon endroit, de la majestueuse amabilité du sage Mentor et de la curiosité studieuse du jeune Anacharsis.

### Passage

Il ne m'offrit absolument rien pour la Revue des Deux-Mondes, mais me posa un certain nombre de questions sur ce qu'avaient été ma vie et mes études, sur mes goûts dont j'entendis parler pour la première fois comme s'il pouvait être raisonnable de les suivre, tandis que j'avais cru jusqu'ici que c'était un devoir de les contrarier. Puisqu'ils me portaient du côté de la littérature, il ne me détourna pas d'elle ; il m'en parla au contraire avec déférence comme d'une personne vénérable et charmante du cercle choisi de laquelle, à Rome ou à Dresde, on a gardé le meilleur souvenir et qu'on regrette par suite des nécessités de la vie de retrouver si rarement. Il semblait m'envier en souriant d'un air presque grivois les bons moments que, plus heureux que lui et plus libre, elle me ferait passer. Mais les termes mêmes dont il se servait me montraient la Littérature comme trop différente de l'image que je m'en étais faite à Combray, et je compris que j'avais eu doublement raison de renoncer à elle. Jusqu'ici je m'étais seulement rendu compte que je n'avais pas le don d'écrire ; maintenant Norpois m'en ôtait même le désir. Je voulus lui exprimer ce que j'avais rêvé ; tremblant d'émotion, je me serais fait un scrupule que toutes mes paroles ne fussent pas l'équivalent le plus sincère possible de ce que j'avais senti et que je n'avais jamais essayé de me formuler ; c'est dire que mes paroles n'eurent aucune netteté. Peut-être par habitude professionnelle, peut-être en vertu du calme qu'acquiert tout homme important dont on sollicite le conseil et qui, sachant qu'il gardera en mains la maîtrise de la conversation, laisse l'interlocuteur s'agiter, s'efforcer, peiner à son aise, peut-être aussi pour faire valoir le caractère de sa tête (selon lui grecque, malgré les grands favoris), Norpois, pendant qu'on lui exposait quelque chose, gardait une immobilité de visage aussi absolue que si vous aviez parlé devant quelque buste antique – et sourd – dans une glyptothèque. Tout à coup, tombant comme le marteau du commissaire-priseur, ou comme un oracle de Delphes, la voix de l'Ambassadeur qui vous répondait vous impressionnait d'autant plus que rien dans sa face ne vous avait laissé soupçonner le genre d'impression que vous aviez produit sur lui, ni l'avis qu'il allait émettre.

– Précisément, me dit-il tout à coup comme si la cause était jugée et après m'avoir laissé bafouiller en face des yeux immobiles qui ne me quittaient pas un instant, j'ai le fils d'un de mes amis qui, mutatis mutandis, est comme vous (et il prit pour parler de nos dispositions communes le même ton rassurant que si elles avaient été des dispositions non pas à la littérature, mais au rhumatisme et s'il avait voulu me montrer qu'on n'en mourait pas). Aussi a-t-il préféré quitter le quai d'Orsay où la voie lui était pourtant toute tracée par son père et, sans se soucier du qu'en dira-t-on, il s'est mis à produire. Il n'a certes pas lieu de s'en repentir. Il a publié il y a deux ans – il est d'ailleurs beaucoup plus âgé que vous, naturellement – un ouvrage relatif au sentiment de l'Infini sur la rive occidentale du lac Victoria-Nyanza et cette année un opuscule moins important, mais conduit d'une plume alerte, parfois même acérée, sur le fusil à répétition dans l'armée bulgare, qui l'ont mis tout à fait hors de pair. Il a déjà fait un joli chemin, il n'est pas homme à s'arrêter en route, et je sais que, sans que l'idée d'une candidature ait été envisagée, on a laissé tomber son nom deux ou trois dans la conversation, et d'une façon qui n'avait rien de défavorable, à l'Académie des Sciences morales. En somme, sans pouvoir dire encore qu'il soit au pinacle, il a conquis de haute lutte une fort jolie position et le succès qui ne va pas toujours qu'aux agités et aux brouillons, aux faiseurs d'embarras qui sont presque toujours des faiseurs, le succès a récompensé son effort.

Mon père, me voyant déjà académicien dans quelques années, respirait une satisfaction que Norpois porta à son comble quand, après un instant d'hésitation pendant lequel il sembla calculer les conséquences de son acte, il me dit, en me tendant sa carte : « Allez donc le voir de ma part, il pourra vous donner d'utiles conseils », me causant par ces mots une agitation aussi pénible que s'il m'avait annoncé qu'on m'embarquait le lendemain comme mousse à bord d'un voilier.

Ma tante Léonie m'avait fait héritier, en même temps que de beaucoup d'objets et de meubles fort embarrassants, de presque toute sa fortune liquide – révélant ainsi après sa mort une affection pour moi que je n'avais guère soupçonnée pendant sa vie. Mon père, qui devait gérer cette fortune jusqu'à ma majorité, consulta Norpois sur un certain nombre de placements. Il conseilla des titres à faible rendement qu'il jugeait particulièrement solides, notamment les Consolidés Anglais et le 4% Russe. « Avec ces valeurs de tout premier ordre, dit Norpois, si le revenu n'est pas très élevé, vous êtes du moins assuré de ne jamais voir fléchir le capital. » Pour le reste, mon père lui dit en gros ce qu'il avait acheté. Norpois eut un imperceptible sourire de félicitations : comme tous les capitalistes, il estimait la fortune une chose enviable, mais trouvait plus délicat de ne complimenter que par un signe d'intelligence à peine avoué, au sujet de celle qu'on possédait ; d'autre part, comme il était lui-même colossalement riche, il trouvait de bon goût d'avoir l'air de juger considérables les revenus moindres d'autrui, avec pourtant un retour joyeux et confortable sur la supériorité des siens. En revanche il n'hésita pas à féliciter mon père de la « composition » de son portefeuille « d'un goût très sûr, très délicat, très fin ». On aurait dit qu'il attribuait aux relations des valeurs de bourse entre elles, et même aux valeurs de bourse en elles-mêmes, quelque chose comme un mérite esthétique. D'une, assez nouvelle et ignorée, dont mon père lui parla, Norpois, pareil à ces gens qui ont lu des livres que vous vous croyez seul à connaître, lui dit : « Mais si, je me suis amusé pendant quelque temps à la suivre dans la Cote, elle était intéressante », avec le sourire rétrospectivement captivé d'un abonné qui a lu le dernier roman d'une revue, par tranches, en feuilleton. « Je ne vous déconseillerais pas de souscrire à l'émission qui va être lancée prochainement. Elle est attrayante, car on vous offre les titres à des prix tentants. » Pour certaines valeurs anciennes au contraire, mon père ne se rappelant plus exactement les noms, faciles à confondre avec ceux d'actions similaires, ouvrit un tiroir et montra les titres eux-mêmes à l'Ambassadeur. Leur vue me charma ; ils étaient enjolivés de flèches de cathédrales et de figures allégoriques comme certaines vieilles publications romantiques que j'avais feuilletées autrefois. Tout ce qui est d'un même temps se ressemble ; les artistes qui illustrent les poèmes d'une époque sont les mêmes que font travailler pour elles les Sociétés financières. Et rien ne fait mieux penser à certaines livraisons de Notre-Dame de Paris et d'oeuvres de Gérard de Nerval, telles qu'elles étaient accrochées à la devanture de l'épicerie de Combray, que, dans son encadrement rectangulaire et fleuri que supportaient des divinités fluviales, une action nominative de la Compagnie des Eaux.

Mon père avait pour mon genre d'intelligence un mépris suffisamment corrigé par la tendresse pour qu'au total, son sentiment sur tout ce que je faisais fût une indulgence aveugle. Aussi n'hésita-t-il pas à m'envoyer chercher un petit poème en prose que j'avais fait autrefois à Combray en revenant d'une promenade. Je l'avais écrit avec une exaltation qu'il me semblait devoir communiquer à ceux qui le liraient. Mais elle ne dut pas gagner Norpois, car ce fut sans me dire une parole qu'il me le rendit.

Ma mère, pleine de respect pour les occupations de mon père, vint demander, timidement, si elle pouvait faire servir. Elle avait peur d'interrompre une conversation où elle n'aurait pas eu à être mêlée. Et, en effet, à tout moment mon père rappelait au marquis quelque mesure utile qu'ils avaient décidé de soutenir à la prochaine séance de Commission, et il le faisait sur le ton particulier qu'ont ensemble dans un milieu différent – pareils en cela à deux collégiens – deux collègues à qui leurs habitudes professionnelles créent des souvenirs communs où n'ont pas accès les autres et auxquels ils s'excusent de se reporter devant eux.

Mais la parfaite indépendance des muscles du visage à laquelle Norpois était arrivé lui permettait d'écouter sans avoir l'air d'entendre. Mon père finissait par se troubler : « J'avais pensé à demander l'avis de la Commission... » disait-il à Norpois après de longs préambules. Alors du visage de l'aristocratique virtuose qui avait gardé l'inertie d'un instrumentiste dont le moment n'est pas venu d'exécuter sa partie sortait avec un débit égal, sur un ton aigu et comme ne faisant que finir, mais confiée cette fois à un autre timbre, la phrase commencée : « Que, bien entendu, vous n'hésiterez pas à réunir, d'autant plus que les membres vous sont individuellement connus et peuvent facilement se déplacer. » Ce n'était pas évidemment en elle-même une terminaison bien extraordinaire. Mais l'immobilité qui l'avait précédée la faisait se détacher avec la netteté cristalline, l'imprévu quasi malicieux de ces phrases par lesquelles le piano, silencieux jusque-là, réplique, au moment voulu, au violoncelle qu'on vient d'entendre, dans un concerto de Mozart.

– Hé bien, as-tu été content de ta matinée ? me dit mon père tandis qu'on passait à table, pour me faire briller en pensant que mon enthousiasme me ferait bien juger par Norpois. « Il est allé entendre la Berma tantôt, vous vous rappelez que nous en avions parlé ensemble », dit-il en se tournant vers le diplomate, du même ton d'allusion rétrospective, technique et mystérieuse que s'il se fût agi d'une séance de la Commission.

– Vous avez dû être enchanté, surtout si c'était la première fois que vous l'entendiez. Monsieur votre père s'alarmait du contre-coup que cette petite escapade pouvait avoir sur votre état de santé, car vous êtes un peu délicat, un peu frêle, je crois. Mais je l'ai rassuré. Les théâtres ne sont plus aujourd'hui ce qu'ils étaient il y a seulement vingt ans. Vous avez des sièges à peu près confortables, une atmosphère renouvelée, quoique nous ayons fort à faire encore pour rejoindre l'Allemagne et l'Angleterre, qui à cet égard comme à bien d'autres ont une formidable avance sur nous. Je n'ai pas vu Mme Berma dans Phèdre, mais j'ai entendu dire qu'elle y était admirable. Et vous avez été ravi, naturellement ?
