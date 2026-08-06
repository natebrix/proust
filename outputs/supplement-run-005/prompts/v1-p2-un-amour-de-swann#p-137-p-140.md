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
  "Odette": {
    "aliases": [
      "Odette",
      "Odette de Crécy",
      "Mme de Crécy",
      "Mme de Crecy"
    ]
  },
  "Mme Verdurin": {
    "aliases": [
      "Mme Verdurin",
      "Madame Verdurin",
      "la Patronne"
    ]
  },
  "M. Verdurin": {
    "aliases": [
      "M. Verdurin",
      "Monsieur Verdurin"
    ]
  },
  "docteur Cottard": {
    "aliases": [
      "le docteur",
      "Cottard",
      "le docteur Cottard"
    ]
  },
  "la jeune ouvriere": {
    "aliases": [
      "la jeune ouvrière",
      "la jeune ouvriere"
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
      "source": "Swann",
      "target": "Odette",
      "type": "prestige_association",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.86,
      "evidence": "« la ressemblance d’Odette avec la Zéphora … lui conférait à elle aussi une beauté, la rendait plus précieuse »; « Le mot d’“œuvre florentine” … elle s’imprégna de noblesse »; « cet amour assuré quand il eut à la place pour base les données d’une esthétique certaine »",
      "explanation": "Swann elevates Odette by associating her with Botticelli’s Zéphora, finding in this resemblance a justification grounded in high art; the narrator signals mild irony about this aesthetic rationalization."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Within Swann’s valuation, Odette gains marked beauty, preciousness, and ‘nobility’ through the Botticelli association."
    },
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "Swann becomes more emotionally bound—his love is ‘assured’ by the aesthetic framing—giving Odette greater leverage over him."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-137-p-140"
}

### Candidate characters

[
  "Rémi",
  "le peintre"
]

### Prior local context (optional)

Laissant à gauche, au rez-de-chaussée surélevé, la chambre à coucher d'Odette qui donnait derrière sur une petite rue parallèle, un escalier droit entre des murs peints de couleur sombre et d'où tombaient des étoffes orientales, des fils de chapelets turcs et une grande lanterne japonaise suspendue à une cordelette de soie (mais qui, pour ne pas priver les visiteurs des derniers conforts de la civilisation occidentale, s'éclairait au gaz) montait au salon et au petit salon. Ils étaient précédés d'un étroit vestibule dont le mur quadrillé d'un treillage de jardin, mais doré, était bordé dans toute sa longueur d'une caisse rectangulaire où fleurissaient comme dans une serre une rangée de ces gros chrysanthèmes encore rares à cette époque, mais bien éloignés cependant de ceux que les horticulteurs réussirent plus tard à obtenir. Swann était agacé par la mode qui depuis l'année dernière se portait sur eux, mais il avait eu plaisir, cette fois, à voir la pénombre de la pièce zébrée de rose, d'oranger et de blanc par les rayons odorants de ces astres éphémères qui s'allument dans les jours gris. Odette l'avait reçu en robe de chambre de soie rose, le cou et les bras nus. Elle l'avait fait asseoir près d'elle dans un des nombreux retraits mystérieux qui étaient ménagés dans les enfoncements du salon, protégés par d'immenses palmiers contenus dans des cache-pot de Chine, ou par des paravents auxquels étaient fixés des photographies, des noeuds de rubans et des éventails. Elle lui avait dit : « Vous n'êtes pas confortable comme cela, attendez, moi je vais bien vous arranger », et avec le petit rire vaniteux qu'elle aurait eu pour quelque invention particulière à elle, avait installé derrière la tête de Swann, sous ses pieds, des coussins de soie japonaise qu'elle pétrissait comme si elle avait été prodigue de ces richesses et insoucieuse de leur valeur. Mais quand le valet de chambre était venu apporter successivement les nombreuses lampes qui, presque toutes enfermées dans des potiches chinoises, brûlaient isolées ou par couples, toutes sur des meubles différents comme sur des autels et qui dans le crépuscule déjà presque nocturne de cette fin d'après-midi d'hiver avaient fait reparaître un coucher de soleil plus durable, plus rose et plus humain – faisant peut-être rêver dans la rue quelque amoureux arrêté devant le mystère de la présence que décelaient et cachaient à la fois les vitres rallumées – elle avait surveillé sévèrement du coin de l'oeil le domestique pour voir s'il les posait bien à leur place consacrée. Elle pensait qu'en en mettant une seule là où il ne fallait pas, l'effet d'ensemble de son salon eût été détruit, et son portrait, placé sur un chevalet oblique drapé de peluche, mal éclairé. Aussi suivait-elle avec fièvre les mouvements de cet homme grossier et le réprimanda-t-elle vivement parce qu'il avait passé trop près de deux jardinières qu'elle se réservait de nettoyer elle-même dans sa peur qu'on ne les abîmât et qu'elle alla regarder de près pour voir s'il ne les avait pas écornées. Elle trouvait à tous ses bibelots chinois des formes « amusantes », et aussi aux orchidées, aux catleyas surtout, qui étaient, avec les chrysanthèmes, ses fleurs préférées, parce qu'ils avaient le grand mérite de ne pas ressembler à des fleurs, mais d'être en soie, en satin. « Celle-là a l'air d'être découpée dans la doublure de mon manteau », dit-elle à Swann en lui montrant une orchidée, avec une nuance d'estime pour cette fleur si « chic », pour cette soeur élégante et imprévue que la nature lui donnait, si loin d'elle dans l'échelle des êtres et pourtant raffinée, plus digne que bien des femmes qu'elle lui fît une place dans son salon. En lui montrant tour à tour des chimères à langues de feu décorant une potiche ou brodées sur un écran, les corolles d'un bouquet d'orchidées, un dromadaire d'argent niellé aux yeux incrustés de rubis qui voisinait sur la cheminée avec un crapaud de jade, elle affectait tour à tour d'avoir peur de la méchanceté, ou de rire de la cocasserie des monstres, de rougir de l'indécence des fleurs et d'éprouver un irrésistible désir d'aller embrasser le dromadaire et le crapaud qu'elle appelait : « chéris ». Et ces affectations contrastaient avec la sincérité de certaines de ses dévotions, notamment à Notre-Dame du Laghet qui l'avait jadis, quand elle habitait Nice, guérie d'une maladie mortelle, et dont elle portait toujours sur elle une médaille d'or à laquelle elle attribuait un pouvoir sans limites. Odette fit à Swann « son » thé, lui demanda : « Citron ou crème ? » et comme il répondit « crème », lui dit en riant : « Un nuage ! » Et comme il le trouvait bon : « Vous voyez que je sais ce que vous aimez. » Ce thé en effet avait paru à Swann quelque chose de précieux comme à elle-même, et l'amour a tellement besoin de se trouver une justification, une garantie de durée, dans des plaisirs qui au contraire sans lui n'en seraient pas et finissent avec lui, que quand il l'avait quittée à sept heures pour rentrer chez lui s'habiller, pendant tout le trajet qu'il fit dans son coupé, ne pouvant contenir la joie que cet après-midi lui avait causée, il se répétait : « Ce serait bien agréable d'avoir ainsi une petite personne chez qui on pourrait trouver cette chose si rare, du bon thé. » Une heure après, il reçut un mot d'Odette, et reconnut tout de suite cette grande écriture dans laquelle une affectation de raideur britannique imposait une apparence de discipline à des caractères informes qui eussent signifié peut-être pour des yeux moins prévenus le désordre de la pensée, l'insuffisance de l'éducation, le manque de franchise et de volonté. Swann avait oublié son étui à cigarettes chez Odette. « Que n'y avez-vous oublié aussi votre coeur, je ne vous aurais pas laissé le reprendre. »

### Passage

Une seconde visite qu'il lui fit eut plus d'importance peut-être. En se rendant chez elle ce jour-là comme chaque fois qu'il devait la voir, d'avance il se la représentait ; et la nécessité où il était pour trouver jolie sa figure de limiter aux seules pommettes roses et fraîches, les joues qu'elle avait si souvent jaunes, languissantes, parfois piquées de petits points rouges, l'affligeait comme une preuve que l'idéal est inaccessible et le bonheur médiocre. Il lui apportait une gravure qu'elle désirait voir. Elle était un peu souffrante ; elle le reçut en peignoir de crêpe de Chine mauve, ramenant sur sa poitrine, comme un manteau, une étoffe richement brodée. Debout à côté de lui, laissant couler le long de ses joues ses cheveux qu'elle avait dénoués, fléchissant une jambe dans une attitude légèrement dansante pour pouvoir se pencher sans fatigue vers la gravure qu'elle regardait, en inclinant la tête, de ses grands yeux, si fatigués et maussades quand elle ne s'animait pas, elle frappa Swann par sa ressemblance avec cette figure de Zéphora, la fille de Jéthro, qu'on voit dans une fresque de la chapelle Sixtine. Swann avait toujours eu ce goût particulier d'aimer à retrouver dans la peinture des maîtres non pas seulement les caractères généraux de la réalité qui nous entoure, mais ce qui semble au contraire le moins susceptible de généralité, les traits individuels des visages que nous connaissons : ainsi, dans la matière d'un buste du doge Loredan par Antoine Rizzo, la saillie des pommettes, l'obliquité des sourcils, enfin la ressemblance criante de son cocher Rémi ; sous les couleurs d'un Ghirlandajo, le nez de M. de Palancy ; dans un portrait de Tintoret, l'envahissement du gras de la joue par l'implantation des premiers poils des favoris, la cassure du nez, la pénétration du regard, la congestion des paupières du docteur du Boulbon. Peut-être ayant toujours gardé un remords d'avoir borné sa vie aux relations mondaines, à la conversation, croyait-il trouver une sorte d'indulgent pardon à lui accordé par les grands artistes, dans ce fait qu'ils avaient eux aussi considéré avec plaisir, fait entrer dans leur oeuvre, de tels visages qui donnent à celle-ci un singulier certificat de réalité et de vie, une saveur moderne ; peut-être aussi s'était-il tellement laissé gagner par la frivolité des gens du monde qu'il éprouvait le besoin de trouver dans une oeuvre ancienne ces allusions anticipées et rajeunissantes à des noms propres d'aujourd'hui. Peut-être au contraire avait-il gardé suffisamment une nature d'artiste pour que ces caractéristiques individuelles lui causassent du plaisir en prenant une signification plus générale, dès qu'il les apercevait déracinées, délivrées, dans la ressemblance d'un portrait plus ancien avec un original qu'il ne représentait pas. Quoi qu'il en soit, et peut-être parce que la plénitude d'impressions qu'il avait depuis quelque temps, et bien qu'elle lui fût venue plutôt avec l'amour de la musique, avait enrichi même son goût pour la peinture, le plaisir fut plus profond et devait exercer sur Swann une influence durable qu'il trouva à ce moment-là dans la ressemblance d'Odette avec la Zéphora de ce Sandro di Mariano auquel on ne donne plus volontiers son surnom populaire de Botticelli depuis que celui-ci évoque au lieu de l'oeuvre véritable du peintre l'idée banale et fausse qui s'en est vulgarisée. Il n'estima plus le visage d'Odette selon la plus ou moins bonne qualité de ses joues et d'après la douceur purement carnée qu'il supposait devoir leur trouver en les touchant avec ses lèvres si jamais il osait l'embrasser, mais comme un écheveau de lignes subtiles et belles que ses regards dévidèrent, poursuivant la courbe de leur enroulement, rejoignant la cadence de la nuque à l'effusion des cheveux et à la flexion des paupières, comme en un portrait d'elle en lequel son type devenait intelligible et clair.

Il la regardait ; un fragment de la fresque apparaissait dans son visage et dans son corps, que dès lors il chercha toujours à y retrouver, soit qu'il fût auprès d'Odette, soit qu'il pensât seulement à elle, et bien qu'il ne tînt sans doute au chef-d'oeuvre florentin que parce qu'il le retrouvait en elle, pourtant cette ressemblance lui conférait à elle aussi une beauté, la rendait plus précieuse. Swann se reprocha d'avoir méconnu le prix d'un être qui eût paru adorable au grand Sandro, et il se félicita que le plaisir qu'il avait à voir Odette trouvât une justification dans sa propre culture esthétique. Il se dit qu'en associant la pensée d'Odette à ses rêves de bonheur, il ne s'était pas résigné à un pis-aller aussi imparfait qu'il l'avait cru jusqu'ici, puisqu'elle contentait en lui ses goûts d'art les plus raffinés. Il oubliait qu'Odette n'était pas plus pour cela une femme selon son désir, puisque précisément son désir avait toujours été orienté dans un sens opposé à ses goûts esthétiques. Le mot d'« oeuvre florentine » rendit un grand service à Swann. Il lui permit, comme un titre, de faire pénétrer l'image d'Odette dans un monde de rêves où elle n'avait pas eu accès jusqu'ici et où elle s'imprégna de noblesse. Et tandis que la vue purement charnelle qu'il avait eue de cette femme, en renouvelant perpétuellement ses doutes sur la qualité de son visage, de son corps, de toute sa beauté, affaiblissait son amour, ces doutes furent détruits, cet amour assuré quand il eut à la place pour base les données d'une esthétique certaine ; sans compter que le baiser et la possession qui semblaient naturels et médiocres s'ils lui étaient accordés par une chair abîmée, venant couronner l'adoration d'une pièce de musée, lui parurent devoir être surnaturels et délicieux.

Et quand il était tenté de regretter que depuis des mois il ne fît plus que voir Odette, il se disait qu'il était raisonnable de donner beaucoup de son temps à un chef-d'oeuvre inestimable, coulé pour une fois dans une matière différente et particulièrement savoureuse, en un exemplaire rarissime qu'il contemplait tantôt avec l'humilité, la spiritualité et le désintéressement d'un artiste, tantôt avec l'orgueil, l'égoïsme et la sensualité d'un collectionneur.

Il plaça sur sa table de travail, comme une photographie d'Odette, une reproduction de la fille de Jéthro. Il admirait les grands yeux, le délicat visage qui laissait deviner la peau imparfaite, les boucles merveilleuses des cheveux le long des joues fatiguées, et adaptant ce qu'il trouvait beau jusque-là d'une façon esthétique à l'idée d'une femme vivante, il le transformait en mérites physiques qu'il se félicitait de trouver réunis dans un être qu'il pourrait posséder. Cette vague sympathie qui nous porte vers un chef-d'oeuvre que nous regardons, maintenant qu'il connaissait l'original charnel de la fille de Jéthro, elle devenait un désir qui suppléa désormais à celui que le corps d'Odette ne lui avait pas d'abord inspiré. Quand il avait regardé longtemps ce Botticelli, il pensait à son Botticelli à lui qu'il trouvait plus beau encore et, approchant de lui la photographie de Zéphora, il croyait serrer Odette contre son coeur.
