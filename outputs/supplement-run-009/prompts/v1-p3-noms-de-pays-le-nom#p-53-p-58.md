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
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "Odette apparaît comme une « arrivée souveraine », « reposait avec abandon » dans une victoria incomparable; « toutes les têtes étaient tournées vers elle »; « Ceux même qui ne la connaissaient pas [...] se demandaient: “Qui est-ce ?” »",
      "explanation": "The narrator magnifies Odette as queen of the Bois de Boulogne, emphasizing the public attention and admiration that surround her and place her above the other female walkers."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Odette is locally elevated as a sovereign and noticed figure, benefiting from collective admiration."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-53-p-58"
}

### Candidate characters

[
  "Françoise",
  "Gilberte",
  "Swann",
  "la Berma",
  "le narrateur"
]

### Prior local context (optional)

Ni elle d'ailleurs ni mon père ne semblaient non plus trouver à parler des grands-parents de Swann, du titre d'agent de change honoraire, un plaisir qui passât tous les autres. Mon imagination avait isolé et consacré dans le Paris social une certaine famille, comme elle avait fait dans le Paris de pierre pour une certaine maison dont elle avait sculpté la porte cochère et rendu précieuses les fenêtres. Mais ces ornements, j'étais seul à les voir. De même que mon père et la mère du narrateur trouvaient la maison qu'habitait Swann pareille aux autres maisons construites en même temps dans le quartier du Bois, de même la famille de Swann leur semblait du même genre que beaucoup d'autres familles d'agents de change. Ils la jugeaient plus ou moins favorablement selon le degré où elle avait participé à des mérites communs au reste de l'univers et ne lui trouvaient rien d'unique. Ce qu'au contraire ils y appréciaient, ils le rencontraient à un degré égal, ou plus élevé, ailleurs. Aussi après avoir trouvé la maison bien située, ils parlaient d'une autre qui l'était mieux, mais qui n'avait rien à voir avec Gilberte, ou de financiers d'un cran supérieur à son grand-père ; et s'ils avaient eu l'air un moment d'être du même avis que moi, c'était par un malentendu qui ne tardait pas à se dissiper. C'est que, pour percevoir dans tout ce qui entourait Gilberte, une qualité inconnue analogue dans le monde des émotions à ce que peut être dans celui des couleurs l'infra-rouge, mes parents étaient dépourvus de ce sens supplémentaire et momentané dont m'avait doté l'amour.

### Passage

Les jours où Gilberte m'avait annoncé qu'elle ne devait pas venir aux Champs-Élysées, je tâchais de faire des promenades qui me rapprochassent un peu d'elle. Parfois j'emmenais Françoise en pèlerinage devant la maison qu'habitaient les Swann. Je lui faisais répéter sans fin ce que, par l'institutrice, elle avait appris relativement à Odette. « Il paraît qu'elle a bien confiance à des médailles. Jamais elle ne partira en voyage si elle a entendu la chouette, ou bien comme un tic-tac d'horloge dans le mur, ou si elle a vu un chat à minuit, ou si le bois d'un meuble, il a craqué. Ah ! c'est une personne très croyante ! » J'étais si amoureux de Gilberte que si sur le chemin j'apercevais leur vieux maître d'hôtel promenant un chien, l'émotion m'obligeait à m'arrêter, j'attachais sur ses favoris blancs des regards pleins de passion. Françoise me disait :

– Qu'est-ce que vous avez ?

Puis, nous poursuivions notre route jusque devant leur porte cochère où un concierge différent de tout concierge, et pénétré jusque dans les galons de sa livrée du même charme douloureux que j'avais ressenti dans le nom de Gilberte, avait l'air de savoir que j'étais de ceux à qui une indignité originelle interdirait toujours de pénétrer dans la vie mystérieuse qu'il était chargé de garder et sur laquelle les fenêtres de l'entresol paraissaient conscientes d'être refermées, ressemblant beaucoup moins entre la noble retombée de leurs rideaux de mousseline à n'importe quelles autres fenêtres, qu'aux regards de Gilberte. D'autres fois nous allions sur les boulevards et je me postais à l'entrée de la rue Duphot ; on m'avait dit qu'on pouvait souvent y voir passer Swann se rendant chez son dentiste ; et mon imagination différenciait tellement le père de Gilberte du reste de l'humanité, sa présence au milieu du monde réel y introduisait tant de merveilleux, que, avant même d'arriver à la Madeleine, j'étais ému à la pensée d'approcher d'une rue où pouvait se produire inopinément l'apparition surnaturelle.

Mais le plus souvent – quand je ne devais pas voir Gilberte – comme j'avais appris que Odette se promenait presque chaque jour dans l'allée « des Acacias », autour du grand Lac, et dans l'allée de la « Reine Marguerite », je dirigeais Françoise du côté du bois de Boulogne. Il était pour moi comme ces jardins zoologiques où l'on voit rassemblés des flores diverses et des paysages opposés ; où, après une colline on trouve une grotte, un pré, des rochers, une rivière, une fosse, une colline, un marais, mais où l'on sait qu'ils ne sont là que pour fournir aux ébats de l'hippopotame, des zèbres, des crocodiles, des lapins russes, des ours et du héron, un milieu approprié ou un cadre pittoresque ; lui, le Bois, complexe aussi, réunissant des petits mondes divers et clos – faisant succéder quelque ferme plantée d'arbres rouges, de chênes d'Amérique, comme une exploitation agricole dans la Virginie, à une sapinière au bord du lac, ou à une futaie d'où surgit tout à coup dans sa souple fourrure, avec les beaux yeux d'une bête, quelque promeneuse rapide – il était le Jardin des femmes ; et – comme l'allée de Myrtes de l'Énéide – plantée pour elles d'arbres d'une seule essence, l'allée des Acacias était fréquentée par les Beautés célèbres. Comme, de loin, la culmination du rocher d'où elle se jette dans l'eau, transporte de joie les enfants qui savent qu'ils vont voir l'otarie, bien avant d'arriver à l'allée des Acacias, leur parfum qui, irradiant alentour, faisait sentir de loin l'approche et la singularité d'une puissante et molle individualité végétale ; puis, quand je me rapprochais, le faîte aperçu de leur frondaison légère et mièvre, d'une élégance facile, d'une coupe coquette et d'un mince tissu, sur laquelle des centaines de fleurs s'étaient abattues comme des colonies ailées et vibratiles de parasites précieux ; enfin jusqu'à leur nom féminin, désoeuvré et doux, me faisaient battre le coeur mais d'un désir mondain, comme ces valses qui ne nous évoquent plus que le nom des belles invitées que l'huissier annonce à l'entrée d'un bal. On m'avait dit que je verrais dans l'allée certaines élégantes que, bien qu'elles n'eussent pas toutes été épousées, l'on citait habituellement à côté de Odette, mais le plus souvent sous leur nom de guerre ; leur nouveau nom, quand il y en avait un, n'était qu'une sorte d'incognito que ceux qui voulaient parler d'elles avaient soin de lever pour se faire comprendre. Pensant que le Beau – dans l'ordre des élégances féminines – était régi par des lois occultes à la connaissance desquelles elles avaient été initiées, et qu'elles avaient le pouvoir de le réaliser, j'acceptais d'avance comme une révélation l'apparition de leur toilette, de leur attelage, de mille détails au sein desquels je mettais ma croyance comme une âme intérieure qui donnait la cohésion d'un chef-d'oeuvre à cet ensemble éphémère et mouvant. Mais c'est Odette que je voulais voir, et j'attendais qu'elle passât, ému comme si ç'avait été Gilberte, dont les parents, imprégnés, comme tout ce qui l'entourait, de son charme, excitaient en moi autant d'amour qu'elle, même un trouble plus douloureux (parce que leur point de contact avec elle était cette partie intestine de sa vie qui m'était interdite), et enfin (car je sus bientôt, comme on le verra, qu'ils n'aimaient pas que je jouasse avec elle), ce sentiment de vénération que nous vouons toujours à ceux qui exercent sans frein la puissance de nous faire du mal.

J'assignais la première place à la simplicité, dans l'ordre des mérites esthétiques et des grandeurs mondaines, quand j'apercevais Odette à pied, dans une polonaise de drap, sur la tête un petit toquet agrémenté d'une aile de lophophore, un bouquet de violettes au corsage, pressée, traversant l'allée des Acacias comme si ç'avait été seulement le chemin le plus court pour rentrer chez elle et répondant d'un clin d'oeil aux messieurs en voiture qui, reconnaissant de loin sa silhouette, la saluaient et se disaient que personne n'avait autant de chic. Mais au lieu de la simplicité, c'est le faste que je mettais au plus haut rang, si, après que j'avais forcé Françoise, qui n'en pouvait plus et disait que les jambes « lui rentraient », à faire les cent pas pendant une heure, je voyais enfin, débouchant de l'allée qui vient de la Porte Dauphine – image pour moi d'un prestige royal, d'une arrivée souveraine telle qu'aucune reine véritable n'a pu m'en donner l'impression dans la suite, parce que j'avais de leur pouvoir une notion moins vague et plus expérimentale – emportée par le vol de deux chevaux ardents, minces et contournés comme on en voit dans les dessins de Constantin Guys, portant établi sur son siège un énorme cocher fourré comme un cosaque, à côté d'un petit groom rappelant le « tigre » de « feu Baudenord », je voyais – ou plutôt je sentais imprimer sa forme dans mon coeur par une nette et épuisante blessure – une incomparable victoria, à dessein un peu haute et laissant passer à travers son luxe « dernier cri » des allusions aux formes anciennes, au fond de laquelle reposait avec abandon Odette, ses cheveux maintenant blonds avec une seule mèche grise ceints d'un mince bandeau de fleurs, le plus souvent des violettes, d'où descendaient de longs voiles, à la main une ombrelle mauve, aux lèvres un sourire ambigu où je ne voyais que la bienveillance d'une Majesté et où il y avait surtout la provocation de la cocotte, et qu'elle inclinait avec douceur sur les personnes qui la saluaient. Ce sourire en réalité disait aux uns : « Je me rappelle très bien, c'était exquis ! » ; à d'autres : « Comme j'aurais aimé ! ç'a été la mauvaise chance ! » ; à d'autres : « Mais si vous voulez ! Je vais suivre encore un moment la file et dès que je pourrai, je couperai. » Quand passaient des inconnus, elle laissait cependant autour de ses lèvres un sourire oisif, comme tourné vers l'attente ou le souvenir d'un ami et qui faisait dire : « Comme elle est belle ! » Et pour certains hommes seulement elle avait un sourire aigre, contraint, timide et froid et qui signifiait : « Oui, rosse, je sais que vous avez une langue de vipère, que vous ne pouvez pas vous tenir de parler ! Est-ce que je m'occupe de vous, moi ! » Coquelin passait en discourant au milieu d'amis qui l'écoutaient et faisait avec la main, à des personnes en voiture, un large bonjour de théâtre. Mais je ne pensais qu'à Odette et je faisais semblant de ne pas l'avoir vue, car je savais qu'arrivée à la hauteur du Tir aux pigeons elle dirait à son cocher de couper la file et de l'arrêter pour qu'elle pût descendre l'allée à pied. Et les jours où je me sentais le courage de passer à côté d'elle, j'entraînais Françoise dans cette direction. À un moment en effet, c'est dans l'allée des piétons, marchant vers nous que j'apercevais Odette laissant s'étaler derrière elle la longue traîne de sa robe mauve, vêtue, comme le peuple imagine les reines, d'étoffes et de riches atours que les autres femmes ne portaient pas, abaissant parfois son regard sur le manche de son ombrelle, faisant peu attention aux personnes qui passaient, comme si sa grande affaire et son but avaient été de prendre de l'exercice, sans penser qu'elle était vue et que toutes les têtes étaient tournées vers elle. Parfois pourtant, quand elle s'était retournée pour appeler son lévrier, elle jetait imperceptiblement un regard circulaire autour d'elle.

Ceux même qui ne la connaissaient pas étaient avertis par quelque chose de singulier et d'excessif – ou peut-être par une radiation télépathique comme celles qui déchaînaient des applaudissements dans la foule ignorante aux moments où la Berma était sublime – que ce devait être quelque personne connue. Ils se demandaient : « Qui est-ce ? », interrogeaient quelquefois un passant, ou se promettaient de se rappeler la toilette comme un point de repère pour des amis plus instruits qui les renseigneraient aussitôt. D'autres promeneurs, s'arrêtant à demi, disaient :
