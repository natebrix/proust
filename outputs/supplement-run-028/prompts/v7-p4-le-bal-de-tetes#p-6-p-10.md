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
      "canonical_name": "Bloch",
      "surface_forms": [
        "Bloch",
        "Jacques du Rozier"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bloch",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "La duchesse croit qu’on parle du père « car il n’a rien d’un jeune homme »; puis Bloch entre « j’eus de la peine à le reconnaître »: pseudonyme « Jacques du Rozier », « redoutable monocle » dispensant d’« exprimer l’esprit, la bienveillance, l’effort »; « ses traits n'exprimaient plus jamais rien »; « mine débile et opinante »; « il paraissait bien son âge » avec des signes « des hommes qui sont vieux ».",
      "explanation": "The passage belittles Bloch by highlighting marked aging and snobbish artificiality (monocle, English chic) that erases his lively expression and renders him unrecognizable; the duchess's prior remark reinforces this perception."
    }
  ],
  "status_effects": [
    {
      "character": "Bloch",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Locally, Bloch appears diminished: aged, frozen, and rendered ridiculous/empty by his preparations, which degrades the esteem attached to his person."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-6-p-10"
}

### Candidate characters

[
  "Legrandin",
  "M. de Marsantes",
  "Robert de Saint-Loup",
  "Swann",
  "duchesse de Guermantes",
  "le narrateur",
  "marquis de Bréauté",
  "marquis de Forestelle"
]

### Prior local context (optional)

Par tous ces côtés, une matinée comme celle où je me trouvais était quelque chose de beaucoup plus précieux qu'une image du passé, m'offrant comme toutes les images successives et que je n'avais jamais vues qui séparaient le passé du présent, mieux encore, le rapport qu'il y avait entre le présent et le passé ; elle était comme ce qu'on appelait autrefois une vue d'optique, mais une vue d'optique des années, la vue non d'un monument, mais d'une personne située dans la perspective déformante du Temps.

### Passage

Quant à la femme dont M. d'Argencourt avait été l'amant, elle n'avait pas beaucoup changé, si on tenait compte du temps passé, c'est-à-dire que son visage n'était pas trop complètement démoli pour celui d'un être qui se déforme tout le long de son trajet dans l'abîme où il est lancé, abîme dont nous ne pouvons exprimer la direction que par des comparaisons également vaines, puisque nous ne pouvons les emprunter qu'au monde de l'espace, et qui, que nous les orientions dans le sens de l'élévation, de la longueur ou de la profondeur, ont comme seul avantage de nous faire sentir que cette dimension inconcevable et sensible existe. La nécessité, pour donner un nom aux figures, de remonter effectivement le cours des années, me forçait, en réaction, de rétablir ensuite, en leur donnant leur place réelle, les années auxquelles je n'avais pensé. À ce point de vue, et pour ne pas me laisser tromper par l'identité apparente de l'espace, l'aspect tout nouveau d'un être comme M. d'Argencourt m'était une révélation frappante de cette réalité du millésime qui d'habitude nous reste abstraite, comme l'apparition de certains arbres nains ou des baobabs géants nous avertit du changement de latitude. Alors la vie nous apparaît comme la féerie où l'on voit d'acte en acte le bébé devenir adolescent, homme mûr et se courber vers la tombe. Et comme c'est par des changements perpétuels qu'on sent que ces êtres prélevés à des distances assez grandes sont si différents, on sent qu'on a suivi la même loi que ces créatures qui se sont tellement transformées qu'elles ne ressemblent plus, sans avoir cessé d'être – justement parce qu'elles n'ont pas cessé d'être – à ce que nous avons vu d'elles jadis.

Une jeune femme que j'avais connue autrefois, maintenant blanche et tassée en petite vieille maléfique, semblait indiquer qu'il est nécessaire que, dans le divertissement final d'une pièce, les êtres fussent travestis à ne pas les reconnaître. Mais son frère était resté si droit, si pareil à lui-même qu'on s'étonnait que sur sa figure jeune il eût fait passer au blanc sa moustache bien relevée. Les parties d'une blancheur de neige de barbes jusque-là entièrement noires rendaient mélancolique le paysage humain de cette matinée, comme les premières feuilles jaunes des arbres alors qu'on croyait encore pouvoir compter sur un long été, et qu'avant d'avoir commencé d'en profiter on voit que c'est déjà l'automne. Alors moi qui, depuis mon enfance, vivais au jour le jour, ayant reçu d'ailleurs de moi-même et des autres une impression définitive, je m'aperçus pour la première fois, d'après les métamorphoses qui s'étaient produites dans tous ces gens, du temps qui avait passé pour eux, ce qui me bouleversa par la révélation qu'il avait passé aussi pour moi. Et indifférente en elle-même, leur vieillesse me désolait en m'avertissant des approches de la mienne. Celles-ci me furent, du reste, proclamées coup sur coup par des paroles qui, à quelques minutes d'intervalle, vinrent me frapper comme les trompettes du Jugement. La première fut prononcée par la Mme de Guermantes ; je venais de la voir, passant entre une double haie de curieux qui, sans se rendre compte des merveilleux artifices de toilette et d'esthétique qui agissaient sur eux, émus devant cette tête rousse, ce corps saumoné émergeant à peine de ses ailerons de dentelle noire, et étranglé de joyaux, le regardaient, dans la sinuosité héréditaire de ses lignes, comme ils eussent fait de quelque vieux poisson sacré, chargé de pierreries, en lequel s'incarnait le Génie protecteur de la famille Guermantes. « Ah ! me dit-elle, quelle joie de vous voir, vous mon plus vieil ami. » Et, dans mon amour-propre de jeune homme de Combray qui ne m'étais jamais compté à aucun moment comme pouvant être un de ses amis, participant vraiment à la vraie vie mystérieuse qu'on menait chez les Guermantes, un de ses amis au même titre que M. de Bréauté, que M. de Forestelle, que Swann, que tous ceux qui étaient morts, j'aurais pu en être flatté, j'en étais surtout malheureux. « Son plus vieil ami ! me dis-je, elle exagère ; peut-être un des plus vieux, mais suis-je donc... » À ce moment un neveu du prince s'approcha de moi : « Vous qui êtes un vieux Parisien », me dit-il. Un instant après on me remit un mot. J'avais rencontré, en arrivant, un jeune Létourville, dont je ne savais plus très bien la parenté avec la duchesse mais qui me connaissait un peu. Il venait de sortir de Saint-Cyr, et, me disant que ce serait pour moi un gentil camarade comme avait été Saint-Loup, qui pourrait m'initier aux choses de l'armée, avec les changements qu'elle avait subis, je lui avais dit que je le retrouverais tout à l'heure et que nous prendrions rendez-vous pour dîner ensemble, ce dont il m'avait beaucoup remercié. Mais j'étais resté trop longtemps à rêver dans la bibliothèque et le petit mot qu'il avait laissé pour moi était pour me dire qu'il n'avait pu m'attendre et me laisser son adresse. La lettre de ce camarade rêvé finissait ainsi : « Avec tout le respect de votre petit ami, LÉTOURVILLE. » « Petit ami ! » C'est ainsi qu'autrefois j'écrivais aux gens qui avaient trente ans de plus que moi, à Legrandin par exemple. Quoi ! ce sous-lieutenant, que je me figurais mon camarade comme Saint-Loup, se disait mon petit ami. Mais alors il n'y avait donc pas que les méthodes militaires qui avaient changé depuis lors, et pour M. de Létourville j'étais donc, non un camarade, mais un vieux monsieur, et de M. de Létourville, dans la compagnie duquel je me figurais, moi, tel que je m'apparaissais à moi-même, un bon camarade, en étais-je donc séparé par l'écartement d'un invisible compas auquel je n'avais pas songé et qui me situait si loin du jeune sous-lieutenant qu'il semblait que pour celui qui se disait mon « petit ami » j'étais un vieux monsieur !

Presque aussitôt après quelqu'un parla de Bloch, je demandai si c'était du jeune homme ou du père (dont j'avais ignoré la mort, pendant la guerre, d'émotion, avait-on dit, de voir la France envahie). « Je ne savais pas qu'il eût des enfants, je ne le savais même pas marié, me dit la duchesse. Mais c'est évidemment du père que nous parlons, car il n'a rien d'un jeune homme, ajouta-t-elle en riant. Il pourrait avoir des fils qui seraient eux-mêmes déjà des hommes. » Et je compris qu'il s'agissait de mon camarade. Il entra, d'ailleurs, au bout d'un instant. J'eus de la peine à le reconnaître. D'ailleurs, il avait pris maintenant non seulement un pseudonyme, mais le nom de Jacques du Rozier, sous lequel il eût fallu le flair de mon grand'père pour reconnaître la douce vallée de l'Hébron et les chaînes d'Israël que mon ami semblait avoir définitivement rompues. Un chic anglais avait, en effet, complètement transformé sa figure et passé au rabot tout ce qui se pouvait effacer. Les cheveux, jadis bouclés, coiffés à plat avec une raie au milieu, brillaient de cosmétique. Son nez restait fort et rouge mais semblait plutôt tuméfié par une sorte de rhume permanent qui pouvait expliquer l'accent nasal dont il débitait paresseusement ses phrases, car il avait trouvé, de même qu'une coiffure appropriée à son teint, une voix à sa prononciation où le nasonnement d'autrefois prenait un air de dédain particulier qui allait avec les ailes enflammées de son nez. Et grâce à la coiffure, à la suppression des moustaches, à l'élégance du type, à la volonté, ce nez juif disparaissait comme semble presque droite une bossue bien arrangée. Mais surtout, dès que Bloch apparaissait, la signification de sa physionomie était changée par un redoutable monocle. La part de machinisme que ce monocle introduisait dans la figure de Bloch la dispensait de tous ces devoirs difficiles auxquels une figure humaine est soumise, devoir d'être belle, d'exprimer l'esprit, la bienveillance, l'effort. La seule présence de ce monocle dans la figure de Bloch dispensait d'abord de se demander si elle était jolie ou non, comme devant ces objets anglais dont un garçon dit, dans un magasin, que c'est le grand chic, après quoi on n'ose plus se demander si cela vous plaît. D'autre part, il s'installait derrière la glace de ce monocle dans une position aussi hautaine, distante et confortable que si ç'avait été la glace d'un huit ressorts, et, pour assortir la figure aux cheveux plats et au monocle, ses traits n'exprimaient plus jamais rien. Sur cette figure de Bloch je vis se superposer cette mine débile et opinante, ces frêles hochements de tête qui trouvent si vite leur cran d'arrêt, et où j'aurais reconnu la docte fatigue des vieillards aimables, si, d'autre part, je n'avais enfin reconnu devant moi mon ami et si mes souvenirs ne l'avaient animé de cet entrain juvénile et ininterrompu dont il semblait actuellement dépossédé. Pour moi qui l'avais connu au seuil de la vie, il était mon camarade, un adolescent dont je mesurais la jeunesse par celle que, n'ayant cru vivre depuis ce moment-là, je me donnais inconsciemment à moi-même. J'entendis dire qu'il paraissait bien son âge, je fus étonné de remarquer sur son visage quelques-uns de ces signes qui sont plutôt la caractéristique des hommes qui sont vieux. Je compris que c'est parce qu'il l'était en effet et que c'est avec des adolescents qui durent un assez grand nombre d'années que la vie fait ses vieillards.

Comme quelqu'un, entendant dire que j'étais souffrant, demanda si je ne craignais pas de prendre la grippe qui régnait à ce moment-là, un autre bienveillant me rassura en me disant : « Non, cela atteint plutôt les personnes encore jeunes, les gens de votre âge ne risquent plus grand'chose. » Et on assura que le personnel m'avait bien reconnu. Ils avaient chuchoté mon nom, et même « dans leur langage », raconta une dame, elle les avait entendus dire : « Voilà le Père... » (cette expression était suivie de mon nom. Et comme je n'avais pas d'enfant, elle ne pouvait se rapporter qu'à l'âge).

En attendant la Mme de Guermantes dire : « Comment, si j'ai connu le maréchal ? Mais j'ai connu des gens bien plus représentatifs, la duchesse de Galliera, Pauline de Périgord, Mgr Dupanloup », je regrettais naïvement de ne pas avoir connu moi-même ceux qu'elle appelait un reste d'ancien régime. J'aurais dû penser qu'on appelle ancien régime ce dont on n'a pu connaître que la fin ; c'est ainsi que ce que nous apercevons à l'horizon prend une grandeur mystérieuse et nous semble se refermer sur un monde qu'on ne reverra plus ; cependant nous avançons, et c'est bientôt nous-même qui sommes à l'horizon pour les générations qui sont derrière nous ; cependant l'horizon recule, et le monde, qui semblait fini, recommence. « J'ai même pu voir, quand j'étais jeune fille, ajouta Mme de Guermantes, la duchesse de Dino. Dame, vous savez que je n'ai plus vingt-cinq ans. » Ces derniers mots me fâchèrent. Elle ne devrait pas dire cela, ce serait bon pour une vieille femme. « Quant à vous, reprit-elle, vous êtes toujours le même, vous n'avez pour ainsi dire pas changé », me dit la duchesse, et cela me fit presque plus de peine que si elle m'avait parlé d'un changement, car cela prouvait, puisqu'il était extraordinaire qu'il s'en fût si peu produit, que bien du temps s'était écoulé. « Ami, me dit-elle, vous êtes étonnant, vous restez toujours jeune », expression si mélancolique puisqu'elle n'a de sens que si nous sommes, en fait sinon d'apparence, devenus vieux. Et elle me donna le dernier coup en ajoutant : « J'ai toujours regretté que vous ne vous soyez pas marié. Au fond, qui sait, c'est peut-être plus heureux. Vous auriez été d'âge à avoir des fils à la guerre, et s'ils avaient été tués, comme l'a été ce pauvre Saint-Loup de Saint-Loup (je pense encore souvent à lui), sensible comme vous êtes, vous ne leur auriez pas survécu. » Et je pus me voir, comme dans la première glace véridique que j'eusse rencontrée dans les yeux de vieillards restés jeunes, à leur avis, comme je le croyais moi-même de moi, et qui, quand je me citais à eux, pour entendre un démenti, comme exemple de vieux, n'avaient pas dans leurs regards, qui me voyaient tel qu'ils ne se voyaient pas eux-mêmes et tel que je les voyais, une seule protestation. Car nous ne voyions pas notre propre aspect, nos propres âges, mais chacun, comme un miroir opposé, voyait celui de l'autre. Et sans doute, à découvrir qu'ils ont vieilli, bien des gens eussent été moins tristes que moi. Mais d'abord il en est de la vieillesse comme de la mort, quelques-uns les affrontent avec indifférence, non pas parce qu'ils ont plus de courage que les autres, mais parce qu'ils ont moins d'imagination. Puis un homme qui depuis son enfance vise une même idée, auquel sa paresse même et jusqu'à son état de santé, en lui faisant remettre sans cesse les réalisations, annule chaque soir le jour écoulé et perdu, si bien que la maladie qui hâte le vieillissement de son corps retarde celui de son esprit, est plus surpris et plus bouleversé de voir qu'il n'a cessé de vivre dans le Temps, que celui qui vit peu en soi-même, se règle sur le calendrier, et ne découvre pas d'un seul coup le total des années dont il a poursuivi quotidiennement l'addition. Mais une raison plus grave expliquait mon angoisse ; je découvrais cette action destructrice du Temps au moment même où je voulais entreprendre de rendre claires, d'intellectualiser dans une oeuvre d'art, des réalités extra-temporelles.
