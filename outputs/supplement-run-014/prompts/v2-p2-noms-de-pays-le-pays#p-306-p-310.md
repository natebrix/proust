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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [],
  "status_effects": [],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-306-p-310"
}

### Candidate characters

[
  "Mme de Villeparisis",
  "Odette",
  "Robert de Saint-Loup",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Le bonheur de connaître ces jeunes filles était-il donc irréalisable ? Certes ce n'eût pas été le premier de ce genre auquel j'eusse renoncé. Je n'avais qu'à me rappeler tant d'inconnues que, même à Balbec, la voiture s'éloignant à toute vitesse m'avait fait à jamais abandonner. Et même le plaisir que me donnait la petite bande, noble comme si elle était composée de vierges helléniques, venait de ce qu'elle avait quelque chose de la fuite des passantes sur la route. Cette fugacité des êtres qui ne sont pas connus de nous, qui nous forcent à démarrer de la vie habituelle où les femmes que nous fréquentons finissent par dévoiler leurs tares, nous met dans cet état de poursuite où rien n'arrête plus l'imagination. Or dépouiller d'elle nos plaisirs, c'est les réduire à eux-mêmes, à rien. Offertes chez une de ces entremetteuses que, par ailleurs, on a vu que je ne méprisais pas, retirées de l'élément qui leur donnait tant de nuances et de vague, ces jeunes filles m'eussent moins enchanté. Il faut que l'imagination, éveillée par l'incertitude de pouvoir atteindre son objet, crée un but qui nous cache l'autre, et en substituant au plaisir sensuel l'idée de pénétrer dans une vie, nous empêche de reconnaître ce plaisir, d'éprouver son goût véritable, de le restreindre à sa portée.

### Passage

Il faut qu'entre nous et le poisson qui si nous le voyions pour la première fois servi sur une table ne paraîtrait pas valoir les mille ruses et détours nécessaires pour nous emparer de lui, s'interpose, pendant les après-midi de pêche, le remous à la surface duquel viennent affleurer, sans que nous sachions bien ce que nous voulons en faire, le poli d'une chair, l'indécision d'une forme, dans la fluidité d'un transparent et mobile azur.

Ces jeunes filles bénéficiaient aussi de ce changement des proportions sociales caractéristiques de la vie des bains de mer. Tous les avantages qui dans notre milieu habituel nous prolongent, nous agrandissent, se trouvent là devenus invisibles, en fait supprimés ; en revanche les êtres à qui on suppose indûment de tels avantages ne s'avancent qu'amplifiés d'une étendue postiche. Elle rendait plus aisé que des inconnues, et ce jour-là ces jeunes filles, prissent à mes yeux une importance énorme, et impossible de leur faire connaître celle que je pouvais avoir.

Mais si la promenade de la petite bande avait pour elle de n'être qu'un extrait de la fuite innombrable de passantes, laquelle m'avait toujours troublé, cette fuite était ici ramenée à un mouvement tellement lent qu'il se rapprochait de l'immobilité. Or, précisément, que dans une phase aussi peu rapide, les visages non plus emportés dans un tourbillon, mais calmes et distincts, me parussent encore beaux, cela m'empêchait de croire, comme je l'avais fait si souvent quand m'emportait la voiture de Mme de Villeparisis, que, de plus près, si je me fusse arrêté un instant, tels détails, une peau grêlée, un défaut dans les ailes du nez, un regard benêt, la grimace du sourire, une vilaine taille, eussent remplacé dans le visage et dans le corps de la femme ceux que j'avais sans doute imaginés ; car il avait suffi d'une jolie ligne de corps, d'un teint frais entrevu, pour que de très bonne foi j'y eusse ajouté quelque ravissante épaule, quelque regard délicieux dont je portais toujours en moi le souvenir ou l'idée préconçue, ces déchiffrages rapides d'un être qu'on voit à la volée nous exposant ainsi aux mêmes erreurs que ces lectures trop rapides où, sur une seule syllabe et sans prendre le temps d'identifier les autres, on met à la place du mot qui est écrit un tout différent que nous fournit notre mémoire. Il ne pouvait en être ainsi maintenant. J'avais bien regardé leurs visages ; chacun d'eux je l'avais vu, non pas dans tous ses profils, et rarement de face, mais tout de même selon deux ou trois aspects assez différents pour que je pusse faire soit la rectification, soit la vérification et la « preuve » des différentes suppositions de lignes et de couleurs que hasarde la première vue, et pour voir subsister en eux, à travers les expressions successives, quelque chose d'inaltérablement matériel. Aussi, je pouvais me dire avec certitude que, ni à Paris, ni à Balbec, dans les hypothèses les plus favorables de ce qu'auraient pu être, même si j'avais pu rester à causer avec elles, les passantes qui avaient arrêté mes yeux, il n'y en avait jamais eu dont l'apparition, puis la disparition sans que je les eusse connues, m'eussent laissé plus de regrets que ne feraient celles-ci, m'eussent donné l'idée que leur amitié pût être une telle ivresse. Ni parmi les actrices, ou les paysannes, ou les demoiselles du pensionnat religieux, je n'avais rien vu d'aussi beau, imprégné d'autant d'inconnu, aussi inestimablement précieux, aussi vraisemblablement inaccessible. Elles étaient, du bonheur inconnu et possible de la vie, un exemplaire si délicieux et en si parfait état, que c'était presque pour des raisons intellectuelles que j'étais désespéré, de peur de ne pas pouvoir faire dans des conditions uniques, ne laissant aucune place à l'erreur possible, l'expérience de ce que nous offre de plus mystérieux la beauté qu'on désire et qu'on se console de ne posséder jamais, en demandant du plaisir – comme Swann avait toujours refusé de faire, avant Odette – à des femmes qu'on n'a pas désirées, si bien qu'on meurt sans avoir jamais su ce qu'était cet autre plaisir. Sans doute, il se pouvait qu'il ne fût pas en réalité un plaisir inconnu, que de près son mystère se dissipât, qu'il ne fût qu'une projection, qu'un mirage du désir. Mais, dans ce cas, je ne pourrais m'en prendre qu'à la nécessité d'une loi de la nature – qui, si elle s'appliquait à ces jeunes filles, s'appliquerait à toutes – et non à la défectuosité de l'objet. Car il était celui que j'eusse choisi entre tous, me rendant bien compte, avec une satisfaction de botaniste, qu'il n'était pas possible de trouver réunies des espèces plus rares que celles de ces jeunes fleurs qui interrompaient en ce moment devant moi la ligne du flot de leur haie légère, pareille à un bosquet de roses de Pennsylvanie, ornement d'un jardin sur la falaise, entre lesquelles tient tout le trajet de l'océan parcouru par quelque steamer, si lent à glisser sur le trait horizontal et bleu qui va d'une tige à l'autre, qu'un papillon paresseux, attardé au fond de la corolle que la coque du navire a depuis longtemps dépassée, peut pour s'envoler en étant sûr d'arriver avant le vaisseau, attendre que rien qu'une seule parcelle azurée sépare encore la proue de celui-ci du premier pétale de la fleur vers laquelle il navigue.

Je rentrai parce que je devais aller dîner à Rivebelle avec Saint-Loup et que ma grand'mère exigeait qu'avant de partir, je m'étendisse ces soirs-là pendant une heure sur mon lit, sieste que le médecin de Balbec m'ordonna bientôt d'étendre à tous les autres soirs.

D'ailleurs, il n'y avait même pas besoin pour rentrer de quitter la digue et de pénétrer dans l'hôtel par le hall, c'est-à-dire par derrière. En vertu d'une avance comparable à celle du samedi où à Combray on déjeunait une heure plus tôt, maintenant avec le plein de l'été les jours étaient devenus si longs que le soleil était encore haut dans le ciel, comme à une heure de goûter, quand on mettait le couvert pour le dîner au Grand-Hôtel de Balbec. Aussi les grandes fenêtres vitrées et à coulisses restaient-elles ouvertes de plain-pied avec la digue. Je n'avais qu'à enjamber un mince cadre de bois pour me trouver dans la salle à manger que je quittais aussitôt pour prendre l'ascenseur.
