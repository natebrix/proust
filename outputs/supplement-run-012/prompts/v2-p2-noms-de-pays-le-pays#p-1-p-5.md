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
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« J'étais arrivé à une presque complète indifférence à l'égard de Gilberte »; « cette souffrance et ce regain d'amour … ne furent pas plus longs »; « Mon voyage à Balbec fut … la première sortie d'un convalescent … il est guéri. »",
      "explanation": "The narrator frames his near-complete indifference and subsequent stabilization at Balbec as recovery, which locally diminishes Gilberte’s emotional hold over him despite a brief, memory-triggered resurgence."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Within this passage Gilberte loses leverage over the narrator; his love has largely cooled and is stabilized by the change of habit at Balbec."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-1-p-5"
}

### Candidate characters

[
  "la grand-mère",
  "le directeur",
  "le narrateur"
]

### Prior local context (optional)

(none provided)

### Passage

J'étais arrivé à une presque complète indifférence à l'égard de Gilberte, quand deux ans plus tard je partis avec ma grand'mère pour Balbec. Quand je subissais le charme d'un visage nouveau, quand c'était à l'aide d'une autre jeune fille que j'espérais connaître les cathédrales gothiques, les palais et les jardins de l'Italie, je me disais tristement que notre amour, en tant qu'il est l'amour d'une certaine créature, n'est peut-être pas quelque chose de bien réel, puisque si des associations de rêveries agréables ou douloureuses peuvent le lier pendant quelque temps à une femme jusqu'à nous faire penser qu'il a été inspiré par elle d'une façon nécessaire, en revanche si nous nous dégageons volontairement ou à notre insu de ces associations, cet amour, comme s'il était au contraire spontané et venait de nous seuls, renaît pour se donner à une autre femme. Pourtant au moment de ce départ pour Balbec, et pendant les premiers temps de mon séjour, mon indifférence n'était encore qu'intermittente. Souvent (notre vie étant si peu chronologique, interférant tant d'anachronismes dans la suite des jours), je vivais dans ceux, plus anciens que la veille ou l'avant-veille, où j'aimais Gilberte. Alors ne plus la voir m'était soudain douloureux, comme c'eût été dans ce temps-là. Le moi qui l'avait aimée, remplacé déjà presque entièrement par un autre, resurgissait, et il m'était rendu beaucoup plus fréquemment par une chose futile que par une chose importante. Par exemple, pour anticiper sur mon séjour en Normandie, j'entendis à Balbec un inconnu que je croisai sur la digue dire : « La famille du directeur du ministère des Postes. » Or (comme je ne savais pas alors l'influence que cette famille devait avoir sur ma vie), ce propos aurait dû me paraître oiseux, mais il me causa une vive souffrance, celle qu'éprouvait un moi, aboli pour une grande part depuis longtemps, à être séparé de Gilberte. C'est que jamais je n'avais repensé à une conversation que Gilberte avait eue devant moi avec son père, relativement à la famille du « directeur du ministère des Postes ». Or, les souvenirs d'amour ne font pas exception aux lois générales de la mémoire, elles-mêmes régies par les lois plus générales de l'habitude. Comme celle-ci affaiblit tout, ce qui nous rappelle le mieux un être, c'est justement ce que nous avions oublié (parce que c'était insignifiant et que nous lui avions ainsi laissé toute sa force). C'est pourquoi la meilleure part de notre mémoire est hors de nous, dans un souffle pluvieux, dans l'odeur de renfermé d'une chambre ou dans l'odeur d'une première flambée, partout où nous retrouvons de nous-même ce que notre intelligence, n'en ayant pas l'emploi, avait dédaigné, la dernière réserve du passé, la meilleure, celle qui quand toutes nos larmes semblent taries, sait nous faire pleurer encore. Hors de nous ? En nous pour mieux dire, mais dérobée à nos propres regards, dans un oubli plus ou moins prolongé. C'est grâce à cet oubli seul que nous pouvons de temps à autre retrouver l'être que nous fûmes, nous placer vis-à-vis des choses comme cet être l'était, souffrir à nouveau, parce que nous ne sommes plus nous, mais lui, et qu'il aimait ce qui nous est maintenant indifférent. Au grand jour de la mémoire habituelle, les images du passé pâlissent peu à peu, s'effacent, il ne reste plus rien d'elles, nous ne le retrouverons plus. Ou plutôt nous ne le retrouverions plus, si quelques mots (comme « directeur au ministère des Postes ») n'avaient été soigneusement enfermés dans l'oubli, de même qu'on dépose à la Bibliothèque Nationale un exemplaire d'un livre qui sans cela risquerait de devenir introuvable.

Mais cette souffrance et ce regain d'amour pour Gilberte ne furent pas plus longs que ceux qu'on a en rêve, et cette fois, au contraire, parce qu'à Balbec l'Habitude ancienne n'était plus là pour les faire durer. Et si ces effets de l'Habitude semblent contradictoires, c'est qu'elle obéit à des lois multiples. À Paris j'étais devenu de plus en plus indifférent à Gilberte, grâce à l'Habitude. Le changement d'habitude, c'est-à-dire la cessation momentanée de l'Habitude, paracheva l'oeuvre de l'Habitude quand je partis pour Balbec. Elle affaiblit mais stabilise, elle amène la désagrégation mais la fait durer indéfiniment. Chaque jour depuis des années je calquais tant bien que mal mon état d'âme sur celui de la veille. À Balbec un lit nouveau à côté duquel on m'apportait le matin un petit déjeuner différent de celui de Paris ne devait plus soutenir les pensées dont s'était nourri mon amour pour Gilberte : il y a des cas (assez rares il est vrai) où, la sédentarité immobilisant les jours, le meilleur moyen de gagner du temps, c'est de changer de place. Mon voyage à Balbec fut comme la première sortie d'un convalescent qui n'attendait plus qu'elle pour s'apercevoir qu'il est guéri.

Ce voyage, on le ferait sans doute aujourd'hui en automobile, croyant le rendre ainsi plus agréable. On verra, qu'accompli de cette façon, il serait même en un sens plus vrai puisqu'on y suivrait de plus près, dans une intimité plus étroite, les diverses gradations par lesquelles change la surface de la terre. Mais enfin le plaisir spécifique du voyage n'est pas de pouvoir descendre en route et de s'arrêter quand on est fatigué, c'est de rendre la différence entre le départ et l'arrivée non pas aussi insensible, mais aussi profonde qu'on peut, de la ressentir dans sa totalité, intacte, telle quelle était dans notre pensée quand notre imagination nous portait du lieu où nous vivions jusqu'au coeur d'un lieu désiré, en un bond qui nous semblait moins miraculeux parce qu'il franchissait une distance que parce qu'il unissait deux individualités distinctes de la terre, qu'il nous menait d'un nom à un autre nom ; et que schématise (mieux qu'une promenade où, comme on débarque où l'on veut, il n'y a guère plus d'arrivée) l'opération mystérieuse qui s'accomplissait dans ces lieux spéciaux, les gares, lesquels ne font pas partie pour ainsi dire de la ville mais contiennent l'essence de sa personnalité de même que sur un écriteau signalétique elles portent son nom.

Mais en tout genre, notre temps a la manie de vouloir ne montrer les choses qu'avec ce qui les entoure dans la réalité, et par là de supprimer l'essentiel, l'acte de l'esprit, qui les isola d'elle. On « présente » un tableau au milieu de meubles, de bibelots, de tentures de la même époque, fade décor qu'excelle à composer dans les hôtels d'aujourd'hui la maîtresse de maison la plus ignorante la veille, passant maintenant ses journées dans les archives et les bibliothèques, et au milieu duquel le chef-d'oeuvre qu'on regarde tout en dînant ne nous donne pas la même enivrante joie qu'on ne doit lui demander que dans une salle de musée, laquelle symbolise bien mieux, par sa nudité et son dépouillement de toutes particularités, les espaces intérieurs où l'artiste s'est abstrait pour créer.

Malheureusement ces lieux merveilleux que sont les gares, d'où l'on part pour une destination éloignée, sont aussi des lieux tragiques, car si le miracle s'y accomplit grâce auquel les pays qui n'avaient encore d'existence que dans notre pensée vont être ceux au milieu desquels nous vivrons, pour cette raison même il faut renoncer au sortir de la salle d'attente à retrouver tout à l'heure la chambre familière où l'on était il y a un instant encore. Il faut laisser toute espérance de rentrer coucher chez soi, une fois qu'on s'est décidé à pénétrer dans l'antre empesté par où l'on accède au mystère, dans un de ces grands ateliers vitrés, comme celui de Saint-Lazare où j'allais chercher le train de Balbec, et qui déployait au-dessus de la ville éventrée un de ces immenses ciels crus et gros de menaces amoncelées de drame, pareils à certains ciels, d'une modernité presque parisienne, de Mantegna ou de Véronèse, et sous lequel ne pouvait s'accomplir que quelque acte terrible et solennel comme un départ en chemin de fer ou l'érection de la Croix.
