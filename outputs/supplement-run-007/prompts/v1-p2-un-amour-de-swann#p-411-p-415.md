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
        "Swann",
        "fils Swann"
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
      "evidence": "« ...quand elle y restait, elle le voyait peu... maintenant, chaque fois qu’il voulait la voir, elle invoquait les convenances ou prétextait des occupations. Quand il parlait d’aller à une fête... elle lui disait qu’il voulait afficher leur liaison, qu’il la traitait comme une fille. »",
      "explanation": "Odette refuses to see him and to appear in public with him, accusing him of flaunting their relationship. This behavior excludes him and places him in a subordinate position."
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
      "explanation": "Locally, Swann is kept at a distance by Odette; he must pay, seek excuses, solicit Charlus, and even the influence of Adolphe without guarantee of being received."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-411-p-415"
}

### Candidate characters

[
  "Françoise",
  "M. Verdurin",
  "baron de Charlus",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le grand-père du narrateur",
  "le narrateur",
  "oncle Adolphe",
  "princesse de Parme"
]

### Prior local context (optional)

Mais elle, de même qu'elle avait cru que son refus d'argent n'était qu'une feinte, ne voyait qu'un prétexte dans le renseignement que Swann venait lui demander sur la voiture à repeindre ou la valeur à acheter. Car elle ne reconstituait pas les diverses phases de ces crises qu'il traversait et, dans l'idée qu'elle s'en faisait, elle omettait d'en comprendre le mécanisme, ne croyant qu'à ce qu'elle connaissait d'avance, à la nécessaire, à l'infaillible et toujours identique terminaison. Idée incomplète – d'autant plus profonde peut-être – si on la jugeait du point de vue de Swann qui eût sans doute trouvé qu'il était incompris d'Odette, comme un morphinomane ou un tuberculeux, persuadés qu'ils ont été arrêtés, l'un par un événement extérieur au moment où il allait se délivrer de son habitude invétérée, l'autre par une indisposition accidentelle au moment où il allait être enfin rétabli, se sentent incompris du médecin qui n'attache pas la même importance qu'eux à ces prétendues contingences, simples déguisements, selon lui, revêtus, pour redevenir sensibles à ses malades, par le vice et l'état morbide qui, en réalité, n'ont pas cessé de peser incurablement sur eux tandis qu'ils berçaient des rêves de sagesse ou de guérison. Et de fait, l'amour de Swann en était arrivé à ce degré où le médecin et, dans certaines affections, le chirurgien le plus audacieux, se demandent si priver un malade de son vice ou lui ôter son mal, est encore raisonnable ou même possible.

### Passage

Certes l'étendue de cet amour, Swann n'en avait pas une conscience directe. Quand il cherchait à le mesurer, il lui arrivait parfois qu'il semblât diminué, presque réduit à rien ; par exemple, le peu de goût, presque le dégoût que lui avaient inspiré, avant qu'il aimât Odette, ses traits expressifs, son teint sans fraîcheur, lui revenait à certains jours. « Vraiment il y a progrès sensible, se disait-il le lendemain ; à voir exactement les choses, je n'avais presque aucun plaisir hier à être dans son lit ; c'est curieux, je la trouvais même laide. » Et certes, il était sincère, mais son amour s'étendait bien au delà des régions du désir physique. La personne même d'Odette n'y tenait plus une grande place. Quand du regard il rencontrait sur sa table la photographie d'Odette, ou quand elle venait le voir, il avait peine à identifier la figure de chair ou de bristol avec le trouble douloureux et constant qui habitait en lui. Il se disait presque avec étonnement : « C'est elle », comme si tout d'un coup on nous montrait extériorisée devant nous une de nos maladies et que nous ne la trouvions pas ressemblante à ce que nous souffrons. « Elle », il essayait de se demander ce que c'était ; car c'est une ressemblance de l'amour et de la mort, plutôt que celles, si vagues, que l'on redit toujours, de nous faire interroger plus avant, dans la peur que sa réalité se dérobe, le mystère de la personnalité. Et cette maladie qu'était l'amour de Swann avait tellement multiplié, il était si étroitement mêlé à toutes les habitudes de Swann, à tous ses actes, à sa pensée, à sa santé, à son sommeil, à sa vie, même à ce qu'il désirait pour après sa mort, il ne faisait tellement plus qu'un avec lui, qu'on n'aurait pas pu l'arracher de lui sans le détruire lui-même à peu près tout entier : comme on dit en chirurgie, son amour n'était plus opérable.

Par cet amour Swann avait été tellement détaché de tous les intérêts, que quand par hasard il retournait dans le monde, en se disant que ses relations, comme une monture élégante qu'elle n'aurait pas d'ailleurs su estimer très exactement, pouvaient lui rendre à lui-même un peu de prix aux yeux d'Odette (et ç'aurait peut-être été vrai en effet si elles n'avaient été avilies par cet amour même, qui pour Odette dépréciait toutes les choses qu'il touchait par le fait qu'il semblait les proclamer moins précieuses), il y éprouvait, à côté de la détresse d'être dans des lieux, au milieu de gens qu'elle ne connaissait pas, le plaisir désintéressé qu'il aurait pris à un roman ou à un tableau où sont peints les divertissements d'une classe oisive ; comme, chez lui, il se complaisait à considérer le fonctionnement de sa vie domestique, l'élégance de sa garde-robe et de sa livrée, le bon placement de ses valeurs, de la même façon qu'à lire dans Saint-Simon, qui était un de ses auteurs favoris, la mécanique des journées, le menu des repas de Mme de Maintenon, ou l'avarice avisée et le grand train de Lulli. Et dans la faible mesure où ce détachement n'était pas absolu, la raison de ce plaisir nouveau que goûtait Swann, c'était de pouvoir émigrer un moment dans les rares parties de lui-même restées presque étrangères à son amour, à son chagrin. À cet égard, cette personnalité que lui attribuait ma grand'tante, de « fils Swann », distincte de sa personnalité plus individuelle de Swann Swann, était celle où il se plaisait maintenant le mieux. Un jour que, pour l'anniversaire de la princesse de Parme (et parce qu'elle pouvait souvent être indirectement agréable à Odette en lui faisant avoir des places pour des galas, des jubilés), il avait voulu lui envoyer des fruits, ne sachant pas trop comment les commander, il en avait chargé une cousine de sa mère qui, ravie de faire une commission pour lui, lui avait écrit, en lui rendant compte qu'elle n'avait pas pris tous les fruits au même endroit, mais les raisins chez Crapote dont c'est la spécialité, les fraises chez Jauret, les poires chez Chevet, où elles étaient plus belles, etc., « chaque fruit visité et examiné un par un par moi ». Et en effet, par les remerciements de la princesse, il avait pu juger du parfum des fraises et du moelleux des poires. Mais surtout le « chaque fruit visité et examiné un par un par moi » avait été un apaisement à sa souffrance, en emmenant sa conscience dans une région où il se rendait rarement, bien qu'elle lui appartînt comme héritier d'une famille de riche et bonne bourgeoisie où s'étaient conservés héréditairement, tout prêts à être mis à son service dès qu'il le souhaitait, la connaissance des « bonnes adresses » et l'art de savoir bien faire une commande.

Certes, il avait trop longtemps oublié qu'il était le « fils Swann » pour ne pas ressentir, quand il le redevenait un moment, un plaisir plus vif que ceux qu'il eût pu éprouver le reste du temps et sur lesquels il était blasé ; et si l'amabilité des bourgeois, pour lesquels il restait surtout cela, était moins vive que celle de l'aristocratie (mais plus flatteuse d'ailleurs, car chez eux du moins elle ne se sépare jamais de la considération), une lettre d'altesse, quelques divertissements princiers qu'elle lui proposât, ne pouvait lui être aussi agréable que celle qui lui demandait d'être témoin, ou seulement d'assister à un mariage dans la famille de vieux amis de ses parents, dont les uns avaient continué à le voir – comme mon grand-père qui, l'année précédente, l'avait invité au mariage de ma mère – et dont certains autres le connaissaient personnellement à peine, mais se croyaient des devoirs de politesse envers le fils, envers le digne successeur de feu Swann.

Mais, par les intimités déjà anciennes qu'il avait parmi eux, les gens du monde, dans une certaine mesure, faisaient aussi partie de sa maison, de son domestique et de sa famille. Il se sentait, à considérer ses brillantes amitiés, le même appui hors de lui-même, le même confort, qu'à regarder les belles terres, la belle argenterie, le beau linge de table, qui lui venaient des siens. Et la pensée que s'il tombait chez lui frappé d'une attaque, ce serait tout naturellement le duc de Chartres, le prince de Reuss, le duc de Luxembourg, et le baron de Charlus, que son valet de chambre courrait chercher, lui apportait la même consolation qu'à notre vieille Françoise de savoir qu'elle serait ensevelie dans des draps fins à elle, marqués, non reprisés (ou si finement que cela ne donnait qu'une plus haute idée du soin de l'ouvrière), linceul de l'image fréquente duquel elle tirait une certaine satisfaction, sinon de bien-être, au moins d'amour-propre. Mais surtout, comme dans toutes celles de ses actions et de ses pensées qui se rapportaient à Odette, Swann était constamment dominé et dirigé par le sentiment inavoué qu'il lui était peut-être pas moins cher, mais moins agréable à voir que quiconque, que le plus ennuyeux fidèle des Verdurin, quand il se reportait à un monde pour qui il était l'homme exquis par excellence, qu'on faisait tout pour attirer, qu'on se désolait de ne pas voir, il recommençait à croire à l'existence d'une vie plus heureuse, presque à en éprouver l'appétit, comme il arrive à un malade alité depuis des mois, à la diète, et qui aperçoit dans un journal le menu d'un déjeuner officiel ou l'annonce d'une croisière en Sicile.

S'il était obligé de donner des excuses aux gens du monde pour ne pas leur faire de visites, c'était de lui en faire qu'il cherchait à s'excuser auprès d'Odette. Encore les payait-il (se demandant à la fin du mois, pour peu qu'il eût un peu abusé de sa patience et fût allé souvent la voir, si c'était assez de lui envoyer quatre mille francs), et pour chacune trouvait un prétexte, un présent à lui apporter, un renseignement dont elle avait besoin, Charlus qu'elle avait rencontré allant chez elle et qui avait exigé qu'il l'accompagnât. Et à défaut d'aucun, il priait Charlus de courir chez elle, de lui dire comme spontanément, au cours de la conversation, qu'il se rappelait avoir à parler à Swann, qu'elle voulût bien lui faire demander de passer tout de suite chez elle ; mais le plus souvent Swann attendait en vain et Charlus lui disait le soir que son moyen n'avait pas réussi. De sorte que si elle faisait maintenant de fréquentes absences, même à Paris, quand elle y restait, elle le voyait peu, et elle qui, quand elle l'aimait, lui disait : « Je suis toujours libre » et « Qu'est-ce que l'opinion des autres peut me faire ? », maintenant, chaque fois qu'il voulait la voir, elle invoquait les convenances ou prétextait des occupations. Quand il parlait d'aller à une fête de charité, à un vernissage, à une première, où elle serait, elle lui disait qu'il voulait afficher leur liaison, qu'il la traitait comme une fille. C'est au point que pour tâcher de n'être pas partout privé de la rencontrer, Swann qui savait qu'elle connaissait et affectionnait beaucoup mon grand-oncle Adolphe dont il avait été lui-même l'ami, alla le voir un jour dans son petit appartement de la rue de Bellechasse afin de lui demander d'user de son influence sur Odette. Comme elle prenait toujours, quand elle parlait à Swann de mon oncle, des airs poétiques, disant : « Ah ! lui, ce n'est pas comme toi, c'est une si belle chose, si grande, si jolie, que son amitié pour moi. Ce n'est pas lui qui me considérerait assez peu pour vouloir se montrer avec moi dans tous les lieux publics », Swann fut embarrassé et ne savait pas à quel ton il devait se hausser pour parler d'elle à mon oncle. Il posa d'abord l'excellence a priori d'Odette, l'axiome de sa supra-humanité séraphique, la révélation de ses vertus indémontrables et dont la notion ne pouvait dériver de l'expérience. « Je veux parler avec vous. Vous, vous savez quelle femme au-dessus de toutes les femmes, quel être adorable, quel ange est Odette. Mais vous savez ce que c'est que la vie de Paris. Tout le monde ne connaît pas Odette sous le jour où nous la connaissons vous et moi. Alors il y a des gens qui trouvent que je joue un rôle un peu ridicule ; elle ne peut même pas admettre que je la rencontre dehors, au théâtre. Vous, en qui elle a tant de confiance, ne pourriez-vous lui dire quelques mots pour moi, lui assurer qu'elle s'exagère le tort qu'un salut de moi lui cause ? »
