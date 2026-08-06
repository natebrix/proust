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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "les Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« Il m'expliqua que la pièce où allait Gilberte était la lingerie, s'offrit à me la montrer et me promit que chaque fois que Gilberte aurait à s'y rendre il la forcerait à m'y emmener... À ce moment-là, j'éprouvai pour lui une tendresse que je crus plus profonde que ma tendresse pour Gilberte. Car maître de sa fille, il me la donnait... »",
      "explanation": "Swann uses his authority to grant the narrator intimate access to Gilberte, dissolving the narrator's anxious 'distance.' The narrator openly registers heightened tenderness and esteem for Swann."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "Swann gains strong positive standing in the narrator's feelings by removing a barrier and promising access to Gilberte."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-170-p-179"
}

### Candidate characters

[
  "Gilberte",
  "Odette",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Mais éclaircir un jour les faits de la vie d'Odette auxquels il avait dû ces souffrances n'avait pas été le seul souhait de Swann ; il avait mis en réserve aussi celui de se venger d'elles, quand n'aimant plus Odette il ne la craindrait plus ; or, d'exaucer ce second souhait, l'occasion se présentait justement car Swann aimait une autre femme, une femme qui ne lui donnait pas de motifs de jalousie mais pourtant de la jalousie parce qu'il n'était plus capable de renouveler sa façon d'aimer, et que c'était celle dont il avait usé pour Odette qui lui servait encore pour une autre. Pour que la jalousie de Swann renaquît, il n'était pas nécessaire que cette femme fût infidèle, il suffisait que pour une raison quelconque elle fût loin de lui, à une soirée par exemple, et eût paru s'y amuser. C'était assez pour réveiller en lui l'ancienne angoisse, lamentable et contradictoire excroissance de son amour, et qui éloignait Swann de ce qu'elle était comme un besoin d'atteindre (le sentiment réel que cette jeune femme avait pour lui, le désir caché de ses journées, le secret de son coeur), car entre Swann et celle qu'il aimait cette angoisse interposait un amas réfractaire de soupçons antérieurs, ayant leur cause en Odette, ou en telle autre peut-être qui avait précédé Odette, et qui ne permettait plus à l'amant vieilli de connaître sa maîtresse d'aujourd'hui qu'à travers le fantôme ancien et collectif de la « femme qui excitait sa jalousie » dans lequel il avait arbitrairement incarné son nouvel amour. Souvent pourtant Swann l'accusait, cette jalousie, de le faire croire à des trahisons imaginaires ; mais alors il se rappelait qu'il avait fait bénéficier Odette du même raisonnement et à tort. Aussi tout ce que la jeune femme qu'il aimait faisait aux heures où il n'était pas avec elle cessait de lui paraître innocent. Mais alors qu'autrefois, il avait fait le serment, si jamais il cessait d'aimer celle qu'il ne devinait pas devoir être un jour sa femme, de lui manifester implacablement son indifférence, enfin sincère, pour venger son orgueil longtemps humilié, ces représailles qu'il pouvait exercer maintenant sans risques (car que pouvait lui faire d'être pris au mot et privé de ces tête-à-tête avec Odette qui lui étaient jadis si nécessaires), ces représailles il n'y tenait plus ; avec l'amour avait disparu le désir de montrer qu'il n'avait plus d'amour. Et lui qui, quand il souffrait par Odette, eût tant désiré de lui laisser voir un jour qu'il était épris d'une autre, maintenant qu'il l'aurait pu, il prenait mille précautions pour que sa femme ne soupçonnât pas ce nouvel amour.

### Passage

Ce ne fut pas seulement à ces goûters, à cause desquels j'avais eu autrefois la tristesse de voir Gilberte me quitter et rentrer plus tôt, que désormais je pris part, mais les sorties qu'elle faisait avec sa mère, soit pour aller en promenade ou à une matinée, et qui en l'empêchant de venir aux Champs-Élysées m'avaient privé d'elle, les jours où je restais seul le long de la pelouse ou devant les chevaux de bois, ces sorties maintenant M. et Odette m'y admettaient, j'avais une place dans leur landau et même c'était à moi qu'on demandait si j'aimais mieux aller au théâtre, à une leçon de danse chez une camarade de Gilberte, à une réunion mondaine chez des amies des Swann (ce que celle-ci appelait « un petit meeting ») ou visiter les Tombeaux de Saint-Denis.

Ces jours où je devais sortir avec les Swann, je venais chez eux pour le déjeuner, que Odette appelait le lunch ; comme on n'était invité que pour midi et demi et qu'à cette époque mes parents déjeunaient à onze heures un quart, c'est après qu'ils étaient sortis de table que je m'acheminais vers ce quartier luxueux, assez solitaire à toute heure, mais particulièrement à celle-là où tout le monde était rentré. Même l'hiver et par la gelée s'il faisait beau, tout en resserrant de temps à autre le noeud d'une magnifique cravate de chez Charvet et en regardant si mes bottines vernies ne se salissaient pas, je me promenais de long en large dans les avenues en attendant midi vingt-sept. J'apercevais de loin, dans le jardinet des Swann, le soleil qui faisait étinceler comme du givre les arbres dénudés. Il est vrai que ce jardinet n'en possédait que deux. L'heure indue faisait nouveau le spectacle. À ces plaisirs de nature (qu'avivait la suppression de l'habitude, et même la faim), la perspective émotionnante de déjeuner chez Odette se mêlait, elle ne les diminuait pas, mais les dominant les asservissait, en faisait des accessoires mondains ; de sorte que si, à cette heure où d'ordinaire je ne les percevais pas, il me semblait découvrir le beau temps, le froid, la lumière hivernale, c'était comme une sorte de préface aux oeufs à la crème, comme une patine, un rose et frais glacis ajoutés au revêtement de cette chapelle mystérieuse qu'était la demeure de Odette et au coeur de laquelle il y avait au contraire tant de chaleur, de parfums et de fleurs.

À midi et demi, je me décidais enfin à entrer dans cette maison qui, comme un gros soulier de Noël, me semblait devoir m'apporter de surnaturels plaisirs. (Le nom de Noël était du reste inconnu à Odette et à Gilberte qui l'avaient remplacé par celui de Christmas, et ne parlaient que du pudding de Christmas, de ce qu'on leur avait donné pour leur Christmas, de s'absenter – ce qui me rendait fou de douleur – pour Christmas. Même à la maison, je me serais cru déshonoré en parlant de Noël et je ne disais plus que Christmas, ce que mon père trouvait extrêmement ridicule.)

Je ne rencontrais d'abord qu'un valet de pied qui, après m'avoir fait traverser plusieurs grands salons, m'introduisait dans un tout petit, vide, que commençait déjà à faire rêver l'après-midi bleu de ses fenêtres ; je restais seul en compagnie d'orchidées, de roses et de violettes qui – pareilles à des personnes qui attendent à côté de vous mais ne vous connaissent pas – gardaient un silence que leur individualité de choses vivantes rendait plus impressionnant et recevaient frileusement la chaleur d'un feu incandescent de charbon, précieusement posé derrière une vitrine de cristal, dans une cuve de marbre blanc où il faisait écouler de temps à autre ses dangereux rubis.

Je m'étais assis, mais je me levais précipitamment en entendant ouvrir la porte ; ce n'était qu'un second valet de pied, puis un troisième, et le mince résultat auquel aboutissaient leurs allées et venues inutilement émouvantes était de remettre un peu de charbon dans le feu ou d'eau dans les vases. Ils s'en allaient, je me retrouvais seul, une fois refermée la porte que Odette finirait bien par ouvrir. Et, certes, j'eusse été moins troublé dans un antre magique que dans ce petit salon d'attente où le feu me semblait procéder à des transmutations, comme dans le laboratoire de Klingsor. Un nouveau bruit de pas retentissait, je ne me levais pas, ce devait être encore un valet de pied, c'était Swann. « Comment ? vous êtes seul ? Que voulez-vous, ma pauvre femme n'a jamais pu savoir ce que c'est que l'heure. Une heure moins dix. Tous les jours c'est plus tard, et vous allez voir, elle arrivera sans se presser en croyant qu'elle est en avance. » Et comme il était resté neuro-arthritique, et devenu un peu ridicule, avoir une femme si inexacte qui rentrait tellement tard du Bois, qui s'oubliait chez sa couturière, et n'était jamais à l'heure pour le déjeuner, cela inquiétait Swann pour son estomac, mais le flattait dans son amour-propre.

Il me montrait des acquisitions nouvelles qu'il avait faites et m'en expliquait l'intérêt, mais l'émotion, jointe au manque d'habitude d'être encore à jeun à cette heure-là, tout en agitant mon esprit y faisait le vide, de sorte que, capable de parler, je ne l'étais pas d'entendre. D'ailleurs les oeuvres que possédait Swann, il suffisait pour moi qu'elles fussent situées chez lui, y fissent partie de l'heure délicieuse qui précédait le déjeuner. La Joconde se serait trouvée là qu'elle ne m'eût pas fait plus de plaisir qu'une robe de chambre de Odette, ou ses flacons de sel.

Je continuais à attendre, seul, ou avec Swann et souvent Gilberte, qui était venue nous tenir compagnie. L'arrivée de Odette, préparée par tant de majestueuses entrées, me paraissait devoir être quelque chose d'immense. J'épiais chaque craquement. Mais on ne trouve jamais aussi hauts qu'on les avait espérés une cathédrale, une vague dans la tempête, le bond d'un danseur ; après ces valets de pied en livrée, pareils aux figurants dont le cortège, au théâtre, prépare, et par là même diminue l'apparition finale de la reine, Odette entrant furtivement en petit paletot de loutre, sa voilette baissée sur un nez rougi par le froid, ne tenait pas les promesses prodiguées dans l'attente à mon imagination.

Mais si elle était restée toute la matinée chez elle, quand elle arrivait dans le salon, c'était vêtue d'un peignoir en crêpe de Chine de couleur claire qui me semblait plus élégant que toutes les robes.

Quelquefois les Swann se décidaient à rester à la maison tout l'après-midi. Et alors, comme on avait déjeuné si tard, je voyais bien vite sur le mur du jardinet décliner le soleil de ce jour qui m'avait paru devoir être différent des autres, et les domestiques avaient beau apporter des lampes de toutes les grandeurs et de toutes les formes, brûlant chacune sur l'autel consacré d'une console, d'un guéridon, d'une « encoignure » ou d'une petite table, comme pour la célébration d'un culte inconnu, rien d'extraordinaire ne naissait de la conversation, et je m'en allais déçu, comme on l'est souvent dès l'enfance après la messe de minuit.

Mais ce désappointement-là n'était guère que spirituel. Je rayonnais de joie dans cette maison où Gilberte, quand elle n'était pas encore avec nous, allait entrer, et me donnerait dans un instant, pour des heures, sa parole, son regard attentif et souriant tel que je l'avais vu pour la première fois à Combray. Tout au plus étais-je un peu jaloux en la voyant souvent disparaître dans de grandes chambres auxquelles on accédait par un escalier intérieur. Obligé de rester au salon, comme l'amoureux d'une actrice qui n'a que son fauteuil à l'orchestre et rêve avec inquiétude de ce qui se passe dans les coulisses, au foyer des artistes, je posai à Swann, au sujet de cette autre partie de la maison, des questions savamment voilées, mais sur un ton duquel je ne parvins pas à bannir quelque anxiété. Il m'expliqua que la pièce où allait Gilberte était la lingerie, s'offrit à me la montrer et me promit que chaque fois que Gilberte aurait à s'y rendre il la forcerait à m'y emmener. Par ces derniers mots et la détente qu'ils me procurèrent, Swann supprima brusquement pour moi une de ces affreuses distances intérieures au terme desquelles une femme que nous aimons nous apparaît si lointaine. À ce moment-là, j'éprouvai pour lui une tendresse que je crus plus profonde que ma tendresse pour Gilberte. Car maître de sa fille, il me la donnait et elle, elle se refusait parfois, je n'avais pas directement sur elle ce même empire qu'indirectement par Swann. Enfin elle, je l'aimais et ne pouvais par conséquent la voir sans ce trouble, sans ce désir de quelque chose de plus, qui ôte, auprès de l'être qu'on aime, la sensation d'aimer.
