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
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.94,
      "evidence": "« Swann comprit que le sentiment qu'Odette avait eu pour lui ne renaîtrait jamais... »; « ...une seule parole d'Odette ... l'immobilisait, durcissait sa fluidité, le faisait geler tout entier »; « Oui, mon petit, nous partons le 19... »",
      "explanation": "The passage depicts Swann’s realization of hopelessness and his emotional collapse when Odette signals her upcoming trip with Forcheville."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.94,
      "explanation": "Swann’s emotional standing sharply declines as he accepts Odette’s lost affection and is devastated by her plan to travel with Forcheville."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-536-p-540"
}

### Candidate characters

[
  "M. Vinteuil",
  "Mme Verdurin",
  "Odette",
  "baron de Charlus",
  "comte de Forcheville",
  "comtesse de Monteriender",
  "le pianiste"
]

### Prior local context (optional)

Swann n'avait donc pas tort de croire que la phrase de la sonate existât réellement. Certes, humaine à ce point de vue, elle appartenait pourtant à un ordre de créatures surnaturelles et que nous n'avons jamais vues, mais que malgré cela nous reconnaissons avec ravissement quand quelque explorateur de l'invisible arrive à en capter une, à l'amener, du monde divin où il a accès, briller quelques instants au-dessus du nôtre. C'est ce que M. Vinteuil avait fait pour la petite phrase. Swann sentait que le compositeur s'était contenté, avec ses instruments de musique, de la dévoiler, de la rendre visible, d'en suivre et d'en respecter le dessin d'une main si tendre, si prudente, si délicate et si sûre que le son s'altérait à tout moment, s'estompant pour indiquer une ombre, revivifié quand il lui fallait suivre à la piste un plus hardi contour. Et une preuve que Swann ne se trompait pas quand il croyait à l'existence réelle de cette phrase, c'est que tout amateur un peu fin se fût tout de suite aperçu de l'imposture, si M. Vinteuil ayant eu moins de puissance pour en voir et en rendre les formes, avait cherché à dissimuler, en ajoutant çà et là des traits de son cru, les lacunes de sa vision ou les défaillances de sa main.

### Passage

Elle avait disparu. Swann savait qu'elle reparaîtrait à la fin du dernier mouvement, après tout un long morceau que le pianiste de Mme Verdurin sautait toujours. Il y avait là d'admirables idées que Swann n'avait pas distinguées à la première audition et qu'il percevait maintenant, comme si elles se fussent, dans le vestiaire de sa mémoire, débarrassées du déguisement uniforme de la nouveauté. Swann écoutait tous les thèmes épars qui entreraient dans la composition de la phrase, comme les prémisses dans la conclusion nécessaire, il assistait à sa genèse. « Ô audace aussi géniale peut-être, se disait-il, que celle d'un Lavoisier, d'un Ampère, l'audace d'un Vinteuil expérimentant, découvrant les lois secrètes d'une force inconnue, menant à travers l'inexploré, vers le seul but possible, l'attelage invisible auquel il se fie et qu'il n'apercevra jamais. » Le beau dialogue que Swann entendit entre le piano et le violon au commencement du dernier morceau ! La suppression des mots humains, loin d'y laisser régner la fantaisie, comme on aurait pu croire, l'en avait éliminée ; jamais le langage parlé ne fut si inflexiblement nécessité, ne connut à ce point la pertinence des questions, l'évidence des réponses. D'abord le piano solitaire se plaignit, comme un oiseau abandonné de sa compagne ; le violon l'entendit, lui répondit comme d'un arbre voisin. C'était comme au commencement du monde, comme s'il n'y avait encore eu qu'eux deux sur la terre, ou plutôt dans ce monde fermé à tout le reste, construit par la logique d'un créateur et où ils ne seraient jamais que tous les deux : cette sonate. Est-ce un oiseau, est-ce l'âme incomplète encore de la petite phrase, est-ce une fée, invisible et gémissant, dont le piano ensuite redisait tendrement la plainte ? Ses cris étaient si soudains que le violoniste devait se précipiter sur son archet pour les recueillir. Merveilleux oiseau ! le violoniste semblait vouloir le charmer, l'apprivoiser, le capter. Déjà il avait passé dans son âme, déjà la petite phrase évoquée agitait comme celui d'un médium le corps vraiment possédé du violoniste. Swann savait qu'elle allait parler encore une fois. Et il s'était si bien dédoublé que l'attente de l'instant imminent où il allait se retrouver en face d'elle le secoua d'un de ces sanglots qu'un beau vers ou une triste nouvelle provoquent en nous, non pas quand nous sommes seuls, mais si nous les apprenons à des amis en qui nous nous apercevons comme un autre dont l'émotion probable les attendrit. Elle reparut, mais cette fois pour se suspendre dans l'air et se jouer un instant seulement, comme immobile, et pour expirer après. Aussi Swann ne perdait-il rien du temps si court où elle se prorogeait. Elle était encore là comme une bulle irisée qui se soutient. Tel un arc-en-ciel, dont l'éclat faiblit, s'abaisse, puis se relève et, avant de s'éteindre, s'exalte un moment comme il n'avait pas encore fait : aux deux couleurs qu'elle avait jusque-là laissé paraître, elle ajouta d'autres cordes diaprées, toutes celles du prisme, et les fit chanter. Swann n'osait pas bouger et aurait voulu faire tenir tranquilles aussi les autres personnes, comme si le moindre mouvement avait pu compromettre le prestige surnaturel, délicieux et fragile qui était si près de s'évanouir. Personne, à dire vrai, ne songeait à parler. La parole ineffable d'un seul absent, peut-être d'un mort (Swann ne savait pas si Vinteuil vivait encore) s'exhalant au-dessus des rites de ces officiants, suffisait à tenir en échec l'attention de trois cents personnes, et faisait de cette estrade où une âme était ainsi évoquée un des plus nobles autels où pût s'accomplir une cérémonie surnaturelle. De sorte que quand la phrase se fut enfin défaite, flottant en lambeaux dans les motifs suivants qui déjà avaient pris sa place, si Swann au premier instant fut irrité de voir la comtesse de Monteriender, célèbre par ses naïvetés, se pencher vers lui pour lui confier ses impressions avant même que la sonate fût finie, il ne put s'empêcher de sourire, et peut-être de trouver aussi un sens profond qu'elle n'y voyait pas, dans les mots dont elle se servit. Émerveillée par la virtuosité des exécutants, la comtesse s'écria en s'adressant à Swann : « C'est prodigieux, je n'ai jamais rien vu d'aussi fort... » Mais un scrupule d'exactitude lui faisant corriger cette première assertion, elle ajouta cette réserve : « rien d'aussi fort... depuis les tables tournantes !

À partir de cette soirée, Swann comprit que le sentiment qu'Odette avait eu pour lui ne renaîtrait jamais, que ses espérances de bonheur ne se réaliseraient plus. Et les jours où par hasard elle avait encore été gentille et tendre avec lui, si elle avait eu quelque attention, il notait ces signes apparents et menteurs d'un léger retour vers lui, avec cette sollicitude attendrie et sceptique, cette joie désespérée de ceux qui, soignant un ami arrivé aux derniers jours d'une maladie incurable, relatent comme des faits précieux : « hier, il a fait ses comptes lui-même et c'est lui qui a relevé une erreur d'addition que nous avions faite ; il a mangé un oeuf avec plaisir, s'il le digère bien on essaiera demain d'une côtelette », quoiqu'ils les sachent dénués de signification à la veille d'une mort inévitable. Sans doute Swann était certain que s'il avait vécu maintenant loin d'Odette, elle aurait fini par lui devenir indifférente, de sorte qu'il aurait été content qu'elle quittât Paris pour toujours ; il aurait eu le courage de rester ; mais il n'avait pas celui de partir.

Il en avait eu souvent la pensée. Maintenant qu'il s'était remis à son étude sur Ver Meer il aurait eu besoin de retourner au moins quelques jours à la Haye, à Dresde, à Brunswick. Il était persuadé qu'une « Toilette de Diane » qui avait été achetée par le Mauritshuis à la vente Goldschmidt comme un Nicolas Maes était en réalité de Ver Meer. Et il aurait voulu pouvoir étudier le tableau sur place pour étayer sa conviction. Mais quitter Paris pendant qu'Odette y était et même quand elle était absente – car dans des lieux nouveaux où les sensations ne sont pas amorties par l'habitude, on retrempe, on ranime une douleur – c'était pour lui un projet si cruel, qu'il ne se sentait capable d'y penser sans cesse que parce qu'il se savait résolu à ne l'exécuter jamais. Mais il arrivait qu'en dormant, l'intention du voyage renaissait en lui – sans qu'il se rappelât que ce voyage était impossible – et elle s'y réalisait. Un jour il rêva qu'il partait pour un an ; penché à la portière du wagon vers un jeune homme qui sur le quai lui disait adieu en pleurant, Swann cherchait à le convaincre de partir avec lui. Le train s'ébranlant, l'anxiété le réveilla, il se rappela qu'il ne partait pas, qu'il verrait Odette ce soir-là, le lendemain et presque chaque jour. Alors, encore tout ému de son rêve, il bénit les circonstances particulières qui le rendaient indépendant, grâce auxquelles il pouvait rester près d'Odette, et aussi réussir à ce qu'elle lui permît de la voir quelquefois ; et, récapitulant tous ces avantages : sa situation – sa fortune, dont elle avait souvent trop besoin pour ne pas reculer devant une rupture (ayant même, disait-on, une arrière-pensée de se faire épouser par lui) – cette amitié de Charlus qui à vrai dire ne lui avait jamais fait obtenir grand'chose d'Odette, mais lui donnait la douceur de sentir qu'elle entendait parler de lui d'une manière flatteuse par cet ami commun pour qui elle avait une si grande estime – et jusqu'à son intelligence enfin, qu'il employait tout entière à combiner chaque jour une intrigue nouvelle qui rendît sa présence sinon agréable, du moins nécessaire à Odette – il songea à ce qu'il serait devenu si tout cela lui avait manqué, il songea que s'il avait été, comme tant d'autres, pauvre, humble, dénué, obligé d'accepter toute besogne, ou lié à des parents, à une épouse, il aurait pu être obligé de quitter Odette, que ce rêve dont l'effroi était encore si proche aurait pu être vrai, et il se dit : « On ne connaît pas son bonheur. On n'est jamais aussi malheureux qu'on croit. » Mais il compta que cette existence durait déjà depuis plusieurs années, que tout ce qu'il pouvait espérer c'est qu'elle durât toujours, qu'il sacrifierait ses travaux, ses plaisirs, ses amis, finalement toute sa vie à l'attente quotidienne d'un rendez-vous qui ne pouvait rien lui apporter d'heureux, et il se demanda s'il ne se trompait pas, si ce qui avait favorisé sa liaison et en avait empêché la rupture n'avait pas desservi sa destinée, si l'événement désirable, ce n'aurait pas été celui dont il se réjouissait tant qu'il n'eût eu lieu qu'en rêve : son départ ; il se dit qu'on ne connaît pas son malheur, qu'on n'est jamais si heureux qu'on croit.

Quelquefois il espérait qu'elle mourrait sans souffrances dans un accident, elle qui était dehors, dans les rues, sur les routes, du matin au soir. Et comme elle revenait saine et sauve, il admirait que le corps humain fût si souple et si fort, qu'il pût continuellement tenir en échec, déjouer tous les périls qui l'environnent (et que Swann trouvait innombrables depuis que son secret désir les avait supputés), et permît ainsi aux êtres de se livrer chaque jour et à peu près impunément à leur oeuvre de mensonge, à la poursuite du plaisir. Et Swann sentait bien près de son coeur ce Mahomet II dont il aimait le portrait par Bellini et qui, ayant senti qu'il était devenu amoureux fou d'une de ses femmes, la poignarda afin, dit naïvement son biographe vénitien, de retrouver sa liberté d'esprit. Puis il s'indignait de ne penser ainsi qu'à soi, et les souffrances qu'il avait éprouvées lui semblaient ne mériter aucune pitié puisque lui-même faisait si bon marché de la vie d'Odette.

Ne pouvant se séparer d'elle sans retour, du moins, s'il l'avait vue sans séparations, sa douleur aurait fini par s'apaiser et peut-être son amour par s'éteindre. Et du moment qu'elle ne voulait pas quitter Paris à jamais, il eût souhaité qu'elle ne le quittât jamais. Du moins comme il savait que la seule grande absence qu'elle faisait était tous les ans celle d'août et septembre, il avait le loisir plusieurs mois d'avance d'en dissoudre l'idée amère dans tout le Temps à venir qu'il portait en lui par anticipation et qui, composé de jours homogènes aux jours actuels, circulait transparent et froid en son esprit où il entretenait la tristesse, mais sans lui causer de trop vives souffrances. Mais cet avenir intérieur, ce fleuve, incolore et libre, voici qu'une seule parole d'Odette venait l'atteindre jusqu'en Swann et, comme un morceau de glace, l'immobilisait, durcissait sa fluidité, le faisait geler tout entier ; et Swann s'était senti soudain rempli d'une masse énorme et infrangible qui pesait sur les parois intérieures de son être jusqu'à le faire éclater : c'est qu'Odette lui avait dit, avec un regard souriant et sournois qui l'observait : « Forcheville va faire un beau voyage, à la Pentecôte. Il va en Égypte », et Swann avait aussitôt compris que cela signifiait : « Je vais aller en Égypte à la Pentecôte avec Forcheville. » Et en effet, si quelques jours après, Swann lui disait : « Voyons, à propos de ce voyage que tu m'as dit que tu ferais avec Forcheville », elle répondait étourdiment : « Oui, mon petit, nous partons le 19, on t'enverra une vue des Pyramides. » Alors il voulait apprendre si elle était la maîtresse de Forcheville, le lui demander à elle-même. Il savait que, superstitieuse comme elle était, il y avait certains parjures qu'elle ne ferait pas et puis la crainte, qui l'avait retenu jusqu'ici, d'irriter Odette en l'interrogeant, de se faire détester d'elle, n'existait plus maintenant qu'il avait perdu tout espoir d'en être jamais aimé.
