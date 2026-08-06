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
      "canonical_name": "M. Vinteuil",
      "surface_forms": [
        "M. Vinteuil",
        "Vinteuil"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.96
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "M. Vinteuil",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "« la pensée de Swann se porta… vers ce M. Vinteuil, vers ce frère inconnu et sublime »; « M. Vinteuil avait été l’un de ces musiciens »; « C’est ce que M. Vinteuil avait fait pour la petite phrase. »",
      "explanation": "The passage exalts Vinteuil as a sublime creator who captures and dignifies inner sorrow; the narrator explicitly affirms his artistic power."
    }
  ],
  "status_effects": [
    {
      "character": "M. Vinteuil",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "He is framed as a rare, powerful artist who reveals a 'real' musical being and enriches inner life."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-531-p-535"
}

### Candidate characters

[
  "M. Verdurin",
  "Odette",
  "Swann"
]

### Prior local context (optional)

Mais tout à coup ce fut comme si elle était entrée, et cette apparition lui fut une si déchirante souffrance qu'il dut porter la main à son coeur. C'est que le violon était monté à des notes hautes où il restait comme pour une attente, une attente qui se prolongeait sans qu'il cessât de les tenir, dans l'exaltation où il était d'apercevoir déjà l'objet de son attente qui s'approchait, et avec un effort désespéré pour tâcher de durer jusqu'à son arrivée, de l'accueillir avant d'expirer, de lui maintenir encore un moment de toutes ses dernières forces le chemin ouvert pour qu'il pût passer, comme on soutient une porte qui sans cela retomberait. Et avant que Swann eût eu le temps de comprendre, et de se dire : « C'est la petite phrase de la sonate de M. Vinteuil, n'écoutons pas ! » tous ses souvenirs du temps où Odette était éprise de lui, et qu'il avait réussi jusqu'à ce jour à maintenir invisibles dans les profondeurs de son être, trompés par ce brusque rayon du temps d'amour qu'ils crurent revenu, s'étaient réveillés et, à tire d'aile, étaient remontés lui chanter éperdument, sans pitié pour son infortune présente, les refrains oubliés du bonheur.

### Passage

Au lieu des expressions abstraites « temps où j'étais heureux », « temps où j'étais aimé », qu'il avait souvent prononcées jusque-là et sans trop souffrir, car son intelligence n'y avait enfermé du passé que de prétendus extraits qui n'en conservaient rien, il retrouva tout ce qui de ce bonheur perdu avait fixé à jamais la spécifique et volatile essence ; il revit tout, les pétales neigeux et frisés du chrysanthème qu'elle lui avait jeté dans sa voiture, qu'il avait gardé contre ses lèvres – l'adresse en relief de la « Maison Dorée » sur la lettre où il avait lu : « Ma main tremble si fort en vous écrivant » – le rapprochement de ses sourcils quand elle lui avait dit d'un air suppliant : « Ce n'est pas dans trop longtemps que vous me ferez signe ? » ; il sentit l'odeur du fer du coiffeur par lequel il se faisait relever sa « brosse » pendant que Lorédan allait chercher la petite ouvrière, les pluies d'orage qui tombèrent si souvent ce printemps-là, le retour glacial dans sa victoria, au clair de lune, toutes les mailles d'habitudes mentales, d'impressions saisonnières, de créations cutanées, qui avaient étendu sur une suite de semaines un réseau uniforme dans lequel son corps se trouvait repris. À ce moment-là, il satisfaisait une curiosité voluptueuse en connaissant les plaisirs des gens qui vivent par l'amour. Il avait cru qu'il pourrait s'en tenir là, qu'il ne serait pas obligé d'en apprendre les douleurs ; comme maintenant le charme d'Odette lui était peu de chose auprès de cette formidable terreur qui le prolongeait comme un trouble halo, cette immense angoisse de ne pas savoir à tous moments ce qu'elle avait fait, de ne pas la posséder partout et toujours ! Hélas, il se rappela l'accent dont elle s'était écriée : « Mais je pourrai toujours vous voir, je suis toujours libre ! » elle qui ne l'était plus jamais ! l'intérêt, la curiosité qu'elle avait eus pour sa vie à lui, le désir passionné qu'il lui fît la faveur – redoutée au contraire par lui en ce temps-là comme une cause d'ennuyeux dérangements – de l'y laisser pénétrer ; comme elle avait été obligée de le prier pour qu'il se laissât mener chez les Verdurin ; et, quand il la faisait venir chez lui une fois par mois, comme il avait fallu, avant qu'il se laissât fléchir, qu'elle lui répétât le délice que serait cette habitude de se voir tous les jours dont elle rêvait alors qu'elle ne lui semblait à lui qu'un fastidieux tracas, puis qu'elle avait prise en dégoût et définitivement rompue, pendant qu'elle était devenue pour lui un si invincible et si douloureux besoin. Il ne savait pas dire si vrai quand, à la troisième fois qu'il l'avait vue, comme elle lui répétait : « Mais pourquoi ne me laissez-vous pas venir plus souvent », il lui avait dit en riant, avec galanterie : « par peur de souffrir ». Maintenant, hélas ! il arrivait encore parfois qu'elle lui écrivît d'un restaurant ou d'un hôtel sur du papier qui en portait le nom imprimé ; mais c'était comme des lettres de feu qui le brûlaient. « C'est écrit de l'hôtel Vouillemont ? Qu'y peut-elle être allée faire ! avec qui ? que s'y est-il passé ? » Il se rappela les becs de gaz qu'on éteignait boulevard des Italiens quand il l'avait rencontrée contre tout espoir parmi les ombres errantes, dans cette nuit qui lui avait semblé presque surnaturelle et qui en effet – nuit d'un temps où il n'avait même pas à se demander s'il ne la contrarierait pas en la cherchant, en la retrouvant, tant il était sûr qu'elle n'avait pas de plus grande joie que de le voir et de rentrer avec lui – appartenait bien à un monde mystérieux où on ne peut jamais revenir quand les portes s'en sont refermées. Et Swann aperçut, immobile en face de ce bonheur revécu, un malheureux qui lui fit pitié parce qu'il ne le reconnut pas tout de suite, si bien qu'il dut baisser les yeux pour qu'on ne vît pas qu'ils étaient pleins de larmes. C'était lui-même.

Quand il l'eut compris, sa pitié cessa, mais il fut jaloux de l'autre lui-même qu'elle avait aimé, il fut jaloux de ceux dont il s'était dit souvent sans trop souffrir, « elle les aime peut-être », maintenant qu'il avait échangé l'idée vague d'aimer, dans laquelle il n'y a pas d'amour, contre les pétales du chrysanthème et l'« en tête » de la Maison d'Or, qui, eux, en étaient pleins. Puis sa souffrance devenant trop vive, il passa sa main sur son front, laissa tomber son monocle, en essuya le verre. Et sans doute s'il s'était vu à ce moment-là, il eût ajouté à la collection de ceux qu'il avait distingués le monocle qu'il déplaçait comme une pensée importune et sur la face embuée duquel, avec un mouchoir, il cherchait à effacer des soucis.

Il y a dans le violon – si, ne voyant pas l'instrument, on ne peut pas rapporter ce qu'on entend à son image, laquelle modifie la sonorité – des accents qui lui sont si communs avec certaines voix de contralto, qu'on a l'illusion qu'une chanteuse s'est ajoutée au concert. On lève les yeux, on ne voit que les étuis, précieux comme des boîtes chinoises, mais, par moments, on est encore trompé par l'appel décevant de la sirène ; parfois aussi on croit entendre un génie captif qui se débat au fond de la docte boîte, ensorcelée et frémissante, comme un diable dans un bénitier ; parfois enfin, c'est dans l'air comme un être surnaturel et pur qui passe en déroulant son message invisible.

Comme si les instrumentistes beaucoup moins jouaient la petite phrase qu'ils n'exécutaient les rites exigés d'elle pour qu'elle apparût, et procédaient aux incantations nécessaires pour obtenir et prolonger quelques instants le prodige de son évocation, Swann, qui ne pouvait pas plus la voir que si elle avait appartenu à un monde ultra-violet, et qui goûtait comme le rafraîchissement d'une métamorphose dans la cécité momentanée dont il était frappé en approchant d'elle, Swann la sentait présente, comme une déesse protectrice et confidente de son amour, et qui pour pouvoir arriver jusqu'à lui devant la foule et l'emmener à l'écart pour lui parler, avait revêtu le déguisement de cette apparence sonore. Et tandis qu'elle passait, légère, apaisante et murmurée comme un parfum, lui disant ce qu'elle avait à lui dire et dont il scrutait tous les mots, regrettant de les voir s'envoler si vite, il faisait involontairement avec ses lèvres le mouvement de baiser au passage le corps harmonieux et fuyant. Il ne se sentait plus exilé et seul puisque, elle, qui s'adressait à lui, lui parlait à mi-voix d'Odette. Car il n'avait plus comme autrefois l'impression qu'Odette et lui n'étaient pas connus de la petite phrase. C'est que si souvent elle avait été témoin de leurs joies ! Il est vrai que souvent aussi elle l'avait averti de leur fragilité. Et même, alors que dans ce temps-là il devinait de la souffrance dans son sourire, dans son intonation limpide et désenchantée, aujourd'hui il y trouvait plutôt la grâce d'une résignation presque gaie. De ces chagrins dont elle lui parlait autrefois et qu'il la voyait, sans qu'il fût atteint par eux, entraîner en souriant dans son cours sinueux et rapide, de ces chagrins qui maintenant étaient devenus les siens sans qu'il eût l'espérance d'en être jamais délivré, elle semblait lui dire comme jadis de son bonheur : « Qu'est-ce cela ? tout cela n'est rien. » Et la pensée de Swann se porta pour la première fois dans un élan de pitié et de tendresse vers ce Vinteuil, vers ce frère inconnu et sublime qui lui aussi avait dû tant souffrir ; qu'avait pu être sa vie ? au fond de quelles douleurs avait-il puisé cette force de dieu, cette puissance illimitée de créer ? Quand c'était la petite phrase qui lui parlait de la vanité de ses souffrances, Swann trouvait de la douceur à cette même sagesse qui tout à l'heure pourtant lui avait paru intolérable, quand il croyait la lire dans les visages des indifférents qui considéraient son amour comme une divagation sans importance. C'est que la petite phrase au contraire, quelque opinion qu'elle pût avoir sur la brève durée de ces états de l'âme, y voyait quelque chose, non pas comme faisaient tous ces gens, de moins sérieux que la vie positive, mais au contraire de si supérieur à elle que seul il valait la peine d'être exprimé. Ces charmes d'une tristesse intime, c'était eux qu'elle essayait d'imiter, de recréer, et jusqu'à leur essence qui est pourtant d'être incommunicables et de sembler frivoles à tout autre qu'à celui qui les éprouve, la petite phrase l'avait captée, rendue visible. Si bien qu'elle faisait confesser leur prix et goûter leur douceur divine, par tous ces mêmes assistants – si seulement ils étaient un peu musiciens – qui ensuite les méconnaîtraient dans la vie, en chaque amour particulier qu'ils verraient naître près d'eux. Sans doute la forme sous laquelle elle les avait codifiés ne pouvait pas se résoudre en raisonnements. Mais depuis plus d'une année que, lui révélant à lui-même bien des richesses de son âme, l'amour de la musique était pour quelque temps au moins né en lui, Swann tenait les motifs musicaux pour de véritables idées, d'un autre monde, d'un autre ordre, idées voilées de ténèbres, inconnues, impénétrables à l'intelligence, mais qui n'en sont pas moins parfaitement distinctes les unes des autres, inégales entre elles de valeur et de signification. Quand après la soirée Verdurin, se faisant rejouer la petite phrase, il avait cherché à démêler comment à la façon d'un parfum, d'une caresse, elle le circonvenait, elle l'enveloppait, il s'était rendu compte que c'était au faible écart entre les cinq notes qui la composaient et au rappel constant de deux d'entre elles qu'était due cette impression de douceur rétractée et frileuse ; mais en réalité il savait qu'il raisonnait ainsi non sur la phrase elle-même mais sur de simples valeurs, substituées pour la commodité de son intelligence à la mystérieuse entité qu'il avait perçue, avant de connaître les Verdurin, à cette soirée où il avait entendu pour la première fois la sonate. Il savait que le souvenir même du piano faussait encore le plan dans lequel il voyait les choses de la musique, que le champ ouvert au musicien n'est pas un clavier mesquin de sept notes, mais un clavier incommensurable, encore presque tout entier inconnu, où seulement çà et là, séparées par d'épaisses ténèbres inexplorées, quelques-unes des millions de touches de tendresse, de passion, de courage, de sérénité, qui le composent, chacune aussi différente des autres qu'un univers d'un autre univers, ont été découvertes par quelques grands artistes qui nous rendent le service, en éveillant en nous le correspondant du thème qu'ils ont trouvé, de nous montrer quelle richesse, quelle variété, cache à notre insu cette grande nuit impénétrée et décourageante de notre âme que nous prenons pour du vide et pour du néant. Vinteuil avait été l'un de ces musiciens. En sa petite phrase, quoiqu'elle présentât à la raison une surface obscure, on sentait un contenu si consistant, si explicite, auquel elle donnait une force si nouvelle, si originale, que ceux qui l'avaient entendue la conservaient en eux de plain-pied avec les idées de l'intelligence. Swann s'y reportait comme à une conception de l'amour et du bonheur dont immédiatement il savait aussi bien en quoi elle était particulière, qu'il le savait pour la « Princesse de Clèves », ou pour « René », quand leur nom se présentait à sa mémoire. Même quand il ne pensait pas à la petite phrase, elle existait latente dans son esprit au même titre que certaines autres notions sans équivalent, comme les notions de la lumière, du son, du relief, de la volupté physique, qui sont les riches possessions dont se diversifie et se pare notre domaine intérieur. Peut-être les perdrons-nous, peut-être s'effaceront-elles, si nous retournons au néant. Mais tant que nous vivons, nous ne pouvons pas plus faire que nous ne les ayons connues que nous ne le pouvons pour quelque objet réel, que nous ne pouvons par exemple douter de la lumière de la lampe qu'on allume devant les objets métamorphosés de notre chambre d'où s'est échappé jusqu'au souvenir de l'obscurité. Par là, la phrase de Vinteuil avait, comme tel thème de Tristan par exemple, qui nous représente aussi une certaine acquisition sentimentale, épousé notre condition mortelle, pris quelque chose d'humain qui était assez touchant. Son sort était lié à l'avenir, à la réalité de notre âme dont elle était un des ornements les plus particuliers, les mieux différenciés. Peut-être est-ce le néant qui est le vrai et tout notre rêve est-il inexistant, mais alors nous sentons qu'il faudra que ces phrases musicales, ces notions qui existent par rapport à lui, ne soient rien non plus. Nous périrons, mais nous avons pour otages ces captives divines qui suivront notre chance. Et la mort avec elle a quelque chose de moins amer, de moins inglorieux, peut-être de moins probable.

Swann n'avait donc pas tort de croire que la phrase de la sonate existât réellement. Certes, humaine à ce point de vue, elle appartenait pourtant à un ordre de créatures surnaturelles et que nous n'avons jamais vues, mais que malgré cela nous reconnaissons avec ravissement quand quelque explorateur de l'invisible arrive à en capter une, à l'amener, du monde divin où il a accès, briller quelques instants au-dessus du nôtre. C'est ce que Vinteuil avait fait pour la petite phrase. Swann sentait que le compositeur s'était contenté, avec ses instruments de musique, de la dévoiler, de la rendre visible, d'en suivre et d'en respecter le dessin d'une main si tendre, si prudente, si délicate et si sûre que le son s'altérait à tout moment, s'estompant pour indiquer une ombre, revivifié quand il lui fallait suivre à la piste un plus hardi contour. Et une preuve que Swann ne se trompait pas quand il croyait à l'existence réelle de cette phrase, c'est que tout amateur un peu fin se fût tout de suite aperçu de l'imposture, si Vinteuil ayant eu moins de puissance pour en voir et en rendre les formes, avait cherché à dissimuler, en ajoutant çà et là des traits de son cru, les lacunes de sa vision ou les défaillances de sa main.
