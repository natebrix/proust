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
        "Bloch"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
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
      "confidence": 0.9,
      "evidence": "Bloch ... bondir ... venir féliciter la récitante ... et fit tant de bruit pour regagner sa place que Rachel dut attendre plus de cinq minutes avant de réciter la seconde poésie.",
      "explanation": "Bloch makes himself socially awkward and ostentatious, disrupting the session and attracting implicit disapproval."
    }
  ],
  "status_effects": [
    {
      "character": "Bloch",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "His loud and inappropriate intervention causes him to be poorly judged in the social context."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-81-p-85"
}

### Candidate characters

[
  "Gilberte",
  "M. Verdurin",
  "M. de Vaugoubert",
  "Robert de Saint-Loup",
  "Swann",
  "comte de Forcheville",
  "duchesse de Guermantes",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

* * *

### Passage

La conversation que nous tenions, Gilberte et moi, fut interrompue par la voix de Rachel qui venait de s'élever. Le jeu de celle-ci était intelligent, car il présupposait la poésie que l'actrice était en train de dire comme un tout existant avant cette récitation et dont nous n'entendions qu'un fragment, comme si l'artiste, passant sur un chemin, s'était trouvée pendant quelques instants à portée de notre oreille. Néanmoins, les auditeurs avaient été stupéfaits en voyant cette femme, avant d'avoir émis un seul son, plier les genoux, tendre les bras, en berçant quelque être invisible, devenir cagneuse, et tout d'un coup, pour dire des vers fort connus, prendre un ton suppliant.

L'annonce d'une poésie que presque tout le monde connaissait avait fait plaisir. Mais quand on avait vu Rachel, avant de commencer, chercher partout des yeux d'un air égaré, lever les mains d'un air suppliant et pousser comme un gémissement à chaque mot, chacun se sentit gêné, presque choqué de cette exhibition de sentiments. Personne ne s'était dit que réciter des vers pouvait être quelque chose comme cela. Peu à peu on s'habitue, c'est-à-dire qu'on oublie la première sensation de malaise, on dégage ce qui est bien, on compare dans son esprit diverses manières de réciter, pour se dire : ceci c'est mieux, ceci moins bien. La première fois de même, dans une cause simple, lorsqu'on voit un avocat s'avancer, lever en l'air un bras d'où retombe la toge, commencer d'un ton menaçant, on n'ose pas regarder les voisins. Car on se figure que c'est grotesque, mais, après tout, c'est peut-être magnifique et on attend d'être fixé. Tout le monde se regardait, ne sachant trop quelle tête faire ; quelques jeunesses mal élevées étouffèrent un fou rire ; chacun jetait à la dérobée sur son voisin le regard furtif que dans les repas élégants, quand on a auprès de soi un instrument nouveau, fourchette à homard, râpe à sucre, etc., dont on ne connaît pas le but et le maniement, on attache sur un convive plus autorisé qui, espère-t-on, s'en servira avant vous et vous donnera ainsi la possibilité de l'imiter. Ainsi fait-on encore quand quelqu'un cite un vers qu'on ignore mais qu'on veut avoir l'air de connaître et à qui, comme en cédant le pas devant une porte, on laisse à un plus instruit, comme une faveur, le plaisir de dire de qui il est. Tel, en entendant l'actrice, chacun attendait, la tête baissée et l'oeil investigateur, que d'autres prissent l'initiative de rire ou de critiquer, ou de pleurer ou d'applaudir. Mme de Forcheville, revenue exprès de Guermantes, d'où la duchesse, comme nous le verrons, était à peu près expulsée, avait pris une mine attentive, tendue, presque carrément désagréable, soit pour montrer qu'elle était connaisseuse et ne venait pas en mondaine, soit par hostilité pour les gens moins versés dans la littérature qui eussent pu lui parler d'autre chose, soit par contention de toute sa personne afin de savoir si elle « aimait » ou si elle n'aimait pas, ou peut-être parce que, tout en trouvant cela « intéressant », elle n'« aimait » pas, du moins, la manière de dire certains vers. Cette attitude eût dû être plutôt adoptée, semble-t-il, par la princesse de Guermantes. Mais comme c'était chez elle, et que, devenue aussi avare que riche, elle était décidée à ne donner que cinq roses à Rachel, elle faisait la claque. Elle provoquait l'enthousiasme et faisait la presse en poussant à tous moments des exclamations ravies. Là seulement elle se retrouvait Verdurin, car elle avait l'air d'écouter les vers pour son propre plaisir, d'avoir eu l'envie qu'on vînt les lui dire, à elle toute seule, et qu'il y eût par hasard là cinq cents personnes, à qui elle avait permis de venir comme en cachette assister à son propre plaisir.

Cependant, je remarquai sans aucune satisfaction d'amour-propre, car elle était devenue vieille et laide, que Rachel me faisait de l'oeil, avec une certaine réserve d'ailleurs. Pendant toute la récitation, elle laissa palpiter dans ses yeux un sourire réprimé et pénétrant qui semblait l'amorce d'un acquiescement qu'elle eût souhaité venir de moi. Cependant, quelques vieilles dames, peu habituées aux récitations poétiques, disaient à un voisin : « Vous avez vu ? », faisant allusion à la mimique solennelle, tragique, de l'actrice, et qu'elles ne savaient comment qualifier. La Mme de Guermantes sentit le léger flottement et décida de la victoire en s'écriant : « C'est admirable ! » au beau milieu du poème, qu'elle crut peut-être terminé. Plus d'un invité tint alors à souligner cette exclamation d'un regard approbateur et d'une inclinaison de tête, pour montrer moins peut-être leur compréhension de la récitante que leurs relations avec la duchesse. Quand le poème fut fini, comme nous étions à côté de Rachel, j'entendis celle-ci remercier Mme de Guermantes et en même temps, profitant de ce que j'étais à côté de la duchesse, elle se tourna vers moi et m'adressa un gracieux bonjour. Je compris alors qu'au contraire des regards passionnés du fils de M. de Vaugoubert, que j'avais pris pour le bonjour de quelqu'un qui se trompait, ce que j'avais pris chez Rachel pour un regard de désir n'était qu'une provocation contenue à se faire reconnaître et saluer par moi. Je répondis par un salut souriant au sien. « Je suis sûre qu'il ne me reconnaît pas, dit en minaudant la récitante à la duchesse. – Mais si, dis-je avec assurance, je vous ai reconnue tout de suite. »

Si, pendant les plus beaux vers de La Fontaine, cette femme, qui les récitait avec tant d'assurance, n'avait pensé, soit par bonté, ou bêtise, ou gêne, qu'à la difficulté de me dire bonjour, pendant les mêmes beaux vers Bloch n'avait songé qu'à faire ses préparatifs pour pouvoir, dès la fin de la poésie, bondir comme un assiégé qui tente une sortie, et passant, sinon sur le corps, du moins sur les pieds de ses voisins, venir féliciter la récitante, soit par une conception erronée du devoir, soit par désir d'ostentation.

« C'était bien beau », dit-il à Rachel, et ayant dit ces simples mots, son désir étant satisfait, il repartit et fit tant de bruit pour regagner sa place que Rachel dut attendre plus de cinq minutes avant de réciter la seconde poésie. Quand elle eut fini celle-ci, les Deux Pigeons, Mme de Monrienval s'approcha de Mme de Saint-Loup, qu'elle savait fort lettrée sans se rappeler assez qu'elle avait l'esprit subtil et sarcastique de son père, et lui demanda : « C'est bien la fable de La Fontaine, n'est-ce pas ? » croyant bien l'avoir reconnue mais n'étant pas absolument certaine, car elle connaissait fort mal les fables de La Fontaine et, de plus, croyait que c'était des choses d'enfants qu'on ne récitait pas dans le monde. Pour avoir un tel succès l'artiste avait sans doute pastiché des fables de La Fontaine, pensait la bonne dame. Or, Gilberte, jusque-là impassible, l'enfonça sans le vouloir dans cette idée, car n'aimant pas Rachel et voulant dire qu'il ne restait rien des fables avec une diction pareille, elle le dit de cette nuance trop subtile qui était celle de son père et qui laissait les personnes naïves dans le doute sur ce qu'il voulait dire. Généralement plus moderne, quoique fille de Swann – comme un canard couvé par une poule – elle était assez lakiste et se contentait de dire : « Je trouve d'un touchant, c'est d'une sensibilité charmante. » Mais à Mme de Morienval Gilberte répondit sous cette forme fantaisiste de Swann à laquelle se trompaient les gens qui prennent tout au pied de la lettre : « Un quart est de l'invention de l'interprète, un quart de la folie, un quart n'a aucun sens, le reste est de La Fontaine », ce qui permit à Mme de Morienval de soutenir que ce qu'on venait d'entendre n'était pas les Deux Pigeons de La Fontaine mais un arrangement où tout au plus un quart était de La Fontaine, ce qui n'étonna personne, vu l'extraordinaire ignorance de ce public.
