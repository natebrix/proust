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
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "l'Ambassadeur",
        "le père Norpois"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Norpois",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.86,
      "evidence": "« parce que Norpois lui avait dit qu'il devrait me laisser entendre la Berma »; « sur une seule parole de Norpois »; « Norpois… lui avait assuré qu'on pouvait, comme écrivain, s'attirer autant de considération… que dans les ambassades »; « il t'y fera entrer, il réglera cela, c'est un vieux malin ».",
      "explanation": "The narrator presents Norpois as a decisive authority whose single word alters the father's decisions (on attending la Berma and on a literary career) and whose connections are treated as able to 'arrange' publication, thereby elevating his local standing and influence."
    }
  ],
  "status_effects": [
    {
      "character": "Norpois",
      "dimension": "rhetorical_position",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Norpois's opinion carries decisive weight; others defer to his judgment and connections."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-6-p-10"
}

### Candidate characters

[
  "Gilberte",
  "Swann",
  "la Berma",
  "la grand-mère",
  "la mère du narrateur",
  "le directeur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Quant à la mère du narrateur, peut-être l'Ambassadeur n'avait-il pas par lui-même le genre d'intelligence vers lequel elle se sentait le plus attirée. Et je dois dire que la conversation de Norpois était un répertoire si complet des formes surannées du langage particulières à une carrière, à une classe, et à un temps – un temps qui, pour cette carrière et cette classe-là, pourrait bien ne pas être tout à fait aboli – que je regrette parfois de n'avoir pas retenu purement et simplement les propos que je lui ai entendu tenir. J'aurais ainsi obtenu un effet de démodé, à aussi bon compte et de la même façon que cet acteur du Palais-Royal à qui on demandait où il pouvait trouver ses surprenants chapeaux et qui répondait : « Je ne trouve pas mes chapeaux. Je les garde. » En un mot, je crois que la mère du narrateur jugeait Norpois un peu « vieux jeu », ce qui était loin de lui sembler déplaisant au point de vue des manières, mais la charmait moins dans le domaine, sinon des idées – car celles de Norpois étaient fort modernes – mais des expressions. Seulement, elle sentait que c'était flatter délicatement son mari que de lui parler avec admiration du diplomate qui lui marquait une prédilection si rare. En fortifiant dans l'esprit de le père du narrateur la bonne opinion qu'il avait de Norpois, et par là en le conduisant à en prendre une bonne aussi de lui-même, elle avait conscience de remplir celui de ses devoirs qui consistait à rendre la vie agréable à son époux, comme elle faisait quand elle veillait à ce que la cuisine fût soignée et le service silencieux. Et comme elle était incapable de mentir à le père du narrateur, elle s'entraînait elle-même à admirer l'Ambassadeur pour pouvoir le louer avec sincérité. D'ailleurs, elle goûtait naturellement son air de bonté, sa politesse un peu désuète (et si cérémonieuse que quand, marchant en redressant sa haute taille, il apercevait la mère du narrateur qui passait en voiture, avant de lui envoyer un coup de chapeau, il jetait au loin un cigare à peine commencé) ; sa conversation si mesurée, où il parlait de lui-même le moins possible et tenait toujours compte de ce qui pouvait être agréable à l'interlocuteur, sa ponctualité tellement surprenante à répondre à une lettre que quand, venant de lui en envoyer une, le père du narrateur reconnaissait l'écriture de Norpois sur une enveloppe, son premier mouvement était de croire que par mauvaise chance leur correspondance s'était croisée : on eût dit qu'il existait, pour lui, à la poste, des levées supplémentaires et de luxe. Ma mère s'émerveillait qu'il fut si exact quoique si occupé, si aimable quoique si répandu, sans songer que les « quoique » sont toujours des « parce que » méconnus, et que (de même que les vieillards sont étonnants pour leur âge, les rois pleins de simplicité, et les provinciaux au courant de tout) c'était les mêmes habitudes qui permettaient à Norpois de satisfaire à tant d'occupations et d'être si ordonné dans ses réponses, de plaire dans le monde et d'être aimable avec nous. De plus, l'erreur de la mère du narrateur comme celle de toutes les personnes qui ont trop de modestie, venait de ce qu'elle mettait les choses qui la concernaient au-dessous, et par conséquent en dehors des autres. La réponse qu'elle trouvait que l'ami de le père du narrateur avait eu tant de mérite à nous adresser rapidement parce qu'il écrivait par jour beaucoup de lettres, elle l'exceptait de ce grand nombre de lettres dont ce n'était que l'une ; de même elle ne considérait pas qu'un dîner chez nous fût pour Norpois un des actes innombrables de sa vie sociale : elle ne songeait pas que l'Ambassadeur avait été habitué autrefois dans la diplomatie à considérer les dîners en ville comme faisant partie de ses fonctions, et à y déployer une grâce invétérée dont c'eût été trop lui demander de se départir par extraordinaire quand il venait dîner chez nous.

### Passage

Le premier dîner que Norpois fit à la maison, une année où je jouais encore aux Champs-Élysées, est resté dans ma mémoire, parce que l'après-midi de ce même jour fut celui où j'allai enfin entendre la Berma, en « matinée », dans Phèdre, et aussi parce qu'en causant avec Norpois je me rendis compte tout d'un coup, et d'une façon nouvelle, combien les sentiments éveillés en moi par tout ce qui concernait Gilberte Swann et ses parents différaient de ceux que cette même famille faisait éprouver à n'importe quelle autre personne.

Ce fut sans doute en remarquant l'abattement où me plongeait l'approche des vacances du jour de l'an pendant lesquelles, comme elle me l'avait annoncé elle-même, je ne devais pas voir Gilberte, qu'un jour, pour me distraire, ma mère me dit : « Si tu as encore le même grand désir d'entendre la Berma, je crois que ton père permettrait peut-être que tu y ailles : ta grand'mère pourrait t'y emmener. »

Mais c'était parce que Norpois lui avait dit qu'il devrait me laisser entendre la Berma, que c'était pour un jeune homme un souvenir à garder, que mon père, jusque-là si hostile à ce que j'allasse perdre mon temps à risquer de prendre du mal pour ce qu'il appelait, au grand scandale de ma grand'mère, des inutilités, n'était plus loin de considérer cette soirée préconisée par l'Ambassadeur comme faisant vaguement partie d'un ensemble de recettes précieuses pour la réussite d'une brillante carrière. Ma grand'mère, qui en renonçant pour moi au profit que, selon elle, j'aurais trouvé à entendre la Berma, avait fait un gros sacrifice à l'intérêt de ma santé, s'étonnait que celui-ci devînt négligeable sur une seule parole de Norpois. Mettant ses espérances invincibles de rationaliste dans le régime de grand air et de coucher de bonne heure qui m'avait été prescrit, elle déplorait comme un désastre cette infraction que j'allais y faire et, sur un ton navré, disait : « Comme vous êtes léger » à mon père qui, furieux, répondait : « Comment, c'est vous maintenant qui ne voulez pas qu'il y aille ! c'est un peu fort, vous qui nous répétiez tout le temps que cela pouvait lui être utile. »

Mais Norpois avait changé, sur un point bien plus important pour moi, les intentions de mon père. Celui-ci avait toujours désiré que je fusse diplomate, et je ne pouvais supporter l'idée que, même si je devais rester quelque temps attaché au ministère, je risquasse d'être envoyé un jour comme ambassadeur dans des capitales que Gilberte n'habiterait pas. J'aurais préféré revenir aux projets littéraires que j'avais autrefois formés et abandonnés au cours de mes promenades du côté de Guermantes. Mais mon père avait fait une constante opposition à ce que je me destinasse à la carrière des lettres qu'il estimait fort inférieure à la diplomatie, lui refusant même le nom de carrière, jusqu'au jour où Norpois, qui n'aimait pas beaucoup les agents diplomatiques de nouvelles couches, lui avait assuré qu'on pouvait, comme écrivain, s'attirer autant de considération, exercer autant d'action et garder plus d'indépendance que dans les ambassades.

« Hé bien ! je ne l'aurais pas cru, le père Norpois n'est pas du tout opposé à l'idée que tu fasses de la littérature », m'avait dit mon père. Et comme, assez influent lui-même, il croyait qu'il n'y avait rien qui ne s'arrangeât, ne trouvât sa solution favorable dans la conversation des gens importants : « Je le ramènerai dîner un de ces soirs en sortant de la Commission. Tu causeras un peu avec lui pour qu'il puisse t'apprécier. Écris quelque chose de bien que tu puisses lui montrer ; il est très lié avec le directeur de la Revue des Deux-Mondes, il t'y fera entrer, il réglera cela, c'est un vieux malin ; et, ma foi, il a l'air de trouver que la diplomatie, aujourd'hui !... »
