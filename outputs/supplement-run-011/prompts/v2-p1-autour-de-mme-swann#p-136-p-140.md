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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Françoise",
      "surface_forms": [
        "Françoise",
        "nurse"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Odette",
      "target": "Françoise",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Odette me fit l’éloge de notre vieille « nurse »… « On sent qu’elle vous est si dévouée, qu’elle est si bien. »",
      "explanation": "Odette explicitly praises Françoise’s devotion and worth. The narrator reports this as credible and consequential, noting it even inspired the Swanns’ sympathy for him."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Françoise’s standing is strongly elevated by Odette’s commendation, which the narrator endorses and which favorably impresses the Swanns."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-136-p-140"
}

### Candidate characters

[
  "Gilberte",
  "Mme Bontemps",
  "Mme Cottard",
  "Mme Verdurin",
  "Swann",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

– Tiens, ça a l'air bon ce que vous mangez là, cela me donne faim de vous voir manger du cake.

### Passage

– Eh bien, maman, nous vous invitons, répondait Gilberte.

– Mais non, mon trésor, qu'est-ce que diraient mes visites, j'ai encore Mme Trombert, Mme Cottard et Mme Bontemps, tu sais que chère Mme Bontemps ne fait pas des visites très courtes et elle vient seulement d'arriver. Qu'est-ce qu'ils diraient toutes ces bonnes gens de ne pas me voir revenir ; s'il ne vient plus personne, je reviendrai bavarder avec vous (ce qui m'amusera beaucoup plus) quand elles seront parties. Je crois que je mérite d'être un peu tranquille, j'ai eu quarante-cinq visites et sur quarante-cinq il y en a eu quarante-deux qui ont parlé du tableau de Gérôme ! Mais venez donc un de ces jours, me disait-elle, prendre votre thé avec Gilberte, elle vous le fera comme vous l'aimez, comme vous le prenez dans votre petit « studio », ajoutait-elle, tout en s'enfuyant vers ses visites et comme si ç'avait été quelque chose d'aussi connu de moi que mes habitudes (fût-ce celle que j'aurais eue de prendre le thé, si j'en avais jamais pris ; quand à un « studio » j'étais incertain si j'en avais un ou non) que j'étais venu chercher dans ce monde mystérieux. « Quand viendrez-vous ? Demain ? On vous fera des toasts aussi bons que chez Colombin. Non ? Vous êtes un vilain », disait-elle, car depuis qu'elle aussi commençait à avoir un salon, elle prenait les façons de Mme Verdurin, son ton de despotisme minaudier. Les toasts m'étant d'ailleurs aussi inconnus que Colombin, cette dernière promesse n'aurait pu ajouter à ma tentation. Il semblera plus étrange, puisque tout le monde parle ainsi et peut-être même maintenant à Combray, que je n'eusse pas à la première minute compris de qui voulait parler Odette, quand je l'entendis me faire l'éloge de notre vieille « nurse ». Je ne savais pas l'anglais, je compris bientôt pourtant que ce mot désignait Françoise. Moi qui, aux Champs-Élysées, avais eu si peur de la fâcheuse impression qu'elle devait produire, j'appris par Odette que c'est tout ce que Gilberte lui avait raconté sur ma « nurse » qui leur avait donné à elle et à son mari de la sympathie pour moi. « On sent qu'elle vous est si dévouée, qu'elle est si bien. » (Aussitôt je changeai entièrement d'avis sur Françoise. Par contre-coup, avoir une institutrice pourvue d'un caoutchouc et d'un plumet ne me sembla plus chose si nécessaire.) Enfin je compris, par quelques mots échappés à Odette sur Mme Blatin dont elle reconnaissait la bienveillance mais redoutait les visites, que des relations personnelles avec cette dame ne m'eussent pas été aussi précieuses que j'avais cru et n'eussent amélioré en rien ma situation chez les Swann.

Si j'avais déjà commencé d'explorer avec ces tressaillements de respect et de joie le domaine féerique qui contre toute attente avait ouvert devant moi ses avenues jusque-là fermées, pourtant c'était seulement en tant qu'ami de Gilberte. Le royaume dans lequel j'étais accueilli était contenu lui-même dans un plus mystérieux encore où Swann et sa femme menaient leur vie surnaturelle, et vers lequel ils se dirigeaient après m'avoir serré la main quand ils traversaient en même temps que moi, en sens inverse, l'antichambre. Mais bientôt je pénétrai aussi au coeur du Sanctuaire. Par exemple, Gilberte n'était pas là, M. ou Odette se trouvait à la maison. Ils avaient demandé qui avait sonné, et apprenant que c'était moi, m'avaient fait prier d'entrer un instant auprès d'eux, désirant que j'usasse dans tel ou tel sens, pour une chose ou pour une autre, de mon influence sur leur fille. Je me rappelais cette lettre si complète, si persuasive, que j'avais naguère écrite à Swann et à laquelle il n'avait même pas daigné répondre. J'admirais l'impuissance de l'esprit, du raisonnement et du coeur à opérer la moindre conversion, à résoudre une seule de ces difficultés, qu'ensuite la vie, sans qu'on sache seulement comment elle s'y est prise, dénoue si aisément. Ma position nouvelle d'ami de Gilberte, doué sur elle d'une excellente influence, me faisait maintenant bénéficier de la même faveur que si ayant eu pour camarade, dans un collège où on m'eût classé toujours premier, le fils d'un roi, j'avais dû à ce hasard mes petites entrées au Palais et des audiences dans la salle du trône ; Swann, avec une bienveillance infinie et comme s'il n'avait pas été surchargé d'occupations glorieuses, me faisait entrer dans sa bibliothèque et m'y laissait pendant une heure répondre par des balbutiements, des silences de timidité coupés de brefs et incohérents élans de courage, à des propos dont mon émoi m'empêchait de comprendre un seul mot ; il me montrait des objets d'art et des livres qu'il jugeait susceptibles de m'intéresser et dont je ne doutais pas d'avance qu'ils ne passassent infiniment en beauté tous ceux que possèdent le Louvre et la Bibliothèque Nationale, mais qu'il m'était impossible de regarder. À ces moments-là son maître d'hôtel m'aurait fait plaisir en me demandant de lui donner ma montre, mon épingle de cravate, mes bottines et de signer un acte qui le reconnaissait pour mon héritier : selon la belle expression populaire dont, comme pour les plus célèbres épopées, on ne connaît pas l'auteur, mais qui comme elles et contrairement à la théorie de Wolf en a eu certainement un (un de ces esprits inventifs et modestes ainsi qu'il s'en rencontre chaque année, lesquels font des trouvailles telles que « mettre un nom sur une figure » ; mais leur nom à eux, ils ne le font pas connaître), je ne savais plus ce que je faisais. Tout au plus m'étonnais-je quand la visite se prolongeait, à quel néant de réalisation, à quelle absence de conclusion heureuse, conduisaient ces heures vécues dans la demeure enchantée. Mais ma déception ne tenait ni à l'insuffisance des chefs-d'oeuvre montrés, ni à l'impossibilité d'arrêter sur eux un regard distrait. Car ce n'était pas la beauté intrinsèque des choses qui me rendait miraculeux d'être dans le cabinet de Swann, c'était l'adhérence à ces choses – qui eussent pu être les plus laides du monde – du sentiment particulier, triste et voluptueux que j'y localisais depuis tant d'années et qui l'imprégnait encore ; de même la multitude des miroirs, des brosses d'argent, des autels à saint Antoine de Padoue sculptés et peints par les plus grands artistes, ses amis, n'étaient pour rien dans le sentiment de mon indignité et de sa bienveillance royale qui m'était inspirés quand Odette me recevait un moment dans sa chambre où trois belles et imposantes créatures, sa première, sa deuxième et sa troisième femmes de chambre préparaient en souriant des toilettes merveilleuses, et vers laquelle, sur l'ordre proféré par le valet de pied en culotte courte que Madame désirait me dire un mot, je me dirigeais par le sentier sinueux d'un couloir tout embaumé à distance des essences précieuses qui exhalaient sans cesse du cabinet de toilette leurs effluves odoriférants.

Quand Odette était retournée auprès de ses visites, nous l'entendions encore parler et rire, car même devant deux personnes et comme si elle avait eu à tenir tête à tous les « camarades », elle élevait la voix, lançait les mots, comme elle avait si souvent, dans le petit clan, entendu faire à la « patronne », dans les moments où celle-ci « dirigeait la conversation ». Les expressions que nous avons récemment empruntées aux autres étant celles, au moins pendant un temps, dont nous aimons le plus à nous servir, Odette choisissait tantôt celles qu'elle avait apprises de gens distingués que son mari n'avait pu éviter de lui faire connaître (c'est d'eux qu'elle tenait le maniérisme qui consiste à supprimer l'article ou le pronom démonstratif devant un adjectif qualifiant une personne), tantôt de plus vulgaires (par exemple : « C'est un rien ! » mot favori d'une de ses amies) et cherchait à les placer dans toutes les histoires que, selon une habitude prise dans le « petit clan », elle aimait à raconter. Elle disait volontiers ensuite : « J'aime beaucoup cette histoire », « ah ! avouez, c'est une bien belle histoire ! » ; ce qui lui venait, par son mari, des Guermantes qu'elle ne connaissait pas.

Odette avait quitté la salle à manger, mais son mari qui venait de rentrer faisait à son tour une apparition auprès de nous. – Sais-tu si ta mère est seule, Gilberte ? – Non, elle a encore du monde, papa. – Comment, encore ? à sept heures ! C'est effrayant. La pauvre femme doit être brisée. C'est odieux. (À la maison j'avais toujours entendu, dans odieux, prononcer l'o long – audieux – mais M. et Odette disaient odieux, en faisant l'o bref.) Pensez, depuis deux heures de l'après-midi ! reprenait-il en se tournant vers moi. Et Camille me disait qu'entre quatre et cinq heures, il est bien venu douze personnes. Qu'est-ce que je dis douze, je crois qu'il m'a dit quatorze. Non, douze ; enfin je ne sais plus. Quand je suis rentré je ne songeais pas que c'était son jour, et en voyant toutes ces voitures devant la porte, je croyais qu'il y avait un mariage dans la maison. Et depuis un moment que je suis dans ma bibliothèque les coups de sonnette n'ont pas arrêté ; ma parole d'honneur, j'en ai mal à la tête. Et il y a encore beaucoup de monde près d'elle ? – Non, deux visites seulement. – Sais-tu qui ? – Mme Cottard et Mme Bontemps. – Ah ! la femme du chef de cabinet du ministre des Travaux publics. – J'sais que son mari est employé dans un ministère, mais j'sais pas au juste comme quoi, disait Gilberte en faisant l'enfant.
