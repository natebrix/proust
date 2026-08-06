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
      "canonical_name": "Andrée",
      "surface_forms": [
        "Andrée"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Andrée",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.91,
      "evidence": "« Andrée consultée comme plus grande et comme plus calée... »; elle refait la lettre, corrige, prescrit un plan, et « garda le flegme souriant d'un dandy femelle »; « un sentiment de bienveillante supériorité ».",
      "explanation": "The passage establishes Andrée as an intellectual and rhetorical authority: she is consulted, corrects and explains, all while displaying a serene superiority, which Albertine's admiration reinforces."
    }
  ],
  "status_effects": [
    {
      "character": "Andrée",
      "dimension": "rhetorical_position",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "She dominates the discussion, sets the standards, and is recognized as the most competent."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-451-p-455"
}

### Candidate characters

[
  "Albertine",
  "Robert de Saint-Loup",
  "le narrateur"
]

### Prior local context (optional)

Les yeux d'Albertine n'avaient cessé d'étinceler pendant qu'elle faisait cette lecture.

### Passage

« C'est à croire qu'elle a copié cela, s'écria-t-elle quand elle eut fini. Jamais je n'aurais cru Gisèle capable de pondre un devoir pareil. Et ces vers qu'elle cite ! Où a-t-elle pu aller chiper ça ? » L'admiration d'Albertine, changeant il est vrai d'objet, mais encore accrue, ne cessa pas, ainsi que l'application la plus soutenue, de lui faire « sortir les yeux de la tête » tout le temps qu'Andrée consultée comme plus grande et comme plus calée, d'abord parla du devoir de Gisèle avec une certaine ironie, puis, avec un air de légèreté qui dissimulait mal un sérieux véritable, refit à sa façon la même lettre. « Ce n'est pas mal, dit-elle à Albertine, mais si j'étais toi et qu'on me donne le même sujet, ce qui peut arriver, car on le donne très souvent, je ne ferais pas comme cela. Voilà comment je m'y prendrais. D'abord si j'avais été Gisèle je ne me serais pas laissée emballer et j'aurais commencé par écrire sur une feuille à part mon plan. En première ligne la position de la question et l'exposition du sujet, puis les idées générales à faire entrer dans le développement. Enfin l'appréciation, le style, la conclusion. Comme cela, en s'inspirant d'un sommaire, on sait où on va. Dès l'exposition du sujet ou si tu aimes mieux, Titine, puisque c'est une lettre, dès l'entrée en matière, Gisèle a gaffé. Écrivant à un homme du XVIIe siècle Sophocle ne devait pas écrire : « Mon cher ami. » – Elle aurait dû, en effet, lui faire dire : mon cher Racine, s'écria fougueusement Albertine. Ç'aurait été bien mieux. – Non, répondit Andrée sur un ton un peu persifleur, elle aurait dû mettre : « Monsieur ». De même pour finir elle aurait dû trouver quelque chose comme : « Souffrez, Monsieur (tout au plus, cher Monsieur), que je vous dise ici les sentiments d'estime avec lesquels j'ai l'honneur d'être votre serviteur. » D'autre part, Gisèle dit que les choeurs sont dans Athalie une nouveauté. Elle oublie Esther, et deux tragédies peu connues, mais qui ont été précisément analysées cette année par le Professeur, de sorte que rien qu'en les citant, comme c'est son dada, on est sûre d'être reçue. Ce sont : Les Juives, de Saint-Loup Garnier, et l'Aman, de Montchrestien. » Andrée cita ces deux titres sans parvenir à cacher un sentiment de bienveillante supériorité qui s'exprima dans un sourire, assez gracieux, d'ailleurs. Albertine n'y tint plus : « Andrée, tu es renversante, s'écria-t-elle. Tu vas m'écrire ces deux titres-là. Crois-tu ? quelle chance si je passais là-dessus, même à l'oral, je les citerais aussitôt et je ferais un effet boeuf. » Mais dans la suite chaque fois qu'Albertine demanda à Andrée de lui redire les noms des deux pièces pour qu'elle les inscrivît, l'amie si savante prétendait les avoir oubliés et ne les lui rappela jamais. « Ensuite, reprit Andrée sur un ton d'imperceptible dédain à l'égard de camarades plus puériles, mais heureuse pourtant de se faire admirer et attachant à la manière dont elle aurait fait sa composition plus d'importance qu'elle ne voulait le laisser voir, Sophocle aux Enfers doit être bien informé. Il doit donc savoir que ce n'est pas devant le grand public, mais devant le Roi-Soleil et quelques courtisans privilégiés que fut représentée Athalie. Ce que Gisèle a dit à ce propos de l'estime des connaisseurs n'est pas mal du tout, mais pourrait être complété. Sophocle devenu immortel peut très bien avoir le don de la prophétie et annoncer que selon Voltaire Athalie ne sera pas seulement « le chef-d'oeuvre de Racine, mais celui de l'esprit humain ». Albertine buvait toutes ces paroles. Ses prunelles étaient en feu. Et c'est avec l'indignation la plus profonde qu'elle repoussa la proposition de Rosemonde de se mettre à jouer. « Enfin, dit Andrée du même ton détaché, désinvolte, un peu railleur et assez ardemment convaincu, si Gisèle avait posément noté d'abord les idées générales qu'elle avait à développer, elle aurait peut-être pensé à ce que j'aurais fait, moi, montrer la différence qu'il y a dans l'inspiration religieuse des choeurs de Sophocle et de ceux de Racine. J'aurais fait faire par Sophocle la remarque que si les choeurs de Racine sont empreints de sentiments religieux comme ceux de la tragédie grecque, pourtant il ne s'agit pas des mêmes dieux. Celui de Joad n'a rien à voir avec celui de Sophocle. Et cela amène tout naturellement, après la fin du développement, la conclusion : « Qu'importe que les croyances soient différentes. » Sophocle se ferait un scrupule d'insister là-dessus. Il craindrait de blesser les convictions de Racine et glissant à ce propos quelques mots sur ses maîtres de Port-Royal, il préfère féliciter son émule de l'élévation de son génie poétique. »

L'admiration et l'attention avaient donné si chaud à Albertine qu'elle suait à grosses gouttes. Andrée gardait le flegme souriant d'un dandy femelle. « Il ne serait pas mauvais non plus de citer quelques jugements des critiques célèbres », dit-elle, avant qu'on se remît à jouer. « Oui, répondit Albertine, on m'a dit cela. Les plus recommandables en général, n'est-ce pas, sont les jugements de Sainte-Beuve et de Merlet ? – Tu ne te trompes pas absolument, répliqua Andrée qui se refusa d'ailleurs à lui écrire les deux autres noms malgré les supplications d'Albertine, Merlet et Sainte-Beuve ne font pas mal. Mais il faut surtout citer Deltour et Gascq-Desfossés. »

Pendant ce temps, je songeais à la petite feuille de bloc-notes que m'avait passée Albertine : « Je vous aime bien », et une heure plus tard, tout en descendant les chemins qui ramenaient, un peu trop à pic à mon gré, vers Balbec, je me disais que c'était avec elle que j'aurais mon roman.

L'état caractérisé par l'ensemble des signes auxquels nous reconnaissons d'habitude que nous sommes amoureux, tels les ordres que je donnais à l'hôtel de ne m'éveiller pour aucune visite, sauf si c'était celle d'une ou l'autre de ces jeunes filles, ces battements de coeur en les attendant (quelle que fût celle qui dût venir), et ces jours-là ma rage si je n'avais pu trouver un coiffeur pour me raser et devais paraître enlaidi devant Albertine, Rosemonde ou Andrée ; sans doute cet état, renaissant alternativement pour l'une ou l'autre, était aussi différent de ce que nous appelons amour, que diffère de la vie humaine celle des zoophytes où l'existence, l'individualité si l'on peut dire, est répartie entre différents organismes. Mais l'histoire naturelle nous apprend qu'une telle organisation animale est observable, et que notre propre vie, pour peu qu'elle soit déjà un peu avancée, n'est pas moins affirmative sur la réalité d'états insoupçonnés de nous autrefois et par lesquels nous devons passer, quitte à les abandonner ensuite. Tel pour moi cet état amoureux divisé simultanément entre plusieurs jeunes filles. Divisé ou plutôt indivisé, car le plus souvent ce qui m'était délicieux, différent du reste du monde, ce qui commençait à me devenir cher au point que l'espoir de le retrouver le lendemain était la meilleure joie de ma vie, c'était plutôt tout le groupe de ces jeunes filles, pris dans l'ensemble de ces après-midi sur la falaise, pendant ces heures éventées, sur cette bande d'herbe où étaient posées ces figures, si excitantes pour mon imagination, d'Albertine, de Rosemonde, d'Andrée ; et cela, sans que j'eusse pu dire laquelle me rendait ces lieux si précieux, laquelle j'avais le plus envie d'aimer. Au commencement d'un amour comme à sa fin, nous ne sommes pas exclusivement attachés à l'objet de cet amour, mais plutôt le désir d'aimer dont il va procéder (et plus tard le souvenir qu'il laisse) erre voluptueusement dans une zone de charmes interchangeables – charmes parfois simplement de nature, de gourmandise, d'habitation – assez harmoniques entre eux pour qu'il ne se sente, auprès d'aucun, dépaysé. D'ailleurs comme, devant elles, je n'étais pas encore blasé par l'habitude, j'avais la faculté de les voir, autant dire d'éprouver un étonnement profond chaque fois que je me retrouvais en leur présence. Sans doute pour une part cet étonnement tient à ce que l'être nous présente alors une nouvelle face de lui-même ; mais tant est grande la multiplicité de chacun, de la richesse des lignes de son visage et de son corps, lignes desquelles si peu se retrouvent aussitôt que nous ne sommes plus auprès de la personne, dans la simplicité arbitraire de notre souvenir, comme la mémoire a choisi telle particularité qui nous a frappés, l'a isolée, l'a exagérée, faisant d'une femme qui nous a paru grande une étude où la longueur de sa taille est démesurée, ou d'une femme qui nous a semblé rose et blonde une pure « Harmonie en rose et or », au moment où de nouveau cette femme est près de nous, toutes les autres qualités oubliées qui font équilibre à celle-là nous assaillent, dans leur complexité confuse, diminuant la hauteur, noyant le rose, et substituant à ce que nous sommes venus exclusivement chercher d'autres particularités que nous nous rappelons avoir remarquées la première fois et dont nous ne comprenons pas que nous ayons pu si peu nous attendre à les revoir. Nous nous souvenons, nous allons au devant d'un paon et nous trouvons une pivoine. Et cet étonnement inévitable n'est pas le seul ; car à côté de celui-là il y en a un autre né de la différence, non plus entre les stylisations du souvenir et la réalité, mais entre l'être que nous avons vu la dernière fois, et celui qui nous apparaît aujourd'hui sous un autre angle, nous montrant un nouvel aspect. Le visage humain est vraiment comme celui du Dieu d'une théogénie orientale, toute une grappe de visages juxtaposés dans des plans différents et qu'on ne voit pas à la fois.

Mais pour une grande part, notre étonnement vient surtout de ce que l'être nous présente aussi une même face. Il nous faudrait un si grand effort pour recréer tout ce qui nous a été fourni par ce qui n'est pas nous – fût-ce le goût d'un fruit – qu'à peine l'impression reçue, nous descendons insensiblement la pente du souvenir et sans nous en rendre compte, en très peu de temps, nous sommes très loin de ce que nous avons senti. De sorte que chaque entrevue est une espèce de redressement qui nous ramène à ce que nous avions bien vu. Nous ne nous en souvenions déjà tant ce qu'on appelle se rappeler un être c'est en réalité l'oublier. Mais aussi longtemps que nous savons encore voir, au moment où le trait oublié nous apparaît, nous le reconnaissons, nous sommes obligés de rectifier la ligne déviée, et ainsi la perpétuelle et féconde surprise qui rendait si salutaires et assouplissants pour moi ces rendez-vous quotidiens avec les belles jeunes filles du bord de la mer était faite, tout autant que de découvertes, de réminiscence. En ajoutant à cela l'agitation éveillée par ce qu'elles étaient pour moi, qui n'était jamais tout à fait ce que j'avais cru et qui faisait que l'espérance de la prochaine réunion n'était plus semblable à la précédente espérance, mais au souvenir encore vibrant du dernier entretien, on comprendra que chaque promenade donnait un violent coup de barre à mes pensées, et non pas du tout dans le sens que, dans la solitude de ma chambre, j'avais pu tracer à tête reposée. Cette direction-là était oubliée, abolie, quand je rentrais vibrant comme une ruche des propos qui m'avaient troublé, et qui retentissaient longtemps en moi. Chaque être est détruit quand nous cessons de le voir ; puis son apparition suivante est une création nouvelle, différente de celle qui l'a immédiatement précédée, sinon de toutes. Car le minimum de variété qui puisse régner dans ces créations est de deux. Nous souvenant d'un coup d'oeil énergique, d'un air hardi, c'est inévitablement la fois suivante par un profil quasi languide, par une sorte de douceur rêveuse, choses négligées par nous dans le précédent souvenir, que nous serons, à la prochaine rencontre, étonnés, c'est-à-dire presque uniquement frappés. Dans la confrontation de notre souvenir à la nouvelle réalité, c'est cela qui marquera notre déception ou notre surprise, nous apparaîtra comme la retouche de la réalité en nous avertissant que nous nous étions mal rappelés. À son tour l'aspect, la dernière fois négligé, du visage, et à cause de cela même le plus saisissant cette fois-ci, le plus réel, le plus rectificatif, deviendra matière à rêverie, à souvenirs. C'est un profil langoureux et rond, une expression douce, rêveuse que nous désirerons revoir. Et alors de nouveau la fois suivante, ce qu'il y a de volontaire dans les yeux perçants, dans le nez pointu, dans les lèvres serrées, viendra corriger l'écart entre notre désir et l'objet auquel il a cru correspondre. Bien entendu, cette fidélité aux impressions premières, et purement physiques, retrouvées à chaque fois auprès de mes amies, ne concernait pas que les traits de leur visage puisqu'on a vu que j'étais aussi sensible à leur voix, plus troublante peut-être (car elle n'offre pas seulement les mêmes surfaces singulières et sensuelles que lui, elle fait partie de l'abîme inaccessible qui donne le vertige des baisers sans espoir), leur voix pareille au son unique d'un petit instrument, où chacune se mettait tout entière et qui n'était qu'à elle. Tracée par une inflexion, telle ligne profonde d'une de ces voix m'étonnait quand je la reconnaissais après l'avoir oubliée. Si bien que les rectifications qu'à chaque rencontre nouvelle j'étais obligé de faire, pour le retour à la parfaite justesse, étaient aussi bien d'un accordeur ou d'un maître de chant que d'un dessinateur.
