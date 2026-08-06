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
      "canonical_name": "Bergotte",
      "surface_forms": [
        "Bergotte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bergotte",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "La malveillance avec laquelle Bergotte parlait ainsi à un étranger d'amis chez qui il était reçu… était aussi nouvelle pour moi que le ton presque tendre que chez les Swann il prenait… « Tout ceci de vous à moi ».",
      "explanation": "The narrator highlights Bergotte's worldly duplicity (tenderness on the surface, gossip behind the scenes), which locally diminishes him in esteem."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "The narrator frames him as malicious and duplicitous, which lowers his local image."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-241-p-245"
}

### Candidate characters

[
  "Gilberte",
  "Norpois",
  "Odette",
  "Swann",
  "docteur Cottard",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Il ne me persuadait certes pas ; et pourtant je me sentais plus heureux, moins à l'étroit. À cause de ce que m'avait dit Norpois, j'avais considéré mes moments de rêverie, d'enthousiasme, de confiance en moi, comme purement subjectifs et sans vérité. Or, selon Bergotte qui avait l'air de connaître mon cas, il semblait que le symptôme à négliger c'était au contraire mes doutes, mon dégoût de moi-même. Surtout ce qu'il avait dit de Norpois ôtait beaucoup de sa force à une condamnation que j'avais crue sans appel.

### Passage

« Êtes-vous bien soigné ? me demanda Bergotte. Qui est-ce qui s'occupe de votre santé ? » Je lui dis que j'avais vu et reverrais sans doute Cottard. « Mais ce n'est pas ce qu'il vous faut ! me répondit-il. Je ne le connais pas comme médecin, Mais je l'ai vu chez Odette. C'est un imbécile. À supposer que cela n'empêche pas d'être un bon médecin, ce que j'ai peine à croire, cela empêche d'être un bon médecin pour artistes, pour gens intelligents. Les gens comme vous ont besoin de médecins appropriés, je dirais presque de régimes, de médicaments particuliers. Cottard vous ennuiera et rien que l'ennui empêchera son traitement d'être efficace. Et puis ce traitement ne peut pas être le même pour vous que pour un individu quelconque. Les trois quarts du mal des gens intelligents viennent de leur intelligence. Il leur faut au moins un médecin qui connaisse ce mal-là. Comment voulez-vous que Cottard puisse vous soigner, il a prévu la difficulté de digérer les sauces, l'embarras gastrique, mais il n'a pas prévu la lecture de Shakespeare... Aussi ses calculs ne sont plus justes avec vous, l'équilibre est rompu, c'est toujours le petit ludion qui remonte. Il vous trouvera une dilatation de l'estomac, il n'a pas besoin de vous examiner puisqu'il l'a d'avance dans son oeil. Vous pouvez la voir, elle se reflète dans son lorgnon. » Cette manière de parler me fatiguait beaucoup, je me disais avec la stupidité du bon sens : « Il n'y a pas plus de dilatation de l'estomac reflétée dans le lorgnon du professeur Cottard que de sottises cachées dans le gilet blanc de Norpois. » « Je vous conseillerais plutôt, poursuivit Bergotte, le docteur du Boulbon, qui est tout à fait intelligent. – C'est un grand admirateur de vos oeuvres », lui répondis-je. Je vis que Bergotte le savait et j'en conclus que les esprits fraternels se rejoignent vite, qu'on a peu de vrais « amis inconnus ». Ce que Bergotte me dit au sujet de Cottard me frappa tout en étant contraire à tout ce que je croyais. Je ne m'inquiétais nullement de trouver mon médecin ennuyeux ; j'attendais de lui que, grâce à un art dont les lois m'échappaient, il rendît au sujet de ma santé un indiscutable oracle en consultant mes entrailles. Et je ne tenais pas à ce que, à l'aide d'une intelligence où j'aurais pu le suppléer, il cherchât à comprendre la mienne que je ne me représentais que comme un moyen indifférent en soi-même de tâcher d'atteindre des vérités extérieures. Je doutais beaucoup que les gens intelligents eussent besoin d'une autre hygiène que les imbéciles et j'étais tout prêt à me soumettre à celle de ces derniers. « Quelqu'un qui aurait besoin d'un bon médecin, c'est notre ami Swann », dit Bergotte. Et comme je demandais s'il était malade. « Hé bien c'est l'homme qui a épousé une fille, qui avale par jour cinquante couleuvres de femmes qui ne veulent pas recevoir la sienne, ou d'hommes qui ont couché avec elle. On les voit, elles lui tordent la bouche. Regardez un jour le sourcil circonflexe qu'il a quand il rentre, pour voir qui il y a chez lui. »

«  La malveillance avec laquelle Bergotte parlait ainsi à un étranger d'amis chez qui il était reçu depuis si longtemps était aussi nouvelle pour moi que le ton presque tendre que chez les Swann il prenait à tous moments avec eux. Certes, une personne comme ma grand'tante, par exemple, eût été incapable, avec aucun de nous, de ces gentillesses que j'avais entendu Bergotte prodiguer à Swann. Même aux gens qu'elle aimait, elle se plaisait à dire des choses désagréables. Mais hors de leur présence elle n'aurait pas prononcé une parole qu'ils n'eussent pu entendre. Rien, moins que notre société de Combray, ne ressemblait au monde. Celle des Swann était déjà un acheminement vers lui, vers ses flots versatiles. Ce n'était pas encore la grande mer, c'était déjà la lagune. « Tout ceci de vous à moi », me dit Bergotte en me quittant devant ma porte. Quelques années plus tard, je lui aurais répondu : « Je ne répète jamais rien. » C'est la phrase rituelle des gens du monde, par laquelle chaque fois le médisant est faussement rassuré. C'est celle que j'aurais déjà ce jour-là adressée à Bergotte car on n'invente pas tout ce qu'on dit, surtout dans les moments où on agit comme personnage social. Mais je ne la connaissais pas encore. D'autre part, celle de ma grand'tante dans une occasion semblable eût été : « Si vous ne voulez pas que ce soit répété, pourquoi le dites-vous ? » C'est la réponse des gens insociables, des « mauvaises têtes ». Je ne l'étais pas : je m'inclinai en silence.

Des gens de lettres qui étaient pour moi des personnages considérables intriguaient pendant des années avant d'arriver à nouer avec Bergotte des relations qui restaient toujours obscurément littéraires et ne sortaient pas de son cabinet de travail, alors que moi, je venais de m'installer parmi les amis du grand écrivain, d'emblée et tranquillement, comme quelqu'un qui, au lieu de faire la queue avec tout le monde pour avoir une mauvaise place, gagne les meilleures, ayant passé par un couloir fermé aux autres. Si Swann me l'avait ainsi ouvert, c'est sans doute parce que, comme un roi se trouve naturellement inviter les amis de ses enfants dans la loge royale, sur le yacht royal, de même les parents de Gilberte recevaient les amis de leur fille au milieu des choses précieuses qu'ils possédaient et des intimités plus précieuses encore qui y étaient encadrées. Mais à cette époque je pensais, et peut-être avec raison, que cette amabilité de Swann était indirectement à l'adresse de mes parents. J'avais cru entendre autrefois à Combray qu'il leur avait offert, voyant mon admiration pour Bergotte, de m'emmener dîner chez lui, et que mes parents avaient refusé, disant que j'étais trop jeune et trop nerveux pour « sortir ». Sans doute, mes parents représentaient-ils pour certaines personnes, justement celles qui me semblaient le plus merveilleuses, quelque chose de tout autre qu'à moi, de sorte que, comme au temps où la dame en rose avait adressé à mon père des éloges dont il s'était montré si peu digne, j'aurais souhaité que mes parents comprissent quel inestimable présent je venais de recevoir et témoignassent leur reconnaissance à ce Swann généreux et courtois qui me l'avait, ou le leur avait offert, sans avoir plus l'air de s'apercevoir de sa valeur que ne fait dans la fresque de Luini, le charmant roi mage, au nez busqué, aux cheveux blonds, et avec lequel on lui avait trouvé autrefois – paraît-il – une grande ressemblance.

Malheureusement, cette faveur que m'avait faite Swann et que, en rentrant, avant même d'ôter mon pardessus, j'annonçai à mes parents, avec l'espoir qu'elle éveillerait dans leur coeur un sentiment aussi ému que le mien et les déterminerait envers les Swann à quelque « politesse » énorme et décisive, cette faveur ne parut pas très appréciée par eux. « Swann t'a présenté à Bergotte ? Excellente connaissance, charmante relation ! s'écria ironiquement mon père. Il ne manquait plus que cela ! » Hélas, quand j'eus ajouté qu'il ne goûtait pas du tout Norpois :

– Naturellement ! reprit-il. Cela prouve bien que c'est un esprit faux et malveillant. Mon pauvre fils, tu n'avais pas déjà beaucoup de sens commun, je suis désolé de te voir tombé dans un milieu qui va achever de te détraquer.
