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
      "canonical_name": "baron de Charlus",
      "surface_forms": [
        "baron de Charlus",
        "le baron",
        "baron"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "baron de Charlus",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "« Presque chaque fois qu'il adresse une déclaration il essuie une avanie »; « il ne se plaisait plus qu'avec des gens du peuple qui l'exploitaient »; « baron de Charlus n'était en art qu'un dilettante, ... n'était pas doué ».",
      "explanation": "The narrator depicts Charlus as naive and repeatedly humiliated, dependent on and exploited by social inferiors, and artistically a mere dilettante, producing a clear local diminishment."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Charlus appears weakened—naive, exploited, frequently humiliated, and judged a dilettante—lowering his overall standing in the passage."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p2-m-de-charlus-pendant-la-guerre#p-66-p-70"
}

### Candidate characters

[
  "Françoise",
  "Jupien",
  "Robert de Saint-Loup",
  "le narrateur",
  "prince de Léon"
]

### Prior local context (optional)

La mauvaise impression du baron fut d'ailleurs accrue par la façon dont le bénéficiaire le remercia, car il dit : « Je vais envoyer ça à mes vieux et j'en garderai aussi un peu pour mon frangin qui est sur le front. » Ces sentiments touchants désappointèrent presque autant baron de Charlus que l'agaçait l'expression d'une paysannerie un peu conventionnelle. Jupien parfois les prévenait qu'« il fallait être plus pervers ». Alors l'un d'eux, de l'air de confesser quelque chose de satanique, aventurait : « Dites donc, baron, vous n'allez pas me croire, mais quand j'étais gosse, je regardais par le trou de la serrure mes parents s'embrasser. C'est vicieux, pas ? Vous avez l'air de croire que c'est un bourrage de crâne, mais non, je vous jure, tel que je vous le dis. » Et baron de Charlus était à la fois désespéré et exaspéré par cet effort factice vers la perversité qui n'aboutissait qu'à révéler tant de sottise et tant d'innocence. Et même le voleur, l'assassin le plus déterminés ne l'eussent pas contenté, car ils ne parlent pas de leur crime ; et il y a, d'ailleurs, chez le sadique – si bon qu'il puisse être, bien plus, d'autant meilleur qu'il est – une soif de mal que les méchants agissant dans d'autres buts ne peuvent contenter.

### Passage

Le jeune homme eut beau, comprenant trop tard son erreur, dire qu'il ne blairait pas les flics et pousser l'audace jusqu'à dire au baron : « Fous-moi un rancart » (un rendez-vous), le charme était dissipé. On sentait le chiqué, comme dans les livres des auteurs qui s'efforcent pour parler argot. C'est en vain que le jeune homme détailla toutes les « saloperies » qu'il faisait avec sa femme. Charlus fut seulement frappé combien ces saloperies se bornaient à peu de chose... Au reste, ce n'était pas seulement par insincérité. Rien n'est plus limité que le plaisir et le vice. On peut vraiment, dans ce sens-là et en changeant le sens de l'expression, dire qu'on tourne toujours dans le même cercle vicieux.

« Comme il est simple ! jamais on ne dirait un prince », dirent quelques habitués quand Charlus fut sorti, reconduit jusqu'en bas par Jupien auquel le baron ne laissa pas de se plaindre de la vertu du jeune homme. À l'air mécontent de Jupien, qui avait dû styler le jeune homme d'avance, on sentit que le faux assassin recevrait tout à l'heure un fameux savon. « C'est tout le contraire de ce que tu m'as dit », ajouta le baron pour que Jupien profitât de la leçon pour une autre fois. « Il a l'air d'une bonne nature, il exprime des sentiments de respect pour sa famille. – Il n'est pourtant pas bien avec son père, objecta Jupien, pris au dépourvu, ils habitent ensemble, mais ils servent chacun dans un bar différent. » C'était évidemment faible comme crime auprès de l'assassinat, mais Jupien se trouvait pris au dépourvu. Le baron n'ajouta rien car, s'il voulait qu'on préparât ses plaisirs, il voulait se donner à lui-même l'illusion que ceux-ci n'étaient pas « préparés ». « C'est un vrai bandit, il vous a dit cela pour vous tromper, vous êtes trop naïf », ajouta Jupien pour se disculper et ne faisant que froisser l'amour-propre de Charlus.

En même temps qu'on croyait Charlus prince, en revanche on regrettait beaucoup, dans l'établissement, la mort de quelqu'un dont les gigolos disaient : « Je ne sais pas son nom, il paraît que c'est un baron » et qui n'était autre que le prince de Foix (le père de l'ami de Saint-Loup). Passant, chez sa femme, pour vivre beaucoup au cercle, en réalité il passait des heures chez Jupien à bavarder, à raconter des histoires du monde devant des voyous. C'était un grand bel homme, comme son fils. Il est extraordinaire que Charlus, sans doute parce qu'il l'avait toujours connu dans le monde, ignorât qu'il partageait ses goûts. On allait même jusqu'à dire qu'il les avait autrefois portés jusque sur son fils encore collégien (l'ami de Saint-Loup), ce qui était probablement faux. Au contraire, très renseigné sur des moeurs que beaucoup ignorent, il veillait beaucoup aux fréquentations de son fils. Un jour qu'un homme, d'ailleurs de basse extraction, avait suivi le jeune prince de Foix jusqu'à l'hôtel de son père, où il avait jeté un billet par la fenêtre, le père l'avait ramassé. Mais le suiveur, bien qu'il ne fût pas aristocratiquement du même monde que M. de Foix le père, l'était à un autre point de vue. Il n'eut pas de peine à trouver dans de communs complices un intermédiaire qui fit taire M. de Foix en lui prouvant que c'était le jeune homme qui avait provoqué cette audace d'un homme âgé. Et c'était possible. Car le prince de Foix avait pu réussir à préserver son fils des mauvaises fréquentations au dehors mais non de l'hérédité. Au reste, le jeune prince de Foix resta, comme son père, ignoré à ce point de vue des gens du monde bien qu'il allât plus loin que personne avec ceux d'un autre.

« Il paraît qu'il a un million à manger par jour », dit le jeune homme de vingt-deux ans auquel l'assertion qu'il émettait ne semblait pas invraisemblable. On entendit bientôt le roulement de la voiture qui était venue chercher Charlus. À ce moment j'aperçus, avec une démarche lente, à côté d'un militaire qui évidemment sortait avec elle d'une chambre voisine, une personne qui me parut une dame assez âgée, en jupe noire. Je reconnus bientôt mon erreur, c'était un prêtre. C'était cette chose si rare, et en France absolument exceptionnelle, qu'est un mauvais prêtre. Évidemment le militaire était en train de railler son compagnon au sujet du peu de conformité que sa conduite offrait avec son habit, car celui-ci, d'un air grave et levant vers son visage hideux un doigt de docteur en théologie, dit sentencieusement : « Que voulez-vous, je ne suis pas (j'attendais « un saint ») un ange. » D'ailleurs il n'avait plus qu'à s'en aller et prit congé de Jupien qui, ayant accompagné le baron, venait de remonter, mais par étourderie le mauvais prêtre oublia de payer sa chambre. Jupien, que son esprit n'abandonnait jamais, agita le tronc dans lequel il mettait la contribution de chaque client, et le fit sonner en disant : « Pour les frais du culte, Monsieur l'Abbé ! » Le vilain personnage s'excusa, donna sa pièce et disparut. Jupien vint me chercher dans l'antre obscur où je n'osais faire un mouvement. « Entrez un moment dans le vestibule où mes jeunes gens font banquette, pendant que je monte fermer la chambre ; puisque vous êtes locataire, c'est tout naturel. » Le patron y était, je le payai. À ce moment un jeune homme en smoking entra et demanda d'un air d'autorité au patron : « Pourrai-je avoir Léon demain matin à onze heures moins le quart au lieu de onze heures parce que je déjeune en ville ? – Cela dépend, répondit le patron, du temps que le gardera l'abbé. » Cette réponse ne parut pas satisfaire le jeune homme en smoking qui semblait déjà prêt à invectiver contre l'abbé, mais sa colère prit un autre cours quand il m'aperçut ; marchant droit au patron : « Qui est-ce ? Qu'est-ce que ça signifie ? », murmura-t-il d'une voix basse mais courroucée. Le patron, très ennuyé, expliqua que ma présence n'avait aucune importance, que j'étais un locataire. Le jeune homme en smoking ne parut nullement apaisé par cette explication. Il ne cessait de répéter : « C'est excessivement désagréable, ce sont des choses qui ne devraient pas arriver, vous savez que je déteste ça et vous ferez si bien que je ne remettrai plus les pieds ici. » L'exécution de cette menace ne parut pas cependant imminente, car il partit furieux mais en recommandant que Léon tâchât d'être libre à 11 h. moins ¼, 10 h. ½ si possible. Jupien revint me chercher et descendit avec moi. « Je ne voudrais pas que vous me jugiez mal, me dit-il, cette maison ne me rapporte pas autant d'argent que vous croyez, je suis forcé d'avoir des locataires honnêtes, il est vrai qu'avec eux seuls on ne ferait que manger de l'argent. Ici c'est le contraire des Carmels, c'est grâce au vice que vit la vertu. Non, si j'ai pris cette maison, ou plutôt si je l'ai fait prendre au gérant que vous avez vu, c'est uniquement pour rendre service au baron et distraire ses vieux jours. » Jupien ne voulait pas parler que de scènes de sadisme comme celles auxquelles j'avais assisté et de l'exercice même du vice du baron. Celui-ci, même pour la conversation, pour lui tenir compagnie, pour jouer aux cartes, ne se plaisait plus qu'avec des gens du peuple qui l'exploitaient. Sans doute le snobisme de la canaille peut aussi bien se comprendre que l'autre. Ils avaient, d'ailleurs, été longtemps unis, alternant l'un avec l'autre, chez Charlus qui ne trouvait personne d'assez élégant pour ses relations mondaines, ni de frisant assez l'apache pour les autres. « Je déteste le genre moyen, disait-il, la comédie bourgeoise est guindée, il me faut ou les princesses de la tragédie classique ou la grosse farce. Pas de milieu, Phèdre ou Les Saltimbanques. » Mais enfin l'équilibre entre ces deux snobismes avait été rompu. Peut-être fatigue de vieillard, ou extension de la sensualité aux relations les plus banales, le baron ne vivait plus qu'avec des « inférieurs », prenant ainsi sans le savoir la succession de tel de ses grands ancêtres, le duc de La Rochefoucauld, le prince d'Harcourt, le duc de Berry, que Saint-Simon nous montre passant leur vie avec leurs laquais, qui tiraient d'eux des sommes énormes, partageant leurs jeux, au point qu'on était gêné pour ces grands seigneurs, quand il fallait les aller voir, de les trouver installés familièrement à jouer aux cartes ou à boire avec leur domesticité. « C'est surtout, ajouta Jupien, pour lui éviter des ennuis, parce que, voyez-vous, le baron, c'est un grand enfant. Même maintenant qu'il a ici tout ce qu'il peut désirer il va encore à l'aventure faire le vilain. Et généreux comme il est, ça pourrait souvent, par le temps qui court, avoir des conséquences. N'y a-t-il pas l'autre jour un chasseur d'hôtel qui mourait de peur à cause de tout l'argent que le baron lui offrait pour venir chez lui. Chez lui, quelle imprudence ! Ce garçon, qui pourtant aime seulement les femmes, a été rassuré quand il a compris ce qu'on voulait de lui. En entendant toutes ces promesses d'argent, il avait pris le baron pour un espion. Et il s'est senti bien à l'aise quand il a vu qu'on ne lui demandait pas de livrer sa patrie mais son corps, ce qui n'est peut-être pas plus moral, mais ce qui est moins dangereux, et surtout plus facile. » Et en écoutant Jupien, je me disais : « Quel malheur que Charlus ne soit pas romancier ou poète, non pas pour décrire ce qu'il verrait, mais le point où se trouve un Charlus par rapport au désir fait naître autour de lui les scandales, le force à prendre la vie sérieusement, à mettre des émotions dans le plaisir, l'empêche de s'arrêter, de s'immobiliser dans une vue ironique et extérieure des choses, rouvre sans cesse en lui un courant douloureux. Presque chaque fois qu'il adresse une déclaration il essuie une avanie, s'il ne risque pas même la prison. » Ce n'est pas que l'éducation des enfants, c'est celle des poètes qui se fait à coups de gifles. Si Charlus avait été romancier, la maison que lui avait aménagée Jupien, en réduisant dans de telles proportions les risques, du moins (car une descente de police était toujours à craindre) les risques à l'égard d'un individu des dispositions duquel, dans la rue, le baron n'eût pas été assuré, eût été pour lui un malheur. Mais Charlus n'était en art qu'un dilettante, qui ne songeait pas à écrire et n'était pas doué pour cela. « D'ailleurs, vous avouerais-je, reprit Jupien, que je n'ai pas un grand scrupule à avoir ce genre de gains ? La chose elle-même qu'on fait ici, je ne peux plus vous cacher que je l'aime, qu'elle est le goût de ma vie. Or, est-il défendu de recevoir un salaire pour des choses qu'on ne juge pas coupables ? Vous êtes plus instruit que moi et vous me direz sans doute que Socrate ne croyait pas pouvoir recevoir d'argent pour ses leçons. Mais de notre temps les professeurs de philosophie ne pensent pas ainsi, ni les médecins, ni les peintres, ni les dramaturges, ni les directeurs de théâtre. Ne croyez pas que ce métier ne fasse fréquenter que des canailles. Sans doute le Directeur d'un établissement de ce genre, comme une grande cocotte, ne reçoit que des hommes, mais il reçoit des hommes marquants dans tous les genres et qui sont généralement, à situation égale, parmi les plus fins, les plus sensibles, les plus aimables de leur profession. Cette maison se transformerait vite, je vous l'assure, en un bureau d'esprit et une agence de nouvelles. » Mais j'étais encore sous l'impression des coups que j'avais vu recevoir à Charlus. Et à vrai dire, quand on connaissait bien Charlus, son orgueil, sa satiété des plaisirs mondains, ses caprices changés facilement en passions pour des hommes de dernier ordre et de la pire espèce, on peut très bien comprendre que la même grosse fortune qui, échue à un parvenu, l'eût charmé en lui permettant de marier sa fille à un duc et d'inviter des Altesses à ses chasses, Charlus était content de la posséder parce qu'elle lui permettait d'avoir ainsi la haute main sur un, peut-être sur plusieurs établissements où étaient en permanence des jeunes gens avec lesquels il se plaisait. Peut-être n'y eut-il même pas besoin de son vice pour cela. Il était l'héritier de tant de grands seigneurs, princes du sang ou ducs, dont Saint-Simon nous raconte qu'ils ne fréquentaient personne « qui se pût nommer ». « En attendant, dis-je à Jupien, cette maison est tout autre chose, plus qu'une maison de fous, puisque la folie des aliénés qui y habitent est mise en scène, reconstituée, visible, c'est un vrai pandémonium. J'avais cru, comme le calife des Mille et une Nuits, arriver à point au secours d'un homme qu'on frappait, et c'est un autre conte des Mille et une Nuits que j'ai vu réaliser devant moi, celui où une femme, transformée en chienne, se fait frapper volontairement pour retrouver sa forme première. » Jupien paraissait fort troublé par mes paroles, car il comprenait que j'avais vu frapper le baron. Il resta un moment silencieux, puis tout d'un coup, avec le joli esprit qui m'avait si souvent frappé chez cet homme qui s'était fait lui-même, quand il avait pour m'accueillir, Françoise ou moi, dans la cour de notre maison, de si gracieuses paroles : « Vous parlez de bien des contes des Mille et une Nuits, me dit-il. Mais j'en connais un qui n'est pas sans rapport avec le titre d'un livre que je crois avoir aperçu chez le baron (il faisait allusion à une traduction de Sésame et les Lys, de Ruskin, que j'avais envoyée à Charlus). Si jamais vous étiez curieux, un soir, de voir, je ne dis pas quarante, mais une dizaine de voleurs, vous n'avez qu'à venir ici ; pour savoir si je suis là vous n'avez qu'à regarder là-haut, je laisse ma petite fenêtre ouverte et éclairée, cela veut dire que je suis venu, qu'on peut entrer ; c'est mon Sésame à moi. Je dis seulement Sésame. Car pour les Lys, si c'est eux que vous voulez, je vous conseille d'aller les chercher ailleurs. » Et me saluant assez cavalièrement, car une clientèle aristocratique et une clique de jeunes gens, qu'il menait comme un pirate, lui avaient donné une certaine familiarité, il prit congé de moi. Il m'avait à peine quitté que la sirène retentit, immédiatement suivie de violents tirs de barrage. On sentait que c'était tout auprès, juste au-dessus de nous, que l'avion allemand se tenait, et soudain le bruit d'une forte détonation montra qu'il venait de lancer une de ses bombes.

Dans une même salle de la maison de Jupien beaucoup d'hommes, qui n'avaient pas voulu fuir, s'étaient réunis. Ils ne se connaissaient pas entre eux, mais étaient pourtant à peu près du même monde, riche et aristocratique. L'aspect de chacun avait quelque chose de répugnant qui devait être la non-résistance à des plaisirs dégradants. L'un, énorme, avait la figure couverte de taches rouges, comme un ivrogne. J'avais appris qu'au début il ne l'était pas et prenait seulement son plaisir à faire boire des jeunes gens. Mais, effrayé par l'idée d'être mobilisé (bien qu'il semblât avoir dépassé la cinquantaine), comme il était très gros il s'était mis à boire sans arrêter pour tâcher de dépasser le poids de cent kilos, au-dessus duquel on était réformé. Et maintenant, ce calcul s'étant changé en passion, où qu'on le quittât, tant qu'on le surveillait, on le retrouvait chez un marchand de vin. Mais dès qu'il parlait on voyait que, médiocre d'ailleurs d'intelligence, c'était un homme de beaucoup de savoir, d'éducation et de culture. Un autre homme du grand monde, celui-là fort jeune et d'une extrême distinction physique, était entré. Chez lui, à vrai dire, il n'y avait encore aucun stigmate extérieur d'un vice, mais, ce qui était plus troublant, d'intérieurs. Très grand, d'un visage charmant, son élocution décelait une tout autre intelligence que celle de son voisin l'alcoolique, et, sans exagérer, vraiment remarquable. Mais à tout ce qu'il disait était ajoutée une expression qui eût convenu à une phrase différente. Comme si, tout en possédant le trésor complet des expressions du visage humain, il eût vécu dans un autre monde, il mettait à jour ces expressions dans l'ordre qu'il ne fallait pas, il semblait effeuiller au hasard des sourires et des regards sans rapport avec le propos qu'il entendait. J'espère pour lui, si, comme il est certain, il vit encore, qu'il était non la proie d'une maladie durable mais d'une intoxication passagère. Il est probable que si l'on avait demandé leur carte de visite à tous ces hommes on eût été surpris de voir qu'ils appartenaient à une haute classe sociale. Mais quelque vice, et le plus grand de tous, le manque de volonté qui empêche de résister à aucun, les réunissait là, dans des chambres isolées il est vrai, mais chaque soir, me dit-on, de sorte que si leur nom était connu des femmes du monde, celles-ci avaient peu à peu perdu de vue leur visage et n'avaient plus jamais l'occasion de recevoir leur visite. Ils recevaient encore des invitations, mais l'habitude les ramenait au mauvais lieu composite. Ils s'en cachaient peu, du reste, au contraire des petits chasseurs, ouvriers, etc. qui servaient à leur plaisir. Et en dehors de beaucoup de raisons que l'on devine, cela se comprend par celle-ci. Pour un employé d'industrie, pour un domestique, aller là c'était, comme pour une femme qu'on croyait honnête, aller dans une maison de passe. Certains qui avouaient y être allés se défendaient d'y être plus jamais retournés, et Jupien lui-même, mentant pour protéger leur réputation ou éviter des concurrences, affirmait : « Oh ! non, il ne vient pas chez moi, il ne voudrait pas y venir. » Pour des hommes du monde, c'est moins grave, d'autant plus que les autres gens du monde qui n'y vont pas ne savent pas ce que c'est et ne s'occupent pas de votre vie.
