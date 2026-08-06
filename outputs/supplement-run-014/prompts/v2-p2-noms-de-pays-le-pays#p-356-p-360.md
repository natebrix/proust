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
      "canonical_name": "la grand-mère",
      "surface_forms": [
        "la grand-mère",
        "ma grand-mère"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "la grand-mère",
      "type": "blame",
      "polarity": "negative",
      "narrative_stance": "uncertain",
      "confidence": 0.68,
      "evidence": "« m'irritant contre la grand-mère, inconsciemment méchante » ; « un mépris qui me semblait procéder de vues un peu étroites »",
      "explanation": "The narrator frames his grandmother as obstructive and narrow in outlook, blaming her for keeping him from pursuing the girls and for disparaging his beach pursuits compared to visiting Elstir."
    }
  ],
  "status_effects": [
    {
      "character": "la grand-mère",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.65,
      "explanation": "Locally diminished because the narrator characterizes her as unconsciously unkind and guided by narrow views."
    }
  ],
  "ambiguities": [
    "The blame is filtered through the narrator's desire-driven perspective (“me semblait”), and he also ends up obeying her, which could indicate countervailing authority not fully captured here."
  ],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-356-p-360"
}

### Candidate characters

[
  "Elstir",
  "le narrateur"
]

### Prior local context (optional)

Ma la grand-mère, à qui j'avais raconté mon entrevue avec Elstir et qui se réjouissait de tout le profit intellectuel que je pouvais tirer de son amitié, trouvait absurde et peu gentil que je ne fusse pas encore allé lui faire une visite. Mais je ne pensais qu'à la petite bande, et incertain de l'heure où ces jeunes filles passeraient sur la digue, je n'osais pas m'éloigner. Ma la grand-mère s'étonnait aussi de mon élégance, car je m'étais soudain souvenu des costumes que j'avais jusqu'ici laissés au fond de ma malle. J'en mettais chaque jour un différent, et j'avais même écrit à Paris pour me faire envoyer de nouveaux chapeaux, et de nouvelles cravates.

### Passage

C'est un grand charme ajouté à la vie dans une station balnéaire comme était Balbec, si le visage d'une jolie fille, une marchande de coquillages, de gâteaux ou de fleurs, peint en vives couleurs dans notre pensée, est quotidiennement pour nous dès le matin le but de chacune de ces journées oisives et lumineuses qu'on passe sur la plage. Elles sont alors, et par là, bien que désoeuvrées, alertes comme des journées de travail, aiguillées, aimantées, soulevées légèrement vers un instant prochain, celui où tout en achetant des sablés, des roses, des ammonites, on se délectera à voir, sur un visage féminin, les couleurs étalées aussi purement que sur une fleur. Mais au moins, ces petites marchandes, d'abord, on peut leur parler, ce qui évite d'avoir à construire avec l'imagination les autres côtés que ceux que nous fournit la simple perception visuelle, et à recréer leur vie, à s'exagérer son charme, comme devant un portrait ; surtout, justement parce qu'on leur parle, on peut apprendre où, à quelles heures on peut les retrouver. Or il n'en était nullement ainsi pour moi en ce qui concernait les jeunes filles de la petite bande. Leurs habitudes m'étant inconnues, quand certains jours je ne les apercevais pas, ignorant la cause de leur absence, je cherchais si celle-ci était quelque chose de fixe, si on ne les voyait que tous les deux jours, ou quand il faisait tel temps, ou s'il y avait des jours où on ne les voyait jamais. Je me figurais d'avance ami avec elles et leur disant : « Mais vous n'étiez pas là tel jour ? – Ah ! oui, c'est parce que c'était un samedi, le samedi nous ne venons jamais parce que... » Encore si c'était aussi simple que de savoir que le triste samedi il est inutile de s'acharner, qu'on pourrait parcourir la plage en tous sens, s'asseoir à la devanture du pâtissier, faire semblant de manger un éclair, entrer chez le marchand de curiosités, attendre l'heure du bain, le concert, l'arrivée de la marée, le coucher du soleil, la nuit, sans voir la petite bande désirée. Mais le jour fatal ne revenait peut-être pas une fois par semaine. Il ne tombait peut-être pas forcément un samedi. Peut-être certaines conditions atmosphériques influaient-elles sur lui ou lui étaient-elles entièrement étrangères. Combien d'observations patientes, mais non point sereines, il faut recueillir sur les mouvements en apparence irréguliers de ces mondes inconnus avant de pouvoir être sûr qu'on ne s'est pas laissé abuser par des coïncidences, que nos prévisions ne seront pas trompées, avant de dégager les lois certaines, acquises au prix d'expériences cruelles, de cette astronomie passionnée. Me rappelant que je ne les avais pas vues le même jour qu'aujourd'hui, je me disais qu'elles ne viendraient pas, qu'il était inutile de rester sur la plage. Et justement je les apercevais. En revanche, un jour où, autant que j'avais pu supposer que des lois réglaient le retour de ces constellations, j'avais calculé devoir être un jour faste, elles ne venaient pas. Mais à cette première incertitude si je les verrais ou non le jour même venait s'en ajouter une plus grave, si je les reverrais jamais, car j'ignorais en somme si elles ne devaient pas partir pour l'Amérique, ou rentrer à Paris. Cela suffisait pour me faire commencer à les aimer. On peut avoir du goût pour une personne. Mais pour déchaîner cette tristesse, ce sentiment de l'irréparable, ces angoisses, qui préparent l'amour, il faut – et il est peut-être ainsi, plutôt que ne l'est une personne, l'objet même que cherche anxieusement à étreindre la passion – le risque d'une impossibilité. Ainsi agissaient déjà ces influences qui se répètent au cours d'amours successives, pouvant du reste se produire, mais alors plutôt dans l'existence des grandes villes, au sujet d'ouvrières dont on ne sait pas les jours de congé et qu'on s'effraye de ne pas avoir vues à la sortie de l'atelier, ou du moins qui se renouvelèrent au cours des miennes. Peut-être sont-elles inséparables de l'amour ; peut-être tout ce qui fut une particularité du premier vient-il s'ajouter aux suivants, par souvenir, suggestion, habitude et, à travers les périodes successives de notre vie, donner à ses aspects différents un caractère général.

Je prenais tous les prétextes pour aller sur la plage aux heures où j'espérais pouvoir les rencontrer. Les ayant aperçues une fois pendant notre déjeuner je n'y arrivais plus qu'en retard, attendant indéfiniment sur la digue qu'elles y passassent ; restant le peu de temps que j'étais assis dans la salle à manger à interroger des yeux l'azur du vitrage ; me levant bien avant le dessert pour ne pas les manquer dans le cas où elles se fussent promenées à une autre heure et m'irritant contre ma grand-mère, inconsciemment méchante, quand elle me faisait rester avec elle au delà de l'heure qui me semblait propice. Je tâchais de prolonger l'horizon en mettant ma chaise de travers ; si par hasard j'apercevais n'importe laquelle des jeunes filles, comme elles participaient toutes à la même essence spéciale, c'était comme si j'avais vu projeté en face de moi dans une hallucination mobile et diabolique un peu de rêve ennemi et pourtant passionnément convoité qui, l'instant d'avant encore, n'existait, y stagnant d'ailleurs d'une façon permanente, que dans mon cerveau.

Je n'en aimais aucune les aimant toutes, et pourtant leur rencontre possible était pour mes journées le seul élément délicieux, faisait seule naître en moi de ces espoirs où on briserait tous les obstacles, espoirs souvent suivis de rage, si je ne les avais pas vues. En ce moment, ces jeunes filles éclipsaient pour moi ma grand-mère ; un voyage m'eût tout de suite souri si ç'avait été pour aller dans un lieu où elles dussent se trouver. C'était à elles que ma pensée s'était agréablement suspendue quand je croyais penser à autre chose ou à rien. Mais quand, même ne le sachant pas, je pensais à elles, plus inconsciemment encore, elles, c'était pour moi les ondulations montueuses et bleues de la mer, le profil d'un défilé devant la mer. C'était la mer que j'espérais retrouver, si j'allais dans quelque ville où elles seraient. L'amour le plus exclusif pour une personne est toujours l'amour d'autre chose.

Ma grand'mère me témoignait, parce que maintenant je m'intéressais extrêmement au golf et au tennis et laissais échapper l'occasion de regarder travailler et entendre discourir un artiste qu'elle savait des plus grands, un mépris qui me semblait procéder de vues un peu étroites. J'avais autrefois entrevu aux Champs-Élysées et je m'étais rendu mieux compte depuis qu'en étant amoureux d'une femme nous projetons simplement en elle un état de notre âme ; que par conséquent l'important n'est pas la valeur de la femme mais la profondeur de l'état ; et que les émotions qu'une jeune fille médiocre nous donne peuvent nous permettre de faire monter à notre conscience des parties plus intimes de nous-même, plus personnelles, plus lointaines, plus essentielles, que ne ferait le plaisir que nous donne la conversation d'un homme supérieur ou même la contemplation admirative de ses oeuvres.

Je dus finir par obéir à ma grand-mère avec d'autant plus d'ennui qu'Elstir habitait assez loin de la digue, dans une des avenues les plus nouvelles de Balbec. La chaleur du jour m'obligea à prendre le tramway qui passait par la rue de la Plage, et je m'efforçais, pour penser que j'étais dans l'antique royaume des Cimmériens, dans la patrie peut-être du roi Mark ou sur l'emplacement de la forêt de Broceliande, de ne pas regarder le luxe de pacotille des constructions qui se développaient devant moi et entre lesquelles la villa d'Elstir était peut-être la plus somptueusement laide, louée malgré cela par lui, parce que de toutes celles qui existaient à Balbec, c'était la seule qui pouvait lui offrir un vaste atelier.
