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
        "grand-mère",
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
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.88,
      "evidence": "Regards ‘imprégnés d’une telle bonté’ comme à un bébé; pain ‘du genre de ceux qu’on jette aux canards’; ‘la grand-mère n’était plus un canard ou une antilope, mais déjà … un “baby”.’",
      "explanation": "The princess’s benevolence is framed by the narrator as miscalibrated and patronizing, animalizing/infantilizing the grandmother and thus locally lowering her standing."
    }
  ],
  "status_effects": [
    {
      "character": "la grand-mère",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "She is included by a princess but treated as an animal/child, which the narrator ironizes as a condescending lowering."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-91-p-95"
}

### Candidate characters

[
  "Mme de Villeparisis",
  "Norpois",
  "Odette",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le directeur",
  "le narrateur",
  "le père du narrateur",
  "princesse de Luxembourg"
]

### Prior local context (optional)

– Il faudra que je pense une fois à lui demander si je me trompe et si elle n'a pas quelque parenté avec les Guermantes, me dit la grand-mère qui excita par là mon indignation. Comment aurais-je pu croire à une communauté d'origine entre deux noms qui étaient entrés en moi l'un par la porte basse et honteuse de l'expérience, l'autre par la porte d'or de l'imagination ?

### Passage

On voyait souvent passer depuis quelques jours, en pompeux équipage, grande, rousse, belle, avec un nez un peu fort, la princesse de Luxembourg, qui était en villégiature pour quelques semaines dans le pays. Sa calèche s'était arrêtée devant l'hôtel, un valet de pied était venu parler au directeur, était retourné à la voiture et avait rapporté des fruits merveilleux (qui unissaient dans une seule corbeille, comme la baie elle-même, diverses saisons), avec une carte : « La princesse de Luxembourg », où étaient écrits quelques mots au crayon. À quel voyageur princier demeurant ici incognito, pouvaient être destinés ces prunes glauques, lumineuses et sphériques comme était à ce moment-là la rotondité de la mer, ces raisins transparents suspendus au bois desséché comme une claire journée d'automne, ces poires d'un outremer céleste ? Car ce ne pouvait être à l'amie de ma grand'mère que la princesse avait voulu faire visite. Pourtant le lendemain soir Mme de Villeparisis nous envoya la grappe de raisins fraîche et dorée et des prunes et des poires que nous reconnûmes aussi, quoique les prunes eussent passé, comme la mer à l'heure de notre dîner, au mauve et que dans l'outremer des poires flottassent quelques formes de nuages roses. Quelques jours après nous rencontrâmes Mme de Villeparisis en sortant du concert symphonique qui se donnait le matin sur la plage. Persuadé que les oeuvres que j'y entendais (le Prélude de Lohengrin, l'ouverture de Tannhauser, etc.) exprimaient les vérités les plus hautes, je tâchais de m'élever autant que je pouvais pour atteindre jusqu'à elles, je tirais de moi pour les comprendre, je leur remettais tout ce que je recélais alors de meilleur, de plus profond.

Or, en sortant du concert, comme, en reprenant le chemin qui va vers l'hôtel, nous nous étions arrêtés un instant sur la digue, ma grand'mère et moi, pour échanger quelques mots avec Mme de Villeparisis qui nous annonçait qu'elle avait commandé pour nous à l'hôtel des « Croque-Monsieur » et des oeufs à la crème, je vis de loin venir dans notre direction la princesse de Luxembourg, à demi appuyée sur une ombrelle de façon à imprimer à son grand et merveilleux corps cette légère inclinaison, à lui faire dessiner cette arabesque si chère aux femmes qui avaient été belles sous l'Empire et qui savaient, les épaules tombantes, le dos remonté, la hanche creuse, la jambe tendue, faire flotter mollement leur corps comme un foulard, autour de l'armature d'une invisible tige inflexible et oblique, qui l'aurait traversé. Elle sortait tous les matins faire son tour de plage presque à l'heure où tout le monde après le bain remontait pour déjeuner, et comme le sien était seulement à une heure et demie, elle ne rentrait à sa villa que longtemps après que les baigneurs avaient abandonné la digue déserte et brûlante. Mme de Villeparisis présenta ma grand'mère, voulut me présenter, mais dut me demander mon nom, car elle ne se le rappelait pas. Elle ne l'avait peut-être jamais su, ou en tous cas avait oublié depuis bien des années à qui ma grand'mère avait marié sa fille. Ce nom parut faire une vive impression sur Mme de Villeparisis. Cependant la princesse de Luxembourg nous avait tendu la main et, de temps en temps, tout en causant avec la marquise, elle se détournait pour poser de doux regards sur ma grand'mère et sur moi, avec cet embryon de baiser qu'on ajoute au sourire quand celui-ci s'adresse à un bébé avec sa nounou. Même dans son désir de ne pas avoir l'air de siéger dans une sphère supérieure à la nôtre, elle avait sans doute mal calculé la distance, car, par une erreur de réglage, ses regards s'imprégnèrent d'une telle bonté que je vis approcher le moment où elle nous flatterait de la main comme deux bêtes sympathiques qui eussent passé la tête vers elle, à travers un grillage, au Jardin d'Acclimatation. Aussitôt du reste cette idée d'animaux et de bois de Boulogne prit plus de consistance pour moi. C'était l'heure où la digue est parcourue par des marchands ambulants et criards qui vendent des gâteaux, des bonbons, des petits pains. Ne sachant que faire pour nous témoigner sa bienveillance, la princesse arrêta le premier qui passa ; il n'avait plus qu'un pain de seigle, du genre de ceux qu'on jette aux canards. La princesse le prit et me dit : « C'est pour votre grand'mère. » Pourtant, ce fut à moi qu'elle le tendit, en me disant avec un fin sourire : « Vous le lui donnerez vous-même », pensant qu'ainsi mon plaisir serait plus complet s'il n'y avait pas d'intermédiaires entre moi et les animaux. D'autres marchands s'approchèrent, elle remplit mes poches de tout ce qu'ils avaient, de paquets tout ficelés, de plaisirs, de babas et de sucres d'orge. Elle me dit : « Vous en mangerez et vous en ferez manger aussi à votre grand'mère » et elle fit payer les marchands par le petit nègre habillé en satin rouge qui la suivait partout et qui faisait l'émerveillement de la plage. Puis elle dit adieu à Mme de Villeparisis et nous tendit la main avec l'intention de nous traiter de la même manière que son amie, en intimes et de se mettre à notre portée. Mais cette fois, elle plaça sans doute notre niveau un peu moins bas dans l'échelle des êtres, car son égalité avec nous fut signifiée par la princesse à ma grand'mère au moyen de ce tendre et maternel sourire qu'on adresse à un gamin quand on lui dit au revoir comme à une grande personne. Par un merveilleux progrès de l'évolution, ma grand'mère n'était plus un canard ou une antilope, mais déjà ce que Odette eût appelé un « baby ». Enfin, nous ayant quittés tous trois, la Princesse reprit sa promenade sur la digue ensoleillée en incurvant sa taille magnifique qui comme un serpent autour d'une baguette s'enlaçait à l'ombrelle blanche imprimée de bleu que Mme de Luxembourg tenait fermée à la main. C'était ma première altesse, je dis la première, car la princesse Mathilde n'était pas altesse du tout de façons. La seconde, on le verra plus tard, ne devait pas moins m'étonner par sa bonne grâce. Une forme de l'amabilité des grands seigneurs, intermédiaires bénévoles entre les souverains et les bourgeois, me fut apprise le lendemain quand Mme de Villeparisis nous dit : « Elle vous a trouvés charmants. C'est une femme d'un grand jugement, de beaucoup de coeur. Elle n'est pas comme tant de souveraines ou d'altesses. Elle a une vraie valeur. » Et Mme de Villeparisis ajouta d'un air convaincu, et toute ravie de pouvoir nous le dire : « Je crois qu'elle serait enchantée de vous revoir. »

Mais ce matin-là même, en quittant la princesse de Luxembourg, Mme de Villeparisis me dit une chose qui me frappa davantage et qui n'était pas du domaine de l'amabilité.

– Est-ce que vous êtes le fils du directeur au Ministère ? me demanda-t-elle. Ah ! il paraît que votre père est un homme charmant. Il fait un bien beau voyage en ce moment.

Quelques jours auparavant nous avions appris par une lettre de maman que mon père et son compagnon Norpois avaient perdu leurs bagages.
