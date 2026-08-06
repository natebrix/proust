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
      "canonical_name": "Aimé",
      "surface_forms": [
        "Aimé"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Aimé",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.77,
      "evidence": "Aimé affirme que « Dreyfus était mille fois coupable » en invoquant « un monsieur très lié dans l'état-major », mime la scène, répète le geste sur l’épaule et ajoute : « Vous voyez, je vous montre exactement comme il a fait », le narrateur suggérant qu’il était « flatté de cette familiarité d’un grand personnage ».",
      "explanation": "The narrator presents Aimé as gullible and flattered by the proximity of a 'great personage', reinforcing the idea that his argument of authority is naïve; the descriptive tone, with mimicry and emphasis on the gesture, slightly exposes him to ridicule."
    }
  ],
  "status_effects": [
    {
      "character": "Aimé",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.75,
      "explanation": "Aimé appears locally less commendable, his gullibility and quest for prestige through mimicry weakening his seriousness in the eyes of the narrator."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-321-p-325"
}

### Candidate characters

[
  "Bergotte",
  "Dreyfus",
  "Robert de Saint-Loup",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

On frappa ; c'était Aimé qui avait tenu à m'apporter lui-même les dernières listes d'étrangers.

### Passage

Aimé, avant de se retirer, tint à me dire que Dreyfus était mille fois coupable. « On saura tout, me dit-il, pas cette année, mais l'année prochaine : c'est un monsieur très lié dans l'état-major qui me l'a dit. Je lui demandais si on ne se déciderait pas à tout découvrir tout de suite avant la fin de l'année. Il a posé sa cigarette », continua Aimé en mimant la scène et en secouant la tête et l'index comme avait fait son client voulant dire : il ne faut pas être trop exigeant. « Pas cette année, Aimé, qu'il m'a dit en me touchant à l'épaule, ce n'est pas possible. Mais à Pâques, oui ! » Et Aimé me frappa légèrement sur l'épaule en me disant : « Vous voyez, je vous montre exactement comme il a fait », soit qu'il fût flatté de cette familiarité d'un grand personnage, soit pour que je pusse mieux apprécier en pleine connaissance de cause la valeur de l'argument et nos raisons d'espérer.

Ce ne fut pas sans un léger choc au coeur qu'à la première page de la liste des étrangers, j'aperçus les mots : « Simonet et famille ». J'avais en moi de vieilles rêveries qui dataient de mon enfance et où toute la tendresse qui était dans mon coeur, mais qui éprouvée par lui ne s'en distinguait pas, m'était apportée par un être aussi différent que possible de moi. Cet être, une fois de plus je le fabriquais en utilisant pour cela le nom de Simonet et le souvenir de l'harmonie qui régnait entre les jeunes corps que j'avais vus se déployer sur la plage, en une procession sportive, digne de l'antique et de Giotto. Je ne savais pas laquelle de ces jeunes filles était Mlle Simonet, si aucune d'elles s'appelait ainsi, mais je savais que j'étais aimé de Mlle Simonet et que j'allais grâce à Saint-Loup essayer de la connaître. Malheureusement n'ayant obtenu qu'à cette condition une prolongation de congé, il était obligé de retourner tous les jours à Doncières : mais pour le faire manquer à ses obligations militaires, j'avais cru pouvoir compter, plus encore que sur son amitié pour moi, sur cette même curiosité de naturaliste humain que si souvent – même sans avoir vu la personne dont il parlait et rien qu'à entendre dire qu'il y avait une jolie caissière chez un fruitier – j'avais eue de faire connaissance avec une nouvelle variété de la beauté féminine. Or, cette curiosité, c'est à tort que j'avais espéré l'exciter chez Saint-Loup en lui parlant de mes jeunes filles. Car elle était pour longtemps paralysée en lui par l'amour qu'il avait pour cette actrice dont il était l'amant. Et même l'eût-il légèrement ressentie qu'il l'eût réprimée, à cause d'une sorte de croyance superstitieuse que de sa propre fidélité pouvait dépendre celle de sa maîtresse. Aussi fût-ce sans qu'il m'eût promis de s'occuper activement de mes jeunes filles que nous partîmes dîner à Rivebelle.

Les premiers temps, quand nous arrivions, le soleil venait de se coucher, mais il faisait encore clair ; dans le jardin du restaurant dont les lumières n'étaient pas encore allumées, la chaleur du jour tombait, se déposait, comme au fond d'un vase le long des parois duquel la gelée transparente et sombre de l'air semblait si consistante qu'un grand rosier appliqué au mur obscurci qu'il veinait de rose avait l'air de l'arborisation qu'on voit au fond d'une pierre d'onyx. Bientôt ce ne fut qu'à la nuit que nous descendions de voiture, souvent même que nous partions de Balbec si le temps était mauvais et que nous eussions retardé le moment de faire atteler, dans l'espoir d'une accalmie. Mais ces jours-là, c'est sans tristesse que j'entendais le vent souffler, je savais qu'il ne signifiait pas l'abandon de mes projets, la réclusion dans une chambre, je savais que, dans la grande salle à manger du restaurant où nous entrerions au son de la musique des tziganes, les innombrables lampes triompheraient aisément de l'obscurité et du froid en leur appliquant leurs larges cautères d'or, et je montais gaiement à côté de Saint-Loup dans le coupé qui nous attendait sous l'averse. Depuis quelque temps, les paroles de Bergotte, se disant convaincu que malgré ce que je prétendais, j'étais fait pour goûter surtout les plaisirs de l'intelligence, m'avaient rendu au sujet de ce que je pourrais faire plus tard une espérance que décevait chaque jour l'ennui que j'éprouvais à me mettre devant une table, à commencer une étude critique ou un roman. « Après tout, me disais-je, peut-être le plaisir qu'on a eu à l'écrire n'est-il pas le critérium infaillible de la valeur d'une belle page ; peut-être n'est-il qu'un état accessoire qui s'y surajoute souvent, mais dont le défaut ne peut préjuger contre elle. Peut-être certains chefs-d'oeuvre ont-ils été composés en bâillant. » Ma grand'mère apaisait mes doutes en me disant que je travaillerais bien et avec joie si je me portais bien. Et, notre médecin ayant trouvé plus prudent de m'avertir des graves risques auxquels pouvait m'exposer mon état de santé, et m'ayant tracé toutes les précautions d'hygiène à suivre pour éviter un accident, je subordonnais tous les plaisirs au but que je jugeais infiniment plus important qu'eux, de devenir assez fort pour pouvoir réaliser l'oeuvre que je portais peut-être en moi, j'exerçais sur moi-même depuis que j'étais à Balbec un contrôle minutieux et constant. On n'aurait pu me faire toucher à la tasse de café qui m'eût privé du sommeil de la nuit, nécessaire pour ne pas être fatigué le lendemain. Mais quand nous arrivions à Rivebelle, aussitôt, à cause de l'excitation d'un plaisir nouveau et me trouvant dans cette zone différente où l'exceptionnel nous fait entrer après avoir coupé le fil, patiemment tissé depuis tant de jours, qui nous conduisait vers la sagesse – comme s'il ne devait plus jamais y avoir de lendemain, ni de fins élevées à réaliser – disparaissait ce mécanisme précis de prudente hygiène qui fonctionnait pour les sauvegarder. Tandis qu'un valet de pied me demandait mon paletot, Saint-Loup me disait :

– Vous n'aurez pas froid ? Vous feriez peut-être mieux de le garder, il ne fait pas très chaud.

Je répondais : « Non, non », et peut-être je ne sentais pas le froid, mais en tous cas je ne savais plus la peur de tomber malade, la nécessité de ne pas mourir, l'importance de travailler. Je donnais mon paletot ; nous entrions dans la salle du restaurant aux sons de quelque marche guerrière jouée par les tziganes, nous nous avancions entre les rangées de tables servies comme dans un facile chemin de gloire, et, sentant l'ardeur joyeuse imprimée à notre corps par les rythmes de l'orchestre qui nous décernait ses honneurs militaires et ce triomphe immérité, nous la dissimulions sous une mine grave et glacée, sous une démarche pleine de lassitude, pour ne pas imiter ces gommeuses de café-concert qui, venant chanter sur un air belliqueux un couplet grivois, entrent en courant sur la scène avec la contenance martiale d'un général vainqueur.
