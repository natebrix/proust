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
      "canonical_name": "Albertine",
      "surface_forms": [
        "Albertine"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
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
      "source": "Albertine",
      "target": "Bloch",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.84,
      "evidence": "« Je ne sais pas pourquoi il me salue puisqu'il ne me connaît pas. Aussi je ne lui ai pas rendu son salut. » ... « Je l'aurais parié que c'était un youpin. C'est bien leur genre de faire les punaises. »",
      "explanation": "Albertine refuses to return Bloch’s greeting and disparages him with insulting, anti-Semitic language. The narrator reports this and later characterizes the 'petite bande' as hard and coarse, suggesting critical distance."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.68,
      "explanation": "Her rudeness and prejudiced remarks lower her local estimation in the narrated framing, despite other charms noted elsewhere in the passage."
    },
    {
      "character": "Bloch",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "He suffers a public social rebuff when Albertine withholds a return greeting and denigrates him, locally excluding him from her circle."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-411-p-415"
}

### Candidate characters

[
  "Andrée",
  "Elstir",
  "M. Vinteuil",
  "Octave",
  "Robert de Saint-Loup",
  "le narrateur"
]

### Prior local context (optional)

Je crus d'abord que j'y échouerais. Comme elle devait rester fort longtemps encore à Balbec et moi aussi, j'avais trouvé que le mieux était de ne pas trop chercher à la voir et d'attendre une occasion qui me fît la rencontrer. Mais cela arrivât-il tous les jours, il était fort à craindre qu'elle se contentât de répondre de loin à mon salut, lequel dans ce cas, répété quotidiennement pendant toute la saison, ne m'avancerait à rien.

### Passage

Peu de temps après, un matin où il avait plu et où il faisait presque froid, je fus abordé sur la digue par une jeune fille portant un toquet et un manchon, si différente de celle que j'avais vue à la réunion d'Elstir que reconnaître en elle la même personne semblait pour l'esprit une opération impossible ; le mien y réussit cependant, mais après une seconde de surprise qui, je crois, n'échappa pas à Albertine. D'autre part me souvenant à ce moment-là des « bonnes façons » qui m'avaient frappé, elle me fit éprouver l'étonnement inverse par son ton rude et ses manières « petite bande ». Au reste la tempe avait cessé d'être le centre optique et rassurant du visage, soit que je fusse placé de l'autre côté, soit que le toquet la recouvrît, soit que son inflammation ne fût pas constante. « Quel temps ! me dit-elle, au fond l'été sans fin à Balbec est une vaste blague. Vous ne faites rien ici ? On ne vous voit jamais au golf, aux bals du Casino ; vous ne montez pas à cheval non plus. Comme vous devez vous raser ! Vous ne trouvez pas qu'on se bêtifie à rester tout le temps sur la plage. Ah ! vous aimez à faire le lézard. Vous avez du temps de reste. Je vois que vous n'êtes pas comme moi, j'adore tous les sports ! Vous n'étiez pas aux courses de la Sogne ? Nous y sommes allés par le tram et je comprends que ça ne vous amuse pas de prendre un tacot pareil ! nous avons mis deux heures ! J'aurais fait trois fois l'aller et retour avec ma bécane. » Moi qui avais admiré Saint-Loup quand il avait appelé tout naturellement le petit chemin de fer d'intérêt local le tortillard, à cause des innombrables détours qu'il faisait, j'étais intimidé par la facilité avec laquelle Albertine disait le « tram », le « tacot ». Je sentais sa maîtrise dans un mode de désignations où j'avais peur qu'elle ne constatât et ne méprisât mon infériorité. Encore la richesse de synonymes que possédait la petite bande pour désigner ce chemin de fer ne m'était-elle pas encore révélée. En parlant, Albertine gardait la tête immobile, les narines serrées, ne faisait remuer que le bout des lèvres. Il en résultait ainsi un son traînard et nasal dans la composition duquel entraient peut-être des hérédités provinciales, une affectation juvénile de flegme britannique, les leçons d'une institutrice étrangère et une hypertrophie congestive de la muqueuse du nez. Cette émission, qui cédait bien vite du reste quand elle connaissait plus les gens et redevenait naturellement enfantine, aurait pu passer pour désagréable. Mais elle était particulière et m'enchantait. Chaque fois que j'étais quelques jours sans la rencontrer, je m'exaltais en me répétant : « On ne vous voit jamais au golf », avec le ton nasal sur lequel elle l'avait dit, toute droite, sans bouger la tête. Et je pensais alors qu'il n'existait pas de personne plus désirable.

Nous formions ce matin-là un de ces couples qui piquent çà et là la digue de leur conjonction, de leur arrêt, juste le temps d'échanger quelques paroles avant de se désunir pour reprendre séparément chacun sa promenade divergente. Je profitai de cette immobilité pour regarder et savoir définitivement où était situé le grain de beauté. Or, comme une phrase de Vinteuil qui m'avait enchanté dans la Sonate et que ma mémoire faisait errer de l'andante au finale jusqu'au jour où, ayant la partition en main, je pus la trouver et l'immobiliser dans mon souvenir à sa place, dans le scherzo, de même le grain de beauté que je m'étais rappelé tantôt sur la joue, tantôt sur le menton, s'arrêta à jamais sur la lèvre supérieure au-dessous du nez. C'est ainsi encore que nous rencontrons avec étonnement des vers que nous savons par coeur, dans une pièce où nous ne soupçonnions pas qu'ils se trouvassent.

À ce moment, comme pour que devant la mer se multipliât en liberté, dans la variété de ses formes, tout le riche ensemble décoratif qu'était le beau déroulement des vierges, à la fois dorées et roses, cuites par le soleil et par le vent, les amies d'Albertine, aux belles jambes, à la taille souple, mais si différentes les unes des autres, montrèrent leur groupe qui se développa, s'avançant dans notre direction, plus près de la mer, sur une ligne parallèle. Je demandai à Albertine la permission de l'accompagner pendant quelques instants. Malheureusement elle se contenta de leur faire bonjour de la main. « Mais vos amies vont se plaindre si vous les laissez », lui dis-je, espérant que nous nous promènerions ensemble. Un jeune homme aux traits réguliers, qui tenait à la main des raquettes, s'approcha de nous. C'était le joueur de baccarat dont les folies indignaient tant la femme du premier président. D'un air froid, impassible, en lequel il se figurait évidemment que consistait la distinction suprême, il dit bonjour à Albertine. « Vous venez du golf, Octave ? lui demanda-t-elle. Ça a-t-il bien marché ? étiez-vous en forme ? – Oh ! ça me dégoûte, je suis dans les choux », répondit-il. – Est-ce qu'Andrée y était ? – Oui, elle a fait soixante-dix-sept. – Oh ! mais c'est un record. – J'avais fait quatre-vingt-deux hier. » Il était le fils d'un très riche industriel qui devait jouer un rôle assez important dans l'organisation de la prochaine Exposition Universelle. Je fus frappé à quel point chez ce jeune homme et les autres très rares amis masculins de ces jeunes filles la connaissance de tout ce qui était vêtements, manière de les porter, cigares, boissons anglaises, cheveux, – et qu'il possédait jusque dans ses moindres détails avec une infaillibilité orgueilleuse qui atteignait à la silencieuse modestie du savant – s'était développée isolément sans être accompagnée de la moindre culture intellectuelle. Il n'avait aucune hésitation sur l'opportunité du smoking ou du pyjama, mais ne se doutait pas du cas où on peut ou non employer tel mot, même des règles les plus simples du français. Cette disparité entre les deux cultures devait être la même chez son père, président du Syndicat des propriétaires de Balbec, car dans une lettre ouverte aux électeurs, qu'il venait de faire afficher sur tous les murs, il disait : « J'ai voulu voir le maire pour lui en causer, il n'a pas voulu écouter mes justes griefs. » Octave obtenait, au casino, des prix dans tous les concours de boston, de tango, etc., ce qui lui ferait faire s'il le voulait un joli mariage dans ce milieu des « bains de mer », où ce n'est pas au figuré mais au propre que les jeunes filles épousent leur « danseur ». Il alluma un cigare en disant à Albertine : « Vous permettez », comme on demande l'autorisation de terminer tout en causant un travail pressé. Car il ne pouvait jamais « rester sans rien faire » quoique il ne fît d'ailleurs jamais rien. Et comme l'inactivité complète finit par avoir les mêmes effets que le travail exagéré, aussi bien dans le domaine moral que dans la vie du corps et des muscles, la constante nullité intellectuelle qui habitait sous le front songeur d'Octave avait fini par lui donner, malgré son air calme, d'inefficaces démangeaisons de penser qui la nuit l'empêchaient de dormir, comme il aurait pu arriver à un métaphysicien surmené.

Pensant que si je connaissais leurs amis j'aurais plus d'occasions de voir ces jeunes filles, j'avais été sur le point de lui demander à être présenté. Je le dis à Albertine, dès qu'il fut parti en répétant : « Je suis dans les choux. » Je pensais lui inculquer ainsi l'idée de le faire la prochaine fois. « Mais voyons, s'écria-t-elle, je ne peux pas vous présenter à un gigolo ! Ici ça pullule de gigolos. Mais ils ne pourraient pas causer avec vous. Celui-ci joue très bien au golf, un point c'est tout. Je m'y connais, il ne serait pas du tout votre genre. – Vos amies vont se plaindre si vous les laissez ainsi, lui dis-je, espérant qu'elle allait me proposer d'aller avec elle les rejoindre. – Mais non, elles n'ont aucun besoin de moi ». Nous croisâmes Bloch qui m'adressa un sourire fin et insinuant, et, embarrassé au sujet d'Albertine qu'il ne connaissait pas ou du moins connaissait « sans la connaître », abaissa sa tête vers son col d'un mouvement raide et rébarbatif. « Comment s'appelle-t-il, cet ostrogoth-là, me demanda Albertine. Je ne sais pas pourquoi il me salue puisqu'il ne me connaît pas. Aussi je ne lui ai pas rendu son salut. » Je n'eus pas le temps de répondre à Albertine, car marchant droit sur nous : « Excuse-moi, dit-il, de t'interrompre, mais je voulais t'avertir que je vais demain à Doncières. Je ne peux plus attendre sans impolitesse et je me demande ce que Saint-Loup-en-bray doit penser de moi. Je te préviens que je prends le train de deux heures. À ta disposition. » Mais je ne pensais plus qu'à revoir Albertine et à tâcher de connaître ses amies, et Doncières, comme elles n'y allaient pas et que je rentrerais après l'heure où elles allaient sur la plage, me paraissait au bout du monde. Je dis à Bloch que cela m'était impossible. « Hé bien, j'irai seul. Selon les deux ridicules alexandrins du sieur Arouet, je dirai à Saint-Loup, pour charmer son cléricalisme : « Apprends que mon devoir ne dépend pas du sien, qu'il y manque s'il veut, je dois faire le mien. » – Je reconnais qu'il est assez joli garçon, me dit Albertine, mais ce qu'il me dégoûte ! » Je n'avais jamais songé que Bloch pût être joli garçon ; il l'était, en effet. Avec une tête un peu proéminente, un nez très busqué, un air d'extrême finesse et d'être persuadé de sa finesse, il avait un visage agréable. Mais il ne pouvait pas plaire à Albertine. C'était peut-être du reste à cause des mauvais côtés de celle-ci, de la dureté, de l'insensibilité de la petite bande, de sa grossièreté avec tout ce qui n'était pas elle. D'ailleurs plus tard quand je les présentai, l'antipathie d'Albertine ne diminua pas. Bloch appartenait à un milieu où, entre la blague exercée contre le monde et pourtant le respect suffisant des bonnes manières que doit avoir un homme qui a « les mains propres », on a fait une sorte de compromis spécial qui diffère des manières du monde et est malgré tout une sorte particulièrement odieuse de mondanité. Quand on le présentait, il s'inclinait à la fois avec un sourire de scepticisme et un respect exagéré, et si c'était à un homme disait : « Enchanté, Monsieur », d'une voix qui se moquait des mots qu'elle prononçait, mais avait conscience d'appartenir à quelqu'un qui n'était pas un mufle. Cette première seconde donnée à une coutume qu'il suivait et raillait à la fois (comme il disait le premier janvier : « Je vous la souhaite bonne et heureuse »), il prenait un air fin et rusé et « proférait des choses subtiles » qui étaient souvent pleines de vérité mais « tapaient sur les nerfs » d'Albertine. Quand je lui dis ce premier jour qu'il s'appelait Bloch, elle s'écria : « Je l'aurais parié que c'était un youpin. C'est bien leur genre de faire les punaises. » Du reste, Bloch devait dans la suite irriter Albertine d'autre façon. Comme beaucoup d'intellectuels, il ne pouvait pas dire simplement les choses simples. Il trouvait pour chacune d'elles un qualificatif précieux, puis généralisait. Cela ennuyait Albertine, laquelle n'aimait pas beaucoup qu'on s'occupât de ce qu'elle faisait, que quand elle s'était foulé le pied et restait tranquille, Bloch dît : « Elle est sur sa chaise longue, mais par ubiquité ne cesse pas de fréquenter simultanément de vagues golfs et de quelconques tennis. » Ce n'était que de la « littérature », mais qui, à cause des difficultés qu'Albertine sentait que cela pouvait lui créer avec des gens chez qui elle avait refusé une invitation en disant qu'elle ne pouvait pas remuer, eût suffi pour lui faire prendre en grippe la figure, le son de la voix, du garçon qui disait ces choses. Nous nous quittâmes, Albertine et moi, en nous promettant de sortir une fois ensemble. J'avais causé avec elle sans plus savoir où tombaient mes paroles, ce qu'elles devenaient, que si j'eusse jeté des cailloux dans un abîme sans fond. Qu'elles soient remplies en général par la personne à qui nous les adressons d'un sens qu'elle tire de sa propre substance et qui est très différent de celui que nous avions mis dans ces mêmes paroles, c'est un fait que la vie courante nous révèle perpétuellement. Mais si de plus nous nous trouvons auprès d'une personne dont l'éducation (comme pour moi celle d'Albertine) nous est inconcevable, inconnus les penchants, les lectures, les principes, nous ne savons pas si nos paroles éveillent en elle quelque chose qui y ressemble plus que chez un animal à qui pourtant on aurait à faire comprendre certaines choses. De sorte qu'essayer de me lier avec Albertine m'apparaissait comme une mise en contact avec l'inconnu sinon avec l'impossible, comme un exercice aussi malaisé que dresser un cheval, aussi reposant qu'élever des abeilles ou que cultiver des rosiers.

J'avais cru, il y avait quelques heures, qu'Albertine ne répondrait à mon salut que de loin. Nous venions de nous quitter en faisant le projet d'une excursion ensemble. Je me promis, quand je rencontrerais Albertine, d'être plus hardi avec elle, et je m'étais tracé d'avance le plan de tout ce que je lui dirais et même (maintenant que j'avais tout à fait l'impression qu'elle devait être légère) de tous les plaisirs que je lui demanderais. Mais l'esprit est influençable comme la plante, comme la cellule, comme les éléments chimiques, et le milieu qui le modifie si on l'y plonge, ce sont des circonstances, un cadre nouveau. Devenu différent par le fait de sa présence même, quand je me trouvai de nouveau avec Albertine, je lui dis tout autre chose que ce que j'avais projeté. Puis me souvenant de la tempe enflammée je me demandais si Albertine n'appréciait pas davantage une gentillesse qu'elle saurait être désintéressée. Enfin j'étais embarrassé devant certains de ses regards, de ses sourires. Ils pouvaient signifier moeurs faciles, mais aussi gaieté un peu bête d'une jeune fille sémillante mais ayant un fond d'honnêteté. Une même expression, de figure comme de langage, pouvant comporter diverses acceptions ; j'étais hésitant comme un élève devant les difficultés d'une version grecque.
