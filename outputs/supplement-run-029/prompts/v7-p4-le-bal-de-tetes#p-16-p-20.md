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
      "presence_confidence": 0.98
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
      "confidence": 0.83,
      "evidence": "Il avait dû s'enfermer près d'une année dans une maison de santé... Il avait répondu à mes condoléances sur un ton à la fois profondément senti et presque hautain... la même colère qui animait Bloch contre M. Nissim Bernard animait Bloch contre son beau-père... offrant la répétition... d'un tableau aussi identique.",
      "explanation": "The narrator uses Bloch as a case of recurring traits across generations, ironizing his cult-like grief and highlighting his unchanged quarrelsomeness, which locally lowers his appraisal."
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
      "confidence": 0.83,
      "explanation": "Bloch is portrayed as repeating the same contentious behaviors and affectations, which diminishes him in the narrator's evaluation."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-16-p-20"
}

### Candidate characters

[
  "Bloch père",
  "Brichot",
  "M. Ski",
  "Mme de Cambremer",
  "Robert de Saint-Loup",
  "Swann",
  "docteur Cottard",
  "duchesse de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

Chez d'autres invités, dont le visage était intact, l'âge se marquait autrement ; ils semblaient seulement embarrassés quand ils avaient à marcher ; on croyait d'abord qu'ils avaient mal aux jambes, et ce n'est qu'ensuite qu'on comprenait que la vieillesse leur avait attaché ses semelles de plomb. Elle en embellissait d'autres, comme le prince d'Agrigente. À cet homme long, mince, au regard terne, aux cheveux qui semblaient devoir rester éternellement rougeâtres, avait succédé, par une métamorphose analogue à celle des insectes, un vieillard chez qui les cheveux rouges, trop longtemps vus, avaient été, comme un tapis de table qui a trop servi, remplacé par des cheveux blancs. Sa poitrine avait pris une corpulence inconnue, robuste, presque guerrière, et qui avait dû nécessiter un véritable éclatement de la frêle chrysalide que j'avais connue ; une gravité consciente d'elle-même baignait les yeux, où elle était teintée d'une bienveillance nouvelle qui s'inclinait vers chacun. Et comme, malgré tout, une certaine ressemblance subsistait entre le puissant prince actuel et le portrait que gardait mon souvenir, j'admirais la force de renouvellement original du temps qui, tout en respectant l'unité de l'être et les lois de la vie, sait changer ainsi le décor et introduire de hardis contrastes dans deux aspects successifs d'un même personnage, car, beaucoup de ces gens, on les identifiait immédiatement, mais comme d'assez mauvais portraits d'eux-mêmes réunis dans l'exposition où un artiste inexact et malveillant durcit les traits de l'un, enlève la fraîcheur du teint ou la légèreté de la taille à celle-ci, assombrit le regard de tel autre. Comparant ces images avec celles que j'avais sous les yeux de ma mémoire, j'aimais moins celles qui m'étaient montrées en dernier lieu. Comme souvent on trouve moins bonne et on refuse une des photographies entre lesquelles un ami vous a prié de choisir. À chaque personne et devant l'image qu'elle me montrait d'elle-même j'aurais voulu dire : « Non, pas celle-ci, vous êtes moins bien, ce n'est pas vous. » Je n'aurais pas osé ajouter : « Au lieu de votre beau nez droit on vous a fait le nez crochu de le père du narrateur que je ne vous ai jamais connu. » En effet, c'était un nez nouveau et familial. Bref, l'artiste le Temps avait « rendu » tous ces modèles de telle façon qu'ils étaient reconnaissables, mais ils n'étaient pas ressemblants, non parce qu'il les avait flattés, mais parce qu'il les avait vieillis. Cet artiste-là, du reste, travaille fort lentement. Ainsi cette réplique du visage d'Odette, dont, le jour où j'avais pour la première fois vu Bergotte, j'avais aperçu l'esquisse à peine ébauchée dans le visage de Gilberte, le temps l'avait enfin poussée jusqu'à la plus parfaite ressemblance, comme on le verra tout à l'heure, pareil à ces peintres qui gardent longtemps une oeuvre et la complètent année par année.

### Passage

En plusieurs, je finissais par reconnaître, non seulement eux-mêmes, mais eux tels qu'ils étaient autrefois, et M. Ski, par exemple, pas plus modifié qu'une fleur ou un fruit qui a séché, type de ces amateurs « célibataires de l'art » qui vieillissent inutiles et insatisfaits. M. Ski était resté ainsi un essai informe, confirmant mes théories sur l'art. D'autres le suivaient qui n'étaient nullement des amateurs ; c'étaient des gens du monde qui ne s'intéressaient à rien, et eux aussi, la vieillesse ne les avait pas mûris et, même s'il s'entourait d'un premier cercle de rides et d'un arc de cheveux blancs, leur même visage poupin gardait l'enjouement de la dix-huitième année. Ils n'étaient pas des vieillards, mais des jeunes gens de dix-huit ans extrêmement fanés. Peu de chose eût suffi à effacer ces flétrissures de la vie, et la mort n'aurait pas plus de peine à rendre au visage sa jeunesse qu'il n'en faut pour nettoyer un portrait que seul un peu d'encrassement empêche de briller comme autrefois. Aussi je pensais à l'illusion dont nous sommes dupes quand, entendant parler d'un célèbre vieillard, nous nous fions d'avance à sa bonté, à sa justice, à sa douceur d'âme ; car je sentais qu'ils avaient été, quarante ans plus tôt, de terribles jeunes gens dont il n'y avait aucune raison pour supposer qu'ils n'avaient pas gardé la vanité, la duplicité, la morgue et les ruses.

Et pourtant, en complet contraste avec ceux-ci, j'eus la surprise de causer avec des hommes et des femmes, jadis insupportables, et qui avaient perdu à peu près tous leurs défauts, soit que la vie, en décevant ou comblant leurs désirs, leur eût enlevé de leur présomption ou de leur amertume. Un riche mariage qui ne nous rend plus nécessaire la lutte ou l'ostentation, l'influence même de la femme, la connaissance lentement acquise de valeurs autres que celles auxquelles croit exclusivement une jeunesse frivole, leur avait permis de détendre leur caractère et de montrer leurs qualités. Ceux-là en vieillissant semblaient avoir une personnalité différente, comme ces arbres dont l'automne, en variant leurs couleurs, semble changer l'essence. Pour eux celle de la vieillesse se manifestait vraiment, mais comme une chose morale (qu'ils ne possédaient pas avant). Chez d'autres elle était plutôt physique, et si nouvelle que la personne – Mme de Souvré par exemple – me semblait à la fois inconnue et connue. Inconnue, car il m'était impossible de soupçonner que ce fût elle, et malgré moi je ne pus m'empêcher, en répondant à son salut, de laisser voir le travail d'esprit qui me faisait hésiter entre trois ou quatre personnes (parmi lesquelles n'était pas Mme de Souvré) pour savoir à qui je le rendais avec une chaleur, du reste, qui dut l'étonner, car dans le doute, ayant peur d'être trop froid si c'était une amie intime, j'avais compensé l'incertitude du regard par la chaleur de la poignée de main et du sourire. Mais, d'autre part, son aspect nouveau ne m'était pas inconnu. C'était celui que j'avais souvent vu, au cours de ma vie, à des femmes âgées et fortes, mais sans soupçonner alors qu'elles avaient pu, beaucoup d'années avant, ressembler à Mme de Souvré. Cet aspect était si différent de celui que j'avais connu dans le passé qu'on eût dit qu'elle était un être condamné, comme un personnage de féerie, à apparaître d'abord en jeune fille, puis en épaisse matrone, et qui reviendrait sans doute bientôt en vieille branlante et courbée. Elle semblait, comme une lourde nageuse qui ne voit plus le rivage qu'à une grande distance, repousser avec peine les flots du temps qui la submergeaient. J'arrivai à force de regarder sa figure hésitante, incertaine comme une mémoire infidèle qui ne peut plus retenir les formes d'autrefois, j'arrivai pourtant à en retrouver quelque chose en me livrant au petit jeu d'éliminer les carrés et les hexagones que l'âge avait ajoutés à ces joues. D'ailleurs, ce qu'il mêlait à celles des femmes n'était pas toujours seulement des figures géométriques. Dans les joues de la Mme de Guermantes, restées si semblables pourtant et pourtant composites maintenant comme un nougat, je distinguais une trace de vert-de-gris, un petit morceau rose de coquillage concassé, une grosseur difficile à définir, plus petite qu'une boule de gui et moins transparente qu'une perle de verre.

Certains hommes boitaient dont on sentait bien que ce n'était pas par suite d'un accident de voiture, mais à cause d'une attaque et parce qu'ils avaient déjà, comme on dit, un pied dans la tombe. Dans l'entrebâillement de la leur, à demi paralysées, certaines femmes, comme Mme de Franquetot, semblaient ne pas pouvoir retirer complètement leur robe restée accrochée à la pierre du caveau, et elles ne pouvaient se redresser, infléchies qu'elles étaient, la tête basse, en une courbe qui était comme celle qu'elles occupaient actuellement entre la vie et la mort, avant la chute dernière. Rien ne pouvait lutter contre le mouvement de cette parabole qui les emportait et, dès qu'elles voulaient se lever, elles tremblaient et leurs doigts ne pouvaient rien retenir.

Certaines figures sous la cagoule de leurs cheveux blancs avaient déjà la rigidité, les paupières scellées de ceux qui vont mourir, et leurs lèvres, agitées d'un tremblement perpétuel, semblaient marmonner la prière des agonisants.

À un visage linéairement le même il suffisait, pour qu'il semblât autre, de cheveux blancs au lieu de cheveux noirs ou blonds. Les costumiers de théâtre savent qu'il suffit d'une perruque poudrée pour déguiser très suffisamment quelqu'un et le rendre méconnaissable. Le jeune marquis de Beausergent, que j'avais vu dans la loge de Mme de Cambremer, alors sous-lieutenant, le jour où Mme de Guermantes était dans la baignoire de sa cousine, avait toujours ses traits aussi parfaitement réguliers, plus même, la rigidité physiologique de l'artério-sclérose exagérant encore la rectitude impassible de la physionomie du dandy et donnant à ces traits l'intense netteté, presque grimaçante à force d'immobilité, qu'ils auraient eue dans une étude de Mantegna ou de Michel-Ange. Son teint jadis d'une rougeur égrillarde était maintenant d'une solennelle pâleur ; des poils argentés, un léger embonpoint, une noblesse de doge, une fatigue qui allait jusqu'à l'envie de dormir, tout concourait chez lui à donner une impression nouvelle de majesté fatale. Au rectangle de sa barbe blonde le rectangle égal de sa barbe blanche se substituait si parfaitement que, remarquant que ce sous-lieutenant que j'avais connu avait cinq galons, ma première pensée fut de le féliciter non d'avoir été promu colonel, mais d'être si bien en colonel, déguisement pour lequel il semblait avoir emprunté l'uniforme, l'air grave et triste de l'officier supérieur qu'avait été son père. Chez un autre, la barbe blanche avait succédé à la barbe blonde, mais comme le visage était resté vif, souriant et jeune, elle le faisait paraître seulement plus rouge et plus militant, augmentant l'éclat des yeux, et donnant au mondain resté jeune l'air inspiré d'un prophète. La transformation que les cheveux blancs et d'autres éléments encore avaient opérée, surtout chez les femmes, m'eussent retenu avec moins de force s'ils n'avaient été qu'un changement de couleur, ce qui peut charmer les yeux, mais parce qu'est troublant pour l'esprit un changement de personnes. En effet, « reconnaître » quelqu'un, et plus encore, après n'avoir pas pu le reconnaître, l'identifier, c'est penser sous une seule dénomination deux choses contradictoires, c'est admettre que ce qui était ici l'être qu'on se rappelle n'est plus, et que ce qui y est, c'est un être qu'on ne connaissait pas, c'est avoir à percer un mystère presque aussi troublant que celui de la mort dont il est, du reste, comme la préface et l'annonciateur. Car, ces changements, je savais ce qu'ils voulaient dire, ce à quoi ils préludaient. Aussi cette blancheur des cheveux impressionnait chez les femmes, jointe à tant d'autres changements. On me disait un nom et je restais stupéfait de penser qu'il s'appliquait à la fois à la blonde valseuse que j'avais connue autrefois et à la lourde dame à cheveux blancs qui passait pesamment près de moi. Avec une certaine roseur de teint ce nom était peut-être la seule chose qu'il y avait de commun entre ces deux femmes, plus différentes – celle de la mémoire et celle de la matinée Guermantes – qu'une ingénue et une douairière de pièce de théâtre. Pour que la vie ait pu arriver à donner à la valseuse ce corps énorme, pour qu'elle eût pu ralentir, comme au métronome, ses mouvements embarrassés, pour qu'avec peut-être comme seule parcelle permanente, les joues – plus larges certes, mais qui dès la jeunesse étaient déjà couperosées – elle eût pu substituer à la légère blonde ce vieux maréchal ventripotent, il lui avait fallu accomplir plus de dévastations et de reconstitutions que pour mettre un dôme à la place d'une flèche, et quand on pensait qu'un pareil travail s'était opéré non sur la matière inerte mais sur une chair qui ne change qu'insensiblement, le contraste bouleversant entre l'apparition présente et l'être que je me rappelais reculait celui-ci dans un passé plus que lointain, presque invraisemblable. On avait peine à réunir les deux aspects, à penser les deux personnes sous une même dénomination ; car de même qu'on a peine à penser qu'un mort fut vivant ou que celui qui était vivant est mort aujourd'hui, il est presque aussi difficile, et du même genre de difficulté (car l'anéantissement de la jeunesse, la destruction d'une personne pleine de forces et de légèreté est déjà un premier néant), de concevoir que celle qui fut jeune est vieille, quand l'aspect de cette vieille, juxtaposé à celui de la jeune, semble tellement l'exclure que tour à tour c'est la vieille, puis la jeune, puis la vieille encore qui vous paraissent un rêve, et qu'on ne croirait pas que ceci peut avoir jamais été cela, que la matière de cela est elle-même, sans se réfugier ailleurs, grâce aux savantes manipulations du temps, devenue ceci, que c'est la même matière n'ayant pas quitté le même corps – si l'on n'avait l'indice du nom pareil et le témoignage affirmatif des amis auquel donne seule une apparence de vraisemblance la couperose, jadis étroite entre l'or des épis, aujourd'hui étalée sous la neige. On était effrayé en pensant aux périodes qui avaient dû s'écouler avant que s'accomplît une pareille révolution dans la géologie d'un visage, et de voir quelles érosions s'étaient faites le long du nez, quelles énormes alluvions, au bord des joues, entouraient toute la figure de leurs masses opaques et réfractaires. J'avais bien considéré toujours notre individu à un moment donné du temps comme un polypier où l'oeil, organisme indépendant bien qu'associé, si une poussière passe, cligne sans que l'intelligence le commande ; bien plus, où l'intestin, parasite enfoui, s'infecte sans que l'intelligence l'apprenne, mais aussi et pareillement pour l'âme, dans la durée de la vie, comme une suite de moi juxtaposés mais distincts qui mourraient les uns après les autres ou même alterneraient entre eux comme ceux qui, à Combray, prenaient pour moi la place l'un de l'autre quand venait le soir. Mais aussi j'avais vu que ces cellules morales qui composent un être sont plus durables que lui. J'avais vu les vices, le courage des Guermantes revenir en Saint-Loup comme en lui-même ses défauts étranges et brefs de caractère, comme le sémitisme de Swann. Je pouvais le voir encore en Bloch. Depuis qu'il avait perdu son père, l'idée, outre les grands sentiments de famille qui existent souvent dans les familles juives, que son père était un homme tellement supérieur à tous, avait donné à son amour pour lui la forme d'un culte. Il n'avait pu supporter l'idée de l'avoir perdu et avait dû s'enfermer près d'une année dans une maison de santé. Il avait répondu à mes condoléances sur un ton à la fois profondément senti et presque hautain, tant il me jugeait enviable d'avoir approché cet homme supérieur dont il eût volontiers donné la voiture à deux chevaux à quelque musée historique. Et maintenant, à sa table de famille (car, contrairement à ce que croyait la Mme de Guermantes, il était marié), la même colère qui animait Bloch contre M. Nissim Bernard animait Bloch contre son beau-père. Il lui faisait les mêmes sorties. De même qu'en écoutant parler Cottard, Brichot, tant d'autres, j'avais senti que, par la culture et la mode, une seule ondulation propage dans toute l'étendue de l'espace les mêmes manières de dire, de penser, de même dans toute la durée du temps de grandes lames de fond soulèvent des profondeurs des âges les mêmes colères, les mêmes tristesses, les mêmes bravoures, les mêmes manies, à travers les générations superposées, chaque section, prise à plusieurs niveaux d'une même série, offrant la répétition, comme des ombres sur des écrans successifs, d'un tableau aussi identique, quoique souvent moins insignifiant, que celui qui mettait aux prises de la même façon Bloch et son beau-père, Bloch père et M. Nissim Bernard et d'autres que je n'avais pas connus.
