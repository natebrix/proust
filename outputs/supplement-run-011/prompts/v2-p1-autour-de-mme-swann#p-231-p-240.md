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
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
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
    },
    {
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.93
    },
    {
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Norpois",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.82,
      "evidence": "« Surtout ce qu'il avait dit de Norpois ôtait beaucoup de sa force à une condamnation que j'avais crue sans appel. »",
      "explanation": "The narrator presents Bergotte’s remarks as undermining Norpois’s earlier judgment, locally reducing Norpois’s rhetorical authority."
    }
  ],
  "status_effects": [
    {
      "character": "Norpois",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Norpois’s prior condemnation is explicitly said to lose much of its force."
    },
    {
      "character": "Bergotte",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "By countering Norpois and being treated as knowing the narrator’s ‘case,’ Bergotte gains local authority as a more credible judge."
    }
  ],
  "ambiguities": [
    "The weakening of Norpois’s authority occurs within the narrator’s perception and via Bergotte’s reported comments, not through a direct public confrontation."
  ],
  "unit_id": "v2-p1-autour-de-mme-swann#p-231-p-240"
}

### Candidate characters

[
  "Odette",
  "Swann",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Swann était un de ces hommes qui, ayant vécu longtemps dans les illusions de l'amour, ont vu le bien-être qu'ils ont donné à nombre de femmes accroître le bonheur de celles-ci sans créer de leur part aucune reconnaissance, aucune tendresse envers eux ; mais dans leur enfant ils croient sentir une affection qui, incarnée dans leur nom même, les fera durer après leur mort. Quand il n'y aurait plus de Swann Swann, il y aurait encore une Gilberte, ou une Mme X., née Swann, qui continuerait à aimer le père disparu. Même à l'aimer trop peut-être, pensait sans doute Swann, car il répondit à Gilberte : « Tu es une bonne fille » de ce ton attendri par l'inquiétude que nous inspire, pour l'avenir, la tendresse trop passionnée d'un être destiné à nous survivre. Pour dissimuler son émotion, il se mêla à notre conversation sur la Berma. Il me fit remarquer, mais d'un ton détaché, ennuyé, comme s'il voulait rester en quelque sorte en dehors de ce qu'il disait, avec quelle intelligence, quelle justesse imprévue l'actrice disait à Œnone : « Tu le savais ! » Il avait raison : cette intonation-là du moins, avait une valeur vraiment intelligible et aurait pu par là satisfaire à mon désir de trouver des raisons irréfutables d'admirer la Berma. Mais c'est à cause de sa clarté même qu'elle ne le contentait point. L'intonation était si ingénieuse, d'une intention, d'un sens si définis, qu'elle semblait exister en elle-même et que toute artiste intelligente eût pu l'acquérir. C'était une belle idée ; mais quiconque la concevrait aussi pleinement la posséderait de même. Il restait à la Berma qu'elle l'avait trouvée, mais peut-on employer ce mot de « trouver » quand il s'agit de quelque chose qui ne serait pas différent si on l'avait reçu, quelque chose qui ne tient pas essentiellement à votre être, puisqu'un autre peut ensuite le reproduire ?

### Passage

Mon Dieu, mais comme votre présence élève le niveau de la conversation ! » me dit, comme pour s'excuser auprès de Bergotte, Swann qui avait pris dans le milieu Guermantes l'habitude de recevoir les grands artistes comme de bons amis à qui on cherche seulement à faire manger les plats qu'ils aiment, jouer aux jeux ou, à la campagne, se livrer aux sports qui leur plaisent. « Il me semble que nous parlons bien d'art, ajouta-t-il. – C'est très bien, j'aime beaucoup ça », dit Odette en me jetant un regard reconnaissant, par bonté et aussi parce qu'elle avait gardé ses anciennes aspirations vers une conversation plus intellectuelle. Ce fut ensuite à d'autres personnes, à Gilberte en particulier, que parla Bergotte. J'avais dit à celui-ci tout ce que je ressentais avec une liberté qui m'avait étonné et qui tenait à ce qu'ayant pris avec lui, depuis des années (au cours de tant d'heures de solitude et de lecture, où il n'était pour moi que la meilleure partie de moi-même), l'habitude de la sincérité, de la franchise, de la confiance, il m'intimidait moins qu'une personne avec qui j'aurais causé pour la première fois. Et cependant pour la même raison j'étais fort inquiet de l'impression que j'avais dû produire sur lui, le mépris que j'avais supposé qu'il aurait pour mes idées ne datant pas d'aujourd'hui, mais des temps déjà anciens où j'avais commencé à lire ses livres, dans notre jardin de Combray. J'aurais peut-être dû pourtant me dire que puisque c'était sincèrement, en m'abandonnant à ma pensée, que d'une part j'avais tant sympathisé avec l'oeuvre de Bergotte et que, d'autre part, j'avais éprouvé au théâtre un désappointement dont je ne connaissais pas les raisons, ces deux mouvements instinctifs qui m'avaient entraîné ne devaient pas être si différents l'un de l'autre, mais obéir aux mêmes lois ; et que cet esprit de Bergotte, que j'avais aimé dans ses livres ne devait pas être quelque chose d'entièrement étranger et hostile à ma déception et à mon incapacité de l'exprimer. Car mon intelligence devait être une, et peut-être même n'en existe-t-il qu'une seule dont tout le monde est co-locataire, une intelligence sur laquelle chacun, du fond de son corps particulier, porte ses regards, comme au théâtre, où si chacun a sa place, en revanche, il n'y a qu'une seule scène. Sans doute, les idées que j'avais le goût de chercher à démêler n'étaient pas celles qu'approfondissait d'ordinaire Bergotte dans ses livres. Mais si c'était la même intelligence que nous avions lui et moi à notre disposition, il devait, en me les entendant exprimer, se les rappeler, les aimer, leur sourire, gardant probablement, malgré ce que je supposais, devant son oeil intérieur, tout une autre partie de l'intelligence que celle dont une découpure avait passé dans ses livres et d'après laquelle j'avais imaginé tout son univers mental. De même que les prêtres, ayant la plus grande expérience du coeur, peuvent le mieux pardonner aux péchés qu'ils ne commettent pas, de même le génie, ayant la plus grande expérience de l'intelligence, peut le mieux comprendre les idées qui sont le plus opposées à celles qui forment le fond de ses propres oeuvres. J'aurais dû me dire tout cela (qui d'ailleurs n'a rien de très agréable, car la bienveillance des hauts esprits a pour corollaire l'incompréhension et l'hostilité des médiocres ; or, on est beaucoup moins heureux de l'amabilité d'un grand écrivain qu'on trouve à la rigueur dans ses livres, qu'on ne souffre de l'hostilité d'une femme qu'on n'a pas choisie pour son intelligence, mais qu'on ne peut s'empêcher d'aimer). J'aurais dû me dire tout cela, mais ne me le disais pas, j'étais persuadé que j'avais paru stupide à Bergotte, quand Gilberte me chuchota à l'oreille :

– Je nage dans la joie, parce que vous avez fait la conquête de mon grand ami Bergotte. Il a dit à maman qu'il vous avait trouvé extrêmement intelligent.

– Où allons-nous ? demandai-je à Gilberte.

– Oh ! où on voudra, moi, vous savez, aller ici ou là...

Mais depuis l'incident qui avait eu lieu le jour de l'anniversaire de la mort de son grand-père, je me demandais si le caractère de Gilberte n'était pas autre que ce que j'avais cru, si cette indifférence à ce qu'on ferait, cette sagesse, ce calme, cette douce soumission constante, ne cachaient pas au contraire des désirs très passionnés que par amour-propre elle ne voulait pas laisser voir et qu'elle ne révélait que par sa soudaine résistance quand ils étaient par hasard contrariés.

Comme Bergotte habitait dans le même quartier que mes parents, nous partîmes ensemble ; en voiture il me parla de ma santé : « Nos amis m'ont dit que vous étiez souffrant. Je vous plains beaucoup. Et puis malgré cela je ne vous plains pas trop, parce que je vois bien que vous devez avoir les plaisirs de l'intelligence et c'est probablement ce qui compte surtout pour vous, comme pour ceux qui les connaissent. »

Hélas ! ce qu'il disait là, combien je sentais que c'était peu vrai pour moi que tout raisonnement, si élevé qu'il fût, laissait froid, qui n'étais heureux que dans des moments de simple flânerie, quand j'éprouvais du bien-être ; je sentais combien ce que je désirais dans la vie était purement matériel, et avec quelle facilité je me serais passé de l'intelligence. Comme je ne distinguais pas entre les plaisirs ceux qui me venaient de sources différentes, plus ou moins profondes et durables, je pensai, au moment de lui répondre, que j'aurais aimé une existence où j'aurais été lié avec la Mme de Guermantes, et où j'aurais souvent senti comme dans l'ancien bureau d'octroi des Champs-Élysées une fraîcheur qui m'eût rappelé Combray. Or, dans cet idéal de vie que je n'osais lui confier, les plaisirs de l'intelligence ne tenaient aucune place.

– Non, Monsieur, les plaisirs de l'intelligence sont bien peu de chose pour moi, ce n'est pas eux que je recherche, je ne sais même pas si je les ai jamais goûtés.

– Vous croyez vraiment ? me répondit-il. Eh bien, écoutez, si, tout de même, cela doit être cela que vous aimez le mieux, moi, je me le figure, voilà ce que je crois.

Il ne me persuadait certes pas ; et pourtant je me sentais plus heureux, moins à l'étroit. À cause de ce que m'avait dit Norpois, j'avais considéré mes moments de rêverie, d'enthousiasme, de confiance en moi, comme purement subjectifs et sans vérité. Or, selon Bergotte qui avait l'air de connaître mon cas, il semblait que le symptôme à négliger c'était au contraire mes doutes, mon dégoût de moi-même. Surtout ce qu'il avait dit de Norpois ôtait beaucoup de sa force à une condamnation que j'avais crue sans appel.
