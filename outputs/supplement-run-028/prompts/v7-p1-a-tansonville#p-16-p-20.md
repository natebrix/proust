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
      "canonical_name": "Mme Bontemps",
      "surface_forms": [
        "Mme Bontemps"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme Bontemps",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "Il n'y aurait plus eu besoin d'offrir de l'argent à Mme Bontemps pour qu'elle me renvoyât Albertine.",
      "explanation": "The narrator asserts that Mme Bontemps would now comply without payment, indicating a loss of leverage and a posture of supplication within the new social dynamic."
    }
  ],
  "status_effects": [
    {
      "character": "Mme Bontemps",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Her bargaining power and autonomy are locally reduced; she is portrayed as ready to yield rather than exact a price."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p1-a-tansonville#p-16-p-20"
}

### Candidate characters

[
  "Albertine",
  "Gilberte",
  "Robert de Saint-Loup",
  "le narrateur"
]

### Prior local context (optional)

Gilberte disait-elle cela pour me cacher qu'elle-même, selon ce qu'Albertine m'avait dit, aimait les femmes et avait fait à Albertine des propositions ? Ou bien (car les autres sont souvent plus renseignés sur notre vie que nous ne croyons) savait-elle que j'avais aimé, que j'avais été jaloux d'Albertine et (les autres pouvant savoir plus de vérité que nous ne croyons, mais l'étendre aussi trop loin et être dans l'erreur par des suppositions excessives, alors que nous les avions espérés dans l'erreur par l'absence de toute supposition) s'imaginait-elle que je l'étais encore et me mettait-elle sur les yeux, par bonté, ce bandeau qu'on a toujours tout prêt pour les jaloux ? En tout cas, les paroles de Gilberte, depuis « le mauvais genre » d'autrefois jusqu'au certificat de bonne vie et moeurs d'aujourd'hui, suivaient une marche inverse des affirmations d'Albertine qui avait fini presque par avouer des demi-rapports avec Gilberte. Albertine m'avait étonné en cela comme sur ce que m'avait dit Andrée, car pour toute cette petite bande, si j'avais d'abord cru, avant de la connaître, à sa perversité, je m'étais rendu compte de mes fausses suppositions, comme il arrive si souvent quand on trouve une honnête fille et presque ignorante des réalités de l'amour dans le milieu qu'on avait cru à tort le plus dépravé. Puis j'avais refait le chemin en sens contraire, reprenant pour vraies mes suppositions du début. Mais peut-être Albertine avait-elle voulu me dire cela pour avoir l'air plus expérimentée qu'elle n'était et pour m'éblouir, à Paris, du prestige de sa perversité comme la première fois, à Balbec, par celui de sa vertu. Et tout simplement, quand je lui avais parlé des femmes qui aimaient les femmes, pour ne pas avoir l'air de ne pas savoir ce que c'était, comme dans une conversation on prend un air entendu si on parle de Fourier ou de Tobolsk encore qu'on ne sache pas ce que c'est. Elle avait peut-être vécu près de l'amie de Mlle M. Vinteuil et d'Andrée, séparée par une cloison étanche d'elles qui croyaient qu'elle n'en était pas, ne s'était renseignée ensuite – comme une femme qui épouse un homme de lettres cherche à se cultiver – qu'afin de me complaire en se faisant capable de répondre à mes questions, jusqu'au jour où elle avait compris qu'elles étaient inspirées par la jalousie et où elle avait fait machine en arrière, à moins que ce ne fût Gilberte qui me mentît. L'idée me vint que c'était pour avoir appris d'elle, au cours d'un flirt qu'il aurait conduit dans le sens qui l'intéressait, qu'elle ne détestait pas les femmes, que Robert de Saint-Loup l'avait épousée, espérant des plaisirs qu'il n'avait pas dû trouver chez lui puisqu'il les prenait ailleurs. Aucune de ces hypothèses n'était absurde, car chez des femmes comme la fille d'Odette ou les jeunes filles de la petite bande il y a une telle diversité, un tel cumul de goûts alternants, si même ils ne sont pas simultanés, qu'elles passent aisément d'une liaison avec une femme à un grand amour pour un homme, si bien que définir le goût réel et dominant reste difficile. C'est ainsi qu'Albertine avait cherché à me plaire pour me décider à l'épouser, mais elle y avait renoncé elle-même à cause de mon caractère indécis et tracassier. C'était, en effet, sous cette forme trop simple que je jugeais mon aventure avec Albertine, maintenant que je ne voyais plus cette aventure que du dehors.

### Passage

Ce qui est curieux et ce sur quoi je ne puis m'étendre, c'est à quel point, vers cette époque-là, toutes les personnes qu'avait aimées Albertine, toutes celles qui auraient pu lui faire faire ce qu'elles auraient voulu, demandèrent, implorèrent, j'oserai dire mendièrent, à défaut de mon amitié, quelques relations avec moi. Il n'y aurait plus eu besoin d'offrir de l'argent à Mme Bontemps pour qu'elle me renvoyât Albertine. Ce retour de la vie, se produisant quand il ne servait plus à rien, m'attristait profondément, non à cause d'Albertine, que j'eusse reçue sans plaisir si elle m'eût été ramenée, non plus de Touraine mais de l'autre monde, mais à cause d'une jeune femme que j'aimais et que je ne pouvais arriver à voir. Je me disais que si elle mourait, ou si je ne l'aimais plus, tous ceux qui eussent pu me rapprocher d'elle tomberaient à mes pieds. En attendant, j'essayais en vain d'agir sur eux, n'étant pas guéri par l'expérience, qui aurait dû m'apprendre – si elle apprenait jamais rien – qu'aimer est un mauvais sort comme ceux qu'il y a dans les contes contre quoi on ne peut rien jusqu'à ce que l'enchantement ait cessé.

– Justement, reprit Gilberte, le livre que je tiens parle de ces choses. C'est un vieux Balzac que je pioche pour me mettre à la hauteur de mes oncles, la Fille aux yeux d'Or. Mais c'est absurde, invraisemblable, un beau cauchemar. D'ailleurs, une femme peut, peut-être, être surveillée ainsi par une autre femme, jamais par un homme. – Vous vous trompez, j'ai connu une femme qu'un homme qui l'aimait était arrivé véritablement à séquestrer ; elle ne pouvait jamais voir personne et sortait seulement avec des serviteurs dévoués. – Hé bien, cela devrait vous faire horreur à vous qui êtes si bon. Justement nous disions avec Saint-Loup que vous devriez vous marier. Votre femme vous guérirait et vous feriez son bonheur. – Non, parce que j'ai trop mauvais caractère. – Quelle idée ! – Je vous assure ! J'ai, du reste, été fiancé, mais je n'ai pas pu.

Je ne voulus pas emprunter à Gilberte la Fille aux yeux d'Or puisqu'elle le lisait. Mais elle me prêta, le dernier soir que je passai chez elle, un livre qui me produisit une impression assez vive et mêlée. C'était un volume du journal inédit des Goncourt.

J'étais triste, ce dernier soir, en remontant dans ma chambre, de penser que je n'avais pas été une seule fois revoir l'église de Combray qui semblait m'attendre au milieu des verdures dans une fenêtre toute violacée. Je me disais : « Tant pis, ce sera pour une autre année si je ne meurs pas d'ici là », ne voyant pas d'autre obstacle que ma mort et n'imaginant pas celle de l'église qui me semblait devoir durer longtemps après ma mort comme elle avait duré longtemps avant ma naissance.

Quand, avant d'éteindre ma bougie, je lus le passage que je transcris plus bas, mon absence de disposition pour les lettres, pressentie jadis du côté de Guermantes, confirmée durant ce séjour dont c'était le dernier soir – ce soir des veilles de départ où, l'engourdissement des habitudes qui vont finir cessant, on essaie de se juger – me parut quelque chose de moins regrettable, comme si la littérature ne révélait pas de vérité profonde, et en même temps il me semblait triste que la littérature ne fût pas ce que j'avais cru. D'autre part, moins regrettable me semblait l'état maladif qui allait me confiner dans une maison de santé, si les belles choses dont parlent les livres n'étaient pas plus belles que ce que j'avais vu. Mais par une contradiction bizarre, maintenant que ce livre en parlait, j'avais envie de les voir. Voici les pages que je lus jusqu'à ce que la fatigue me fermât les yeux :
