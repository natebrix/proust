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
      "narrative_stance": "uncertain",
      "confidence": 0.78,
      "evidence": "À la présentation, le narrateur voit \"un homme jeune, rude, petit... à nez rouge... et à barbiche noire\" et conclut que ses livres \"déclinèrent pour moi... jusqu'à n'avoir été que quelque médiocre divertissement d'homme à barbiche.\"",
      "explanation": "The encounter shatters the ideal image of a \"divine old man\"; the narrator reads Bergotte's appearance and attitude (worried about lunch) as incompatible with the wisdom of his works, which greatly diminishes Bergotte in his eyes."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "Locally, Bergotte clearly loses esteem because the narrator requalifies his books as mediocre entertainment produced by a man without apparent nobility."
    }
  ],
  "ambiguities": [
    "Le rabaissement repose sur la première impression du narrateur et sur une projection de l'œuvre sur le corps de l'écrivain; la narration au passé ('je me disais') peut ironiser cette réaction, rendant incertaine son plein aval par le texte."
  ],
  "unit_id": "v2-p1-autour-de-mme-swann#p-201-p-205"
}

### Candidate characters

[
  "Gilberte",
  "Odette",
  "Swann",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Qu'est-ce que cela peut me faire ce que les autres pensent ? Je trouve ça grotesque de s'occuper des autres dans les choses de sentiment. On sent pour soi, pas pour le public. Mademoiselle, qui a peu de distractions, se fait une fête d'aller au concert, je ne vais pas l'en priver pour faire plaisir au public.

### Passage

Elle prit son chapeau.

– Mais Gilberte, lui dis-je en lui prenant le bras, ce n'est pas pour faire plaisir au public, c'est pour faire plaisir à votre père.

– Vous n'allez pas me faire d'observations, j'espère, me cria-t-elle, d'une voix dure et en se dégageant vivement.

Faveur plus précieuse encore que de m'emmener avec eux au Jardin d'Acclimatation ou au concert, les Swann ne m'excluaient même pas de leur amitié avec Bergotte, laquelle avait été à l'origine du charme que je leur avais trouvé quand, avant même de connaître Gilberte, je pensais que son intimité avec le divin vieillard eût fait d'elle pour moi la plus passionnante des amies, si le dédain que je devais lui inspirer ne m'eût pas interdit l'espoir qu'elle m'emmenât jamais avec lui visiter les villes qu'il aimait. Or, un jour, Odette m'invita à un grand déjeuner. Je ne savais pas quels devaient être les convives. En arrivant, je fus, dans le vestibule, déconcerté par un incident qui m'intimida. Odette manquait rarement d'adopter les usages qui passent pour élégants pendant une saison et ne parvenant pas à se maintenir sont bientôt abandonnés (comme beaucoup d'années auparavant elle avait eu son « handsome cab », ou faisait imprimer sur une invitation à déjeuner que c'était « to meet » un personnage plus ou moins important). Souvent ces usages n'avaient rien de mystérieux et n'exigeaient pas d'initiation. C'est ainsi que, mince innovation de ces années-là et importée d'Angleterre, Odette avait fait faire à son mari des cartes où le nom de Swann Swann était précédé de « Mr ». Après la première visite que je lui avais faite, Odette avait corné chez moi un de ces « cartons » comme elle disait. Jamais personne ne m'avait déposé de cartes ; je ressentis tant de fierté, d'émotion, de reconnaissance, que, réunissant tout ce que je possédais d'argent, je commandai une superbe corbeille de camélias et l'envoyai à Odette. Je suppliai mon père d'aller mettre une carte chez elle, mais de s'en faire vite graver d'abord où son nom fût précédé de « Mr ». Il n'obéit à aucune de mes deux prières, j'en fus désespéré pendant quelques jours, et me demandai ensuite s'il n'avait pas eu raison. Mais l'usage du « Mr », s'il était inutile, était clair. Il n'en était pas ainsi d'un autre qui, le jour de ce déjeuner, me fut révélé, mais non pourvu de signification. Au moment où j'allais passer de l'antichambre dans le salon, le maître d'hôtel me remit une enveloppe mince et longue sur laquelle mon nom était écrit. Dans ma surprise, je le remerciai, cependant je regardais l'enveloppe. Je ne savais pas plus ce que j'en devais faire qu'un étranger d'un de ces petits instruments que l'on donne aux convives dans les dîners chinois. Je vis qu'elle était fermée, je craignis d'être indiscret en l'ouvrant tout de suite et je la mis dans ma poche d'un air entendu. Odette m'avait écrit quelques jours auparavant de venir déjeuner « en petit comité ». Il y avait pourtant seize personnes, parmi lesquelles j'ignorais absolument que se trouvât Bergotte. Odette qui venait de me « nommer » comme elle disait à plusieurs d'entre elles, tout à coup, à la suite de mon nom, de la même façon qu'elle venait de le dire (et comme si nous étions seulement deux invités du déjeuner qui devaient être chacun également contents de connaître l'autre), prononça le nom du doux Chantre aux cheveux blancs. Ce nom de Bergotte me fit tressauter comme le bruit d'un revolver qu'on aurait déchargé sur moi, mais instinctivement pour faire bonne contenance je saluai ; devant moi, comme ces prestidigitateurs qu'on aperçoit intacts et en redingote dans la poussière d'un coup de feu d'où s'envole une colombe, mon salut m'était rendu par un homme jeune, rude, petit, râblé et myope, à nez rouge en forme de coquille de colimaçon et à barbiche noire. J'étais mortellement triste, car ce qui venait d'être réduit en poudre, ce n'était pas seulement le langoureux vieillard, dont il ne restait plus rien, c'était aussi la beauté d'une oeuvre immense que j'avais pu loger dans l'organisme défaillant et sacré que j'avais, comme un temple, construit expressément pour elle, mais à laquelle aucune place n'était réservée dans le corps trapu, rempli de vaisseaux, d'os, de ganglions, du petit homme à nez camus et à barbiche noire qui était devant moi. Tout le Bergotte que j'avais lentement et délicatement élaboré moi-même, goutte à goutte, comme une stalactite, avec la transparente beauté de ses livres, ce Bergotte-là se trouvait d'un seul coup ne plus pouvoir être d'aucun usage, du moment qu'il fallait conserver le nez en colimaçon et utiliser la barbiche noire ; comme n'est plus bonne à rien la solution que nous avions trouvée pour un problème dont nous avions lu incomplètement la donnée et sans tenir compte que le total devait faire un certain chiffre. Le nez et la barbiche étaient des éléments aussi inéluctables et d'autant plus gênants que, me forçant à réédifier entièrement le personnage de Bergotte, ils semblaient encore impliquer, produire, sécréter incessamment un certain genre d'esprit actif et satisfait de soi, ce qui n'était pas de jeu, car cet esprit-là n'avait rien à voir avec la sorte d'intelligence répandue dans ces livres, si bien connus de moi et que pénétrait une douce et divine sagesse. En partant d'eux, je ne serais jamais arrivé à ce nez en colimaçon ; mais en partant de ce nez qui n'avait pas l'air de s'en inquiéter, faisait cavalier seul et « fantaisie », j'allais dans une tout autre direction que l'oeuvre de Bergotte, j'aboutirais, semblait-il, à quelque mentalité d'ingénieur pressé, de la sorte de ceux qui quand on les salue croient comme il faut de dire : « Merci et vous » avant qu'on leur ait demandé de leurs nouvelles, et si on leur déclare qu'on a été enchanté de faire leur connaissance, répondent par une abréviation qu'ils se figurent bien portée, intelligente et moderne en ce qu'elle évite de perdre en de vaines formules un temps précieux : « Également ». Sans doute, les noms sont des dessinateurs fantaisistes, nous donnant des gens et des pays des croquis si peu ressemblants que nous éprouvons souvent une sorte de stupeur quand nous avons devant nous, au lieu du monde imaginé, le monde visible (qui d'ailleurs n'est pas le monde vrai, nos sens ne possédant pas beaucoup plus le don de la ressemblance que l'imagination, si bien que les dessins enfin approximatifs qu'on peut obtenir de la réalité sont au moins aussi différents du monde vu que celui-ci l'était du monde imaginé). Mais pour Bergotte la gêne du nom préalable n'était rien auprès de celle que me causait l'oeuvre connue, à laquelle j'étais obligé d'attacher, comme après un ballon, l'homme à barbiche sans savoir si elle garderait la force de s'élever. Il semblait bien pourtant que ce fût lui qui eût écrit les livres que j'avais tant aimés, car Odette ayant cru devoir lui dire mon goût pour l'un d'eux, il ne montra nul étonnement qu'elle en eût fait part à lui plutôt qu'à un autre convive, et ne sembla pas voir là l'effet d'une méprise ; mais, emplissant la redingote qu'il avait mise en l'honneur de tous ces invités, d'un corps avide du déjeuner prochain, ayant son attention occupée d'autres réalités importantes, ce ne fut que comme à un épisode révolu de sa vie antérieure, et comme si on avait fait allusion à un costume du duc de Guise qu'il eût mis une certaine année à un bal costumé, qu'il sourit en se reportant à l'idée de ses livres, lesquels aussitôt déclinèrent pour moi (entraînant dans leur chute toute la valeur du Beau, de l'univers, de la vie), jusqu'à n'avoir été que quelque médiocre divertissement d'homme à barbiche. Je me disais qu'il avait dû s'y appliquer, mais que s'il avait vécu dans une île entourée par des bancs d'huîtres perlières, il se fût à la place livré avec succès au commerce des perles. Son oeuvre ne me semblait plus aussi inévitable. Et alors je me demandais si l'originalité prouve vraiment que les grands écrivains soient des dieux régnant chacun dans un royaume qui n'est qu'à lui, ou bien s'il n'y a pas dans tout cela un peu de feinte, si les différences entre les oeuvres ne seraient pas le résultat du travail, plutôt que l'expression d'une différence radicale d'essence entre les diverses personnalités.

Cependant on était passé à table. À côté de mon assiette je trouvai un oeillet dont la tige était enveloppée dans du papier d'argent. Il m'embarrassa moins que n'avait fait l'enveloppe remise dans l'antichambre et que j'avais complètement oubliée. L'usage, pourtant aussi nouveau pour moi, me parut plus intelligible quand je vis tous les convives masculins s'emparer d'un oeillet semblable qui accompagnait leur couvert et l'introduire dans la boutonnière de leur redingote. Je fis comme eux avec cet air naturel d'un libre penseur dans une église, lequel ne connaît pas la messe, mais se lève quand tout le monde se lève et se met à genoux un peu après que tout le monde s'est mis à genoux. Un autre usage inconnu et moins éphémère me déplut davantage. De l'autre côté de mon assiette il y en avait une plus petite remplie d'une matière noirâtre que je ne savais pas être du caviar. J'étais ignorant de ce qu'il fallait en faire, mais résolu à n'en pas manger.
