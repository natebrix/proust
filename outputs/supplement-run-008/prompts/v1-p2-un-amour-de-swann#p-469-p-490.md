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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "duchesse de Guermantes",
      "surface_forms": [
        "princesse des Laumes",
        "princesse"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Mme de Cambremer",
      "surface_forms": [
        "Mme de Cambremer",
        "petite Mme de Cambremer"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "duchesse de Guermantes",
      "target": "Mme de Cambremer",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.89,
      "evidence": "« c'est une petite Mme de Cambremer »; « Ça doit être des ‘gens de la campagne’ ! »; « Elle se met trop en avant... Pas agréable... pour son mari ! »",
      "explanation": "The princesse dismisses and belittles Mme de Cambremer and her milieu with snobbish remarks, marking social exclusion; the narration frames these as affected and spiteful."
    },
    {
      "event_id": "E2",
      "source": "collective_social_voice",
      "target": "Mme de Cambremer",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.86,
      "evidence": "« l'initiative hardie ... produisirent une impression généralement favorable »; le général: « Elle est jolie à croquer. »",
      "explanation": "Mme de Cambremer’s bold move at the piano and brief closeness to the pianist win general approval, reinforced by the general’s overt admiration."
    }
  ],
  "status_effects": [
    {
      "character": "Mme de Cambremer",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "She is socially undercut by a high-ranking princess’s public snub and characterization as provincial and forward."
    },
    {
      "character": "Mme de Cambremer",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.83,
      "explanation": "She gains local credit and favor for her initiative and is openly admired by a high-status observer."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-469-p-490"
}

### Candidate characters

[
  "duc de Guermantes",
  "général de Froberville",
  "le narrateur",
  "le pianiste",
  "marquise de Gallardon",
  "marquise de Saint-Euverte",
  "prince des Laumes"
]

### Prior local context (optional)

Et quittant sa cousine mortifiée, elle éclata de nouveau d'un rire qui scandalisa les personnes qui écoutaient la musique, mais attira l'attention de marquise de Saint-Euverte, restée par politesse près du piano et qui aperçut seulement alors la princesse des Laumes. marquise de Saint-Euverte était d'autant plus ravie de voir princesse des Laumes qu'elle la croyait encore à Guermantes en train de soigner son beau-père malade.

### Passage

– Mais comment, princesse, vous étiez là ?

– Oui, je m'étais mise dans un petit coin, j'ai entendu de belles choses.

– Comment, vous êtes là depuis déjà un long moment !

– Mais oui, un très long moment qui m'a semblé très court, long seulement parce que je ne vous voyais pas.

Mme de Saint-Euverte voulut donner son fauteuil à la princesse qui répondit :

– Mais pas du tout ! Pourquoi ? Je suis bien n'importe où !

Et, avisant, avec intention, pour mieux manifester sa simplicité de grande dame, un petit siège sans dossier :

– Tenez, ce pouf, c'est tout ce qu'il me faut. Cela me fera tenir droite. Oh ! mon Dieu, je fais encore du bruit, je vais me faire conspuer.

Cependant le pianiste redoublant de vitesse, l'émotion musicale était à son comble, un domestique passait des rafraîchissements sur un plateau et faisait tinter des cuillers et, comme chaque semaine, Mme de Saint-Euverte lui faisait, sans qu'il la vît, des signes de s'en aller. Une nouvelle mariée, à qui on avait appris qu'une jeune femme ne doit pas avoir l'air blasé, souriait de plaisir, et cherchait des yeux la maîtresse de maison pour lui témoigner par son regard sa reconnaissance d'avoir « pensé à elle » pour un pareil régal. Pourtant, quoique avec plus de calme que Mme de Franquetot, ce n'est pas sans inquiétude qu'elle suivait le morceau ; mais la sienne avait pour objet, au lieu du pianiste, le piano sur lequel une bougie tressautant à chaque fortissimo risquait, sinon de mettre le feu à l'abat-jour, du moins de faire des taches sur le palissandre. À la fin elle n'y tint plus et, escaladant les deux marches de l'estrade, sur laquelle était placé le piano, se précipita pour enlever la bobèche. Mais à peine ses mains allaient-elles la toucher que, sur un dernier accord, le morceau finit et le pianiste se leva. Néanmoins l'initiative hardie de cette jeune femme, la courte promiscuité qui en résulta entre elle et l'instrumentiste, produisirent une impression généralement favorable.

– Vous avez remarqué ce qu'a fait cette personne, princesse, dit le général de Froberville à la princesse des Laumes qu'il était venu saluer et que Mme de Saint-Euverte quitta un instant. C'est curieux. Est-ce donc une artiste ?

– Non, c'est une petite Mme de Cambremer, répondit étourdiment la princesse et elle ajouta vivement : Je vous répète ce que j'ai entendu dire, je n'ai aucune espèce de notion de qui c'est, on a dit derrière moi que c'étaient des voisins de campagne de Mme de Saint-Euverte, mais je ne crois pas que personne les connaisse. Ça doit être des « gens de la campagne » ! Du reste, je ne sais pas si vous êtes très répandu dans la brillante société qui se trouve ici, mais je n'ai pas idée du nom de toutes ces étonnantes personnes. À quoi pensez-vous qu'ils passent leur vie en dehors des soirées de Mme de Saint-Euverte ? Elle a dû les faire venir avec les musiciens, les chaises et les rafraîchissements. Avouez que ces « invités de chez Belloir » sont magnifiques. Est-ce que vraiment elle a le courage de louer ces figurants toutes les semaines. Ce n'est pas possible !

– Ah ! Mais Cambremer, c'est un nom authentique et ancien, dit le général.

– Je ne vois aucun mal à ce que ce soit ancien, répondit sèchement la princesse, mais en tous cas ce n'est pas euphonique, ajouta-t-elle en détachant le mot euphonique comme s'il était entre guillemets, petite affectation de dépit qui était particulière à la coterie Guermantes.

– Vous trouvez ? Elle est jolie à croquer, dit le général qui ne perdait pas Mme de Cambremer de vue. Ce n'est pas votre avis, princesse ?

– Elle se met trop en avant, je trouve que chez une si jeune femme, ce n'est pas agréable, car je ne crois pas qu'elle soit ma contemporaine, répondit Mme des Laumes (cette expression étant commune aux Gallardon et aux Guermantes).

Mais la princesse voyant que M. de Froberville continuait à regarder Mme de Cambremer, ajouta moitié par méchanceté pour celle-ci, moitié par amabilité pour le général : « Pas agréable... pour son mari ! Je regrette de ne pas la connaître puisqu'elle vous tient à coeur, je vous aurais présenté », dit la princesse qui probablement n'en aurait rien fait si elle avait connu la jeune femme. « Je vais être obligée de vous dire bonsoir, parce que c'est la fête d'une amie à qui je dois aller la souhaiter, dit-elle d'un ton modeste et vrai, réduisant la réunion mondaine à laquelle elle se rendait à la simplicité d'une cérémonie ennuyeuse, mais où il était obligatoire et touchant d'aller. D'ailleurs je dois y retrouver duc de Guermantes qui, pendant que j'étais ici, est allé voir ses amis que vous connaissez, je crois, qui ont un nom de pont, les Iéna. »

– Ç'a été d'abord un nom de victoire, princesse, dit le général. Qu'est-ce que vous voulez, pour un vieux briscard comme moi, ajouta-t-il en ôtant son monocle pour l'essuyer, comme il aurait changé un pansement, tandis que la princesse détournait instinctivement les yeux, cette noblesse d'Empire, c'est autre chose bien entendu, mais enfin, pour ce que c'est, c'est très beau dans son genre, ce sont des gens qui en somme se sont battus en héros.

– Mais je suis pleine de respect pour les héros, dit la princesse, sur un ton légèrement ironique : si je ne vais pas avec duc de Guermantes chez cette princesse d'Iéna, ce n'est pas du tout pour ça, c'est tout simplement parce que je ne les connais pas. duc de Guermantes les connaît, les chérit. Oh ! non, ce n'est pas ce que vous pouvez penser, ce n'est pas un flirt, je n'ai pas à m'y opposer ! Du reste, pour ce que cela sert quand je veux m'y opposer ! ajouta-t-elle d'une voix mélancolique, car tout le monde savait que dès le lendemain du jour où le prince des Laumes avait épousé sa ravissante cousine, il n'avait pas cessé de la tromper. Mais enfin ce n'est pas le cas, ce sont des gens qu'il a connus autrefois, il en fait ses choux gras, je trouve cela très bien. D'abord je vous dirai que rien que ce qu'il m'a dit de leur maison... Pensez que tous leurs meubles sont « Empire » !

– Mais, princesse, naturellement, c'est parce que c'est le mobilier de leurs grands-parents.

– Mais je ne vous dis pas, mais ça n'est pas moins laid pour ça. Je comprends très bien qu'on ne puisse pas avoir de jolies choses, mais au moins qu'on n'ait pas de choses ridicules. Qu'est-ce que vous voulez ? je ne connais rien de plus pompier, de plus bourgeois que cet horrible style avec ces commodes qui ont des têtes de cygnes comme des baignoires.

– Mais je crois même qu'ils ont de belles choses, ils doivent avoir la fameuse table de mosaïque sur laquelle a été signé le traité de...

– Ah ! Mais qu'ils aient des choses intéressantes au point de vue de l'histoire, je ne vous dis pas. Mais ça ne peut pas être beau... puisque c'est horrible ! Moi j'ai aussi des choses comme ça que duc de Guermantes a héritées des Montesquiou. Seulement elles sont dans les greniers de Guermantes où personne ne les voit. Enfin, du reste, ce n'est pas la question, je me précipiterais chez eux avec duc de Guermantes, j'irais les voir même au milieu de leurs sphinx et de leur cuivre si je les connaissais, mais... je ne les connais pas ! Moi, on m'a toujours dit quand j'étais petite que ce n'était pas poli d'aller chez les gens qu'on ne connaissait pas, dit-elle en prenant un ton puéril. Alors, je fais ce qu'on m'a appris. Voyez-vous ces braves gens s'ils voyaient entrer une personne qu'ils ne connaissent pas ? Ils me recevraient peut-être très mal ! dit la princesse.
