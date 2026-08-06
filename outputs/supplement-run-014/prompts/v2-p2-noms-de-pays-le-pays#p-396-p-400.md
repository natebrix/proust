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
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Robert de Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.78,
      "evidence": "« Robert de Saint-Loup avait si peur d'avoir mal remercié la grand-mère… »; sa lettre pleine de gratitude et de tendresse (« J'espère qu'elle ne finira jamais… esprit subtil et coeur ultra-sensitif… ») et le papier « aux armes de M. de Marsantes ».",
      "explanation": "The narrator presents Robert as delicate, grateful, and affectionate, reinforcing his personal value locally (and insinuating his distinction through the family arms)."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "His letter highlights his sensitivity, gratitude, and attachment, which improves his local appreciation."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-396-p-400"
}

### Candidate characters

[
  "Bloch",
  "M. de Marsantes",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Il fut pris d'une joie dont il ne fut pas plus le maître que d'un état physique qui se produit sans intervention de la volonté, il devint écarlate comme un enfant qu'on vient de punir, et la grand-mère fut beaucoup plus touchée de voir tous les efforts qu'il avait faits (sans y réussir) pour contenir la joie qui le secouait, que par tous les remerciements qu'il aurait pu proférer. Mais lui, craignant d'avoir mal témoigné sa reconnaissance, me priait encore de l'en excuser, le lendemain, penché à la fenêtre du petit chemin de fer d'intérêt local qu'il prit pour rejoindre sa garnison. Celle-ci était, en effet, très peu éloignée. Il avait pensé s'y rendre, comme il faisait souvent, quand il devait revenir le soir et qu'il ne s'agissait pas d'un départ définitif, en voiture. Mais il eût fallu cette fois-ci qu'il mît ses nombreux bagages dans le train. Et il trouva plus simple d'y monter aussi lui-même, suivant en cela l'avis du le directeur qui, consulté, répondit que, voiture ou petit chemin de fer, « ce serait à peu près équivoque ». Il entendait signifier par là que ce serait équivalent (en somme, à peu près ce que Françoise eût exprimé en disant que « cela reviendrait du pareil au même »).

### Passage

« Soit, avait conclu Saint-Loup, je prendrai le petit « tortillard ». Je l'aurais pris aussi si je n'avais été fatigué et aurais accompagné mon ami jusqu'à Doncières ; je lui promis du moins, tout le temps que nous restâmes à la gare de Balbec – c'est-à-dire que le chauffeur du petit train passa à attendre des amis retardataires, sans lesquels il ne voulait pas s'en aller, et aussi à prendre quelques rafraîchissements – d'aller le voir plusieurs fois par semaine. Comme Bloch était venu aussi à la gare – au grand ennui de Saint-Loup – ce dernier voyant que notre camarade l'entendait me prier de venir déjeuner, dîner, habiter à Doncières, finit par lui dire d'un ton extrêmement froid, lequel était chargé de corriger l'amabilité forcée de l'invitation et d'empêcher Bloch de la prendre au sérieux : « Si jamais vous passez par Doncières une après-midi où je sois libre, vous pourrez me demander au quartier, mais libre, je ne le suis à peu près jamais. » Peut-être aussi Saint-Loup craignait-il que, seul, je ne vinsse pas et pensant que j'étais plus lié avec Bloch que je ne le disais, me mettait-il ainsi en mesure d'avoir un compagnon de route, un entraîneur.

J'avais peur que ce ton, cette manière d'inviter quelqu'un en lui conseillant de ne pas venir, n'eût froissé Bloch, et je trouvais que Saint-Loup eût mieux fait de ne rien dire. Mais je m'étais trompé, car après le départ du train, tant que nous fîmes route ensemble jusqu'au croisement de deux avenues où il fallait nous séparer, l'une allant à l'hôtel, l'autre à la villa de Bloch, celui-ci ne cessa de me demander quel jour nous irions à Doncières, car après « toutes les amabilités que Saint-Loup lui avait faites », il eût été « trop grossier de sa part » de ne pas se rendre à son invitation. J'étais content qu'il n'eût pas remarqué, ou fût assez peu mécontent pour désirer feindre de ne pas avoir remarqué, sur quel ton moins que pressant, à peine poli, l'invitation avait été faite. J'aurais pourtant voulu pour Bloch qu'il s'évitât le ridicule d'aller tout de suite à Doncières. Mais je n'osais pas lui donner un conseil qui n'eût pu que lui déplaire en lui montrant que Saint-Loup avait été moins pressant que lui n'était empressé. Il l'était beaucoup trop, et bien que tous les défauts qu'il avait dans ce genre fussent compensés chez lui par de remarquables qualités que d'autres plus réservés n'auraient pas eues, il poussait l'indiscrétion à un point dont on était agacé. La semaine ne pouvait, à l'entendre, se passer sans que nous allions à Doncières (il disait « nous », car je crois qu'il comptait un peu sur ma présence pour excuser la sienne). Tout le long de la route, devant le gymnase perdu dans ses arbres, devant le terrain de tennis, devant la maison, devant le marchand de coquillages, il m'arrêta, me suppliant de fixer un jour, et comme je ne le fis pas, me quitta fâché en me disant : « À ton aise, messire. Moi en tous cas, je suis obligé d'y aller puisqu'il m'a invité. »

Saint-Loup avait si peur d'avoir mal remercié ma grand-mère qu'il me chargeait encore de lui dire sa gratitude le surlendemain, dans une lettre que je reçus de lui de la ville où il était en garnison et qui semblait, sur l'enveloppe où la poste en avait timbré le nom, accourir vite vers moi, me dire qu'entre ses murs, dans le quartier de cavalerie Louis XVI, il pensait à moi. Le papier était aux armes de Marsantes dans lesquelles je distinguais un lion que surmontait une couronne formée par un bonnet de pair de France.

« Après un trajet qui, me disait-il, s'est bien effectué, en lisant un livre acheté à la gare, qui est par Arvède Barine (c'est un auteur russe, je pense, cela m'a paru remarquablement écrit pour un étranger, mais donnez-moi votre appréciation, car vous devez connaître cela, vous, puits de science qui avez tout lu), me voici revenu au milieu de cette vie grossière, où hélas, je me sens bien exilé, n'y ayant pas ce que j'ai laissé à Balbec ; cette vie où je ne retrouve aucun souvenir d'affection, aucun charme d'intellectualité ; vie dont vous mépriseriez sans doute l'ambiance et qui n'est pourtant pas sans charme. Tout m'y semble avoir changé depuis que j'en étais parti, car dans l'intervalle, une des ères les plus importantes de ma vie, celle d'où notre amitié date, a commencé. J'espère qu'elle ne finira jamais. Je n'ai parlé d'elle, de vous, qu'à une seule personne, qu'à mon amie qui m'a fait la surprise de venir passer une heure auprès de moi. Elle aimerait beaucoup vous connaître et je crois que vous vous accorderiez, car elle est aussi extrêmement littéraire. En revanche, pour repenser à nos causeries, pour revivre ces heures que je n'oublierai jamais, je me suis isolé de mes camarades, excellents garçons, mais qui eussent été bien incapables de comprendre cela. Ce souvenir des instants passés avec vous, j'aurais presque mieux aimé, pour le premier jour, l'évoquer pour moi seul et sans vous écrire. Mais j'ai craint que vous, esprit subtil et coeur ultra-sensitif, ne vous mettiez martel en tête en ne recevant pas de lettre, si toutefois vous avez daigné abaisser votre pensée sur le rude cavalier que vous aurez fort à faire pour dégrossir et rendre un peu plus subtil et plus digne de vous. »

Au fond cette lettre ressemblait beaucoup par sa tendresse à celles que, quand je ne connaissais pas encore Saint-Loup, je m'étais imaginé qu'il m'écrirait, dans ces songeries d'où la froideur de son premier accueil m'avait tiré en me mettant en présence d'une réalité glaciale qui ne devait pas être définitive. Une fois que je l'eus reçue, chaque fois qu'à l'heure du déjeuner on apportait le courrier, je reconnaissais tout de suite quand c'était de lui que venait une lettre, car elle avait toujours ce second visage qu'un être montre quand il est absent et dans les traits duquel (les caractères de l'écriture) il n'y a aucune raison pour que nous ne croyions pas saisir une âme individuelle aussi bien que dans la ligne du nez ou les inflexions de la voix.
