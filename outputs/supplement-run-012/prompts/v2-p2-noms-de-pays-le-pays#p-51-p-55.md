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
      "canonical_name": "M. de Stermaria",
      "surface_forms": [
        "M. de Stermaria",
        "de Stermaria",
        "Stermaria"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "M. de Stermaria",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "Il « gardait l'air glacial, pressé, distant, rude, pointilleux et malintentionné » et, « sans le moindre geste d'excuse », fit lever les nouveaux venus en déclarant qu'il lui déplaisait que « des gens qu'il ne connaissait pas » eussent pris sa table.",
      "explanation": "The narrator frames Mr. de Stermaria as haughty and discourteous; the scene highlights his arrogance and his dry exclusion of others, which devalues him locally."
    }
  ],
  "status_effects": [
    {
      "character": "M. de Stermaria",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "His attitude judged cold and harsh by the narrator lowers his perceived value in this passage."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-51-p-55"
}

### Candidate characters

[
  "Mlle de Stermaria",
  "le directeur"
]

### Prior local context (optional)

– Naturellement ! On la loue pour vingt francs. Vous pouvez la prendre si cela vous fait plaisir. Et je sais pertinemment que, lui, avait fait demander une audience au roi qui lui a fait savoir qu'il n'avait pas à connaître ce souverain de Guignol.

### Passage

– Ah, vraiment, c'est intéressant ! il y a tout de même des gens !...

Et sans doute tout cela était vrai, mais c'était aussi par ennui de sentir que pour une bonne partie de la foule ils n'étaient, eux, que de bons bourgeois qui ne connaissaient pas ce roi et cette reine prodigues de leur monnaie, que le notaire, le président, le bâtonnier, au passage de ce qu'ils appelaient un carnaval, éprouvaient tant de mauvaise humeur et manifestaient tout haut une indignation au courant de laquelle était leur ami le maître d'hôtel, qui, obligé de faire bon visage aux souverains plus généreux qu'authentiques, cependant tout en prenant leur commande, adressait de loin à ses vieux clients un clignement d'oeil significatif. Peut-être y avait-il aussi un peu de ce même ennui d'être par erreur crus moins « chic » et de ne pouvoir expliquer qu'ils l'étaient davantage, au fond du « Joli Monsieur ! » dont ils qualifiaient un jeune gommeux, fils poitrinaire et fêtard d'un grand industriel et qui, tous les jours, dans un veston nouveau, une orchidée à la boutonnière, déjeunait au champagne, et allait, pâle, impassible, un sourire d'indifférence aux lèvres, jeter au Casino sur la table de baccarat des sommes énormes « qu'il n'a pas les moyens de perdre » disait d'un air renseigné le notaire au premier président duquel la femme « tenait de bonne source » que ce jeune homme « fin de siècle » faisait mourir de chagrin ses parents.

D'autre part, le bâtonnier et ses amis ne tarissaient pas de sarcasmes, au sujet d'une vieille dame riche et titrée, parce qu'elle ne se déplaçait qu'avec tout son train de maison. Chaque fois que la femme du notaire et la femme du premier président la voyaient dans la salle à manger au moment des repas, elles l'inspectaient insolemment avec leur face à main du même air minutieux et défiant que si elle avait été quelque plat au nom pompeux mais à l'apparence suspecte qu'après le résultat défavorable d'une observation méthodique on fait éloigner, avec un geste distant et une grimace de dégoût.

Sans doute par là voulaient-elles seulement montrer, que s'il y avait certaines choses dont elles manquaient – dans l'espèce certaines prérogatives de la vieille dame, et être en relations avec elle – c'était non pas parce qu'elles ne pouvaient, mais ne voulaient pas les posséder. Mais elles avaient fini par s'en convaincre elles-mêmes ; et c'est la suppression de tout désir, de la curiosité pour les formes de la vie qu'on ne connaît pas, de l'espoir de plaire à de nouveaux êtres, remplacés chez ces femmes par un dédain simulé, par une allégresse factice, qui avait l'inconvénient de leur faire mettre du déplaisir sous l'étiquette de contentement et se mentir perpétuellement à elles-mêmes, deux conditions pour qu'elles fussent malheureuses. Mais tout le monde dans cet hôtel agissait sans doute de la même manière qu'elles, bien que sous d'autres formes, et sacrifiait sinon à l'amour-propre, du moins à certains principes d'éducations ou à des habitudes intellectuelles, le trouble délicieux de se mêler à une vie inconnue. Sans doute le microcosme dans lequel s'isolait la vieille dame n'était pas empoisonné de virulentes aigreurs comme le groupe où ricanaient de rage la femme du notaire et du premier président. Il était au contraire embaumé d'un parfum fin et vieillot mais qui n'était pas moins factice. Car au fond la vieille dame eût probablement trouvé à séduire, à s'attacher, en se renouvelant pour cela elle-même, la sympathie mystérieuse d'êtres nouveaux, un charme dont est dénué le plaisir qu'il y a à ne fréquenter que des gens de son monde et à se rappeler que, ce monde étant le meilleur qui soit, le dédain mal informé d'autrui est négligeable. Peut-être sentait-elle que, si elle était arrivée inconnue au Grand-Hôtel de Balbec elle eût avec sa robe de laine noire et son bonnet démodé fait sourire quelque noceur qui de son « rocking » eût murmuré « quelle purée ! » ou surtout quelque homme de valeur ayant gardé comme le premier président, entre ses favoris poivre et sel, un visage frais et des yeux spirituels comme elle les aimait, et qui eût aussitôt désigné à la lentille rapprochante du face-à-main conjugal l'apparition de ce phénomène insolite ; et peut-être était-ce par inconsciente appréhension de cette première minute qu'on sait courte mais qui n'est pas moins redoutée – comme la première tête qu'on pique dans l'eau – que cette dame envoyait d'avance un domestique mettre l'hôtel au courant de sa personnalité et de ses habitudes, et coupant court aux salutations du directeur gagnait avec une brièveté où il y avait plus de timidité que d'orgueil sa chambre où des rideaux personnels remplaçant ceux qui pendaient aux fenêtres, des paravents, des photographies, mettaient si bien entre elle et le monde extérieur auquel il eût fallu s'adapter la cloison de ses habitudes, que c'était son chez elle, au sein duquel elle était restée, qui voyageait plutôt qu'elle-même...

Dès lors, ayant placé entre elle d'une part, le personnel de l'hôtel et les fournisseurs de l'autre, ses domestiques qui recevaient à sa place le contact de cette humanité nouvelle et entretenaient autour de leur maîtresse l'atmosphère accoutumée, ayant mis ses préjugés entre elle et les baigneurs, insoucieuse de déplaire à des gens que ses amies n'auraient pas reçus, c'est dans son monde qu'elle continuait à vivre par la correspondance avec ses amies, par le souvenir, par la conscience intime qu'elle avait de sa situation, de la qualité de ses manières, de la compétence de sa politesse. Et tous les jours, quand elle descendait pour aller dans sa calèche faire une promenade, sa femme de chambre qui portait ses affaires derrière elle, son valet de pied qui la devançait semblaient comme ces sentinelles qui, aux portes d'une ambassade pavoisée aux couleurs du pays dont elle dépend, garantissent pour elle, au milieu d'un sol étranger, le privilège de son exterritorialité. Elle ne quitta pas sa chambre avant le milieu de l'après-midi, le jour de notre arrivée, et nous ne l'aperçûmes pas dans la salle à manger où le directeur, comme nous étions nouveaux venus, nous conduisit, sous sa protection, à l'heure du déjeuner, comme un gradé qui mène des bleus chez le caporal tailleur pour les faire habiller ; mais nous y vîmes, en revanche, au bout d'un instant un hobereau et sa fille, d'une obscure mais très ancienne famille de Bretagne, M. et Mlle de Stermaria, dont on nous avait fait donner la table, croyant qu'ils ne rentreraient que le soir. Venus seulement à Balbec pour retrouver des châtelains qu'ils connaissaient dans le voisinage, ils ne passaient dans la salle à manger de l'hôtel, entre les invitations acceptées au dehors et les visites rendues que le temps strictement nécessaire. C'était leur morgue qui les préservait de toute sympathie humaine, de tout intérêt pour les inconnus assis autour d'eux, et au milieu desquels M. de Stermaria gardait l'air glacial, pressé, distant, rude, pointilleux et malintentionné, qu'on a dans un buffet de chemin de fer au milieu de voyageurs qu'on n'a jamais vus, qu'on ne reverra pas, et avec qui on ne conçoit d'autres rapports que de défendre contre eux son poulet froid et son coin dans le wagon. À peine commencions-nous à déjeuner qu'on vint nous faire lever sur l'ordre de M. de Stermaria, lequel venait d'arriver et, sans le moindre geste d'excuse à notre adresse, pria à haute voix le maître d'hôtel de veiller à ce qu'une pareille erreur ne se renouvelât pas, car il lui était désagréable que « des gens qu'il ne connaissait pas » eussent pris sa table.
