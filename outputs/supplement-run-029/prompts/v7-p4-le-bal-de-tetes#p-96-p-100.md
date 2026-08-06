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
      "canonical_name": "la Berma",
      "surface_forms": [
        "la Berma"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "la Berma",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "« ...ce qu'il fit sans se faire prier, ruinant d'un seul coup la situation sociale de la Berma comme il avait détruit sa santé. »",
      "explanation": "The humiliating, uninvited appeal by the Berma’s daughter and son-in-law, and Rachel’s public, condescending reception, are presented as socially compromising for la Berma; the narrator states the reputational damage as a fact."
    }
  ],
  "status_effects": [
    {
      "character": "la Berma",
      "dimension": "social_status",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "Her social standing is explicitly said to be ‘ruined’ by the children’s conduct and Rachel’s staged condescension."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-96-p-100"
}

### Candidate characters

[
  "Gilberte",
  "Mme Verdurin",
  "Odette",
  "comte de Forcheville",
  "duc de Guermantes",
  "duchesse de Guermantes",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

Je dis à la duchesse de Guermantes, en lui racontant que Bloch avait cru que c'était l'ancienne princesse de Guermantes qui recevait : « Cela me rappelle la première soirée où je suis allé chez la princesse de Guermantes, où je croyais ne pas être invité et qu'on allait me mettre à la porte, et où vous aviez une robe toute rouge et des souliers rouges. – Mon Dieu, que c'est vieux, tout cela », me répondit duchesse de Guermantes, accentuant pour moi l'impression du temps écoulé. Elle regardait dans le lointain avec mélancolie et pourtant insista particulièrement sur la robe rouge. Je lui demandai de me la décrire, ce qu'elle fit complaisamment. « Maintenant cela ne se porterait plus du tout. C'étaient des robes qui se portaient dans ce temps-là. – Mais est-ce que ce n'était pas joli ? » lui dis-je. Elle avait toujours peur de donner un avantage contre elle par ses paroles, de dire quelque chose qui la diminuât. « Mais si, moi je trouvais cela très joli. On n'en porte pas parce que cela ne se fait plus en ce moment. Mais cela se reportera, toutes les modes reviennent, en robes, en musique, en peinture », ajouta-t-elle avec force, car elle croyait une certaine originalité à cette philosophie. Cependant la tristesse de vieillir lui rendit sa lassitude qu'un sourire lui disputa : « Vous êtes sûr que c'étaient des souliers rouges ? Je croyais que c'étaient des souliers d'or. » J'assurai que cela m'était infiniment présent à l'esprit, sans dire la circonstance qui me permettait de l'affirmer. « Vous êtes gentil de vous rappeler cela », me dit-elle d'un air tendre, car les femmes appellent gentillesse se souvenir de leur beauté comme les artistes admirer leurs oeuvres. D'ailleurs, si lointain que soit le passé, quand on est une femme de tête comme duchesse de Guermantes, il peut ne pas être oublié. « Vous rappelez-vous, me dit-elle en remerciement de mon souvenir pour sa robe et ses souliers, que nous vous avons ramené, duc de Guermantes et moi ? Vous aviez une jeune fille qui devait venir vous voir après minuit. duc de Guermantes riait de tout son coeur en pensant qu'on vous faisait des visites à cette heure-là. » Je me rappelais, en effet, que ce soir-là Albertine était venue me voir après la soirée de la princesse de Guermantes, je me le rappelais aussi bien que duchesse de Guermantes, moi à qui Albertine était maintenant aussi indifférente qu'elle l'eût été à duchesse de Guermantes, si duchesse de Guermantes eût su que la jeune fille à cause de qui je n'avais pas pu entrer chez eux était Albertine. C'est que longtemps après que les pauvres morts sont sortis de nos coeurs, leur poussière indifférente continue à être mêlée, à servir d'alliage, aux circonstances du passé. Et, sans plus les aimer, il arrive qu'en évoquant une chambre, une allée, un chemin, où ils furent à une certaine heure, nous sommes obligés, pour que la place qu'ils occupaient soit remplie, de faire allusion à eux, même sans les regretter, même sans les nommer, même sans permettre qu'on les identifie. (duchesse de Guermantes n'identifiait guère la jeune fille qui devait venir ce soir-là, n'avait jamais su son nom et n'en parlait qu'à cause de la bizarrerie de l'heure et de la circonstance.) Telles sont les formes dernières et peu enviables de la survivance.

### Passage

Si les jugements que la duchesse porta ensuite sur Rachel furent en eux-mêmes médiocres, ils m'intéressèrent en ce que, eux aussi, marquaient une heure nouvelle sur le cadran. Car la duchesse n'avait pas plus complètement que Rachel perdu le souvenir de la soirée que celle-ci avait passée chez elle, mais ce souvenir n'y avait pas subi une moindre transformation. « Je vous dirai, me dit-elle, que cela m'intéresse d'autant plus de l'entendre, et de l'entendre acclamer, que je l'ai dénichée, appréciée, prônée, imposée à une époque où personne ne la connaissait et où tout le monde se moquait d'elle. Oui, mon petit, cela va vous étonner, mais la première maison où elle s'est fait entendre en public, c'est chez moi ! Oui, pendant que tous les gens prétendus d'avant-garde, comme ma nouvelle cousine, dit-elle en montrant ironiquement la princesse de Guermantes qui, pour Mme de Guermantes, restait Mme Verdurin, l'auraient laissée crever de faim sans daigner l'entendre, je l'avais trouvée intéressante et je lui avais fait offrir un cachet pour venir jouer chez moi devant tout ce que nous faisions de mieux comme gratin. Je peux dire, d'un mot un peu bête et prétentieux, car, au fond, le talent n'a besoin de personne, que je l'ai lancée. Bien entendu, elle n'avait pas besoin de moi. » J'esquissai un geste de protestation et je vis que Mme de Guermantes était toute prête à accueillir la thèse opposée : « Si ? Vous croyez que le talent a besoin d'un appui ? Au fond, vous avez peut-être raison. C'est curieux, vous dites justement ce que Dumas me disait autrefois. Dans ce cas je suis extrêmement flattée si je suis pour quelque chose, pour si peu que ce soit, non pas évidemment dans le talent, mais dans la renommée d'une telle artiste. » Mme de Guermantes préférait abandonner son idée que le talent perce tout seul comme un abcès, parce que c'était plus flatteur pour elle, mais aussi parce que depuis quelque temps, recevant des nouveaux venus, et étant du reste fatiguée, elle s'était faite assez humble, interrogeant les autres, leur demandant leur opinion pour s'en former une. « Je n'ai pas besoin de vous dire, reprit-elle, que cet intelligent public, qui s'appelle le monde, ne comprenait absolument rien à cela. On protestait, on riait. J'avais beau leur dire : « C'est curieux, c'est intéressant, c'est quelque chose qui n'a encore jamais été fait », on ne me croyait pas, comme on ne m'a jamais crue pour rien. C'est comme la chose qu'elle jouait, c'était une chose de Maeterlinck, maintenant c'est très connu, mais à ce moment-là tout le monde s'en moquait, eh bien, moi je trouvais ça admirable. Ça m'étonne même, quand j'y pense, qu'une paysanne comme moi, qui n'ai que l'éducation des filles de province, ait aimé du premier coup ces choses-là. Naturellement, je n'aurais pas pu dire pourquoi, mais ça me plaisait, ça me remuait ; tenez, duc de Guermantes qui n'a rien d'un sensible avait été frappé de l'effet que ça me produisait. Il m'avait dit : « Je ne veux plus que vous entendiez ces absurdités, ça vous rend malade. » Et c'était vrai parce qu'on me prend pour une femme sèche et que je suis, au fond, un paquet de nerfs. »

* * *

À ce moment se produisit un incident inattendu. Un valet de pied vint dire à Rachel que la fille de la Berma et son gendre demandaient à lui parler. On a vu que la fille de la Berma avait résisté au désir qu'avait son mari de faire demander une invitation à Rachel. Mais après le départ du jeune homme invité, l'ennui du jeune couple auprès de leur mère s'était accru, la pensée que d'autres s'amusaient les tourmentait, bref, profitant d'un moment où la Berma s'était retirée dans sa chambre, crachant un peu de sang, ils avaient quatre à quatre revêtu des vêtements plus élégants, fait appeler une voiture et étaient venus chez la princesse de Guermantes sans être invités. Rachel, se doutant de la chose et secrètement flattée, prit un ton arrogant et dit au valet de pied qu'elle ne pouvait pas se déranger, qu'ils écrivissent un mot pour dire l'objet de leur démarche insolite. Le valet de pied revint portant une carte où la fille de la Berma avait griffonné qu'elle et son mari n'avaient pu résister au désir d'entendre Rachel et lui demandaient de les laisser entrer. Rachel sourit de la niaiserie de leur prétexte et de son propre triomphe. Elle fit répondre qu'elle était désolée, mais qu'elle avait terminé ses récitations. Déjà, dans l'antichambre, où l'attente du couple s'était prolongée, les valets de pied commençaient à se gausser des deux solliciteurs éconduits. La honte d'une avanie, le souvenir du rien qu'était Rachel auprès de sa mère, poussèrent la fille de la Berma à poursuivre à fond une démarche que lui avait fait risquer d'abord le simple besoin du plaisir. Elle fit demander comme un service à Rachel, dût-elle ne pas avoir à l'entendre, la permission de lui serrer la main. Rachel était en train de causer avec un prince italien qu'on disait séduit par l'attrait de sa grande fortune, dont quelques relations mondaines dissimulaient un peu l'origine ; elle mesura le renversement des situations qui mettait maintenant les enfants de l'illustre Berma à ses pieds. Après avoir narré à tout le monde, d'une façon plaisante, cet incident, elle fit dire au jeune couple d'entrer, ce qu'il fit sans se faire prier, ruinant d'un seul coup la situation sociale de la Berma comme il avait détruit sa santé. Rachel l'avait compris, et que son amabilité condescendante donnerait la réputation, à elle de plus de bonté, au jeune couple de plus de bassesse que n'eût fait son refus. Aussi les reçut-elle à bras ouverts, avec affectation, disant d'un air de protectrice en vue et qui sait oublier sa grandeur : « Mais je crois bien ! c'est une joie. La princesse sera ravie. » Ne sachant pas qu'on croyait, au Théâtre, que c'était elle qui invitait, peut-être avait-elle craint qu'en refusant l'entrée aux enfants de la Berma ceux-ci doutassent, au lieu de sa bonne volonté, ce qui lui eût été bien égal, de son influence. La Mme de Guermantes s'éloigna instinctivement, car au fur et à mesure que quelqu'un avait l'air de rechercher le monde, il baissait dans l'estime de la duchesse. Elle n'en avait plus en ce moment que pour la bonté de Rachel et eût tourné le dos aux enfants de la Berma si on les lui avait présentés. Rachel, cependant, composait déjà dans sa tête la phrase gracieuse dont elle accablerait le lendemain la Berma dans les coulisses : « J'ai été navrée, désolée, que votre fille fasse antichambre. Si j'avais compris ! Elle m'envoyait bien cartes sur cartes. » Elle était ravie de porter ce coup à la Berma. Peut-être eût-elle reculé si elle eût su que ce serait un coup mortel. On aime à faire des victimes, mais sans se mettre précisément dans son tort, et en les laissant vivre. D'ailleurs, où était son tort ? Elle devait dire en riant, quelques jours plus tard : « C'est un peu fort, j'ai voulu être plus aimable pour ses enfants qu'elle n'a jamais été pour moi, et pour un peu on m'accuserait de l'avoir assassinée. Je prends la duchesse à témoin. » Il semble pour les grands artistes que tous les mauvais sentiments et tout le factice de la vie de théâtre passent en leurs enfants sans que chez eux le travail obstiné soit un dérivatif comme chez la mère ; les grandes tragédiennes meurent souvent victimes de complots domestiques noués autour d'elles, comme il leur arrivait tant de fois à la fin des pièces qu'elles jouaient.

Gilberte, nous l'avons vu, avait voulu éviter un conflit avec sa tante au sujet de Rachel. Elle avait bien fait : il n'était déjà pas facile de prendre devant Mme de Guermantes la défense de la fille d'Odette, tant son animosité était grande, et cela parce que la manière nouvelle dont la duchesse m'avait dit être trompée était la manière dont le duc la trompait, si extraordinaire que cela pût paraître à qui savait l'âge d'Odette, avec Mme de Forcheville.

Quand on pensait à l'âge que devait avoir maintenant Mme de Forcheville, cela semblait, en effet, extraordinaire. Mais peut-être Odette avait-elle commencé la vie de femme galante très jeune. Et puis il y a des femmes qu'à chaque décade on retrouve en une nouvelle incarnation, ayant de nouvelles amours, parfois alors qu'on les croyait mortes, faisant le désespoir d'une jeune femme que pour elles abandonne son mari.
