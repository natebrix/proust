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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "Swann multiplies gifts and money to ‘occuper sa pensée’ et obtenir sa gratitude; il envisage qu’‘elle l’aime’ pour son influence ou sa fortune, se demande si cela n’est pas ‘l’entretenir’, puis, une ‘paresse d’esprit’ éteint l’idée et il décide d’envoyer ‘six ou sept mille francs’; il est ‘souffrant et triste’ et n’ose quitter Paris pendant qu’Odette y est.",
      "explanation": "The narrator presents Swann as dependent and buying Odette's recognition, half-aware that he 'supports' her, then repressing this lucidity to pay more. This staging highlights his emotional weakness and submission."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Locally, Swann appears to be suffering, anxious, and held by his dependence on Odette, going so far as to increase his expenses to maintain her favor."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-306-p-310"
}

### Candidate characters

[
  "M. Verdurin",
  "Odette",
  "comte de Forcheville",
  "princesse de Parme"
]

### Prior local context (optional)

Swann ignorait encore la disgrâce dont il était menacé chez les M. Verdurin et continuait à voir leurs ridicules en beau, au travers de son amour.

### Passage

Il n'avait de rendez-vous avec Odette, au moins le plus souvent, que le soir ; mais le jour, ayant peur de la fatiguer de lui en allant chez elle, il aurait aimé du moins ne pas cesser d'occuper sa pensée, et à tous moments il cherchait à trouver une occasion d'y intervenir, mais d'une façon agréable pour elle. Si, à la devanture d'un fleuriste ou d'un joaillier, la vue d'un arbuste ou d'un bijou le charmait, aussitôt il pensait à les envoyer à Odette, imaginant le plaisir qu'ils lui avaient procuré, ressenti par elle, venant accroître la tendresse qu'elle avait pour lui, et les faisait porter immédiatement rue La Pérouse, pour ne pas retarder l'instant où, comme elle recevrait quelque chose de lui, il se sentirait en quelque sorte près d'elle. Il voulait surtout qu'elle les reçût avant de sortir pour que la reconnaissance qu'elle éprouverait lui valût un accueil plus tendre quand elle le verrait chez les Verdurin, ou même, qui sait ? si le fournisseur faisait assez diligence, peut-être une lettre qu'elle lui enverrait avant le dîner, ou sa venue à elle en personne chez lui, en une visite supplémentaire, pour le remercier. Comme jadis quand il expérimentait sur la nature d'Odette les réactions du dépit, il cherchait par celles de la gratitude à tirer d'elle des parcelles intimes de sentiment qu'elle ne lui avait pas révélées encore.

Souvent elle avait des embarras d'argent et, pressée par une dette, le priait de lui venir en aide. Il en était heureux comme de tout ce qui pouvait donner à Odette une grande idée de l'amour qu'il avait pour elle, ou simplement une grande idée de son influence, de l'utilité dont il pouvait lui être. Sans doute si on lui avait dit au début : « c'est ta situation qui lui plaît », et maintenant : « c'est pour ta fortune qu'elle t'aime », il ne l'aurait pas cru, et n'aurait pas été d'ailleurs très mécontent qu'on se la figurât tenant à lui – qu'on les sentît unis l'un à l'autre – par quelque chose d'aussi fort que le snobisme ou l'argent. Mais, même s'il avait pensé que c'était vrai, peut-être n'eût-il pas souffert de découvrir à l'amour d'Odette pour lui cet état plus durable que l'agrément ou les qualités qu'elle pouvait lui trouver : l'intérêt, l'intérêt qui empêcherait de venir jamais le jour où elle aurait pu être tentée de cesser de le voir. Pour l'instant, en la comblant de présents, en lui rendant des services, il pouvait se reposer sur des avantages extérieurs à sa personne, à son intelligence, du soin épuisant de lui plaire par lui-même. Et cette volupté d'être amoureux, de ne vivre que d'amour, de la réalité de laquelle il doutait parfois, le prix dont en somme il la payait, en dilettante, de sensations immatérielles, lui en augmentait la valeur – comme on voit des gens incertains si le spectacle de la mer et le bruit de ses vagues sont délicieux, s'en convaincre ainsi que de la rare qualité de leurs goûts désintéressés, en louant cent francs par jour la chambre d'hôtel qui leur permet de les goûter.

Un jour que des réflexions de ce genre le ramenaient encore au souvenir du temps où on lui avait parlé d'Odette comme d'une femme entretenue, et où une fois de plus il s'amusait à opposer cette personnification étrange : la femme entretenue – chatoyant amalgame d'éléments inconnus et diaboliques, serti, comme une apparition de Gustave Moreau, de fleurs vénéneuses entrelacées à des joyaux précieux – et cette Odette sur le visage de qui il avait vu passer les mêmes sentiments de pitié pour un malheureux, de révolte contre une injustice, de gratitude pour un bienfait, qu'il avait vu éprouver autrefois par sa propre mère, par ses amis, cette Odette dont les propos avaient si souvent trait aux choses qu'il connaissait le mieux lui-même, à ses collections, à sa chambre, à son vieux domestique, au banquier chez qui il avait ses titres, il se trouva que cette dernière image du banquier lui rappela qu'il aurait à y prendre de l'argent. En effet, si ce mois-ci il venait moins largement à l'aide d'Odette dans ses difficultés matérielles qu'il n'avait fait le mois dernier où il lui avait donné cinq mille francs, et s'il ne lui offrait pas une rivière de diamants qu'elle désirait, il ne renouvellerait pas en elle cette admiration qu'elle avait pour sa générosité, cette reconnaissance, qui le rendaient si heureux, et même il risquerait de lui faire croire que son amour pour elle, comme elle en verrait les manifestations devenir moins grandes, avait diminué. Alors, tout d'un coup, il se demanda si cela, ce n'était pas précisément l'« entretenir » (comme si, en effet, cette notion d'entretenir pouvait être extraite d'éléments non pas mystérieux ni pervers, mais appartenant au fond quotidien et privé de sa vie, tels que ce billet de mille francs, domestique et familier, déchiré et recollé, que son valet de chambre, après lui avoir payé les comptes du mois et le terme, avait serré dans le tiroir du vieux bureau où Swann l'avait repris pour l'envoyer avec quatre autres à Odette) et si on ne pouvait pas appliquer à Odette, depuis qu'il la connaissait (car il ne soupçonna pas un instant qu'elle eût jamais pu recevoir d'argent de personne avant lui), ce mot qu'il avait cru si inconciliable avec elle, de « femme entretenue ». Il ne put approfondir cette idée, car un accès d'une paresse d'esprit, qui était chez lui congénitale, intermittente et providentielle, vint à ce moment éteindre toute lumière dans son intelligence, aussi brusquement que, plus tard, quand on eut installé partout l'éclairage électrique, on put couper l'électricité dans une maison. Sa pensée tâtonna un instant dans l'obscurité, il retira ses lunettes, en essuya les verres, se passa la main sur les yeux, et ne revit la lumière que quand il se retrouva en présence d'une idée toute différente, à savoir qu'il faudrait tâcher d'envoyer le mois prochain six ou sept mille francs à Odette au lieu de cinq, à cause de la surprise et de la joie que cela lui causerait.

Le soir, quand il ne restait pas chez lui à attendre l'heure de retrouver Odette chez les Verdurin ou plutôt dans un des restaurants d'été qu'ils affectionnaient au Bois et surtout à Saint-Cloud, il allait dîner dans quelqu'une de ces maisons élégantes dont il était jadis le convive habituel. Il ne voulait pas perdre contact avec des gens qui – savait-on ? – pourraient peut-être un jour être utiles à Odette, et grâce auxquels en attendant il réussissait souvent à lui être agréable. Puis l'habitude qu'il avait eue longtemps du monde, du luxe, lui en avait donné, en même temps que le dédain, le besoin, de sorte qu'à partir du moment où les réduits les plus modestes lui étaient apparus exactement sur le même pied que les plus princières demeures, ses sens étaient tellement accoutumés aux secondes qu'il eût éprouvé quelque malaise à se trouver dans les premiers. Il avait la même considération – à un degré d'identité qu'ils n'auraient pu croire – pour des petits bourgeois qui faisaient danser au cinquième étage d'un escalier D, palier à gauche, que pour la princesse de Parme qui donnait les plus belles fêtes de Paris ; mais il n'avait pas la sensation d'être au bal en se tenant avec les pères dans la chambre à coucher de la maîtresse de la maison, et la vue des lavabos recouverts de serviettes, des lits transformés en vestiaires, sur le couvre-pied desquels s'entassaient les pardessus et les chapeaux lui donnait la même sensation d'étouffement que peut causer aujourd'hui à des gens habitués à vingt ans d'électricité l'odeur d'une lampe qui charbonne ou d'une veilleuse qui file.

Le jour où il dînait en ville, il faisait atteler pour sept heures et demie ; il s'habillait tout en songeant à Odette et ainsi il ne se trouvait pas seul, car la pensée constante d'Odette donnait aux moments où il était loin d'elle le même charme particulier qu'à ceux où elle était là. Il montait en voiture, mais il sentait que cette pensée y avait sauté en même temps et s'installait sur ses genoux comme une bête aimée qu'on emmène partout et qu'il garderait avec lui à table, à l'insu des convives. Il la caressait, se réchauffait à elle, et éprouvant une sorte de langueur, se laissait aller à un léger frémissement qui crispait son cou et son nez, et était nouveau chez lui, tout en fixant à sa boutonnière le bouquet d'ancolies. Se sentant souffrant et triste depuis quelque temps, surtout depuis qu'Odette avait présenté Forcheville aux Verdurin, Swann aurait aimé aller se reposer un peu à la campagne. Mais il n'aurait pas eu le courage de quitter Paris un seul jour pendant qu'Odette y était. L'air était chaud ; c'étaient les plus beaux jours du printemps. Et il avait beau traverser une ville de pierre pour se rendre en quelque hôtel clos, ce qui était sans cesse devant ses yeux, c'était un parc qu'il possédait près de Combray, où, dès quatre heures, avant d'arriver au plant d'asperges, grâce au vent qui vient des champs de Méséglise, on pouvait goûter sous une charmille autant de fraîcheur qu'au bord de l'étang cerné de myosotis et de glaïeuls, et où, quand il dînait, enlacées par son jardinier, couraient autour de la table les groseilles et les roses.
