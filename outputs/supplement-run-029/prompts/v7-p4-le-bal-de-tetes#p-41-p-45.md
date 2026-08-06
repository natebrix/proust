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
      "canonical_name": "Morel",
      "surface_forms": [
        "Morel"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Morel",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.72,
      "evidence": "J’étais peut‑être seul à savoir qu’il avait été entretenu par baron de Charlus, puis par Robert de Saint‑Loup et en même temps par un ami de Robert de Saint‑Loup.",
      "explanation": "The narrator undercuts the public aura of moral purity by recalling Morel’s past as a kept man tied to Charlus and Saint‑Loup, a socially compromising association in this milieu."
    },
    {
      "event_id": "E2",
      "source": "collective_social_voice",
      "target": "Morel",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.93,
      "evidence": "Un homme considérable… sa haute moralité devant laquelle les juges et les avocats s’étaient unanimement inclinés… Aussi y eut‑il un mouvement de curiosité et de déférence quand il entra. C’était Morel.",
      "explanation": "The assembled world reacts with curiosity and deference to Morel because of his reputed moral authority in a famous trial."
    }
  ],
  "status_effects": [
    {
      "character": "Morel",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.93,
      "explanation": "Within the scene, collective deference elevates Morel’s standing as a figure of moral authority."
    },
    {
      "character": "Morel",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "The narrator’s reminder of compromising patronage introduces a countervailing diminishment of his moral image."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-41-p-45"
}

### Candidate characters

[
  "Dreyfus",
  "Jupien",
  "Odette",
  "Robert de Saint-Loup",
  "baron de Charlus",
  "duchesse de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

Bloch m'interrogeait comme moi je faisais autrefois en entrant dans le monde, comme il m'arrivait encore de faire sur les gens que j'y avais connus alors et qui étaient aussi loin, aussi à part de tout, que ces gens de Combray qu'il m'était souvent arrivé de vouloir « situer » exactement. Mais Combray avait pour moi une forme si à part, si impossible à confondre avec le reste, que c'était un puzzle que je ne pouvais jamais arriver à faire rentrer dans la carte de France. « Alors je ne peux avoir aucune idée de ce qu'était jadis le prince de Guermantes en me représentant Swann, ou baron de Charlus ? me demandait Bloch à qui j'avais longtemps emprunté sa manière de parler et qui maintenant imitait souvent la mienne. – Nullement. – Mais en quoi consiste la différence ? – Il aurait fallu les entendre parler entre eux, pour la saisir, mais c'est maintenant impossible, Swann est mort et baron de Charlus ne vaut guère mieux. Mais ces différences étaient énormes. » Et tandis que l'oeil de Bloch brillait en pensant à ce que pouvait être la conversation de ces personnages merveilleux, je pensais que je lui exagérais le plaisir que j'avais eu à me trouver avec eux, n'en ayant jamais ressenti que quand j'étais seul, et l'impression des différenciations véritables n'ayant lieu que dans notre imagination. Bloch s'en aperçut-il ? « Tu me peins peut-être cela trop en beau, me dit-il ; ainsi la maîtresse de maison d'ici, la princesse de Guermantes, je sais bien qu'elle n'est plus jeune, mais enfin il n'y a pas tellement longtemps que tu me parlais de son charme incomparable, de sa merveilleuse beauté. Certes, je reconnais qu'elle a grand air, et elle a bien ces yeux extraordinaires dont tu me parlais, mais enfin je ne la trouve pas tellement inouïe que tu disais. Évidemment elle est très racée, mais enfin... » Je fus obligé de dire à Bloch qu'il ne me parlait pas de la même personne. La princesse de Guermantes, en effet, était morte et c'est l'ex-Mme Verdurin que le prince, ruiné par la défaite allemande, avait épousée et que Bloch ne reconnaissait pas. « Tu te trompes, j'ai cherché dans le Gotha de cette année, me confessa naïvement Bloch, et j'ai trouvé le prince de Guermantes, habitant l'hôtel où nous sommes et marié à tout ce qu'il y a de plus grandiose, attends un peu que je me rappelle, marié à Sidonie, duchesse de Duras, née des Baux. » En effet, Mme Verdurin, peu après la mort de son mari, avait épousé le vieux duc de Duras, ruiné, qui l'avait faite cousine du prince de Guermantes, et était mort après deux ans de mariage. Il avait été pour Mme Verdurin une transition fort utile, et maintenant celle-ci, par un troisième mariage, était princesse de Guermantes et avait dans le faubourg Saint-Germain une grande situation qui eût fort étonné à Combray, où les dames de la rue de l'Oiseau, la fille de Mme Goupil et la belle-fille de Mme Sazerat, toutes ces dernières années, avant que Mme Verdurin ne fût princesse de Guermantes, avaient dit en ricanant : « duchesse de Guermantes de Duras », comme si c'eût été un rôle que Mme Verdurin eût tenu au théâtre. Même, le principe des castes voulant qu'elle mourût Mme Verdurin, ce titre, qu'on ne s'imaginait lui conférer aucun pouvoir mondain nouveau, faisait plutôt mauvais effet. « Faire parler d'elle », cette expression qui dans tous les mondes est appliquée à une femme qui a un amant, pouvait l'être dans le faubourg Saint-Germain à celles qui publient des livres, dans la bourgeoisie de Combray à celles qui font des mariages dans un sens ou dans l'autre « disproportionnés ». Quand elle eut épousé le prince de Guermantes, on dut se dire que c'était un faux Guermantes, un escroc. Pour moi, à me figurer cette identité de titre, de nom, qui faisait qu'il y avait encore une princesse de Guermantes et qu'elle n'avait aucun rapport avec celle qui m'avait tant charmé et qui n'était plus, qui était comme une morte sans défense à qui on l'eût volé, il y avait quelque chose d'aussi douloureux qu'à voir les objets qu'avait possédés la princesse Hedwige, comme son château, comme tout ce qui avait été à elle et dont une autre jouissait. La succession au nom est triste comme toutes les successions, comme toutes les usurpations de propriété ; et toujours sans interruptions viendraient, comme un flot, de nouvelles princesses de Guermantes, ou plutôt, millénaire, remplacée d'âge en âge dans son emploi par une femme différente, vivrait une seule princesse de Guermantes, ignorante de la mort, indifférente à tout ce qui change et blesse nos coeurs, et le nom comme la mer refermerait sur celles qui sombrent de temps à autre sa toujours pareille et immémoriale placidité.

### Passage

Mais – contradiction avec cette permanence – les anciens habitués assuraient que dans le monde tout était changé, qu'on y recevait des gens que jamais de leur temps on n'aurait reçus et, comme on dit : « c'était vrai, et ce n'était pas vrai ». Ce n'était pas vrai parce qu'ils ne se rendaient pas compte de la courbe du temps qui faisait que ceux d'aujourd'hui voyaient ces gens nouveaux à leur point d'arrivée tandis qu'eux se les rappelaient à leur point de départ. Et quand eux, les anciens, étaient entrés dans le monde, il y avait là des gens arrivés dont d'autres se rappelaient le départ. Une génération suffit pour que s'y ramène ce changement qui en des siècles s'est fait pour le nom bourgeois d'un Colbert devenu nom noble. Et, d'autre part, cela pourrait être vrai, car si les personnes changent de situation, les idées et les coutumes les plus indéracinables (de même que les fortunes et les alliances de pays et les haines de pays) changent aussi, parmi lesquelles même celles de ne recevoir que des gens chic. Non seulement le snobisme change de forme, mais il pourrait disparaître, comme la guerre même, et les radicaux, les juifs être reçus au Jockey.

Certes, même ce changement extérieur dans les figures que j'avais connues n'était que le symbole d'un changement intérieur qui s'était effectué jour par jour. Peut-être ces gens avaient-ils continué à accomplir les mêmes choses, mais, jour par jour, l'idée qu'ils se faisaient d'elles et des êtres qu'ils fréquentaient, ayant un peu de vie, au bout de quelques années, sous les mêmes noms c'était d'autres choses, d'autres gens qu'ils aimaient, et étant devenus d'autres personnes, il eût été étonnant qu'ils n'eussent pas eu de nouveaux visages.

Si, dans ces périodes de vingt ans, les conglomérats de coteries se défaisaient et se reformaient selon l'attraction d'astres nouveaux destinés, d'ailleurs, eux aussi, à s'éloigner puis à reparaître, des cristallisations, puis des émiettements suivis de cristallisations nouvelles avaient lieu dans l'âme des êtres. Si pour moi la Mme de Guermantes avait été bien des personnes, pour la Mme de Guermantes, pour Odette, etc., telle personne donnée avait été un favori d'une époque précédant l'Affaire Dreyfus, puis un fanatique ou un imbécile à partir de l'affaire Dreyfus, qui avait changé pour eux la valeur des êtres et reclassé autour les partis, lesquels s'étaient depuis encore défaits et refaits. Ce qui y sert puissamment et y ajoute son influence aux pures affinités intellectuelles, c'est le temps écoulé, qui nous fait oublier nos antipathies, nos dédains, les raisons mêmes qui expliquaient nos antipathies et nos dédains. Si on eût jadis analysé l'élégance de la jeune Mme Léonor de Cambremer, on y eût trouvé qu'elle était la nièce du marchand de notre maison, Jupien, et que ce qui avait pu s'ajouter à cela pour la rendre brillante, c'était que son oncle procurait des hommes à Charlus. Mais tout cela combiné avait produit des effets scintillants, alors que les causes déjà lointaines, non seulement étaient inconnues de beaucoup de nouveaux, mais encore que ceux qui les avaient connues les avaient oubliées, pensant beaucoup plus à l'éclat actuel qu'aux hontes passées, car on prend toujours un nom dans son acception actuelle. Et c'était l'intérêt de ces transformations des salons qu'elles étaient aussi un effet du temps perdu et un phénomène de mémoire.

Parmi les personnes présentes se trouvait un homme considérable qui venait, dans un procès fameux, de donner un témoignage dont la seule valeur résidait dans sa haute moralité devant laquelle les juges et les avocats s'étaient unanimement inclinés et qui avait entraîné la condamnation de deux personnes. Aussi y eut-il un mouvement de curiosité et de déférence quand il entra. C'était Morel. J'étais peut-être seul à savoir qu'il avait été entretenu par Charlus, puis par Saint-Loup et en même temps par un ami de Saint-Loup. Malgré ces souvenirs, il me dit bonjour avec plaisir quoique avec réserve. Il se rappelait le temps où nous nous étions vus à Balbec, et ces souvenirs avaient pour lui la poésie et la mélancolie de la jeunesse.

Mais il y avait aussi des personnes que je ne pouvais pas reconnaître pour la raison que je ne les avais pas connues, car, aussi bien que sur les êtres eux-mêmes, le temps avait aussi, dans ce salon, exercé sa chimie sur la société. Ce milieu, en la nature spécifique duquel, définie par certaines affinités qui lui attiraient tous les grands noms princiers de l'Europe et par la répulsion qui éloignait d'elle tout élément non aristocratique, j'avais trouvé un refuge matériel pour ce nom de Guermantes auquel il prêtait sa dernière réalité, ce milieu avait lui-même subi, dans sa constitution intime et que j'avais crue stable, une altération profonde. La présence de gens que j'avais vus dans de tout autres sociétés et qui me semblaient ne devoir jamais pénétrer dans celle-là m'étonna moins encore que l'intime familiarité avec laquelle ils y étaient reçus, appelés par leur prénom ; un certain ensemble de préjugés aristocratiques, de snobisme, qui jadis écartait automatiquement du nom de Guermantes tout ce qui ne s'harmonisait pas avec lui, avait cessé de fonctionner.
