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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« Les femmes élégantes n'allaient pas chez elle »; Lady Israels fit en sorte que « personne qu'elle connaissait ne reçût Odette »; chez Mme de M. de Marsantes, « elle n'adressa pas une fois la parole à Odette »; l'ignorance d’Odette était telle qu’elle dit des Guermantes: « de l’Aisne ».",
      "explanation": "The narrator insists on Odette's failure to penetrate the suburb and reports an episode of explicit snobbery, while highlighting her ignorance of titles and worldly hierarchies."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.85,
      "explanation": "Locally, his standing is weakened by the exclusion of \"elegant women\" and a public episode of snobbery, despite some successes in the \"official world\"."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-161-p-165"
}

### Candidate characters

[
  "Dreyfus",
  "M. Verdurin",
  "M. de Marsantes",
  "Mme Cottard",
  "Swann",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le père du narrateur",
  "princesse de Guermantes",
  "le narrateur"
]

### Prior local context (optional)

Et toutes les personnes nouvelles que je lui disais avoir vues dans ce milieu un peu composite et artificiel où elles avaient souvent été amenées assez difficilement et de mondes assez différents, elle en devinait tout de suite l'origine et parlait d'elles comme elle aurait fait de trophées chèrement achetés ; elle disait :

### Passage

– Rapporté d'une Expédition chez les un Tel.

Pour Mme Cottard, mon père s'étonnait que Odette pût trouver quelque avantage à attirer cette bourgeoise peu élégante et disait : « Malgré la situation du professeur, j'avoue que je ne comprends pas. » Ma mère, elle, au contraire, comprenait très bien ; elle savait qu'une grande partie des plaisirs qu'une femme trouve à pénétrer dans un milieu différent de celui où elle vivait autrefois lui manquerait si elle ne pouvait informer ses anciennes relations de celles, relativement plus brillantes, par lesquelles elle les a remplacées. Pour cela il faut un témoin qu'on laisse pénétrer dans ce monde nouveau et délicieux, comme dans une fleur un insecte bourdonnant et volage, qui ensuite, au hasard de ses visites, répandra, on l'espère du moins, la nouvelle, le germe dérobé d'envie et d'admiration. Mme Cottard toute trouvée pour remplir ce rôle rentrait dans cette catégorie spéciale d'invités que maman, qui avait certains côtés de la tournure d'esprit de son père, appelait des : « Étranger, va dire à Sparte ! » D'ailleurs – en dehors d'une autre raison qu'on ne sut que bien des années après – Odette en conviant cette amie bienveillante, réservée et modeste, n'avait pas craint d'introduire chez soi, à ses « jours » brillants, un traître ou une concurrente. Elle savait le nombre énorme de calices bourgeois que pouvait, quand elle était armée de l'aigrette et du porte-cartes, visiter en un seul après-midi cette active ouvrière. Elle en connaissait le pouvoir de dissémination et, en se basant sur le calcul des probabilités, était fondée à penser que, très vraisemblablement, tel habitué des Verdurin apprendrait dès le surlendemain que le gouverneur de Paris avait mis des cartes chez elle, ou que M. Verdurin lui-même entendrait raconter que M. Le Hault de Pressagny, président du Concours hippique, les avait emmenés, elle et Swann, au gala du roi Théodose ; elle ne supposait les Verdurin informés que de ces deux événements flatteurs pour elle, parce que les matérialisations particulières sous lesquelles nous nous représentons et nous poursuivons la gloire sont peu nombreuses par le défaut de notre esprit, qui n'est pas capable d'imaginer à la fois toutes les formes que nous espérons bien d'ailleurs – en gros – que, simultanément, elle ne manquera pas de revêtir pour nous.

D'ailleurs, Odette n'avait obtenu de résultats que dans ce qu'on appelait le « monde officiel ». Les femmes élégantes n'allaient pas chez elle. Ce n'était pas la présence de notabilités républicaines qui les avaient fait fuir. Au temps de ma petite enfance, tout ce qui appartenait à la société conservatrice était mondain, et dans un salon bien posé on n'eût pas pu recevoir un républicain. Les personnes qui vivaient dans un tel milieu s'imaginaient que l'impossibilité de jamais inviter un « opportuniste », à plus forte raison un affreux « radical », était une chose qui durerait toujours, comme les lampes à huile et les omnibus à chevaux. Mais pareille aux kaléidoscopes qui tournent de temps en temps, la société place successivement de façon différente des éléments qu'on avait cru immuables et compose une autre figure. Je n'avais pas encore fait ma première communion, que des dames bien pensantes avaient la stupéfaction de rencontrer en visite une Juive élégante.

Ces dispositions nouvelles du kaléidoscope sont produites par ce qu'un philosophe appellerait un changement de critère. L'affaire Dreyfus en amena un nouveau, à une époque un peu postérieure à celle où je commençais à aller chez Odette, et le kaléidoscope renversa une fois de plus ses petits losanges colorés. Tout ce qui était juif passa en bas, fût-ce la dame élégante, et des nationalistes obscurs montèrent prendre sa place. Le salon le plus brillant de Paris fut celui d'un prince autrichien et ultra-catholique. Qu'au lieu de l'affaire Dreyfus il fût survenu une guerre avec l'Allemagne, le tour du kaléidoscope se fût produit dans un autre sens. Les Juifs ayant, à l'étonnement général, montré qu'ils étaient patriotes, auraient gardé leur situation, et personne n'aurait plus voulu aller ni même avouer être jamais allé chez le prince autrichien. Cela n'empêche pas que chaque fois que la société est momentanément immobile, ceux qui y vivent s'imaginent qu'aucun changement n'aura plus lieu, de même qu'ayant vu commencer le téléphone, ils ne veulent pas croire à l'aéroplane. Cependant, les philosophes du journalisme flétrissent la période précédente, non seulement le genre de plaisirs que l'on y prenait et qui leur semble le dernier mot de la corruption, mais même les oeuvres des artistes et des philosophes qui n'ont plus à leurs yeux aucune valeur, comme si elles étaient reliées indissolublement aux modalités successives de la frivolité mondaine. La seule chose qui ne change pas est qu'il semble chaque fois qu'il y ait « quelque chose de changé en France ». Au moment où j'allai chez Odette, l'affaire Dreyfus n'avait pas encore éclaté, et certains grands Juifs étaient fort puissants. Aucun ne l'était plus que sir Rufus Israels dont la femme, lady Israels, était tante de Swann. Elle n'avait pas personnellement des intimités aussi élégantes que son neveu qui, d'autre part, ne l'aimant pas, ne l'avait jamais beaucoup cultivée, quoiqu'il dût vraisemblablement être son héritier. Mais c'était la seule des parentes de Swann qui eût conscience de la situation mondaine de celui-ci, les autres étant toujours restées à cet égard dans la même ignorance qui avait été longtemps la nôtre. Quand, dans une famille, un des membres émigre dans la haute société – ce qui lui semble à lui un phénomène unique, mais ce qu'à dix ans de distance il constate avoir été accompli d'une autre façon et pour des raisons différentes par plus d'un jeune homme avec qui il avait été élevé – il décrit autour de lui une zone d'ombre, une terra incognita, fort visible en ses moindres nuances pour tous ceux qui l'habitent, mais qui n'est que nuit et pur néant pour ceux qui n'y pénètrent pas et la côtoient sans en soupçonner, tout près d'eux, l'existence. Aucune Agence Havas n'ayant renseigné les cousines de Swann sur les gens qu'il fréquentait, c'est (avant son horrible mariage, bien entendu) avec des sourires de condescendance qu'on se racontait dans les dîners de famille qu'on avait « vertueusement » employé son dimanche à aller voir le « cousin Swann » que, le croyant un peu envieux et parent pauvre, on appelait spirituellement, en jouant sur le titre du roman de Balzac : « Le Cousin Bête ». Lady Rufus Israels, elle, savait à merveille qui étaient ces gens qui prodiguaient à Swann une amitié dont elle était jalouse. La famille de son mari, qui était à peu près l'équivalent des Rothschild, faisait depuis plusieurs générations les affaires des princes d'Orléans. Lady Israels, excessivement riche, disposait d'une grande influence et elle l'avait employée à ce qu'aucune personne qu'elle connaissait ne reçût Odette. Une seule avait désobéi, en cachette. C'était la comtesse de Marsantes. Or, le malheur avait voulu qu'Odette étant allé faire visite à Mme De Marsantes, lady Israels était entrée presque en même temps. Mme de Marsantes était sur des épines. Avec la lâcheté des gens qui pourtant pourraient tout se permettre, elle n'adressa pas une fois la parole à Odette qui ne fut pas encouragée à pousser désormais plus loin une incursion dans un monde qui du reste n'était nullement celui où elle eût aimé être reçue. Dans ce complet désintéressement du faubourg Saint-Germain, Odette continuait à être la cocotte illettrée bien différente des bourgeois ferrés sur les moindres points de généalogie et qui trompent dans la lecture des anciens mémoires la soif des relations aristocratiques que la vie réelle ne leur fournit pas. Et Swann, d'autre part, continuait sans doute d'être l'amant à qui toutes ces particularités d'une ancienne maîtresse semblent agréables ou inoffensives, car souvent j'entendis sa femme proférer de vraies hérésies mondaines sans que (par un reste de tendresse, un manque d'estime, ou la paresse de la perfectionner) il cherchât à les corriger. C'était peut-être aussi là une forme de cette simplicité qui nous avait si longtemps trompés à Combray et qui faisait maintenant que, continuant à connaître, au moins pour son compte, des gens très brillants, il ne tenait pas à ce que dans la conversation on eût l'air dans le salon de sa femme de leur trouver quelque importance. Ils en avaient d'ailleurs moins que jamais pour Swann, le centre de gravité de sa vie s'étant déplacé. En tous cas l'ignorance d'Odette en matière mondaine était telle que, si le nom de la princesse de Guermantes venait dans la conversation après celui de la duchesse, sa cousine : « Tiens, ceux-là sont princes, ils ont donc monté en grade, disait Odette. » Si quelqu'un disait : « le prince » en parlant du duc de Chartres, elle rectifiait : « Le duc, il est duc de Chartres et non prince. » Pour le duc d'Orléans, fils du comte de Paris : « C'est drôle, le fils est plus que le père », tout en ajoutant, comme elle était anglomane : « On s'y embrouille dans ces « Royalties » ; et à une personne qui lui demandait de quelle province étaient les Guermantes, elle répondit : « de l'Aisne ».

Swann était du reste aveugle, en ce qui concernait Odette, non seulement devant ces lacunes de son éducation, mais aussi devant la médiocrité de son intelligence. Bien plus, chaque fois qu'Odette racontait une histoire bête, Swann écoutait sa femme avec une complaisance, une gaieté, presque une admiration où il devait entrer des restes de volupté ; tandis que, dans la même conversation, ce que lui-même pouvait dire de fin, même de profond, était écouté par Odette, habituellement sans intérêt, assez vite, avec impatience et quelquefois contredit avec sévérité. Et on conclura que cet asservissement de l'élite à la vulgarité est de règle dans bien des ménages, si l'on pense, inversement, à tant de femmes supérieures qui se laissent charmer par un butor, censeur impitoyable de leurs plus délicates paroles, tandis qu'elles s'extasient, avec l'indulgence infinie de la tendresse, devant ses facéties les plus plates. Pour revenir aux raisons qui empêchèrent à cette époque Odette de pénétrer dans le faubourg Saint-Germain, il faut dire que le plus récent tour du kaléidoscope mondain avait été provoqué par une série de scandales. Des femmes chez qui on allait en toute confiance avaient été reconnues être des filles publiques, des espionnes anglaises. On allait pendant quelque temps demander aux gens, on le croyait du moins, d'être avant tout, bien posés, bien assis... Odette représentait exactement tout ce avec quoi on venait de rompre et d'ailleurs immédiatement de renouer (car les hommes, ne changeant pas du jour au lendemain, cherchent dans un nouveau régime la continuation de l'ancien, mais en le cherchant sous une forme différente qui permît d'être dupe et de croire que ce n'était plus la société d'avant la crise). Or, aux dames « brûlées » de cette société Odette ressemblait trop. Les gens du monde sont fort myopes ; au moment où ils cessent toutes relations avec des dames israélites qu'ils connaissaient, pendant qu'ils se demandent comment remplacer ce vide, ils aperçoivent, poussée là comme à la faveur d'une nuit d'orage, une dame nouvelle, israélite aussi ; mais grâce à sa nouveauté, elle n'est pas associée dans leur esprit, comme les précédentes, avec ce qu'ils croient devoir détester. Elle ne demande pas qu'on respecte son Dieu. On l'adopte. Il ne s'agissait pas d'antisémitisme à l'époque où je commençai d'aller chez Odette. Mais elle était pareille à ce qu'on voulait fuir pour un temps.
