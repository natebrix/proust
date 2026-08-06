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
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "Gilberte",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.88,
      "evidence": "Aussi parlait-on dans les journaux avec les plus grands éloges de son admirable conduite et il était question de la décorer. ... elle n'avait plus cessé ... de mener, comme elle disait cette fois en toute vérité, la vie du front.",
      "explanation": "The newspapers publicly praise Gilberte’s wartime conduct and there is talk of decorating her; the narrator underscores the truth of her front-line involvement, endorsing the elevation."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.87,
      "explanation": "Public acclaim and prospective decoration significantly raise her standing in this local scene."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p2-m-de-charlus-pendant-la-guerre#p-26-p-30"
}

### Candidate characters

[
  "Françoise",
  "Mme Verdurin",
  "Morel",
  "Robert de Saint-Loup",
  "docteur Cottard",
  "le narrateur"
]

### Prior local context (optional)

De même que les héros d'un esprit médiocre et banal écrivant des poèmes pendant leur convalescence se plaçaient pour décrire la guerre non au niveau des événements, qui en eux-mêmes ne sont rien, mais de la banale esthétique, dont ils avaient suivi les règles jusque-là, parlant, comme ils eussent fait dix ans plus tôt, de la « sanglante aurore », du « vol frémissant de la victoire », etc., Robert de Saint-Loup, lui, beaucoup plus intelligent et artiste, restait intelligent et artiste, et notait avec goût pour moi des paysages pendant qu'il était immobilisé à la lisière d'une forêt marécageuse, mais comme si ç'avait été pour une chasse au canard. Pour me faire comprendre certaines oppositions d'ombre et de lumière qui avaient été « l'enchantement de sa matinée », il me citait certains tableaux que nous aimions l'un et l'autre et ne craignait pas de faire allusion à une page de Romain Rolland, voire de Nietzsche, avec cette indépendance des gens du front qui n'avaient pas la même peur de prononcer un nom allemand que ceux de l'arrière, et même avec cette pointe de coquetterie à citer un ennemi que mettait, par exemple, le colonel du Paty de Clam, dans la salle des témoins de l'affaire Zola, à réciter en passant devant Pierre Quillard, poète dreyfusard de la plus extrême violence et que, d'ailleurs, il ne connaissait pas, des vers de son drame symboliste : La Fille aux mains coupées. Robert de Saint-Loup me parlait-il d'une mélodie de Schumann, il n'en donnait le titre qu'en allemand et ne prenait aucune circonlocution pour me dire que quand, à l'aube, il avait entendu un premier gazouillement à la lisière d'une forêt, il avait été enivré comme si lui avait parlé l'oiseau de ce « sublime Siegfried » qu'il espérait bien entendre après la guerre.

### Passage

Et maintenant, à mon second retour à Paris, j'avais reçu dès le lendemain de mon arrivée, une nouvelle lettre de Gilberte, qui sans doute avait oublié celle, ou du moins le sens de celle que j'ai rapportée, car son départ de Paris à la fin de 1914 y était représenté rétrospectivement d'une manière assez différente. « Vous ne savez peut-être pas, mon cher ami, me disait-elle, que voilà bientôt deux ans que je suis à Tansonville. J'y suis arrivée en même temps que les Allemands. Tout le monde avait voulu m'empêcher de partir. On me traitait de folle. – Comment, me disait-on, vous êtes en sûreté à Paris et vous partez pour ces régions envahies, juste au moment où tout le monde cherche à s'en échapper. – Je ne méconnaissais pas tout ce que ce raisonnement avait de juste. Mais, que voulez-vous, je n'ai qu'une seule qualité, je ne suis pas lâche, ou, si vous aimez mieux, je suis fidèle, et quand j'ai su mon cher Tansonville menacé, je n'ai pas voulu que notre vieux régisseur restât seul à le défendre. Il m'a semblé que ma place était à ses côtés. Et c'est, du reste, grâce à cette résolution que j'ai pu sauver à peu près le château – quand tous les autres dans le voisinage, abandonnés par leurs propriétaires affolés, ont été presque tous détruits de fond en comble – et non seulement le château, mais les précieuses collections auxquelles mon cher Papa tenait tant. » En un mot, Gilberte était persuadée maintenant qu'elle n'était pas allée à Tansonville, comme elle me l'avait écrit en 1914, pour fuir les Allemands et pour être à l'abri, mais au contraire pour les rencontrer et défendre contre eux son château.

Ils n'étaient pas restés à Tansonville, d'ailleurs, mais elle n'avait plus cessé d'avoir chez elle un va-et-vient constant de militaires qui dépassait de beaucoup celui qui tirait les larmes à Françoise dans la rue de Combray, et de mener, comme elle disait cette fois en toute vérité, la vie du front. Aussi parlait-on dans les journaux avec les plus grands éloges de son admirable conduite et il était question de la décorer. La fin de sa lettre était entièrement exacte. « Vous n'avez pas idée de ce que c'est que cette guerre, mon cher ami, et de l'importance qu'y prend une route, un pont, une hauteur. Que de fois j'ai pensé à vous, aux promenades, grâce à vous rendues délicieuses, que nous faisions ensemble dans tout ce pays aujourd'hui ravagé, alors que d'immenses combats se livrent pour la possession de tel chemin, de tel coteau que vous aimiez, où nous sommes allés si souvent ensemble. Probablement vous comme moi, vous ne vous imaginiez pas que l'obscur Roussainville et l'assommant Méséglise, d'où on nous portait nos lettres, et où on était allé chercher le docteur quand vous avez été souffrant, seraient jamais des endroits célèbres. Eh bien, mon cher ami, ils sont à jamais entrés dans la gloire au même titre qu'Austerlitz ou Valmy. La bataille de Méséglise a duré plus de huit mois, les Allemands y ont perdu plus de cent mille hommes, ils ont détruit Méséglise, mais ils ne l'ont pas pris. Le petit chemin que vous aimiez tant, que nous appelions le raidillon aux aubépines et où vous prétendez que vous êtes tombé dans votre enfance amoureux de moi, alors que je vous assure en toute vérité que c'était moi qui étais amoureuse de vous, je ne peux pas vous dire l'importance qu'il a prise. L'immense champ de blé auquel il aboutit, c'est la fameuse cote 307 dont vous avez dû voir le nom revenir si souvent dans les communiqués. Les Français ont fait sauter le petit pont sur la Vivonne qui, disiez-vous, ne vous rappelait pas votre enfance autant que vous l'auriez voulu, les Allemands en ont jeté d'autres ; pendant un an et demi ils ont eu une moitié de Combray et les Français l'autre moitié. »

Le lendemain du jour où j'avais reçu cette lettre, c'est-à-dire l'avant-veille de celui où, cheminant dans l'obscurité, j'entendais sonner le bruit de mes pas, tout en remâchant tous ces souvenirs, Saint-Loup venu du front, sur le point d'y retourner, m'avait fait une visite de quelques secondes seulement, dont l'annonce seule m'avait violemment ému. Françoise avait d'abord voulu se précipiter sur lui, espérant qu'il pourrait faire réformer le timide garçon boucher, dont, dans un an, la classe allait partir. Mais elle fut arrêtée elle-même en pensant à l'inutilité de cette démarche, car depuis longtemps le timide tueur d'animaux avait changé de boucherie, et soit que la patronne de la nôtre craignît de perdre notre clientèle, soit qu'elle fût de bonne foi, elle avait déclaré à Françoise qu'elle ignorait où ce garçon, « qui, d'ailleurs, ne ferait jamais un bon boucher », était employé. Françoise avait bien cherché partout, mais Paris est grand, les boucheries nombreuses, et elle avait eu beau entrer dans un grand nombre, elle n'avait pu retrouver le jeune homme timide et sanglant.

Quand Saint-Loup était entré dans ma chambre, je l'avais approché avec ce sentiment de timidité, avec cette impression de surnaturel que donnaient au fond tous les permissionnaires et qu'on éprouve quand on est introduit auprès d'une personne atteinte d'un mal mortel et qui cependant se lève, s'habille, se promène encore. Il semblait (il avait surtout semblé au début, car pour qui n'avait pas vécu comme moi loin de Paris, l'habitude était venue qui retranche aux choses que nous avons vues plusieurs fois la racine d'impression profonde et de pensée qui leur donne leur sens réel), il semblait presque qu'il y eût quelque chose de cruel dans ces permissions données aux combattants. Aux premières, on se disait : « Ils ne voudront pas repartir, ils déserteront. » Et en effet, ils ne venaient pas seulement de lieux qui nous semblaient irréels parce que nous n'en avions entendu parler que par les journaux et que nous ne pouvions nous figurer qu'on eût pris part à ces combats titaniques et revenir seulement avec une contusion à l'épaule ; c'était des rivages de la mort, vers lesquels ils allaient retourner, qu'ils venaient un instant parmi nous, incompréhensibles pour nous, nous remplissant de tendresse, d'effroi, et d'un sentiment de mystère, comme ces morts que nous évoquons, qui nous apparaissent une seconde, que nous n'osons pas interroger et qui, du reste, pourraient tout au plus nous répondre : « Vous ne pourriez pas vous figurer. » Car il est extraordinaire à quel point chez les rescapés du front que sont les permissionnaires parmi les vivants, ou chez les morts qu'un médium hypnotise ou évoque, le seul effet d'un contact avec le mystère soit d'accroître s'il est possible l'insignifiance des propos. Tel j'abordai Saint-Loup qui avait encore au front une cicatrice plus auguste et plus mystérieuse pour moi que l'empreinte laissée sur la terre par le pied d'un géant. Et je n'avais pas osé lui poser de question et il ne m'avait dit que de simples paroles. Encore étaient-elles fort peu différentes de ce qu'elles eussent été avant la guerre, comme si les gens, malgré elle, continuaient à être ce qu'ils étaient ; le ton des entretiens était le même, la matière seule différait, et encore !

Je crus comprendre que Saint-Loup avait trouvé aux armées des ressources qui lui avaient fait peu à peu oublier que Morel s'était aussi mal conduit avec lui qu'avec son oncle. Pourtant il lui gardait une grande amitié et était pris de brusques désirs de le revoir, qu'il ajournait sans cesse. Je crus plus délicat envers Gilberte de ne pas indiquer à Saint-Loup que pour retrouver Morel il n'avait qu'à aller chez Mme Verdurin.
