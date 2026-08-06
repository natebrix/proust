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
      "canonical_name": "Mme Verdurin",
      "surface_forms": [
        "Mme Verdurin"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Mme Verdurin",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« elle était, avec Mme Bontemps, une des reines de ce Paris de la guerre »; « pas une duchesse ne se serait couchée sans avoir appris de ... Mme Verdurin ... ce qu'il y avait dans le communiqué du soir »; « j’ai téléphoné au G.Q.G. »; « au fur et à mesure qu’augmenta le nombre des gens brillants ... le nombre des “ennuyeux” diminua ».",
      "explanation": "The narrator presents Mme Verdurin as a dominant wartime salon authority whose circle now attracts duchesses and confers informational prestige, marking a clear local rise in standing."
    }
  ],
  "status_effects": [
    {
      "character": "Mme Verdurin",
      "dimension": "social_status",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "She is framed as a reigning figure of wartime Paris with duchesses depending on her for privileged information, reducing her circle’s ‘ennuyeux’ and boosting her prestige."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p2-m-de-charlus-pendant-la-guerre#p-6-p-10"
}

### Candidate characters

[
  "Albertine",
  "Brichot",
  "Dreyfus",
  "Gilberte",
  "Mme Bontemps",
  "Morel",
  "Odette",
  "Robert de Saint-Loup",
  "baron de Charlus",
  "duchesse de Guermantes",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Chapitre II

### Passage

Charlus pendant la guerre ; ses opinions, ses plaisirs

Un des premiers soirs dès mon nouveau retour à Paris en 1916, ayant envie d'entendre parler de la seule chose qui m'intéressait alors, la guerre, je sortis, après le dîner, pour aller voir Mme Verdurin, car elle était, avec Mme Bontemps, une des reines de ce Paris de la guerre qui faisait penser au Directoire. Comme par l'ensemencement d'une petite quantité de levure, en apparence de génération spontanée, des jeunes femmes allaient tout le jour coiffées de hauts turbans cylindriques comme aurait pu l'être une contemporaine de Mme Tallien. Par civisme, ayant des tuniques égyptiennes droites, sombres, très « guerre », sur des jupes très courtes, elles chaussaient des lanières rappelant le cothurne selon Talma, ou de hautes guêtres rappelant celles de nos chers combattants ; c'est, disaient-elles, parce qu'elles n'oubliaient pas qu'elles devaient réjouir les yeux de ces combattants qu'elles se paraient encore, non seulement de toilettes « floues », mais encore de bijoux évoquant les armées par leur thème décoratif, si même leur matière ne venait pas des armées, n'avait pas été travaillée aux armées ; au lieu d'ornements égyptiens rappelant la campagne d'Égypte, c'étaient des bagues ou des bracelets faits avec des fragments d'obus ou des ceintures de 75, des allume-cigarettes composés de deux sous anglais, auxquels un militaire était arrivé à donner, dans sa cagna, une patine si belle que le profil de la reine Victoria y avait l'air tracé par Pisanello ; c'est encore parce qu'elles y pensaient sans cesse, disaient-elles, qu'elles portaient à peine le deuil quand l'un des leurs tombait, sous le prétexte qu'il était « mêlé de fierté », ce qui permettait un bonnet de crêpe anglais blanc (du plus gracieux effet et autorisant tous les espoirs), dans l'invincible certitude du triomphe définitif, et permettait ainsi de remplacer le cachemire d'autrefois par le satin et la mousseline de soie, et même de garder ses perles, « tout en observant le tact et la correction qu'il est inutile de rappeler à des Françaises ».

Le Louvre, tous les musées étaient fermés, et quand on lisait en tête d'un article de journal : « Une exposition sensationnelle », on pouvait être sûr qu'il s'agissait d'une exposition non de tableaux, mais de robes, de robes destinées, d'ailleurs, à éveiller « ces délicates joies d'art dont les Parisiennes étaient depuis trop longtemps sevrées ». C'est ainsi que l'élégance et le plaisir avaient repris ; l'élégance, à défaut des arts, cherchait à s'excuser comme ceux-ci en 1793, année où les artistes exposant au Salon révolutionnaire proclamaient que ce serait à tort qu'il paraîtrait « étrange à d'austères républicains que nous nous occupions des arts quand l'Europe coalisée assiège le territoire de la liberté ». Ainsi faisaient en 1916 les couturiers qui, d'ailleurs, avec une orgueilleuse conscience d'artistes, avouaient que « chercher du nouveau, s'écarter de la banalité, préparer la victoire, dégager pour les générations d'après la guerre une formule nouvelle du beau, telle était l'ambition qui les tourmentait, la chimère qu'ils poursuivaient, ainsi qu'on pouvait s'en rendre compte en venant visiter leurs salons délicieusement installés rue de la..., où effacer par une note lumineuse et gaie les lourdes tristesses de l'heure semble être le mot d'ordre, avec la discrétion toutefois qu'imposent les circonstances. Les tristesses de l'heure, il est vrai, pourraient avoir raison des énergies féminines si nous n'avions tant de hauts exemples de courage et d'endurance à méditer. Aussi en pensant à nos combattants qui au fond de leur tranchée rêvent de plus de confort et de coquetterie pour la chère absente laissée au foyer, ne cesserons-nous pas d'apporter toujours plus de recherche dans la création de robes répondant aux nécessités du moment. La vogue, cela se conçoit, est surtout aux maisons anglaises, donc alliées, et on raffole cette année de la robe-tonneau dont le joli abandon nous donne à toutes un amusant petit cachet de rare distinction. Ce sera même une des plus heureuses conséquences de cette triste guerre, ajoutait le charmant chroniqueur (en attendant la reprise des provinces perdues, le réveil du sentiment national), ce sera même une des plus heureuses conséquences de cette guerre que d'avoir obtenu de jolis résultats en fait de toilette, sans luxe inconsidéré et de mauvais aloi, avec très peu de chose, d'avoir créé de la coquetterie avec des riens. À la robe du grand couturier éditée à plusieurs exemplaires on préfère en ce moment les robes faites chez soi, parce qu'affirmant l'esprit, le goût et les tendances indiscutables de chacun. » Quant à la charité, en pensant à toutes les misères nées de l'invasion, à tant de mutilés, il était bien naturel qu'elle fût obligée de se faire « plus ingénieuse encore », ce qui obligeait les dames à hauts turbans à passer la fin de l'après-midi dans les thés autour d'une table de bridge, en commentant les nouvelles du « front », tandis qu'à la porte les attendaient leurs automobiles ayant sur le siège un beau militaire qui bavardait avec le chasseur. Ce n'était pas, du reste, seulement les coiffures surmontant les visages de leur étrange cylindre qui étaient nouvelles. Les visages l'étaient aussi. Les dames à nouveaux chapeaux étaient des jeunes femmes venues on ne savait trop d'où et qui étaient la fleur de l'élégance, les unes depuis six mois, les autres depuis deux ans, les autres depuis quatre. Ces différences avaient, d'ailleurs, pour elles autant d'importance qu'au temps où j'avais débuté dans le monde en avaient entre deux familles comme les Guermantes et les La Rochefoucauld trois ou quatre siècles d'ancienneté prouvée. La dame qui connaissait les Guermantes depuis 1914 regardait comme une parvenue celle qu'on présentait chez eux en 1916, lui faisait un bonjour de douairière, la dévisageait de son face-à-main et avouait dans une moue qu'on ne savait même pas au juste si cette dame était ou non mariée. « Tout cela est assez nauséabond », concluait la dame de 1914, qui eût voulu que le cycle des nouvelles admissions s'arrêtât après elle. Ces personnes nouvelles, que les jeunes gens trouvaient fort anciennes, et que d'ailleurs certains vieillards qui n'avaient pas été que dans le grand monde croyaient bien reconnaître pour ne pas être si nouvelles que cela, n'offraient pas seulement à la société les divertissements de conversation politique et de musique dans l'intimité qui lui convenaient ; il fallait encore que ce fussent elles qui les offrissent, car pour que les choses paraissent nouvelles, même si elles sont anciennes, et même si elles sont nouvelles, il faut en art, comme en médecine, comme en mondanité, des noms nouveaux (ils étaient d'ailleurs nouveaux en certaines choses). Ainsi Mme Verdurin était allée à Venise pendant la guerre, mais comme ces gens qui veulent éviter de parler chagrin et sentiment, quand elle disait que c'était épatant, ce qu'elle admirait ce n'était ni Venise, ni Saint-Marc, ni les palais, tout ce qui m'avait tant plu et dont elle faisait bon marché, mais l'effet des projecteurs dans le ciel, des projecteurs sur lesquels elle donnait des renseignements appuyés de chiffres. (Ainsi d'âge en âge renaît un certain réalisme en réaction contre l'art admiré jusque-là.)

Le salon Sainte-Euverte était une étiquette défraîchie, sous laquelle la présence des plus grands artistes, des ministres les plus influents, n'eût attiré personne. On courait, au contraire, pour écouter un mot prononcé par le secrétaire des uns ou le sous-chef de cabinet des autres, chez les nouvelles dames à turban, dont l'invasion ailée et jacassante emplissait Paris. Les dames du Premier Directoire avaient une reine qui était jeune et belle et s'appelait Madame Tallien. Celles du second en avaient deux qui étaient vieilles et laides et qui s'appelaient Mme Verdurin et Mme Bontemps. Qui eût pu tenir rigueur à Mme Bontemps que son mari eût joué un rôle, âprement critiqué par l'Écho de Paris, dans l'affaire Dreyfus ? Toute la Chambre étant à un certain moment devenue révisionniste, c'était forcément parmi d'anciens révisionnistes, comme parmi d'anciens socialistes, qu'on avait été obligé de recruter le parti de l'Ordre social, de la Tolérance religieuse, de la Préparation militaire. On aurait détesté autrefois M. Bontemps parce que les antipatriotes avaient alors le nom de dreyfusards. Mais bientôt ce nom avait été oublié et remplacé par celui d'adversaire de la loi de trois ans. M. Bontemps était, au contraire, un des auteurs de cette loi, c'était donc un patriote. Dans le monde (et ce phénomène social n'est, d'ailleurs, qu'une application d'une loi psychologique bien plus générale), les nouveautés coupables ou non n'excitent l'horreur que tant qu'elles ne sont pas assimilées et entourées d'éléments rassurants. Il en était du dreyfusisme comme du mariage de Saint-Loup avec la fille d'Odette, mariage qui avait d'abord fait crier. Maintenant qu'on voyait chez les Saint-Loup tous les gens « qu'on connaissait », Gilberte aurait pu avoir les moeurs d'Odette elle-même que, malgré cela, on y serait « allé » et qu'on eût approuvé Gilberte de blâmer comme une douairière des nouveautés morales non assimilées. Le dreyfusisme était maintenant intégré dans une série de choses respectables et habituelles. Quant à se demander ce qu'il valait en soi, personne n'y songeait, pas plus pour l'admettre maintenant qu'autrefois pour le condamner. Il n'était plus « shocking ». C'était tout ce qu'il fallait. À peine se rappelait-on qu'il l'avait été, comme on ne sait plus au bout de quelque temps si le père d'une jeune fille fut un voleur ou non. Au besoin, on peut dire : « Non, c'est du beau-frère, ou d'un homonyme que vous parlez, mais contre celui-là il n'y a jamais eu rien à dire. » De même il y avait certainement eu dreyfusisme et dreyfusisme, et celui qui allait chez la duchesse de Montmorency et faisait passer la loi de trois ans ne pouvait être mauvais. En tout cas, à tout péché miséricorde. Cet oubli qui était octroyé au dreyfusisme l'était a fortiori aux dreyfusards. Il n'y avait plus qu'eux, du reste, dans la politique, puisque tous à un moment l'avaient été s'il voulaient être du Gouvernement, même ceux qui représentaient le contraire de ce que le dreyfusisme, dans sa choquante nouveauté, avait incarné (au temps où Saint-Loup était sur une mauvaise pente) : l'antipatriotisme, l'irréligion, l'anarchie, etc. Ainsi le dreyfusisme de M. Bontemps, invisible et contemplatif comme celui de tous les hommes politiques, ne se voyait pas plus que les os sous la peau. Personne ne se fût rappelé qu'il avait été dreyfusard, car les gens du monde sont distraits et oublieux, parce qu'aussi il y avait de cela un temps fort long, et qu'ils affectaient de croire plus long, car c'était une des idées les plus à la mode de dire que l'avant-guerre était séparé de la guerre par quelque chose d'aussi profond, simulant autant de durée qu'une période géologique, et Brichot lui-même, ce nationaliste, quand il faisait allusion à l'affaire Dreyfus disait : « Dans ces temps préhistoriques ». À vrai dire, ce changement profond opéré par la guerre était en raison inverse de la valeur des esprits touchés, du moins à partir d'un certain degré, car, tout en bas, les purs sots, les purs gens de plaisir ne s'occupaient pas qu'il y eût la guerre. Mais tout en haut, ceux qui se sont fait une vie intérieure ambiante ont peu d'égard à l'importance des événements. Ce qui modifie profondément pour eux l'ordre des pensées, c'est bien plutôt quelque chose qui semble en soi n'avoir aucune importance et qui renverse pour eux l'ordre du temps en les faisant contemporains d'un autre temps de leur vie. Un chant d'oiseau dans le parc de Montboissier, ou une brise chargée de l'odeur de réséda, sont évidemment des événements de moindre conséquence que les plus grandes dates de la Révolution et de l'Empire. Ils ont cependant inspiré à Chateaubriand, dans les Mémoires d'Outre-tombe, des pages d'une valeur infiniment plus grande.

M. Bontemps ne voulait pas entendre parler de paix avant que l'Allemagne eût été réduite au même morcellement qu'au moyen âge, la déchéance de la maison de Hohenzollern prononcée, Guillaume ayant reçu douze balles dans la peau. En un mot, il était ce que Brichot appelait un « Jusquauboutiste », c'était le meilleur brevet de civisme qu'on pouvait lui donner. Sans doute, les trois premiers jours, Mme Bontemps avait été un peu dépaysée au milieu des personnes qui avaient demandé à Mme Verdurin à la connaître, et ce fut d'un ton légèrement aigre que Mme Verdurin répondit : « Le comte, ma chère », à Mme Bontemps qui lui disait : « C'est bien le duc d'Haussonville que vous venez de me présenter », soit par entière ignorance et absence de toute association entre le nom Haussonville et un titre quelconque, soit, au contraire, par excessive instruction et association d'idées avec le « Parti des Ducs », dont on lui avait dit que M. d'Haussonville était un des membres à l'Académie. À partir du quatrième jour elle avait commencé d'être solidement installée dans le faubourg Saint-Germain. Quelquefois encore on voyait autour d'elle les fragments inconnus d'un monde qu'on ne connaissait pas et qui n'étonnaient pas plus que des débris de coquille autour du poussin, ceux qui savaient l'oeuf d'où Mme Bontemps était sortie. Mais dès le quinzième jour, elle les avait secoués, et avant la fin du premier mois, quand elle disait : « Je vais chez les Lévi », tout le monde comprenait, sans qu'elle eût besoin de préciser, qu'il s'agissait des Lévis-Mirepoix, et pas une duchesse ne se serait couchée sans avoir appris de Mme Bontemps ou de Mme Verdurin, au moins par téléphone, ce qu'il y avait dans le communiqué du soir, ce qu'on y avait omis, où on en était avec la Grèce, quelle offensive on préparait, en un mot tout ce que le public ne saurait que le lendemain ou plus tard, et dont on avait ainsi comme une sorte de répétition des couturières. Dans la conversation, Mme Verdurin, pour communiquer les nouvelles, disait : « nous » en parlant de la France. « Hé bien, voici : nous exigeons du roi de Grèce qu'il se retire du Péloponèse, etc. ; nous lui envoyons, etc. » Et dans tous ses récits revenait tout le temps le G.Q.G. (j'ai téléphoné au G.Q.G.), abréviation qu'elle avait à prononcer le même plaisir qu'avaient naguère les femmes qui ne connaissaient pas le prince d'Agrigente à demander en souriant, quand on parlait de lui et pour montrer qu'elles étaient au courant : « Grigri ? », un plaisir qui dans les époques peu troublées n'est connu que par les mondains, mais que dans ces grandes crises le peuple même connaît. Notre maître d'hôtel, par exemple, si on parlait du roi de Grèce, était capable, grâce aux journaux, de dire comme Guillaume II : « Tino », tandis que jusque-là sa familiarité avec les rois était restée plus vulgaire, ayant été inventée par lui, comme quand jadis, pour parler du Roi d'Espagne, il disait : « Fonfonse ». On peut remarquer, d'ailleurs, qu'au fur et à mesure qu'augmenta le nombre des gens brillants qui firent des avances à Mme Verdurin, le nombre de ceux qu'elle appelait les « ennuyeux » diminua. Par une sorte de transformation magique, tout ennuyeux qui était venu lui faire une visite et avait sollicité une invitation devenait subitement quelqu'un d'agréable, d'intelligent. Bref, au bout d'un an le nombre des ennuyeux était réduit dans une proportion tellement forte, que la « peur et l'impossibilité de s'ennuyer », qui avait tenu une si grande place dans la conversation et joué un si grand rôle dans la vie de Mme Verdurin, avait presque entièrement disparu. On eût dit que sur le tard cette impossibilité de s'ennuyer (qu'autrefois, d'ailleurs, elle assurait ne pas avoir éprouvée dans sa prime jeunesse) la faisait moins souffrir, comme certaines migraines, certains asthmes nerveux qui perdent de leur force quand on vieillit. Et l'effroi de s'ennuyer eût sans doute entièrement abandonné Mme Verdurin, faute d'ennuyeux, si elle n'avait, dans une faible mesure, remplacé ceux qui ne l'étaient plus par d'autres recrutés parmi les anciens fidèles. Du reste, pour en finir avec les duchesses qui fréquentaient maintenant chez Mme Verdurin, elles venaient y chercher, sans qu'elles s'en doutassent, exactement la même chose que les dreyfusards autrefois, c'est-à-dire un plaisir mondain composé de telle manière que sa dégustation assouvît les curiosités politiques et rassasiât le besoin de commenter entre soi les incidents lus dans les journaux. Mme Verdurin disait : « Vous viendrez à 5 heures parler de la guerre », comme autrefois « parler de l'affaire », et dans l'intervalle : « Vous viendrez entendre Morel ». Or Morel n'aurait pas dû être là, pour la raison qu'il n'était nullement réformé. Simplement il n'avait pas rejoint et était déserteur, mais personne ne le savait. Une autre étoile du salon était « dans les choux », qui malgré ses goûts sportifs s'était fait réformer. Il était devenu tellement pour moi l'auteur d'une oeuvre admirable à laquelle je pensais constamment que ce n'est que par hasard, quand j'établissais un courant transversal entre deux séries de souvenirs, que je songeais qu'il était celui qui avait amené le départ d'Albertine de chez moi. Et encore ce courant transversal aboutissait, en ce qui concernait ces reliques de souvenirs d'Albertine, à une voie s'arrêtant en pleine friche à plusieurs années de distance. Car je ne pensais plus jamais à elle. C'était une voie non fréquentée de souvenirs, une ligne que je n'empruntais plus. Tandis que les oeuvres de « dans les choux » étaient récentes et cette ligne de souvenirs perpétuellement fréquentée et utilisée par mon esprit.
