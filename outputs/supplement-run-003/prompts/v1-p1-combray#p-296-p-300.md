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
        "Gilberte",
        "une fillette d'un blond roux"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« la petite fille privilégiée qui avait Bergotte pour ami » … « je … fus … amoureux, en elle, de ses yeux bleus »",
      "explanation": "The narrator ascribes prestige and beauty to Gilberte and explicitly falls in love with her, conferring positive valuation on her."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "She is locally elevated by the narrator’s intense admiration and prestige-laden framing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-296-p-300"
}

### Candidate characters

[
  "Bergotte",
  "Swann",
  "le grand-père du narrateur",
  "le narrateur",
  "le peintre",
  "le père du narrateur"
]

### Prior local context (optional)

Devant nous, une allée bordée de capucines montait en plein soleil vers le château. À droite, au contraire, le parc s'étendait en terrain plat. Obscurcie par l'ombre des grands arbres qui l'entouraient, une pièce d'eau avait été creusée par les parents de Swann ; mais dans ses créations les plus factices, c'est sur la nature que l'homme travaille ; certains lieux font toujours régner autour d'eux leur empire particulier, arborent leurs insignes immémoriaux au milieu d'un parc comme ils auraient fait loin de toute intervention humaine, dans une solitude qui revient partout les entourer, surgie des nécessités de leur exposition et superposée à l'oeuvre humaine. C'est ainsi qu'au pied de l'allée qui dominait l'étang artificiel, s'était composée sur deux rangs, tressés de fleurs de myosotis et de pervenches, la couronne naturelle, délicate et bleue qui ceint le front clair-obscur des eaux, et que le glaïeul, laissant fléchir ses glaives avec un abandon royal, étendait sur l'eupatoire et la grenouillette au pied mouillé les fleurs de lis en lambeaux, violettes et jaunes, de son sceptre lacustre.

### Passage

Le départ de Gilberte qui – en m'ôtant la chance terrible de la voir apparaître dans une allée, d'être connu et méprisé par la petite fille privilégiée qui avait Bergotte pour ami et allait avec lui visiter des cathédrales – me rendait la contemplation de Tansonville indifférente la première fois où elle m'était permise, semblait au contraire ajouter à cette propriété, aux yeux de mon grand-père et de mon père, des commodités, un agrément passager, et, comme fait, pour une excursion en pays de montagnes, l'absence de tout nuage, rendre cette journée exceptionnellement propice à une promenade de ce côté ; j'aurais voulu que leurs calculs fussent déjoués, qu'un miracle fît apparaître Gilberte avec son père, si près de nous que nous n'aurions pas le temps de l'éviter et serions obligés de faire sa connaissance. Aussi, quand tout d'un coup, j'aperçus sur l'herbe, comme un signe de sa présence possible, un koufin oublié à côté d'une ligne dont le bouchon flottait sur l'eau, je m'empressai de détourner d'un autre côté les regards de mon père et de mon grand-père. D'ailleurs Swann nous ayant dit que c'était mal à lui de s'absenter, car il avait pour le moment de la famille à demeure, la ligne pouvait appartenir à quelque invité. On n'entendait aucun bruit de pas dans les allées. Divisant la hauteur d'un arbre incertain, un invisible oiseau s'ingéniait à faire trouver la journée courte, explorait d'une note prolongée la solitude environnante, mais il recevait d'elle une réplique si unanime, un choc en retour si redoublé de silence et d'immobilité qu'on aurait dit qu'il venait d'arrêter pour toujours l'instant qu'il avait cherché à faire passer plus vite. La lumière tombait si implacable du ciel devenu fixe que l'on aurait voulu se soustraire à son attention, et l'eau dormante elle-même, dont des insectes irritaient perpétuellement le sommeil, rêvant sans doute de quelque Maelstrôm imaginaire, augmentait le trouble où m'avait jeté la vue du flotteur de liège en semblant l'entraîner à toute vitesse sur les étendues silencieuses du ciel reflété ; presque vertical il paraissait prêt à plonger et déjà je me demandais, si, sans tenir compte du désir et de la crainte que j'avais de la connaître, je n'avais pas le devoir de faire prévenir Gilberte que le poisson mordait – quand il me fallut rejoindre en courant mon père et mon grand-père qui m'appelaient, étonnés que je ne les eusse pas suivis dans le petit chemin qui monte vers les champs et où ils s'étaient engagés. Je le trouvai tout bourdonnant de l'odeur des aubépines. La haie formait comme une suite de chapelles qui disparaissaient sous la jonchée de leurs fleurs amoncelées en reposoir ; au-dessous d'elles, le soleil posait à terre un quadrillage de clarté, comme s'il venait de traverser une verrière ; leur parfum s'étendait aussi onctueux, aussi délimité en sa forme que si j'eusse été devant l'autel de la Vierge, et les fleurs, aussi parées, tenaient chacune d'un air distrait son étincelant bouquet d'étamines, fines et rayonnantes nervures de style flamboyant comme celles qui à l'église ajouraient la rampe du jubé ou les meneaux du vitrail et qui s'épanouissaient en blanche chair de fleur de fraisier. Combien naïves et paysannes en comparaison sembleraient les églantines qui, dans quelques semaines, monteraient elles aussi en plein soleil le même chemin rustique, en la soie unie de leur corsage rougissant qu'un souffle défait.

Mais j'avais beau rester devant les aubépines à respirer, à porter devant ma pensée qui ne savait ce qu'elle devait en faire, à perdre, à retrouver leur invisible et fixe odeur, à m'unir au rythme qui jetait leurs fleurs, ici et là, avec une allégresse juvénile et à des intervalles inattendus comme certains intervalles musicaux, elles m'offraient indéfiniment le même charme avec une profusion inépuisable, mais sans me laisser approfondir davantage, comme ces mélodies qu'on rejoue cent fois de suite sans descendre plus avant dans leur secret. Je me détournais d'elles un moment, pour les aborder ensuite avec des forces plus fraîches. Je poursuivais jusque sur le talus qui, derrière la haie, montait en pente raide vers les champs, quelques coquelicots perdus, quelques bluets restés paresseusement en arrière, qui le décoraient çà et là de leurs fleurs comme la bordure d'une tapisserie où apparaît clairsemé le motif agreste qui triomphera sur le panneau ; rares encore, espacés comme les maisons isolées qui annoncent déjà l'approche d'un village, ils m'annonçaient l'immense étendue où déferlent les blés, où moutonnent les nuages, et la vue d'un seul coquelicot hissant au bout de son cordage et faisant cingler au vent sa flamme rouge, au-dessus de sa bouée graisseuse et noire, me faisait battre le coeur, comme au voyageur qui aperçoit sur une terre basse une première barque échouée que répare un calfat, et s'écrie, avant de l'avoir encore vue : « La Mer ! »

Puis je revenais devant les aubépines comme devant ces chefs-d'oeuvre dont on croit qu'on saura mieux les voir quand on a cessé un moment de les regarder, mais j'avais beau me faire un écran de mes mains pour n'avoir qu'elles sous les yeux, le sentiment qu'elles éveillaient en moi restait obscur et vague, cherchant en vain à se dégager, à venir adhérer à leurs fleurs. Elles ne m'aidaient pas à l'éclaircir, et je ne pouvais demander à d'autres fleurs de le satisfaire. Alors me donnant cette joie que nous éprouvons quand nous voyons de notre peintre préféré une oeuvre qui diffère de celles que nous connaissions, ou bien si l'on nous mène devant un tableau dont nous n'avions vu jusque-là qu'une esquisse au crayon, si un morceau entendu seulement au piano nous apparaît ensuite revêtu des couleurs de l'orchestre, mon grand-père m'appelant et me désignant la haie de Tansonville, me dit : « Toi qui aimes les aubépines, regarde un peu cette épine rose ; est-elle jolie ! » En effet c'était une épine, mais rose, plus belle encore que les blanches. Elle aussi avait une parure de fête, de ces seules vraies fêtes que sont les fêtes religieuses, puisqu'un caprice contingent ne les applique pas comme les fêtes mondaines à un jour quelconque qui ne leur est pas spécialement destiné, qui n'a rien d'essentiellement férié – mais une parure plus riche encore, car les fleurs attachées sur la branche, les unes au-dessus des autres, de manière à ne laisser aucune place qui ne fût décorée, comme des pompons qui enguirlandent une houlette rococo, étaient « en couleur », par conséquent d'une qualité supérieure selon l'esthétique de Combray, si l'on en jugeait par l'échelle des prix dans le « magasin » de la Place ou chez Camus où étaient plus chers ceux des biscuits qui étaient roses. Moi-même j'appréciais plus le fromage à la crème rose, celui où l'on m'avait permis d'écraser des fraises. Et justement ces fleurs avaient choisi une de ces teintes de chose mangeable, ou de tendre embellissement à une toilette pour une grande fête, qui, parce qu'elles leur présentent la raison de leur supériorité, sont celles qui semblent belles avec le plus d'évidence aux yeux des enfants, et à cause de cela, gardent toujours pour eux quelque chose de plus vif et de plus naturel que les autres teintes, même lorsqu'ils ont compris qu'elles ne promettaient rien à leur gourmandise et n'avaient pas été choisies par la couturière. Et certes, je l'avais tout de suite senti, comme devant les épines blanches mais avec plus d'émerveillement, que ce n'était pas facticement, par un artifice de fabrication humaine, qu'était traduite l'intention de festivité dans les fleurs, mais que c'était la nature qui, spontanément, l'avait exprimée avec la naïveté d'une commerçante de village travaillant pour un reposoir, en surchargeant l'arbuste de ces rosettes d'un ton trop tendre et d'un pompadour provincial. Au haut des branches, comme autant de ces petits rosiers aux pots cachés dans des papiers en dentelles, dont aux grandes fêtes on faisait rayonner sur l'autel les minces fusées, pullulaient mille petits boutons d'une teinte plus pâle qui, en s'entr'ouvrant, laissaient voir, comme au fond d'une coupe de marbre rose, de rouges sanguines, et trahissaient, plus encore que les fleurs, l'essence particulière, irrésistible, de l'épine, qui, partout où elle bourgeonnait, où elle allait fleurir, ne le pouvait qu'en rose. Intercalé dans la haie, mais aussi différent d'elle qu'une jeune fille en robe de fête au milieu de personnes en négligé qui resteront à la maison, tout prêt pour le mois de Marie, dont il semblait faire partie déjà, tel brillait en souriant dans sa fraîche toilette rose l'arbuste catholique et délicieux.

La haie laissait voir à l'intérieur du parc une allée bordée de jasmins, de pensées et de verveines entre lesquelles des giroflées ouvraient leurs bourses fraîches du rose odorant et passé d'un cuir ancien de Cordoue, tandis que sur le gravier un long tuyau d'arrosage peint en vert, déroulant ses circuits, dressait aux points où il était percé au-dessus des fleurs, dont il imbibait les parfums, l'éventail vertical et prismatique de ses gouttelettes multicolores. Tout à coup, je m'arrêtai, je ne pus plus bouger, comme il arrive quand une vision ne s'adresse pas seulement à nos regards, mais requiert des perceptions plus profondes et dispose de notre être tout entier. Une fillette d'un blond roux, qui avait l'air de rentrer de promenade et tenait à la main une bêche de jardinage, nous regardait, levant son visage semé de taches roses. Ses yeux noirs brillaient et, comme je ne savais pas alors, ni ne l'ai appris depuis, réduire en ses éléments objectifs une impression forte, comme je n'avais pas, ainsi qu'on dit, assez « d'esprit d'observation » pour dégager la notion de leur couleur, pendant longtemps, chaque fois que je repensai à elle, le souvenir de leur éclat se présentait aussitôt à moi comme celui d'un vif azur, puisqu'elle était blonde : de sorte que, peut-être si elle n'avait pas eu des yeux aussi noirs – ce qui frappait tant la première fois qu'on la voyait – je n'aurais pas été, comme je le fus, plus particulièrement amoureux, en elle, de ses yeux bleus.

Je la regardai, d'abord de ce regard qui n'est pas que le porte-parole des yeux, mais à la fenêtre duquel se penchent tous les sens, anxieux et pétrifiés, le regard qui voudrait toucher, capturer, emmener le corps qu'il regarde et l'âme avec lui ; puis, tant j'avais peur que d'une seconde à l'autre mon grand-père et mon père, apercevant cette jeune fille, me fissent éloigner en me disant de courir un peu devant eux, d'un second regard, inconsciemment supplicateur, qui tâchait de la forcer à faire attention à moi, à me connaître ! Elle jeta en avant et de côté ses pupilles pour prendre connaissance de mon grand'père et de mon père, et sans doute l'idée qu'elle en rapporta fut celle que nous étions ridicules, car elle se détourna, et d'un air indifférent et dédaigneux, se plaça de côté pour épargner à son visage d'être dans leur champ visuel ; et tandis que continuant à marcher et ne l'ayant pas aperçue, ils m'avaient dépassé, elle laissa ses regards filer de toute leur longueur dans ma direction, sans expression particulière, sans avoir l'air de me voir, mais avec une fixité et un sourire dissimulé, que je ne pouvais interpréter d'après les notions que l'on m'avait données sur la bonne éducation que comme une preuve d'outrageant mépris ; et sa main esquissait en même temps un geste indécent, auquel quand il était adressé en public à une personne qu'on ne connaissait pas, le petit dictionnaire de civilité que je portais en moi ne donnait qu'un seul sens, celui d'une intention insolente.
