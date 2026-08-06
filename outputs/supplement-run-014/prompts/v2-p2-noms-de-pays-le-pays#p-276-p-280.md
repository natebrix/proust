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
      "source": "collective_social_voice",
      "target": "Robert de Saint-Loup",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.86,
      "evidence": "Sa famille déplore sa liaison avec une femme « de théâtre », l'accusant de l'avoir « dévoyé » et de risquer qu'il se « déclassât »; on répète : « Cette gueuse le tuera, et en attendant elle le déshonore. »",
      "explanation": "The social discourse stigmatizes Robert because of his liaison, attaching to him a loss of social credit; the narrator reports these judgments to better contest them."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "In the eyes of his family and the neighborhood, his prestige is diminished by the opprobrium associated with his affair."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-276-p-280"
}

### Candidate characters

[
  "Françoise",
  "le narrateur",
  "princesse de Luxembourg"
]

### Prior local context (optional)

Or la sincérité et le désintéressement de Robert de Saint-Loup étaient au contraire absolus et c'était cette grande pureté morale qui, ne pouvant se satisfaire entièrement dans un sentiment égoïste comme l'amour, ne rencontrant pas d'autre part en lui l'impossibilité qui existait par exemple en moi de trouver sa nourriture spirituelle autre part qu'en soi-même, le rendait vraiment capable, autant que moi incapable, d'amitié.

### Passage

Françoise ne se trompait pas moins sur Saint-Loup quand elle disait qu'il avait l'air comme ça de ne pas dédaigner le peuple, mais que ce n'est pas vrai et qu'il n'y avait qu'à le voir quand il était en colère après son cocher. Il était arrivé en effet quelquefois à Saint-Loup de le gronder avec une certaine rudesse, qui prouvait chez lui moins le sentiment de la différence que de l'égalité entre les classes. « Mais, me dit-il en réponse aux reproches que je lui faisais d'avoir traité un peu durement ce cocher, pourquoi affecterais-je de lui parler poliment ? N'est-il pas mon égal ? N'est-il pas aussi près de moi que mes oncles ou mes cousins ? Vous avez l'air de trouver que je devrais le traiter avec égards, comme un inférieur ! Vous parlez comme un aristocrate », ajouta-t-il avec dédain.

En effet, s'il y avait une classe contre laquelle il eût de la prévention et de la partialité, c'était l'aristocratie, et jusqu'à croire aussi difficilement à la supériorité d'un homme du monde, qu'il croyait facilement à celle d'un homme du peuple. Comme je lui parlais de la princesse de Luxembourg que j'avais rencontrée avec sa tante :

– Une carpe, me dit-il, comme toutes ses pareilles. C'est d'ailleurs un peu ma cousine.

Ayant un préjugé contre les gens qui le fréquentaient, il allait rarement dans le monde et l'attitude méprisante ou hostile qu'il y prenait augmentait encore chez tous ses proches parents le chagrin de sa liaison avec une femme « de théâtre », liaison qu'ils accusaient de lui être fatale et notamment d'avoir développé chez lui cet esprit de dénigrement, ce mauvais esprit, de l'avoir « dévoyé », en attendant qu'il se « déclassât » complètement. Aussi, bien des hommes légers du faubourg Saint-Germain étaient-ils sans pitié quand ils parlaient de la maîtresse de Saint-Loup. « Les grues font leur métier, disait-on, elles valent autant que d'autres ; mais celle-là, non ! Nous ne lui pardonnerons pas ! Elle a fait trop de mal à quelqu'un que nous aimons. » Certes, il n'était pas le premier qui eût un fil à la patte. Mais les autres s'amusaient en hommes du monde, continuaient à penser en hommes du monde sur la politique, sur tout. Lui, sa famille le trouvait « aigri ». Elle ne se rendait pas compte que pour bien des jeunes gens du monde, lesquels sans cela resteraient incultes d'esprit, rudes dans leurs amitiés, sans douceur et sans goût, c'est bien souvent leur maîtresse qui est leur vrai maître et les liaisons de ce genre la seule école morale où ils soient initiés à une culture supérieure, où ils apprennent le prix des connaissances désintéressées. Même dans le bas-peuple (qui au point de vue de la grossièreté ressemble si souvent au grand monde), la femme, plus sensible, plus fine, plus oisive, a la curiosité de certaines délicatesses, respecte certaines beautés de sentiment et d'art que, ne les comprît-elle pas, elle place pourtant au-dessus de ce qui semblait le plus désirable à l'homme, l'argent, la situation. Or, qu'il s'agisse de la maîtresse d'un jeune clubman comme Saint-Loup ou d'un jeune ouvrier (les électriciens par exemple comptent aujourd'hui dans les rangs de la Chevalerie véritable), son amant a pour elle trop d'admiration et de respect pour ne pas les étendre à ce qu'elle-même respecte et admire ; et pour lui l'échelle des valeurs s'en trouve renversée. À cause de son sexe même elle est faible, elle a des troubles nerveux, inexplicables, qui chez un homme, et même chez une autre femme, chez une femme dont il est neveu ou cousin auraient fait sourire ce jeune homme robuste. Mais il ne peut voir souffrir celle qu'il aime. Le jeune noble qui comme Saint-Loup a une maîtresse prend l'habitude quand il va dîner avec elle au cabaret d'avoir dans sa poche le valérianate dont elle peut avoir besoin, d'enjoindre au garçon, avec force et sans ironie, de faire attention à fermer les portes sans bruit, à ne pas mettre de mousse humide sur la table, afin d'éviter à son amie ces malaises que pour sa part il n'a jamais ressentis, qui composent pour lui un monde occulte à la réalité duquel elle lui a appris à croire, malaises qu'il plaint maintenant sans avoir besoin pour cela de les connaître, qu'il plaindra même quand ce sera d'autres qu'elle qui les ressentiront. La maîtresse de Saint-Loup – comme les premiers moines du moyen âge, à la chrétienté – lui avait enseigné la pitié envers les animaux, car elle en avait la passion, ne se déplaçant jamais sans son chien, ses serins, ses perroquets ; Saint-Loup veillait sur eux avec des soins maternels et traitait de brutes les gens qui ne sont pas bons avec les bêtes. D'autre part, une actrice, ou soi-disant telle, comme celle qui vivait avec lui – qu'elle fût intelligente ou non, ce que j'ignorais – en lui faisant trouver ennuyeuse la société des femmes du monde et considérer comme une corvée l'obligation d'aller dans une soirée, l'avait préservé du snobisme et guéri de la frivolité. Si grâce à elle les relations mondaines tenaient moins de place dans la vie de son jeune amant, en revanche tandis que s'il avait été un simple homme de salon, la vanité ou l'intérêt auraient dirigé ses amitiés comme la rudesse les aurait empreintes, sa maîtresse lui avait appris à y mettre de la noblesse et du raffinement. Avec son instinct de femme et appréciant plus chez les hommes certaines qualités de sensibilité que son amant eût peut-être sans elle méconnues ou plaisantées, elle avait toujours vite fait de distinguer entre les autres celui des amis de Saint-Loup qui avait pour lui une affection vraie, et de le préférer. Elle savait le forcer à éprouver pour celui-là de la reconnaissance, à la lui témoigner, à remarquer les choses qui lui faisaient plaisir, celles qui lui faisaient de la peine. Et bientôt Saint-Loup, sans plus avoir besoin qu'elle l'avertît, commença à se soucier de tout cela et à Balbec où elle n'était pas, pour moi qu'elle n'avait jamais vu et dont il ne lui avait même peut-être pas encore parlé dans ses lettres, de lui-même il fermait la fenêtre d'une voiture où j'étais, emportait les fleurs qui me faisaient mal, et quand il eut à dire au revoir à la fois à plusieurs personnes, à son départ, s'arrangea à les quitter un peu plus tôt afin de rester seul et en dernier avec moi, de mettre cette différence entre elles et moi, de me traiter autrement que les autres. Sa maîtresse avait ouvert son esprit à l'invisible, elle avait mis du sérieux dans sa vie, des délicatesses dans son coeur, mais tout cela échappait à la famille en larmes qui répétait : « Cette gueuse le tuera, et en attendant elle le déshonore. » Il est vrai qu'il avait fini de tirer d'elle tout le bien qu'elle pouvait lui faire ; et maintenant elle était cause seulement qu'il souffrait sans cesse, car elle l'avait pris en horreur et le torturait. Elle avait commencé un beau jour à le trouver bête et ridicule parce que les amis qu'elle avait parmi les jeunes auteurs et acteurs, lui avaient assuré qu'il l'était, et elle répétait à son tour ce qu'ils avaient dit avec cette passion, cette absence de réserve qu'on montre chaque fois qu'on reçoit du dehors et qu'on adopte des opinions ou des usages qu'on ignorait entièrement. Elle professait volontiers, comme ces comédiens, qu'entre elle et Saint-Loup le fossé était infranchissable, parce qu'ils étaient d'une autre race, qu'elle était une intellectuelle et que lui, quoi qu'il prétendît, était, de naissance, un ennemi de l'intelligence. Cette vue lui semblait profonde et elle en cherchait la vérification dans les paroles les plus insignifiantes, les moindres gestes de son amant. Mais quand les mêmes amis l'eurent en outre convaincue qu'elle détruisait dans une compagnie aussi peu faite pour elle les grandes espérances qu'elle avait, disaient-ils, données, que son amant finirait par déteindre sur elle, qu'à vivre avec lui elle gâchait son avenir d'artiste, à son mépris pour Saint-Loup s'ajouta la même haine que s'il s'était obstiné à vouloir lui inoculer une maladie mortelle. Elle le voyait le moins possible tout en reculant encore le moment d'une rupture définitive, laquelle me paraissait à moi bien peu vraisemblable. Saint-Loup faisait pour elle de tels sacrifices que, à moins qu'elle fût ravissante (mais il n'avait jamais voulu me montrer sa photographie, me disant : « D'abord ce n'est pas une beauté et puis elle vient mal en photographie, ce sont des instantanés que j'ai faits moi-même avec mon Kodak et ils vous donneraient une fausse idée d'elle »), il semblait difficile qu'elle trouvât un second homme qui en consentît de semblables. Je ne songeais pas qu'une certaine toquade de se faire un nom, même quand on n'a pas de talent, que l'estime, rien que l'estime privée, de personnes qui vous imposent, peuvent (ce n'était peut-être du reste pas le cas pour la maîtresse de Saint-Loup) être même pour une petite cocotte des motifs plus déterminants que le plaisir de gagner de l'argent. Saint-Loup qui sans bien comprendre ce qui se passait dans la pensée de sa maîtresse, ne la croyait complètement sincère ni dans les reproches injustes ni dans les promesses d'amour éternel, avait pourtant à certains moments le sentiment qu'elle romprait quand elle le pourrait, et à cause de cela, mû sans doute par l'instinct de conservation de son amour, plus clairvoyant peut-être que Saint-Loup n'était lui-même, usant d'ailleurs d'une habileté pratique qui se conciliait chez lui avec les plus grands et les plus aveugles élans du coeur, il s'était refusé à lui constituer un capital, avait emprunté un argent énorme pour qu'elle ne manquât de rien, mais ne le lui remettait qu'au jour le jour. Et sans doute, au cas où elle eût vraiment songé à le quitter, attendait-elle froidement d'avoir « fait sa pelotte », ce qui avec les sommes données par Saint-Loup demanderait sans doute un temps fort court, mais tout de même concédé en supplément pour prolonger le bonheur de mon nouvel ami – ou son malheur.

Cette période dramatique de leur liaison – et qui était arrivée maintenant à son point le plus aigu, le plus cruel pour Saint-Loup, car elle lui avait défendu de rester à Paris où sa présence l'exaspérait et l'avait forcé de prendre son congé à Balbec, à côté de sa garnison – avait commencé un soir chez une tante de Saint-Loup, lequel avait obtenu d'elle que son amie viendrait pour de nombreux invités dire des fragments d'une pièce symboliste qu'elle avait jouée une fois sur une scène d'avant-garde et pour laquelle elle lui avait fait partager l'admiration qu'elle éprouvait elle-même.
