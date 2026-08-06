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
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.83,
      "evidence": "« cet appartement … où étaient venus se confondre … l'appartement commun à Odette et à lui … ce paradis inespéré … “Madame est-elle prête ?” … prononcer maintenant avec une légère impatience mêlée de quelque satisfaction d'amour-propre »",
      "explanation": "The narrator frames Swann as having attained the once-inaccessible domestic intimacy with Odette; the realized scene, and Swann’s satisfied tone, mark a local elevation in his affective position."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "Swann locally gains affective satisfaction by inhabiting the previously fantasized shared apartment with Odette, now realized in everyday domestic life."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-186-p-190"
}

### Candidate characters

[
  "Gilberte",
  "Odette",
  "comte de Forcheville",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

– Oui, pauvre papa, c'est ces jours-ci l'anniversaire de la mort de son père. Vous pouvez comprendre ce qu'il doit éprouver, vous comprenez cela, vous, nous sentons de même sur ces choses-là. Alors, je tâche d'être moins méchante que d'habitude. – Mais il ne vous trouve pas méchante, il vous trouve parfaite. – Pauvre papa, c'est parce qu'il est trop bon.

### Passage

Ses parents ne me firent pas seulement l'éloge des vertus de Gilberte – cette même Gilberte qui même avant que je l'eusse jamais vue m'apparaissait devant une église, dans un paysage de l'Île-de-France, et qui ensuite m'évoquant non plus mes rêves, mais mes souvenirs, était toujours devant la haie d'épines roses, dans le raidillon que je prenais pour aller du côté de Méséglise ; comme j'avais demandé à Odette, en m'efforçant de prendre le ton indifférent d'un ami de la famille, curieux des préférences d'une enfant, quels étaient parmi les camarades de Gilberte ceux qu'elle aimait le mieux, Odette me répondit :

– Mais vous devez être plus avancé que moi dans ses confidences, vous qui êtes le grand favori, le grand crack comme disent les Anglais.

Sans doute dans ces coïncidences tellement parfaites, quand la réalité se replie et s'applique sur ce que nous avons si longtemps rêvé, elle nous le cache entièrement, se confond avec lui, comme deux figures égales et superposées qui n'en font plus qu'une, alors qu'au contraire, pour donner à notre joie toute sa signification, nous voudrions garder à tous ces points de notre désir, dans le moment même où nous y touchons – et pour être plus certain que ce soit bien eux – le prestige d'être intangibles. Et la pensée ne peut même pas reconstituer l'état ancien pour le confronter au nouveau, car elle n'a plus le champ libre : la connaissance que nous avons faite, le souvenir des premières minutes inespérées, les propos que nous avons entendus, sont là qui obstruent l'entrée de notre conscience, et commandent beaucoup plus les issues de notre mémoire que celles de notre imagination, ils rétroagissent davantage sur notre passé que nous ne sommes plus maîtres de voir sans tenir compte d'eux, que sur la forme, restée libre, de notre avenir. J'avais pu croire pendant des années qu'aller chez Odette était une vague chimère que je n'atteindrais jamais ; après avoir passé un quart d'heure chez elle, c'est le temps où je ne la connaissais pas qui était devenu chimérique et vague comme un possible que la réalisation d'un autre possible a anéanti. Comment aurais-je encore pu rêver de la salle à manger comme d'un lieu inconcevable, quand je ne pouvais pas faire un mouvement dans mon esprit sans y rencontrer les rayons infrangibles qu'émettait à l'infini derrière lui, jusque dans mon passé le plus ancien, le homard à l'américaine que je venais de manger ? Et Swann avait dû voir, pour ce qui le concernait lui-même, se produire quelque chose d'analogue : car cet appartement où il me recevait pouvait être considéré comme le lieu où étaient venus se confondre, et coïncider, non pas seulement l'appartement idéal que mon imagination avait engendré, mais un autre encore, celui que l'amour jaloux de Swann, aussi inventif que mes rêves, lui avait si souvent décrit, cet appartement commun à Odette et à lui qui lui était apparu si inaccessible, tel soir où Odette l'avait ramené avec Forcheville prendre de l'orangeade chez elle ; et ce qui était venu s'absorber, pour lui, dans le plan de la salle à manger où nous déjeunions, c'était ce paradis inespéré où jadis il ne pouvait sans trouble imaginer qu'il aurait dit à leur maître d'hôtel ces mêmes mots : « Madame est-elle prête ? » que je lui entendais prononcer maintenant avec une légère impatience mêlée de quelque satisfaction d'amour-propre. Pas plus que ne le pouvait sans doute Swann, je n'arrivais à connaître mon bonheur, et quand Gilberte elle-même s'écriait : « Qu'est-ce qui vous aurait dit que la petite fille que vous regardiez, sans lui parler, jouer aux barres serait votre grande amie chez qui vous iriez tous les jours où cela vous plairait », elle parlait d'un changement que j'étais bien obligé de constater du dehors, mais que je ne possédais pas intérieurement, car il se composait de deux états que je ne pouvais, sans qu'ils cessassent d'être distincts l'un de l'autre, réussir à penser à la fois.

Et pourtant cet appartement, parce qu'il avait été si passionnément désiré par la volonté de Swann, devait conserver pour lui quelque douceur, si j'en jugeais par moi pour qui il n'avait pas perdu tout mystère. Ce charme singulier dans lequel j'avais pendant si longtemps supposé que baignait la vie des Swann, je ne l'avais pas entièrement chassé de leur maison en y pénétrant ; je l'avais fait reculer, dompté qu'il était par cet étranger, ce paria que j'avais été et à qui Gilberte avançait maintenant gracieusement pour qu'il y prît place un fauteuil délicieux, hostile et scandalisé ; mais tout autour de moi, ce charme, dans mon souvenir, je le perçois encore. Est-ce parce que, ces jours où M. et Odette m'invitaient à déjeuner, pour sortir ensuite avec eux et Gilberte, j'imprimais avec mon regard – pendant que j'attendais seul – sur le tapis, sur les bergères, sur les consoles, sur les paravents, sur les tableaux, l'idée gravée en moi que Odette, ou son mari, ou Gilberte allaient entrer ? Est-ce parce que ces choses ont vécu depuis dans ma mémoire à côté des Swann et ont fini par prendre quelque chose d'eux ? Est-ce que, sachant qu'ils passaient leur existence au milieu d'elles, je faisais de toutes comme les emblèmes de leur vie particulière, de leurs habitudes dont j'avais été trop longtemps exclu pour qu'elles ne continuassent pas à me sembler étrangères même quand on me fit la faveur de m'y mêler ? Toujours est-il que chaque fois que je pense à ce salon que Swann (sans que cette critique impliquât de sa part l'intention de contrarier en rien les goûts de sa femme) trouvait si disparate – parce que tout conçu qu'il était encore dans le goût moitié serre, moitié atelier qui était celui de l'appartement où il avait connu Odette, elle avait pourtant commencé à remplacer dans ce fouillis nombre des objets chinois qu'elle trouvait maintenant un peu « toc », bien « à côté », par une foule de petits meubles tendus de vieilles soies Louis XIV (sans compter les chefs-d'oeuvre apportés par Swann de l'hôtel du quai d'Orléans) – il a au contraire dans mon souvenir, ce salon composite, une cohésion, une unité, un charme individuel que n'ont jamais même les ensembles les plus intacts que le passé nous a légués, ni les plus vivants où se marque l'empreinte d'une personne ; car nous seuls pouvons, par la croyance qu'elles ont une existence à elles, donner à certaines choses que nous voyons une âme qu'elles gardent ensuite et qu'elles développent en nous. Toutes les idées que je m'étais faites des heures, différentes de celles qui existent pour les autres hommes, que passaient les Swann dans cet appartement qui était pour le temps quotidien de leur vie ce que le corps est pour l'âme, et qui devait en exprimer la singularité, toutes ces idées étaient réparties, amalgamées – partout également troublantes et indéfinissables – dans la place des meubles, dans l'épaisseur des tapis, dans l'orientation des fenêtres, dans le service des domestiques. Quand, après le déjeuner, nous allions, au soleil, prendre le café, dans la grande baie du salon, tandis que Odette me demandait combien je voulais de morceaux de sucre dans mon café, ce n'était pas seulement le tabouret de soie qu'elle poussait vers moi qui dégageait, avec le charme douloureux que j'avais perçu autrefois – sous l'épine rose, puis à côté du massif de lauriers – dans le nom de Gilberte, l'hostilité que m'avaient témoignée ses parents et que ce petit meuble semblait avoir si bien sue et partagée, que je ne me sentais pas digne et que je me trouvais un peu lâche d'imposer mes pieds à son capitonnage sans défense ; une âme personnelle le reliait secrètement à la lumière de deux heures de l'après-midi, différente de ce qu'elle était partout ailleurs dans le golfe où elle faisait jouer à nos pieds ses flots d'or parmi lesquels les canapés bleuâtres et les vaporeuses tapisseries émergeaient comme des îles enchantées ; et il n'était pas jusqu'au tableau de Rubens accroché au-dessus de la cheminée qui ne possédât lui aussi le même genre et presque la même puissance de charme que les bottines à lacets de Swann et ce manteau à pèlerine, dont j'avais tant désiré porter le pareil et que maintenant Odette demandait à son mari de remplacer par un autre, pour être plus élégant, quand je leur faisais l'honneur de sortir avec eux. Elle allait s'habiller elle aussi, bien que j'eusse protesté qu'aucune robe « de ville » ne vaudrait à beaucoup près la merveilleuse robe de chambre de crêpe de Chine ou de soie, vieux rose, cerise, rose Tiepolo, blanche, mauve, verte, rouge, jaune unie ou à dessins, dans laquelle Odette avait déjeuné et qu'elle allait ôter. Quand je disais qu'elle aurait dû sortir ainsi, elle riait, par moquerie de mon ignorance ou plaisir de mon compliment. Elle s'excusait de posséder tant de peignoirs parce qu'elle prétendait qu'il n'y avait que là dedans qu'elle se sentait bien et elle nous quittait pour aller mettre une de ces toilettes souveraines qui s'imposaient à tous, et entre lesquelles pourtant j'étais parfois appelé à choisir celle que je préférais qu'elle revêtit.

Au Jardin d'Acclimatation, que j'étais fier, quand nous étions descendus de voiture, de m'avancer à côté de Odette ! Tandis que dans sa démarche nonchalante elle laissait flotter son manteau, je jetais sur elle des regards d'admiration auxquels elle répondait coquettement par un long sourire. Maintenant si nous rencontrions l'un ou l'autre des camarades, fille ou garçon, de Gilberte, qui nous saluait de loin, j'étais à mon tour regardé par eux comme un de ces êtres que j'avais enviés, un de ces amis de Gilberte qui connaissaient sa famille et étaient mêlés à l'autre partie de sa vie, celle qui ne se passait pas aux Champs-Élysées.
