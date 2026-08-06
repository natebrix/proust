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
      "canonical_name": "la mère du narrateur",
      "surface_forms": [
        "Maman",
        "la mère du narrateur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "la mère du narrateur",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« …une lectrice admirable par le respect et la simplicité de l’interprétation, par la beauté et la douceur du son… ces phrases… semblaient écrites pour sa voix… elle insufflait à cette prose… une sorte de vie sentimentale et continue. »",
      "explanation": "The narrator strongly elevates the mother for her moral sensibility and exceptional reading voice, presenting her conduct and delivery as ideally suited to George Sand's prose."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "la mère du narrateur",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "uncertain",
      "confidence": 0.7,
      "evidence": "« …c'était une première abdication de sa part… elle… s'avouait vaincue. Il me semblait que si je venais de remporter une victoire c'était contre elle… »",
      "explanation": "The narrator frames the mother's choice to stay as a painful concession and a 'defeat' of her educational ideal. This locally lowers her, but the language ('il me semblait') marks it as the child's perception."
    }
  ],
  "status_effects": [
    {
      "character": "la mère du narrateur",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.7,
      "explanation": "Locally portrayed as having 'abdicated' her ideal, a concession framed as a defeat in the child's eyes."
    },
    {
      "character": "la mère du narrateur",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Her authority and excellence as a reader are emphatically affirmed, granting her elevated rhetorical standing."
    }
  ],
  "ambiguities": [
    "The 'defeat/abdication' of the mother is explicitly presented as the child's feeling ('il me semblait'), not an asserted fact; the passage later counterbalances with strong praise."
  ],
  "unit_id": "v1-p1-combray#p-36-p-40"
}

### Candidate characters

[
  "Françoise",
  "Robert de Saint-Loup",
  "Swann",
  "la grand-mère",
  "le narrateur",
  "le peintre",
  "le père du narrateur"
]

### Prior local context (optional)

On ne pouvait pas remercier le père du narrateur ; on l'eût agacé par ce qu'il appelait des sensibleries. Je restai sans oser faire un mouvement ; il était encore devant nous, grand, dans sa robe de nuit blanche sous le cachemire de l'Inde violet et rose qu'il nouait autour de sa tête depuis qu'il avait des névralgies, avec le geste d'Abraham dans la gravure d'après Benozzo Gozzoli que m'avait donnée Swann, disant à Sarah qu'elle a à se départir du côté d'Isaac. Il y a bien des années de cela. La muraille de l'escalier où je vis monter le reflet de sa bougie n'existe plus depuis longtemps. En moi aussi bien des choses ont été détruites que je croyais devoir durer toujours, et de nouvelles se sont édifiées donnant naissance à des peines et à des joies nouvelles que je n'aurais pu prévoir alors, de même que les anciennes me sont devenues difficiles à comprendre. Il y a bien longtemps aussi que le père du narrateur a cessé de pouvoir dire à la mère du narrateur : « Va avec le petit. » La possibilité de telles heures ne renaîtra jamais pour moi. Mais depuis peu de temps, je recommence à très bien percevoir si je prête l'oreille, les sanglots que j'eus la force de contenir devant le père du narrateur et qui n'éclatèrent que quand je me retrouvai seul avec la mère du narrateur. En réalité ils n'ont jamais cessé ; et c'est seulement parce que la vie se tait maintenant davantage autour de moi que je les entends de nouveau, comme ces cloches de couvents que couvrent si bien les bruits de la ville pendant le jour qu'on les croirait arrêtées mais qui se remettent à sonner dans le silence du soir.

### Passage

Maman passa cette nuit-là dans ma chambre ; au moment où je venais de commettre une faute telle que je m'attendais à être obligé de quitter la maison, mes parents m'accordaient plus que je n'eusse jamais obtenu d'eux comme récompense d'une belle action. Même à l'heure où elle se manifestait par cette grâce, la conduite de mon père à mon égard gardait ce quelque chose d'arbitraire et d'immérité qui la caractérisait, et qui tenait à ce que généralement elle résultait plutôt de convenances fortuites que d'un plan prémédité. Peut-être même que ce que j'appelais sa sévérité, quand il m'envoyait me coucher, méritait moins ce nom que celle de ma mère ou de ma grand'mère, car sa nature, plus différente en certains points de la mienne que n'était la leur, n'avait probablement pas deviné jusqu'ici combien j'étais malheureux tous les soirs, ce que ma mère et ma grand'mère savaient bien ; mais elles m'aimaient assez pour ne pas consentir à m'épargner de la souffrance, elles voulaient m'apprendre à la dominer afin de diminuer ma sensibilité nerveuse et fortifier ma volonté. Pour mon père, dont l'affection pour moi était d'une autre sorte, je ne sais pas s'il aurait eu ce courage : pour une fois où il venait de comprendre que j'avais du chagrin, il avait dit à ma mère : « Va donc le consoler. » Maman resta cette nuit-là dans ma chambre et, comme pour ne gâter d'aucun remords ces heures si différentes de ce que j'avais eu le droit d'espérer, quand Françoise, comprenant qu'il se passait quelque chose d'extraordinaire en voyant maman assise près de moi, qui me tenait la main et me laissait pleurer sans me gronder, lui demanda : « Mais Madame, qu'a donc Monsieur à pleurer ainsi ? » maman lui répondit : « Mais il ne sait pas lui-même, Françoise, il est énervé ; préparez-moi vite le grand lit et montez vous coucher. » Ainsi, pour la première fois, ma tristesse n'était plus considérée comme une faute punissable mais comme un mal involontaire qu'on venait de reconnaître officiellement, comme un état nerveux dont je n'étais pas responsable ; j'avais le soulagement de n'avoir plus à mêler de scrupules à l'amertume de mes larmes, je pouvais pleurer sans péché. Je n'étais pas non plus médiocrement fier vis-à-vis de Françoise de ce retour des choses humaines, qui, une heure après que maman avait refusé de monter dans ma chambre et m'avait fait dédaigneusement répondre que je devrais dormir, m'élevait à la dignité de grande personne et m'avait fait atteindre tout d'un coup à une sorte de puberté du chagrin, d'émancipation des larmes. J'aurais dû être heureux : je ne l'étais pas. Il me semblait que ma mère venait de me faire une première concession qui devait lui être douloureuse, que c'était une première abdication de sa part devant l'idéal qu'elle avait conçu pour moi, et que pour la première fois, elle, si courageuse, s'avouait vaincue. Il me semblait que si je venais de remporter une victoire c'était contre elle, que j'avais réussi comme auraient pu faire la maladie, des chagrins, ou l'âge, à détendre sa volonté, à faire fléchir sa raison, et que cette soirée commençait une ère, resterait comme une triste date. Si j'avais osé maintenant, j'aurais dit à maman : « Non je ne veux pas, ne couche pas ici. » Mais je connaissais la sagesse pratique, réaliste comme on dirait aujourd'hui, qui tempérait en elle la nature ardemment idéaliste de ma grand'mère, et je savais que, maintenant que le mal était fait, elle aimerait mieux m'en laisser du moins goûter le plaisir calmant et ne pas déranger mon père. Certes, le beau visage de ma mère brillait encore de jeunesse ce soir-là où elle me tenait si doucement les mains et cherchait à arrêter mes larmes ; mais justement il me semblait que cela n'aurait pas dû être, sa colère eût été moins triste pour moi que cette douceur nouvelle que n'avait pas connue mon enfance ; il me semblait que je venais d'une main impie et secrète de tracer dans son âme une première ride et d'y faire apparaître un premier cheveu blanc. Cette pensée redoubla mes sanglots, et alors je vis maman, qui jamais ne se laissait aller à aucun attendrissement avec moi, être tout d'un coup gagnée par le mien et essayer de retenir une envie de pleurer. Comme elle sentit que je m'en étais aperçu, elle me dit en riant : « Voilà mon petit jaunet, mon petit serin, qui va rendre sa maman aussi bêtasse que lui, pour peu que cela continue. Voyons, puisque tu n'as pas sommeil ni ta maman non plus, ne restons pas à nous énerver, faisons quelque chose, prenons un de tes livres. » Mais je n'en avais pas là. « Est-ce que tu aurais moins de plaisir si je sortais déjà les livres que ta grand'mère doit te donner pour ta fête ? Pense bien : tu ne seras pas déçu de ne rien avoir après-demain ? » J'étais au contraire enchanté et maman alla chercher un paquet de livres dont je ne pus deviner, à travers le papier qui les enveloppait, que la taille courte et large, mais qui, sous ce premier aspect, pourtant sommaire et voilé, éclipsaient déjà la boîte à couleurs du Jour de l'An et les vers à soie de l'an dernier. C'était la Mare au Diable, François le Champi, la Petite Fadette et les Maîtres Sonneurs. Ma grand'mère, ai-je su depuis, avait d'abord choisi les poésies de Musset, un volume de Rousseau et Indiana ; car si elle jugeait les lectures futiles aussi malsaines que les bonbons et les pâtisseries, elles ne pensait pas que les grands souffles du génie eussent sur l'esprit même d'un enfant une influence plus dangereuse et moins vivifiante que sur son corps le grand air et le vent du large. Mais mon père l'ayant presque traitée de folle en apprenant les livres qu'elle voulait me donner, elle était retournée elle-même à Jouy-le-Vicomte chez le libraire pour que je ne risquasse pas de ne pas avoir mon cadeau (c'était un jour brûlant et elle était rentrée si souffrante que le médecin avait averti ma mère de ne pas la laisser se fatiguer ainsi) et elle s'était rabattue sur les quatre romans champêtres de George Sand. « Ma fille, disait-elle à maman, je ne pourrais me décider à donner à cet enfant quelque chose de mal écrit. »

En réalité, elle ne se résignait jamais à rien acheter dont on ne pût tirer un profit intellectuel, et surtout celui que nous procurent les belles choses en nous apprenant à chercher notre plaisir ailleurs que dans les satisfactions du bien-être et de la vanité. Même quand elle avait à faire à quelqu'un un cadeau dit utile, quand elle avait à donner un fauteuil, des couverts, une canne, elle les cherchait « anciens », comme si leur longue désuétude ayant effacé leur caractère d'utilité, ils paraissaient plutôt disposés pour nous raconter la vie des hommes d'autrefois que pour servir aux besoins de la nôtre. Elle eût aimé que j'eusse dans ma chambre des photographies des monuments ou des paysages les plus beaux. Mais au moment d'en faire l'emplette, et bien que la chose représentée eût une valeur esthétique, elle trouvait que la vulgarité, l'utilité reprenaient trop vite leur place dans le mode mécanique de représentation, la photographie. Elle essayait de ruser et, sinon d'éliminer entièrement la banalité commerciale, du moins de la réduire, d'y substituer, pour la plus grande partie, de l'art encore, d'y introduire comme plusieurs « épaisseurs » d'art : au lieu de photographies de la Cathédrale de Chartres, des Grandes Eaux de Saint-Cloud, du Vésuve, elle se renseignait auprès de Swann si quelque grand peintre ne les avait pas représentés, et préférait me donner des photographies de la Cathédrale de Chartres par Corot, des Grandes Eaux de Saint-Cloud par Hubert Saint-Loup, du Vésuve par Turner, ce qui faisait un degré d'art de plus. Mais si le photographe avait été écarté de la représentation du chef-d'oeuvre ou de la nature et remplacé par un grand artiste, il reprenait ses droits pour reproduire cette interprétation même. Arrivée à l'échéance de la vulgarité, ma grand'mère tâchait de la reculer encore. Elle demandait à Swann si l'oeuvre n'avait pas été gravée, préférant, quand c'était possible, des gravures anciennes et ayant encore un intérêt au delà d'elles-mêmes, par exemple celles qui représentent un chef-d'oeuvre dans un état où nous ne pouvons plus le voir aujourd'hui (comme la gravure de la Cène de Léonard avant sa dégradation, par Morgan). Il faut dire que les résultats de cette manière de comprendre l'art de faire un cadeau ne furent pas toujours très brillants. L'idée que je pris de Venise d'après un dessin du Titien qui est censé avoir pour fond la lagune, était certainement beaucoup moins exacte que celle que m'eussent donnée de simples photographies. On ne pouvait plus faire le compte à la maison, quand ma grand'tante voulait dresser un réquisitoire contre ma grand'mère, des fauteuils offerts par elle à de jeunes fiancés ou à de vieux époux, qui, à la première tentative qu'on avait faite pour s'en servir, s'étaient immédiatement effondrés sous le poids d'un des destinataires. Mais ma grand'mère aurait cru mesquin de trop s'occuper de la solidité d'une boiserie où se distinguaient encore une fleurette, un sourire, quelquefois une belle imagination du passé. Même ce qui dans ces meubles répondait à un besoin, comme c'était d'une façon à laquelle nous ne sommes plus habitués, la charmait comme les vieilles manières de dire où nous voyons une métaphore, effacée, dans notre moderne langage, par l'usure de l'habitude. Or, justement, les romans champêtres de George Sand qu'elle me donnait pour ma fête, étaient pleins, ainsi qu'un mobilier ancien, d'expressions tombées en désuétude et redevenues imagées, comme on n'en trouve plus qu'à la campagne. Et ma grand'mère les avait achetés de préférence à d'autres, comme elle eût loué plus volontiers une propriété où il y aurait eu un pigeonnier gothique, ou quelqu'une de ces vieilles choses qui exercent sur l'esprit une heureuse influence en lui donnant la nostalgie d'impossibles voyages dans le temps. Maman s'assit à côté de mon lit ; elle avait pris François le Champi à qui sa couverture rougeâtre et son titre incompréhensible donnaient pour moi une personnalité distincte et un attrait mystérieux. Je n'avais jamais lu encore de vrais romans. J'avais entendu dire que George Sand était le type du romancier. Cela me disposait déjà à imaginer dans François le Champi quelque chose d'indéfinissable et de délicieux. Les procédés de narration destinés à exciter la curiosité ou l'attendrissement, certaines façons de dire qui éveillent l'inquiétude et la mélancolie, et qu'un lecteur un peu instruit reconnaît pour communs à beaucoup de romans, me paraissaient simples – à moi qui considérais un livre nouveau non comme une chose ayant beaucoup de semblables, mais comme une personne unique, n'ayant de raison d'exister qu'en soi – une émanation troublante de l'essence particulière à François le Champi. Sous ces événements si journaliers, ces choses si communes, ces mots si courants, je sentais comme une intonation, une accentuation étrange. L'action s'engagea ; elle me parut d'autant plus obscure que dans ce temps-là, quand je lisais, je rêvassais souvent, pendant des pages entières, à tout autre chose. Et aux lacunes que cette distraction laissait dans le récit, s'ajoutait, quand c'était maman qui me lisait à haute voix, qu'elle passait toutes les scènes d'amour. Aussi tous les changements bizarres qui se produisent dans l'attitude respective de la meunière et de l'enfant et qui ne trouvent leur explication que dans les progrès d'un amour naissant me paraissaient empreints d'un profond mystère dont je me figurais volontiers que la source devait être dans ce nom inconnu et si doux de « Champi » qui mettait sur l'enfant, qui le portait sans que je susse pourquoi, sa couleur vive, empourprée et charmante. Si ma mère était une lectrice infidèle, c'était aussi, pour les ouvrages où elle trouvait l'accent d'un sentiment vrai, une lectrice admirable par le respect et la simplicité de l'interprétation, par la beauté et la douceur du son. Même dans la vie, quand c'étaient des êtres et non des oeuvres d'art qui excitaient ainsi son attendrissement ou son admiration, c'était touchant de voir avec quelle déférence elle écartait de sa voix, de son geste, de ses propos, tel éclat de gaîté qui eût pu faire mal à cette mère qui avait autrefois perdu un enfant, tel rappel de fête, d'anniversaire, qui aurait pu faire penser ce vieillard à son grand âge, tel propos de ménage qui aurait paru fastidieux à ce jeune savant. De même, quand elle lisait la prose de George Sand, qui respire toujours cette bonté, cette distinction morale que maman avait appris de ma grand'mère à tenir pour supérieures à tout dans la vie, et que je ne devais lui apprendre que bien plus tard à ne pas tenir également pour supérieures à tout dans les livres, attentive à bannir de sa voix toute petitesse, toute affectation qui eût pu empêcher le flot puissant d'y être reçu, elle fournissait toute la tendresse naturelle, toute l'ample douceur qu'elles réclamaient à ces phrases qui semblaient écrites pour sa voix et qui pour ainsi dire tenaient tout entières dans le registre de sa sensibilité. Elle retrouvait pour les attaquer dans le ton qu'il faut l'accent cordial qui leur préexiste et les dicta, mais que les mots n'indiquent pas ; grâce à lui elle amortissait au passage toute crudité dans les temps des verbes, donnait à l'imparfait et au passé défini la douceur qu'il y a dans la bonté, la mélancolie qu'il y a dans la tendresse, dirigeait la phrase qui finissait vers celle qui allait commencer, tantôt pressant, tantôt ralentissant la marche des syllabes pour les faire entrer, quoique leurs quantités fussent différentes, dans un rythme uniforme, elle insufflait à cette prose si commune une sorte de vie sentimentale et continue. Mes remords étaient calmés, je me laissais aller à la douceur de cette nuit où j'avais ma mère auprès de moi. Je savais qu'une telle nuit ne pourrait se renouveler ; que le plus grand désir que j'eusse au monde, garder ma mère dans ma chambre pendant ces tristes heures nocturnes, était trop en opposition avec les nécessités de la vie et le voeu de tous, pour que l'accomplissement qu'on lui avait accordé ce soir pût être autre chose que factice et exceptionnel. Demain mes angoisses reprendraient et maman ne resterait pas là. Mais quand mes angoisses étaient calmées, je ne les comprenais plus ; puis demain soir était encore lointain ; je me disais que j'aurais le temps d'aviser, bien que ce temps-là ne pût m'apporter aucun pouvoir de plus, puisqu'il s'agissait de choses qui ne dépendaient pas de ma volonté et que seul me faisait paraître plus évitables l'intervalle qui les séparait encore de moi.

C'est ainsi que, pendant longtemps, quand, réveillé la nuit, je me ressouvenais de Combray, je n'en revis jamais que cette sorte de pan lumineux, découpé au milieu d'indistinctes ténèbres, pareil à ceux que l'embrasement d'un feu de bengale ou quelque projection électrique éclairent et sectionnent dans un édifice dont les autres parties restent plongées dans la nuit : à la base assez large, le petit salon, la salle à manger, l'amorce de l'allée obscure par où arriverait Swann, l'auteur inconscient de mes tristesses, le vestibule où je m'acheminais vers la première marche de l'escalier, si cruel à monter, qui constituait à lui seul le tronc fort étroit de cette pyramide irrégulière ; et, au faîte, ma chambre à coucher avec le petit couloir à porte vitrée pour l'entrée de maman ; en un mot, toujours vu à la même heure, isolé de tout ce qu'il pouvait y avoir autour, se détachant seul sur l'obscurité, le décor strictement nécessaire (comme celui qu'on voit indiqué en tête des vieilles pièces pour les représentations en province) au drame de mon déshabillage ; comme si Combray n'avait consisté qu'en deux étages reliés par un mince escalier et comme s'il n'y avait jamais été que sept heures du soir. À vrai dire, j'aurais pu répondre à qui m'eût interrogé que Combray comprenait encore autre chose et existait à d'autres heures. Mais comme ce que je m'en serais rappelé m'eût été fourni seulement par la mémoire volontaire, la mémoire de l'intelligence, et comme les renseignements qu'elle donne sur le passé ne conservent rien de lui, je n'aurais jamais eu envie de songer à ce reste de Combray. Tout cela était en réalité mort pour moi.

Mort à jamais ? C'était possible.

Il y a beaucoup de hasard en tout ceci, et un second hasard, celui de notre mort, souvent ne nous permet pas d'attendre longtemps les faveurs du premier.
