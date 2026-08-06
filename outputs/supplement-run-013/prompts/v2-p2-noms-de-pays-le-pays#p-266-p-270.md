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
      "canonical_name": "Bloch père",
      "surface_forms": [
        "Bloch père",
        "Bloch le père",
        "son père"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bloch père",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Cette importance illusoire de Bloch père… »; surnommé un « faux duc d’Aumale »; il loue une victoria pour paraître chic; la famille s’émerveille; il parle d’« une recommandation de sir Rufus » grâce à une carte prêtée.",
      "explanation": "The narrator systematically exposes the artificial nature of Bloch père's prestige, based on familial simulacra and borrowed prestige."
    }
  ],
  "status_effects": [
    {
      "character": "Bloch père",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "His social status appears manufactured and derived from familial artifices and symbolic borrowings."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-266-p-270"
}

### Candidate characters

[
  "Bergotte",
  "Bloch",
  "M. de Marsantes",
  "Robert de Saint-Loup",
  "duchesse de Guermantes",
  "le grand-père du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

L'égocentrisme permettant de la sorte à chaque humain de voir l'univers étagé au-dessous de lui qui est roi, Bloch se donnait le luxe d'en être un impitoyable quand le matin en prenant son chocolat, voyant la signature de Bergotte au bas d'un article dans le journal à peine entr'ouvert, il lui accordait dédaigneusement une audience écourtée, prononçait sa sentence, et s'octroyait le confortable plaisir de répéter entre chaque gorgée du breuvage bouillant : « Ce Bergotte est devenu illisible. Ce que cet animal-là peut être embêtant. C'est à se désabonner. Comme c'est emberlificoté, quelle tartine ! » Et il reprenait une beurrée.

### Passage

Cette importance illusoire de Bloch père était d'ailleurs étendue un peu au delà du cercle de sa propre perception. D'abord ses enfants le considéraient comme un homme supérieur. Les enfants ont toujours une tendance soit à déprécier, soit à exalter leurs parents, et pour un bon fils, son père est toujours le meilleur des pères, en dehors même de toutes raisons objectives de l'admirer. Or celles-ci ne manquaient pas absolument pour Bloch, lequel était instruit, fin, affectueux pour les siens. Dans la famille la plus proche, on se plaisait d'autant plus avec lui que si dans la « société », on juge les gens d'après un étalon, d'ailleurs absurde, et selon des règles fausses mais fixes, par comparaison avec la totalité des autres gens élégants, en revanche dans le morcellement de la vie bourgeoise, les dîners, les soirées de famille tournent autour de personnes qu'on déclare agréables, amusantes, et qui dans le monde ne tiendraient pas l'affiche deux soirs. Enfin, dans ce milieu où les grandeurs factices de l'aristocratie n'existent pas, on les remplace par des distinctions plus folles encore. C'est ainsi que pour sa famille et jusqu'à un degré de parenté fort éloigné, une prétendue ressemblance dans la façon de porter la moustache et dans le haut du nez faisait qu'on appelait Bloch un « faux duc d'Aumale ». (Dans le monde des « chasseurs » de cercle, l'un porte sa casquette de travers et sa vareuse très serrée de manière à se donner l'air, croit-il, d'un officier étranger, n'est-il pas une manière de personnage pour ses camarades ?)

La ressemblance était des plus vagues, mais on eût dit que ce fût un titre. On répétait : « Bloch ? lequel ? le duc d'Aumale ? » Comme on dit : « La princesse Murat ? laquelle ? la Reine (de Naples) ? »

Un certain nombre d'autres infimes indices achevaient de lui donner aux yeux du cousinage une prétendue distinction. N'allant pas jusqu'à avoir une voiture, Bloch louait à certains jours une victoria découverte à deux chevaux de la Compagnie et traversait le bois de Boulogne, mollement étendu de travers, deux doigts sur la tempe, deux autres sous le menton et si les gens qui ne le connaissaient pas le trouvaient à cause de cela « faiseur d'embarras », on était persuadé dans la famille que pour le chic, l'oncle Salomon aurait pu en remontrer à Gramont-Caderousse. Il était de ces personnes qui quand elles meurent et à cause d'une table commune avec le rédacteur en chef de cette feuille dans un restaurant des boulevards, sont qualifiés de physionomie bien connue des Parisiens, par la Chronique mondaine du Radical. Bloch nous dit à Saint-Loup et à moi que Bergotte savait si bien pourquoi lui, Bloch, ne le saluait pas, que dès qu'il l'apercevait au théâtre ou au cercle, il fuyait son regard. Saint-Loup rougit, car il réfléchit que ce cercle ne pouvait pas être le Jockey dont son père avait été président. D'autre part ce devait être un cercle relativement fermé, car Bloch avait dit que Bergotte n'y serait plus reçu aujourd'hui. Aussi est-ce en tremblant de « sous-estimer l'adversaire » que Saint-Loup demanda si ce cercle était le cercle de la rue Royale, lequel était jugé « déclassant » par la famille de Saint-Loup et où il savait qu'étaient reçus certains Israélites. « Non, répondit Bloch d'un air négligent, fier et honteux, c'est un petit cercle, mais beaucoup plus agréable, le Cercle des Ganaches. On y juge sévèrement la galerie. – Est-ce que sir Rufus Israël n'en est pas président ? » demanda Bloch fils à son père, pour lui fournir l'occasion d'un mensonge honorable et sans se douter que ce financier n'avait pas le même prestige aux yeux de Saint-Loup qu'aux siens. En réalité, il y avait au Cercle des Ganaches non point sir Rufus Israël, mais un de ses employés. Mais comme il était fort bien avec le patron, il avait à sa disposition des cartes du grand financier, et en donnait une à Bloch, quand celui-ci partait en voyage sur une ligne dont sir Rufus était administrateur, ce qui faisait dire au père Bloch : « Je vais passer au cercle demander une recommandation de sir Rufus. » Et la carte lui permettait d'éblouir les chefs de train. Les demoiselles Bloch furent plus intéressées par Bergotte et revenant à lui au lieu de poursuivre sur les « Ganaches », la cadette demanda à son frère du ton le plus sérieux du monde car elle croyait qu'il n'existait pas au monde pour désigner les gens de talent d'autres expressions que celles qu'il employait : « Est-ce un coco vraiment étonnant, ce Bergotte ? Est-il de la catégorie des grands bonshommes, des cocos comme Villiers ou Catulle ? – Je l'ai rencontré à plusieurs générales, dit M. Nissim Bernard. Il est gauche, c'est une espèce de Schlemihl. » Cette allusion au conte de Chamisso n'avait rien de bien grave, mais l'épithète de Schlemihl faisait partie de ce dialecte mi-allemand, mi-juif, dont l'emploi ravissait Bloch dans l'intimité, mais qu'il trouvait vulgaire et déplacé devant des étrangers. Aussi jeta-t-il un regard sévère sur son oncle. « Il a du talent, dit Bloch. – Ah ! fit gravement sa soeur comme pour dire que dans ces conditions j'étais excusable. – Tous les écrivains ont du talent, dit avec mépris Bloch père. – Il paraît même, dit son fils en levant sa fourchette et en plissant ses yeux d'un air diaboliquement ironique, qu'il va se présenter à l'Académie. – Allons donc ! il n'a pas un bagage suffisant, répondit Bloch le père qui ne semblait pas avoir pour l'Académie le mépris de son fils et de ses filles. Il n'a pas le calibre nécessaire. – D'ailleurs l'Académie est un salon et Bergotte ne jouit d'aucune surface », déclara l'oncle à héritage de Mme Bloch, personnage inoffensif et doux dont le nom de Bernard eût peut-être à lui seul éveillé les dons de diagnostic de mon grand-père, mais eût paru insuffisamment en harmonie avec un visage qui semblait rapporté du palais de Darius et reconstitué par Mme Dieulafoy, si, choisi par quelque amateur désireux de donner un couronnement oriental à cette figure de Suse, ce prénom de Nissim n'avait fait planer au-dessus d'elle les ailes de quelque taureau androcéphale de Khorsabad. Mais Bloch ne cessait d'insulter son oncle, soit qu'il fût excité par la bonhomie sans défense de son souffre-douleur, soit que, la villa étant payée par M. Nissim Bernard, le bénéficiaire voulût montrer qu'il gardait son indépendance et surtout qu'il ne cherchait pas par des cajoleries à s'assurer l'héritage à venir du richard. Celui-ci était surtout froissé qu'on le traitât si grossièrement devant le maître d'hôtel. Il murmura une phrase inintelligible où on distinguait seulement : « Quand les Meschorès sont là. » Meschorès désigne dans la Bible le serviteur de Dieu. Entre eux les Bloch s'en servaient pour désigner les domestiques et en étaient toujours égayés, parce que leur certitude de n'être pas compris ni des chrétiens ni des domestiques eux-mêmes exaltait chez M. Nissim Bernard et Bloch leur double particularisme de « maîtres » et de « juifs ». Mais cette dernière cause de satisfaction en devenait une de mécontentement quand il y avait du monde. Alors Bloch entendant son oncle dire « Meschorès » trouvait qu'il laissait trop paraître son côté oriental, de même qu'une cocotte qui invite ses amies avec des gens comme il faut est irritée si elles font allusion à leur métier de cocotte, ou emploient des mots malsonnants. Aussi, bien loin que la prière de son oncle produisît quelque effet sur Bloch, celui-ci, hors de lui, ne put plus se contenir. Il ne perdit plus une occasion d'invectiver le malheureux oncle. « Naturellement, quand il y a quelque bêtise prudhommesque à dire, on peut être sûr que vous ne la ratez pas. Vous seriez le premier à lui lécher les pieds s'il était là », cria Bloch tandis que M. Nissim Bernard attristé inclinait vers son assiette la barbe annelée du roi Sargon. Mon camarade depuis qu'il portait la sienne qu'il avait aussi crépue et bleutée ressemblait beaucoup à son grand-oncle.

– Comment, vous êtes le fils du marquis de Marsantes ? mais je l'ai très bien connu, dit à Saint-Loup M. Nissim Bernard. Je crus qu'il voulait dire « connu » au sens où le père de Bloch disait qu'il connaissait Bergotte, c'est-à-dire de vue. Mais il ajouta : « Votre père était un de mes bons amis. » Cependant Bloch était devenu excessivement rouge, son père avait l'air profondément contrarié, les demoiselles Bloch riaient en s'étouffant. C'est que chez M. Nissim Bernard le goût de l'ostentation, contenu chez Bloch le père et chez ses enfants, avait engendré l'habitude du mensonge perpétuel. Par exemple, en voyage à l'hôtel, M. Nissim Bernard, comme aurait pu faire Bloch le père, se faisait apporter tous ses journaux par son valet de chambre dans la salle à manger, au milieu du déjeuner, quand tout le monde était réuni, pour qu'on vît bien qu'il voyageait avec un valet de chambre. Mais aux gens avec qui il se liait dans l'hôtel, l'oncle disait, ce que le neveu n'eût jamais fait, qu'il était sénateur. Il avait beau être certain qu'on apprendrait un jour que le titre était usurpé, il ne pouvait au moment même résister au besoin de se le donner. Bloch souffrait beaucoup des mensonges de son oncle et de tous les ennuis qu'ils lui causaient. « Ne faites pas attention, il est extrêmement blagueur, dit-il à mi-voix à Saint-Loup qui n'en fut que plus intéressé, étant très curieux de la psychologie des menteurs. – Plus menteur encore que l'Ithaquesien Odysseus qu'Athènes appelait pourtant le plus menteur des hommes, compléta notre camarade Bloch. – Ah ! par exemple ! s'écria M. Nissim Bernard, si je m'attendais à dîner avec le fils de mon ami ! Mais j'ai à Paris chez moi, une photographie de votre père et combien de lettres de lui. Il m'appelait toujours « mon oncle », on n'a jamais su pourquoi. C'était un homme charmant, étincelant. Je me rappelle un dîner chez moi, à Nice, où il y avait Sardou, Labiche, Augier... – Molière, Racine, Corneille, continua ironiquement Bloch le père dont le fils acheva l'énumération en ajoutant : Plaute, Ménandre, Kalidasa. » M. Nissim Bernard blessé arrêta brusquement son récit et, se privant ascétiquement d'un grand plaisir, resta muet jusqu'à la fin du dîner.

– Saint-Loup au casque d'airain, dit Bloch, reprenez un peu de ce canard aux cuisses lourdes de graisse sur lesquelles l'illustre sacrificateur des volailles a répandu de nombreuses libations de vin rouge.
