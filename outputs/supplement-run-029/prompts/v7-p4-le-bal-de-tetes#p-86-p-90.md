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
      "canonical_name": "la Berma",
      "surface_forms": [
        "la Berma",
        "Berma"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "la Berma",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "la Berma était, comme on dit, à cent pics au-dessus de Rachel, et le temps ... avait consacré son génie.",
      "explanation": "The narrator explicitly ranks Berma far above Rachel and affirms that time has consecrated her genius."
    }
  ],
  "status_effects": [
    {
      "character": "la Berma",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "She is strongly elevated as a consecrated genius, explicitly set far above Rachel."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-86-p-90"
}

### Candidate characters

[
  "Bergotte",
  "Bloch",
  "Brichot",
  "Elstir",
  "Gilberte",
  "M. de Marsantes",
  "Mme de Cambremer",
  "Mme de Villeparisis",
  "Robert de Saint-Loup",
  "Swann",
  "duc de Guermantes",
  "duchesse de Guermantes",
  "le directeur",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

« C'était bien beau », dit-il à Rachel, et ayant dit ces simples mots, son désir étant satisfait, il repartit et fit tant de bruit pour regagner sa place que Rachel dut attendre plus de cinq minutes avant de réciter la seconde poésie. Quand elle eut fini celle-ci, les Deux Pigeons, Mme de Monrienval s'approcha de Mme de Robert de Saint-Loup, qu'elle savait fort lettrée sans se rappeler assez qu'elle avait l'esprit subtil et sarcastique de son père, et lui demanda : « C'est bien la fable de La Fontaine, n'est-ce pas ? » croyant bien l'avoir reconnue mais n'étant pas absolument certaine, car elle connaissait fort mal les fables de La Fontaine et, de plus, croyait que c'était des choses d'enfants qu'on ne récitait pas dans le monde. Pour avoir un tel succès l'artiste avait sans doute pastiché des fables de La Fontaine, pensait la bonne dame. Or, Gilberte, jusque-là impassible, l'enfonça sans le vouloir dans cette idée, car n'aimant pas Rachel et voulant dire qu'il ne restait rien des fables avec une diction pareille, elle le dit de cette nuance trop subtile qui était celle de son père et qui laissait les personnes naïves dans le doute sur ce qu'il voulait dire. Généralement plus moderne, quoique fille de Swann – comme un canard couvé par une poule – elle était assez lakiste et se contentait de dire : « Je trouve d'un touchant, c'est d'une sensibilité charmante. » Mais à Mme de Morienval Gilberte répondit sous cette forme fantaisiste de Swann à laquelle se trompaient les gens qui prennent tout au pied de la lettre : « Un quart est de l'invention de l'interprète, un quart de la folie, un quart n'a aucun sens, le reste est de La Fontaine », ce qui permit à Mme de Morienval de soutenir que ce qu'on venait d'entendre n'était pas les Deux Pigeons de La Fontaine mais un arrangement où tout au plus un quart était de La Fontaine, ce qui n'étonna personne, vu l'extraordinaire ignorance de ce public.

### Passage

Mais un des amis de Bloch étant arrivé en retard, celui-ci eut la joie de lui demander s'il n'avait jamais entendu Rachel, de lui faire une peinture extraordinaire de sa diction, en exagérant et en trouvant tout d'un coup à raconter, à révéler à autrui cette diction moderniste, un plaisir étrange, qu'il n'avait nullement éprouvé à l'entendre. Puis Bloch, avec une émotion exagérée, félicita de nouveau Rachel sur un ton de fausset et de proclamer son génie, présenta son ami qui déclara n'admirer personne autant qu'elle, et Rachel, qui connaissait maintenant des dames de la haute société et, sans s'en rendre compte, les copiait, répondit : « Oh ! je suis très flattée, très honorée par votre appréciation. » L'ami de Bloch lui demanda ce qu'elle pensait de la Berma. « Pauvre femme, il paraît qu'elle est dans la dernière misère. Elle n'a pas été, je ne dirai pas sans talent, car ce n'était pas au fond du vrai talent, elle n'aimait que des horreurs, mais enfin elle a été utile, certainement ; elle jouait d'une façon assez vivante, et puis c'était une brave personne, généreuse, qui s'est ruinée pour les autres. Voilà bien longtemps qu'elle ne fait plus un sou, parce que le public n'aime pas du tout ce qu'elle fait. Du reste, ajouta-t-elle en riant, je vous dirai que mon âge ne m'a permis de l'entendre, naturellement, que tout à fait dans les derniers temps et quand j'étais moi-même trop jeune pour me rendre compte. – Elle ne disait pas très bien les vers ? hasarda l'ami de Bloch pour flatter Rachel, qui répondit : – Oh ! ça, elle n'a jamais su en dire un ; c'était de la prose, du chinois, du volapük, tout, excepté un vers. D'ailleurs, je vous dirai que, bien entendu, je ne l'ai entendue que très peu, sur sa fin, ajouta-t-elle pour se rajeunir, mais on m'a dit qu'autrefois ce n'était pas mieux, au contraire. »

Je me rendais compte que le temps qui passe n'amène pas forcément le progrès dans les arts. Et de même que tel auteur du XVIIe siècle, qui n'a connu ni la Révolution française, ni les découvertes scientifiques, ni la guerre, peut être supérieur à tel écrivain d'aujourd'hui, et que peut-être même Fagon était un aussi grand médecin que du Boulbon (la supériorité du génie compensant ici l'infériorité du savoir), de même la Berma était, comme on dit, à cent pics au-dessus de Rachel, et le temps, en la mettant en vedette en même temps qu'Elstir, avait consacré son génie.

Il ne faut pas s'étonner que l'ancienne maîtresse de Saint-Loup débinât la Berma. Elle l'eût fait quand elle était jeune. Ne l'eût-elle pas fait alors, qu'elle l'eût fait maintenant. Qu'une femme du monde de la plus haute intelligence, de la plus grande bonté se fasse actrice, déploie dans ce métier nouveau pour elle de grands talents, n'y rencontre que des succès, on s'étonnera, si on se trouve auprès d'elle après longtemps, d'entendre non son langage à elle, mais celui des comédiennes, leur rosserie spéciale envers les camarades, tout ce qu'ajoutent à l'être humain, quand ils ont passé sur lui, « trente ans de théâtre ». Rachel se comportait de même tout en ne sortant pas du monde.

Mme de Guermantes, au déclin de sa vie, avait senti s'éveiller en soi des curiosités nouvelles. Le monde n'avait plus rien à lui apprendre. L'idée qu'elle y avait la première place était, nous l'avons vu, aussi évidente pour elle que la hauteur du ciel bleu par-dessus la terre. Elle ne croyait pas avoir à affermir une position qu'elle jugeait inébranlable. En revanche, lisant, allant au théâtre, elle eût souhaité avoir un prolongement de ces lectures, de ces spectacles ; comme jadis dans l'étroit petit jardin où on prenait de l'orangeade, tout ce qu'il y avait de plus exquis dans le grand monde venait familièrement, parmi les brises parfumées du soir et les nuages de pollen, entretenir en elle le goût du grand monde, de même maintenant un autre appétit lui faisait souhaiter savoir les raisons de telle polémique littéraire, connaître des auteurs, voir des actrices. Son esprit fatigué réclamait une nouvelle alimentation. Elle se rapprocha, pour connaître les uns et les autres, de femmes avec qui jadis elle n'eût pas voulu échanger de cartes et qui faisaient valoir leur intimité avec le directeur de telle revue dans l'espoir d'avoir la duchesse. La première actrice invitée crut être la seule dans un milieu extraordinaire, lequel parut plus médiocre à la seconde quand elle vit celle qui l'y avait précédée. La duchesse, parce qu'à certains soirs elle recevait des souverains, croyait que rien n'était changé à sa situation. En réalité, elle, la seule d'un sang vraiment sans alliage, elle qui, étant née Guermantes, pouvait signer : Guermantes – Guermantes quand elle ne signait pas : la Mme de Guermantes – elle qui à ses belles-soeurs mêmes semblait quelque chose de plus précieux que tout, comme un Moïse sauvé des eaux, un Christ échappé en Égypte, un Louis XVII enfui du Temple, le pur du pur, maintenant sacrifiant sans doute à ce besoin héréditaire de nourriture spirituelle qui avait fait la décadence sociale de Mme de Villeparisis, elle était devenue elle-même une Mme de Villeparisis, chez qui les femmes snobs redoutaient de rencontrer telle ou tel, et de laquelle les jeunes gens, constatant le fait accompli sans savoir ce qui l'a précédé, croyaient que c'était une Guermantes d'une moins bonne cuvée, d'une moins bonne année, une Guermantes déclassée. Dans les milieux nouveaux qu'elle fréquentait, restée bien plus la même qu'elle ne croyait, elle continuait à croire que s'ennuyer facilement était une supériorité intellectuelle, mais elle l'exprimait avec une sorte de violence qui donnait à sa voix quelque chose de rauque. Comme je lui parlais de Brichot : « Il m'a assez embêtée pendant vingt ans », et comme Mme de Cambremer disait : « Relisez ce que Schopenhauer dit de la musique », elle nous fit remarquer cette phrase en disant avec violence : « Relisez est un chef-d'oeuvre ! Ah ! non, ça, par exemple, il ne faut pas nous la faire. » Alors le vieux d'Albon sourit en reconnaissant une des formes de l'esprit Guermantes.

« On peut dire ce qu'on veut, c'est admirable, cela a de la ligne, du caractère, c'est intelligent, personne n'a jamais dit les vers comme ça », dit la duchesse en parlant de Rachel, craignant que Gilberte ne la débinât. Celle-ci s'éloigna vers un autre groupe pour éviter un conflit avec sa tante, laquelle, d'ailleurs, ne dit sur Rachel que des choses fort ordinaires. Mais puisque les meilleurs écrivains cessent souvent aux approches de la vieillesse, ou après un excès de production, d'avoir du talent, on peut bien excuser les femmes du monde de cesser, à partir d'un certain moment, d'avoir de l'esprit. Swann ne retrouvait plus dans l'esprit dur de la Mme de Guermantes le « fondu » de la jeune princesse des Laumes. Sur le tard, fatiguée au moindre effort, Mme de Guermantes disait énormément de bêtises. Certes, à tout moment et bien des fois au cours même de cette matinée, elle redevenait la femme que j'avais connue et parlait des choses mondaines avec esprit. Mais à côté de cela, bien souvent il arrivait que cette parole pétillante sous un beau regard, et qui pendant tant d'années avait tenu sous son sceptre spirituel les hommes les plus éminents de Paris, scintillât encore mais, pour ainsi dire, à vide. Quand le moment de placer un mot venait, elle s'interrompait pendant le même nombre de secondes qu'autrefois, elle avait l'air d'hésiter, de produire, mais le mot qu'elle lançait alors ne valait rien. Combien peu de personnes, d'ailleurs, s'en apercevaient, la continuité du procédé leur faisant croire à la survivance de l'esprit, comme il arrive à ces gens qui, superstitieusement attachés à une marque de pâtisserie, continuent à faire venir leurs petits fours d'une même maison sans s'apercevoir qu'ils sont devenus détestables. Déjà, pendant la guerre, la duchesse avait donné des marques de cet affaiblissement. Si quelqu'un disait le mot culture, elle l'arrêtait, souriait, allumait son beau regard, et lançait : « la KKKKultur », ce qui faisait rire les amis, qui croyaient retrouver là l'esprit des Guermantes. Et certes, c'était le même moule, la même intonation, le même sourire qui avaient jadis ravi Bergotte, lequel, du reste, s'il avait vécu, eût aussi gardé ses coupes de phrase, ses interjections, ses points suspensifs, ses épithètes, mais pour ne rien dire. Mais les nouveaux venus s'étonnaient et parfois disaient, s'ils n'étaient pas tombés un jour où elle était drôle et en pleine possession de ses moyens : « Comme elle est bête ! » La duchesse, d'ailleurs, s'arrangeait pour canaliser son encanaillement et ne pas le laisser s'étendre à celles des personnes de sa famille desquelles elle tirait une gloire aristocratique. Si au théâtre elle avait, pour remplir son rôle de protectrice des arts, invité un ministre ou un peintre et que celui-ci ou celui-là lui demandât naïvement si sa belle-soeur ou son mari n'étaient pas dans la salle, la duchesse, timorée, avec les apparences superbes de l'audace, répondait insolemment : « Je n'en sais rien. Dès que je sors de chez moi, je ne sais plus ce que fait ma famille. Pour tous les hommes politiques, pour tous les artistes, je suis veuve. » Ainsi s'évitait-elle que le parvenu trop empressé s'attirât des rebuffades – et lui attirât à elle-même des réprimandes – de M. de Marsantes et de duc de Guermantes.
