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
      "source": "narrator",
      "target": "Gilberte",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.83,
      "evidence": "« le ton despotique avec lequel la mère de Gilberte lui avait parlé … en me la montrant comme forcée d'obéir à quelqu'un, comme n'étant pas supérieure à tout, calma un peu ma souffrance … et diminua mon amour »",
      "explanation": "Seeing Gilberte commanded and obedient lowers her imagined superiority in the narrator’s eyes, briefly cooling his passion."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.83,
      "explanation": "Her perceived need to obey her mother makes her seem less exalted to the narrator, briefly lowering her standing in his eyes."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-301-p-305"
}

### Candidate characters

[
  "Swann",
  "baron de Charlus",
  "le grand-père du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Je la regardai, d'abord de ce regard qui n'est pas que le porte-parole des yeux, mais à la fenêtre duquel se penchent tous les sens, anxieux et pétrifiés, le regard qui voudrait toucher, capturer, emmener le corps qu'il regarde et l'âme avec lui ; puis, tant j'avais peur que d'une seconde à l'autre mon grand-père et le père du narrateur, apercevant cette jeune fille, me fissent éloigner en me disant de courir un peu devant eux, d'un second regard, inconsciemment supplicateur, qui tâchait de la forcer à faire attention à moi, à me connaître ! Elle jeta en avant et de côté ses pupilles pour prendre connaissance de mon grand'père et de le père du narrateur, et sans doute l'idée qu'elle en rapporta fut celle que nous étions ridicules, car elle se détourna, et d'un air indifférent et dédaigneux, se plaça de côté pour épargner à son visage d'être dans leur champ visuel ; et tandis que continuant à marcher et ne l'ayant pas aperçue, ils m'avaient dépassé, elle laissa ses regards filer de toute leur longueur dans ma direction, sans expression particulière, sans avoir l'air de me voir, mais avec une fixité et un sourire dissimulé, que je ne pouvais interpréter d'après les notions que l'on m'avait données sur la bonne éducation que comme une preuve d'outrageant mépris ; et sa main esquissait en même temps un geste indécent, auquel quand il était adressé en public à une personne qu'on ne connaissait pas, le petit dictionnaire de civilité que je portais en moi ne donnait qu'un seul sens, celui d'une intention insolente.

### Passage

– Allons, Gilberte, viens ; qu'est-ce que tu fais, cria d'une voix perçante et autoritaire une dame en blanc que je n'avais pas vue, et à quelque distance de laquelle un monsieur habillé de coutil et que je ne connaissais pas fixait sur moi des yeux qui lui sortaient de la tête ; et cessant brusquement de sourire, la jeune fille prit sa bêche et s'éloigna sans se retourner de mon côté, d'un air docile, impénétrable et sournois.

Ainsi passa près de moi ce nom de Gilberte, donné comme un talisman qui me permettait peut-être de retrouver un jour celle dont il venait de faire une personne et qui, l'instant d'avant, n'était qu'une image incertaine. Ainsi passa-t-il, proféré au-dessus des jasmins et des giroflées, aigre et frais comme les gouttes de l'arrosoir vert ; imprégnant, irisant la zone d'air pur qu'il avait traversée – et qu'il isolait – du mystère de la vie de celle qu'il désignait pour les êtres heureux qui vivaient, qui voyageaient avec elle ; déployant sous l'épinier rose, à hauteur de mon épaule, la quintessence de leur familiarité, pour moi si douloureuse, avec elle, avec l'inconnu de sa vie où je n'entrerais pas.

Un instant (tandis que nous nous éloignions et que mon grand-père murmurait : « Ce pauvre Swann, quel rôle ils lui font jouer : on le fait partir pour qu'elle reste seule avec son Charlus, car c'est lui, je l'ai reconnu ! Et cette petite, mêlée à toute cette infamie ! ») l'impression laissée en moi par le ton despotique avec lequel la mère de Gilberte lui avait parlé sans qu'elle répliquât, en me la montrant comme forcée d'obéir à quelqu'un, comme n'étant pas supérieure à tout, calma un peu ma souffrance, me rendit quelque espoir et diminua mon amour. Mais bien vite cet amour s'éleva de nouveau en moi comme une réaction par quoi mon coeur humilié voulait se mettre de niveau avec Gilberte ou l'abaisser jusqu'à lui. Je l'aimais, je regrettais de ne pas avoir eu le temps et l'inspiration de l'offenser, de lui faire mal, et de la forcer à se souvenir de moi. Je la trouvais si belle que j'aurais voulu pouvoir revenir sur mes pas, pour lui crier en haussant les épaules : « Comme je vous trouve laide, grotesque, comme vous me répugnez ! » Cependant je m'éloignais, emportant pour toujours, comme premier type d'un bonheur inaccessible aux enfants de mon espèce de par des lois naturelles impossibles à transgresser, l'image d'une petite fille rousse, à la peau semée de taches roses, qui tenait une bêche et qui riait en laissant filer sur moi de longs regards sournois et inexpressifs. Et déjà le charme dont son nom avait encensé cette place sous les épines roses où il avait été entendu ensemble par elle et par moi, allait gagner, enduire, embaumer tout ce qui l'approchait, ses grands-parents que les miens avaient eu l'ineffable bonheur de connaître, la sublime profession d'agent de change, le douloureux quartier des Champs-Élysées qu'elle habitait à Paris.

« Léonie, dit mon grand-père en rentrant, j'aurais voulu t'avoir avec nous tantôt. Tu ne reconnaîtrais pas Tansonville. Si j'avais osé, je t'aurais coupé une branche de ces épines roses que tu aimais tant. » Mon grand-père racontait ainsi notre promenade à ma tante Léonie, soit pour la distraire, soit qu'on n'eût pas perdu tout espoir d'arriver à la faire sortir. Or elle aimait beaucoup autrefois cette propriété, et d'ailleurs les visites de Swann avaient été les dernières qu'elle avait reçues, alors qu'elle fermait déjà sa porte à tout le monde. Et de même que, quand il venait maintenant prendre de ses nouvelles (elle était la seule personne de chez nous qu'il demandât encore à voir), elle lui faisait répondre qu'elle était fatiguée, mais qu'elle le laisserait entrer la prochaine fois, de même elle dit ce soir-là : « Oui, un jour qu'il fera beau, j'irai en voiture jusqu'à la porte du parc. » C'est sincèrement qu'elle le disait. Elle eût aimé revoir Swann et Tansonville ; mais le désir qu'elle en avait suffisait à ce qui lui restait de forces ; sa réalisation les eût excédées. Quelquefois le beau temps lui rendait un peu de vigueur, elle se levait, s'habillait ; la fatigue commençait avant qu'elle fût passée dans l'autre chambre et elle réclamait son lit. Ce qui avait commencé pour elle – plus tôt seulement que cela n'arrive d'habitude – c'est ce grand renoncement de la vieillesse qui se prépare à la mort, s'enveloppe dans sa chrysalide, et qu'on peut observer, à la fin des vies qui se prolongent tard, même entre les anciens amants qui se sont le plus aimés, entre les amis unis par les liens les plus spirituels, et qui, à partir d'une certaine année cessent de faire le voyage ou la sortie nécessaire pour se voir, cessent de s'écrire et savent qu'ils ne communiqueront plus en ce monde. Ma tante devait parfaitement savoir qu'elle ne reverrait pas Swann, qu'elle ne quitterait plus jamais la maison, mais cette réclusion définitive devait lui être rendue assez aisée pour la raison même qui, selon nous, aurait dû la lui rendre plus douloureuse : c'est que cette réclusion lui était imposée par la diminution qu'elle pouvait constater chaque jour dans ses forces, et qui, en faisant de chaque action, de chaque mouvement, une fatigue, sinon une souffrance, donnait pour elle à l'inaction, à l'isolement, au silence, la douceur réparatrice et bénie du repos.

Ma tante n'alla pas voir la haie d'épines roses, mais à tous moments je demandais à mes parents si elle n'irait pas, si autrefois elle allait souvent à Tansonville, tâchant de les faire parler des parents et grands-parents de Gilberte qui me semblaient grands comme des Dieux. Ce nom, devenu pour moi presque mythologique, de Swann, quand je causais avec mes parents, je languissais du besoin de le leur entendre dire, je n'osais pas le prononcer moi-même, mais je les entraînais sur des sujets qui avoisinaient Gilberte et sa famille, qui la concernaient, où je ne me sentais pas exilé trop loin d'elle ; et je contraignais tout d'un coup mon père, en feignant de croire par exemple que la charge de mon grand-père avait été déjà avant lui dans notre famille, ou que la haie d'épines roses que voulait voir ma tante Léonie se trouvait en terrain communal, à rectifier mon assertion, à me dire, comme malgré moi, comme de lui-même : « Mais non, cette charge-là était au père de Swann, cette haie fait partie du parc de Swann. » Alors j'étais obligé de reprendre ma respiration, tant, en se posant sur la place où il était toujours écrit en moi, pesait à m'étouffer ce nom qui, au moment où je l'entendais, me paraissait plus plein que tout autre, parce qu'il était lourd de toutes les fois où, d'avance, je l'avais mentalement proféré. Il me causait un plaisir que j'étais confus d'avoir osé réclamer à mes parents, car ce plaisir était si grand qu'il avait dû exiger d'eux pour qu'ils me le procurassent beaucoup de peine, et sans compensation, puisqu'il n'était pas un plaisir pour eux. Aussi je détournais la conversation par discrétion. Par scrupule aussi. Toutes les séductions singulières que je mettais dans ce nom de Swann, je les retrouvais en lui dès qu'ils le prononçaient. Il me semblait alors tout d'un coup que mes parents ne pouvaient pas ne pas les ressentir, qu'ils se trouvaient placés à mon point de vue, qu'ils apercevaient à leur tour, absolvaient, épousaient mes rêves, et j'étais malheureux comme si je les avais vaincus et dépravés.
