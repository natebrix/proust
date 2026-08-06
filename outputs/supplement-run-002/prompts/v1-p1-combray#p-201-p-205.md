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
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Il avait l'air de ne pas oser avoir une opinion »; « se livrer avec une politesse pointilleuse à des occupations dont il professait en même temps qu'elles sont ridicules »; « Je trouvais tout cela contradictoire. »",
      "explanation": "The narrator locally brings Swann down by stressing his refusal to make clear judgments and the contradiction between his remarks and his society life."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "His image is diminished by the insistent description of his defensive irony and his contradictions."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-201-p-205"
}

### Candidate characters

[
  "Bergotte",
  "Gilberte",
  "Odette",
  "baron de Charlus",
  "duchesse de Guermantes",
  "la Berma",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur",
  "prince de Léon"
]

### Prior local context (optional)

– Non monsieur, mes parents ne me permettent pas d'aller au théâtre.

### Passage

– C'est malheureux. Vous devriez leur demander. La Berma dans Phèdre, dans le Cid, ce n'est qu'une actrice si vous voulez, mais vous savez je ne crois pas beaucoup à la « hiérarchie ! » des arts.

(Et je remarquai, comme cela m'avait souvent frappé dans ses conversations avec les soeurs de ma grand'mère, que quand il parlait de choses sérieuses, quand il employait une expression qui semblait impliquer une opinion sur un sujet important, il avait soin de l'isoler dans une intonation spéciale, machinale et ironique, comme s'il l'avait mise entre guillemets, semblant ne pas vouloir la prendre à son compte, et dire : « la hiérarchie, vous savez, comme disent les gens ridicules » ? Mais alors, si c'était ridicule, pourquoi disait-il la hiérarchie ?). Un instant après il ajouta : « Cela vous donnera une vision aussi noble que n'importe quel chef-d'oeuvre, je ne sais pas moi... que – et il se mit à rire – les Reines de Chartres ! » Jusque-là cette horreur d'exprimer sérieusement son opinion m'avait paru quelque chose qui devait être élégant et parisien et qui s'opposait au dogmatisme provincial des soeurs de ma grand'mère ; et je soupçonnais aussi que c'était une des formes de l'esprit dans la coterie où vivait Swann et où par réaction sur le lyrisme des générations antérieures on réhabilitait à l'excès les petits faits précis, réputés vulgaires autrefois, et on proscrivait les « phrases ». Mais maintenant je trouvais quelque chose de choquant dans cette attitude de Swann en face des choses. Il avait l'air de ne pas oser avoir une opinion et de n'être tranquille que quand il pouvait donner méticuleusement des renseignements précis. Mais il ne se rendait donc pas compte que c'était professer l'opinion, postuler que l'exactitude de ces détails avait de l'importance. Je repensai alors à ce dîner où j'étais si triste parce que maman ne devait pas monter dans ma chambre et où il avait dit que les bals chez la princesse de Léon n'avaient aucune importance. Mais c'était pourtant à ce genre de plaisirs qu'il employait sa vie. Je trouvais tout cela contradictoire. Pour quelle autre vie réservait-il de dire enfin sérieusement ce qu'il pensait des choses, de formuler des jugements qu'il pût ne pas mettre entre guillemets, et de ne plus se livrer avec une politesse pointilleuse à des occupations dont il professait en même temps qu'elles sont ridicules ? Je remarquai aussi dans la façon dont Swann me parla de Bergotte quelque chose qui en revanche ne lui était pas particulier, mais au contraire était dans ce temps-là commun à tous les admirateurs de l'écrivain, à l'amie de ma mère, au docteur du Boulbon. Comme Swann, ils disaient de Bergotte : « C'est un charmant esprit, si particulier, il a une façon à lui de dire les choses un peu cherchée, mais si agréable. On n'a pas besoin de voir la signature, on reconnaît tout de suite que c'est de lui. » Mais aucun n'aurait été jusqu'à dire : « C'est un grand écrivain, il a un grand talent. » Ils ne disaient même pas qu'il avait du talent. Ils ne le disaient pas parce qu'ils ne le savaient pas. Nous sommes très longs à reconnaître dans la physionomie particulière d'un nouvel écrivain le modèle qui porte le nom de « grand talent » dans notre musée des idées générales. Justement parce que cette physionomie est nouvelle, nous ne la trouvons pas tout à fait ressemblante à ce que nous appelons talent. Nous disons plutôt originalité, charme, délicatesse, force ; et puis un jour nous nous rendons compte que c'est justement tout cela le talent.

– Est-ce qu'il y a des ouvrages de Bergotte où il ait parlé de la Berma ? demandai-je à Swann.

– Je crois dans sa petite plaquette sur Racine, mais elle doit être épuisée. Il y a peut-être eu cependant une réimpression. Je m'informerai. Je peux d'ailleurs demander à Bergotte tout ce que vous voulez, il n'y a pas de semaine dans l'année où il ne dîne à la maison. C'est le grand ami de ma fille. Ils vont ensemble visiter les vieilles villes, les cathédrales, les châteaux.

Comme je n'avais aucune notion sur la hiérarchie sociale, depuis longtemps l'impossibilité que mon père trouvait à ce que nous fréquentions Mme et Gilberte avait eu plutôt pour effet, en me faisant imaginer entre elles et nous de grandes distances, de leur donner à mes yeux du prestige. Je regrettais que ma mère ne se teignît pas les cheveux et ne se mît pas de rouge aux lèvres comme j'avais entendu dire par notre voisine Mme Sazerat que Odette le faisait pour plaire, non à son mari, mais à Charlus, et je pensais que nous devions être pour elle un objet de mépris, ce qui me peinait surtout à cause de Gilberte qu'on m'avait dit être une si jolie petite fille et à laquelle je rêvais souvent en lui prêtant chaque fois un même visage arbitraire et charmant. Mais quand j'eus appris ce jour-là que Gilberte était un être d'une condition si rare, baignant comme dans son élément naturel au milieu de tant de privilèges, que quand elle demandait à ses parents s'il y avait quelqu'un à dîner, on lui répondait par ces syllabes remplies de lumière, par le nom de ce convive d'or qui n'était pour elle qu'un vieil ami de sa famille : Bergotte ; que, pour elle, la causerie intime à table, ce qui correspondait à ce qu'était pour moi la conversation de ma grand'tante, c'étaient des paroles de Bergotte, sur tous ces sujets qu'il n'avait pu aborder dans ses livres, et sur lesquels j'aurais voulu l'écouter rendre ses oracles ; et qu'enfin, quand elle allait visiter des villes, il cheminait à côté d'elle, inconnu et glorieux, comme les Dieux qui descendaient au milieu des mortels ; alors je sentis en même temps que le prix d'un être comme Gilberte, combien je lui paraîtrais grossier et ignorant, et j'éprouvai si vivement la douceur et l'impossibilité qu'il y aurait pour moi à être son ami, que je fus rempli à la fois de désir et de désespoir. Le plus souvent maintenant quand je pensais à elle, je la voyais devant le porche d'une cathédrale, m'expliquant la signification des statues, et, avec un sourire qui disait du bien de moi, me présentant comme son ami, à Bergotte. Et toujours le charme de toutes les idées que faisaient naître en moi les cathédrales, le charme des coteaux de l'Ile-de-France et des plaines de la Normandie faisait refluer ses reflets sur l'image que je me formais de Gilberte : c'était être tout prêt à l'aimer. Que nous croyions qu'un être participe à une vie inconnue où son amour nous ferait pénétrer, c'est, de tout ce qu'exige l'amour pour naître, ce à quoi il tient le plus, et qui lui fait faire bon marché du reste. Même les femmes qui prétendent ne juger un homme que sur son physique, voient en ce physique l'émanation d'une vie spéciale. C'est pourquoi elles aiment les militaires, les pompiers ; l'uniforme les rend moins difficiles pour le visage ; elles croient baiser sous la cuirasse un coeur différent, aventureux et doux ; et un jeune souverain, un prince héritier, pour faire les plus flatteuses conquêtes, dans les pays étrangers qu'il visite, n'a pas besoin du profil régulier qui serait peut-être indispensable à un coulissier.
