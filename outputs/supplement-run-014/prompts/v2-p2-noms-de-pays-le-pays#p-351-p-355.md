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
      "canonical_name": "Elstir",
      "surface_forms": [
        "Elstir"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Elstir",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.74,
      "evidence": "« vécut-il seul, non par indifférence, mais par amour des autres »; « en produisant, lui avait vécu pour lui-même, loin de la société à laquelle il était indifférent; la pratique de la solitude lui en avait donné l'amour »",
      "explanation": "The narrator reframes Elstir’s solitude as disinterested and self-sustaining rather than misanthropic, suggesting he first sought to give others a higher idea of him through his works and then came to love solitude for itself. This sympathetic account raises Elstir’s standing."
    }
  ],
  "status_effects": [
    {
      "character": "Elstir",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.74,
      "explanation": "Elstir is locally valued as noble and independent in motive, countering prior social misreadings of his isolation."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-351-p-355"
}

### Candidate characters

[
  "Gilberte",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Dans les quelques mots qu'Elstir vint nous dire, en s'asseyant à notre table, il ne me répondit jamais, les diverses fois où je lui parlai de Swann. Je commençai à croire qu'il ne le connaissait pas. Il ne m'en demanda pas moins d'aller le voir à son atelier de Balbec, invitation qu'il n'adressa pas à Robert de Saint-Loup, et que me valurent, ce que n'aurait peut-être pas fait la recommandation de Swann si Elstir eût été lié avec lui (car la part des sentiments désintéressés est plus grande qu'on ne croit dans la vie des hommes), quelques paroles qui lui firent penser que j'aimais les arts. Il prodigua pour moi une amabilité, qui était aussi supérieure à celle de Robert de Saint-Loup que celle-ci à l'affabilité d'un petit bourgeois. À côté de celle d'un grand artiste, l'amabilité d'un grand seigneur, si charmante soit-elle, a l'air d'un jeu d'acteur, d'une simulation. Robert de Saint-Loup cherchait à plaire, Elstir aimait à donner, à se donner. Tout ce qu'il possédait, idées, oeuvres, et le reste qu'il comptait pour bien moins, il l'eût donné avec joie à quelqu'un qui l'eût compris. Mais faute d'une société supportable, il vivait dans un isolement, avec une sauvagerie que les gens du monde appelaient de la pose et de la mauvaise éducation, les pouvoirs publics un mauvais esprit, ses voisins de la folie, sa famille de l'égoïsme et de l'orgueil.

### Passage

Et sans doute les premiers temps avait-il pensé, dans la solitude même, avec plaisir que, par le moyen de ses oeuvres, il s'adressait à distance, il donnait une plus haute idée de lui, à ceux qui l'avaient méconnu ou froissé. Peut-être alors vécut-il seul, non par indifférence, mais par amour des autres, et, comme j'avais renoncé à Gilberte pour lui réapparaître un jour sous des couleurs plus aimables, destinait-il son oeuvre à certains, comme un retour vers eux, où sans le revoir lui-même, on l'aimerait, on l'admirerait, on s'entretiendrait de lui ; un renoncement n'est pas toujours total dès le début, quand nous le décidons avec notre âme ancienne et avant que par réaction il n'ait agi sur nous, qu'il s'agisse du renoncement d'un malade, d'un moine, d'un artiste, d'un héros. Mais s'il avait voulu produire en vue de quelques personnes, en produisant, lui avait vécu pour lui-même, loin de la société à laquelle il était indifférent ; la pratique de la solitude lui en avait donné l'amour comme il arrive pour toute grande chose que nous avons crainte d'abord, parce que nous la savions incompatible avec de plus petites auxquelles nous tenions et dont elle nous prive moins qu'elle ne nous détache. Avant de la connaître, toute notre préoccupation est de savoir dans quelle mesure nous pourrons la concilier avec certains plaisirs qui cessent d'en être dès que nous l'avons connue.

Elstir ne resta pas longtemps à causer avec nous. Je me promettais d'aller à son atelier dans les deux ou trois jours suivants, mais le lendemain de cette soirée, comme j'avais accompagné ma grand-mère tout au bout de la digue vers les falaises de Canapville, en revenant, au coin d'une des petites rues qui débouchent perpendiculairement sur la plage, nous croisâmes une jeune fille qui, tête basse comme un animal qu'on fait rentrer malgré lui dans l'étable, et tenant des clubs de golf, marchait devant une personne autoritaire, vraisemblablement son « anglaise », ou celle de ses amies, laquelle ressemblait au portrait de Jeffries par Hogarth, le teint rouge comme si sa boisson favorite avait été plutôt le gin que le thé, et prolongeant par le croc noir d'un reste de chique une moustache grise, mais bien fournie. La fillette qui la précédait ressemblait à celle de la petite bande qui, sous un polo noir, avait dans un visage immobile et joufflu des yeux rieurs. Or, celle qui rentrait en ce moment avait aussi un polo noir, mais elle me semblait encore plus jolie que l'autre, la ligne de son nez était plus droite, à la base l'aile en était plus large et plus charnue. Puis l'autre m'était apparue comme une fière jeune fille pâle, celle-ci comme une enfant domptée et de teint rose. Pourtant, comme elle poussait une bicyclette pareille et comme elle portait les mêmes gants de renne, je conclus que les différences tenaient peut-être à la façon dont j'étais placé et aux circonstances, car il était peu probable qu'il y eût à Balbec une seconde jeune fille, de visage malgré tout si semblable, et qui dans son accoutrement réunît les mêmes particularités. Elle jeta dans ma direction un regard rapide ; les jours suivants, quand je revis la petite bande sur la plage, et même plus tard quand je connus toutes les jeunes filles qui la composaient, je n'eus jamais la certitude absolue qu'aucune d'elles – même celle qui de toutes lui ressemblait le plus, la jeune fille à la bicyclette – fût bien celle que j'avais vue ce soir-là au bout de la plage, au coin de la rue, jeune fille qui n'était guère, mais qui était tout de même un peu différente de celle que j'avais remarquée dans le cortège.

À partir de cet après-midi-là, moi, qui les jours précédents avais surtout pensé à la grande, ce fut celle aux clubs de golf, présumée être Mlle Simonet, qui recommença à me préoccuper. Au milieu des autres, elle s'arrêtait souvent, forçant ses amies qui semblaient la respecter beaucoup, à interrompre aussi leur marche. C'est ainsi, faisant halte, les yeux brillants sous son « polo » que je la revois encore maintenant silhouettée sur l'écran que lui fait, au fond, la mer, et séparée de moi par un espace transparent et azuré, le temps écoulé depuis lors, première image, toute mince dans mon souvenir, désirée, poursuivie, puis oubliée, puis retrouvée, d'un visage que j'ai souvent depuis projeté dans le passé pour pouvoir me dire d'une jeune fille qui était dans ma chambre : « C'est elle ! »

Mais c'est peut-être encore celle au teint de géranium, aux yeux verts, que j'aurais le plus désiré connaître. Quelle que fût, d'ailleurs, tel jour donné, celle que je préférais apercevoir, les autres, sans celle-là, suffisaient à m'émouvoir ; mon désir même se portant une fois plutôt sur l'une, une fois plutôt sur l'autre, continuait – comme le premier jour ma confuse vision – à les réunir, à faire d'elles le petit monde à part, animé d'une vie commune qu'elles avaient, sans doute, d'ailleurs, la prétention de constituer ; j'eusse pénétré en devenant l'ami de l'une elle – comme un païen raffiné ou un chrétien scrupuleux chez les barbares – dans une société rajeunissante où régnaient la santé, l'inconscience, la volupté, la cruauté, l'inintellectualité et la joie.

Ma grand-mère, à qui j'avais raconté mon entrevue avec Elstir et qui se réjouissait de tout le profit intellectuel que je pouvais tirer de son amitié, trouvait absurde et peu gentil que je ne fusse pas encore allé lui faire une visite. Mais je ne pensais qu'à la petite bande, et incertain de l'heure où ces jeunes filles passeraient sur la digue, je n'osais pas m'éloigner. Ma grand-mère s'étonnait aussi de mon élégance, car je m'étais soudain souvenu des costumes que j'avais jusqu'ici laissés au fond de ma malle. J'en mettais chaque jour un différent, et j'avais même écrit à Paris pour me faire envoyer de nouveaux chapeaux, et de nouvelles cravates.
