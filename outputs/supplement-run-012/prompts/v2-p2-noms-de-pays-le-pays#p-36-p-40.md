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
      "canonical_name": "la grand-mère",
      "surface_forms": [
        "la grand-mère",
        "grand-mère"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "la grand-mère",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "Sa robe de chambre devient « sa blouse de servante et de garde, son habit de religieuse »; la narrateur sent en elle « une pitié plus vaste » et « un désir de conservation et d’accroissement de ma propre vie autrement fort que le mien »; tout ce qu’elle touche est « spiritualisé, sanctifié »; ses réponses aux coups sont d’« une calme autorité ».",
      "explanation": "The narrator extols the kindness, tenderness, and benevolent authority of the grandmother, presenting her as devoted, sanctifying, and self-assured, which strongly elevates her moral and emotional value."
    }
  ],
  "status_effects": [
    {
      "character": "la grand-mère",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "Locally, she is presented as exemplary by her dedication and effective tenderness, which significantly increases her esteem."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-36-p-40"
}

### Candidate characters

[
  "le narrateur"
]

### Prior local context (optional)

Il n'est peut-être rien qui donne plus l'impression de la réalité de ce qui nous est extérieur, que le changement de la position, par rapport à nous, d'une personne même insignifiante, avant que nous l'ayons connue, et après. J'étais le même homme qui avait pris à la fin de l'après-midi le petit chemin de fer de Balbec, je portais en moi la même âme. Mais dans cette âme, à l'endroit où, à six heures, il y avait avec l'impossibilité d'imaginer le directeur, le Palace, son personnel, une attente vague et craintive du moment où j'arriverais, se trouvaient maintenant les boutons extirpés dans la figure du le directeur cosmopolite (en réalité naturalisé Monégasque, bien qu'il fût – comme il disait parce qu'il employait toujours des expressions qu'il croyait distinguées, sans s'apercevoir qu'elles étaient vicieuses – « d'originalité roumaine ») – son geste pour sonner le lift, le lift lui-même, toute une frise de personnages de guignol sortis de cette boîte de Pandore qu'était le Grand-Hôtel, indéniables, inamovibles, et, comme tout ce qui est réalisé, stérilisants. Mais du moins ce changement dans lequel je n'étais pas intervenu me prouvait qu'il s'était passé quelque chose d'extérieur à moi – si dénuée d'intérêt que cette chose fût en soi – et j'étais comme le voyageur qui, ayant eu le soleil devant lui en commençant une course, constate que les heures sont passées quand il le voit derrière lui. J'étais brisé par la fatigue, j'avais la fièvre ; je me serais bien couché, mais je n'avais rien de ce qu'il eût fallu pour cela. J'aurais voulu au moins m'étendre un instant sur le lit, mais à quoi bon puisque je n'aurais pu y faire trouver de repos à cet ensemble de sensations qui est pour chacun de nous son corps conscient, sinon son corps matériel, et puisque les objets inconnus qui l'encerclaient, en le forçant à mettre ses perceptions sur le pied permanent d'une défensive vigilante, auraient maintenu mes regards, mon ouïe, tous mes sens, dans une position aussi réduite et incommode (même si j'avais allongé mes jambes) que celle du cardinal La Balue dans la cage où il ne pouvait ni se tenir debout ni s'asseoir. C'est notre attention qui met des objets dans une chambre, et l'habitude qui les en retire, et nous y fait de la place. De la place, il n'y en avait pas pour moi dans ma chambre de Balbec (mienne de nom seulement), elle était pleine de choses qui, ne me connaissant pas, me rendirent le coup d'oeil méfiant que je leur jetai et, sans tenir aucun compte de mon existence, témoignèrent que je dérangeais le train-train de la leur. La pendule – alors qu'à la maison je n'entendais la mienne que quelques secondes par semaine, seulement quand je sortais d'une profonde méditation – continua sans s'interrompre un instant à tenir dans une langue inconnue des propos qui devaient être désobligeants pour moi, car les grands rideaux violets l'écoutaient sans répondre, mais dans une attitude analogue à celle des gens qui haussent les épaules pour montrer que la vue d'un tiers les irrite. Ils donnaient à cette chambre si haute un caractère quasi historique qui eût pu la rendre appropriée à l'assassinat du duc de Guise, et plus tard à une visite de touristes conduits par un guide de l'agence Cook, mais nullement à mon sommeil. J'étais tourmenté par la présence de petites bibliothèques à vitrines, qui couraient le long des murs, mais surtout par une grande glace à pieds, arrêtée en travers de la pièce et avant le départ de laquelle je sentais qu'il n'y aurait pas pour moi de détente possible. Je levais à tout moment mes regards – que les objets de ma chambre de Paris ne gênaient pas plus que ne faisaient mes propres prunelles, car ils n'étaient plus que des annexes de mes organes, un agrandissement de moi-même – vers le plafond surélevé de ce belvédère situé au sommet de l'hôtel et que la grand-mère avait choisi pour moi ; et, jusque dans cette région plus intime que celle où nous voyons et où nous entendons, dans cette région où nous éprouvons la qualité des odeurs, c'était presque à l'intérieur de mon moi que celle du vétiver venait pousser dans mes derniers retranchements son offensive, à laquelle j'opposais non sans fatigue la riposte inutile et incessante d'un reniflement alarmé. N'ayant plus d'univers, plus de chambre, plus de corps que menacé par les ennemis qui m'entouraient, qu'envahi jusque dans les os par la fièvre, j'étais seul, j'avais envie de mourir. Alors la grand-mère entra ; et à l'expansion de mon coeur refoulé s'ouvrirent aussitôt des espaces infinis.

### Passage

Elle portait une robe de chambre de percale qu'elle revêtait à la maison chaque fois que l'un de nous était malade (parce qu'elle s'y sentait plus à l'aise, disait-elle, attribuant toujours à ce qu'elle faisait des mobiles égoïstes), et qui était pour nous soigner, pour nous veiller, sa blouse de servante et de garde, son habit de religieuse. Mais tandis que les soins de celles-là, la bonté qu'elles ont, le mérite qu'on leur trouve et la reconnaissance qu'on leur doit augmentent encore l'impression qu'on a d'être, pour elles, un autre, de se sentir seul, gardant pour soi la charge de ses pensées, de son propre désir de vivre, je savais, quand j'étais avec ma grand'mère, si grand chagrin qu'il y eût en moi, qu'il serait reçu dans une pitié plus vaste encore ; que tout ce qui était mien, mes soucis, mon vouloir, serait, en ma grand'mère, étayé sur un désir de conservation et d'accroissement de ma propre vie autrement fort que celui que j'avais de moi-même ; et mes pensées se prolongeaient en elle sans subir de déviation parce qu'elles passaient de mon esprit dans le sien sans changer de milieu, de personne. Et – comme quelqu'un qui veut nouer sa cravate devant une glace sans comprendre que le bout qu'il voit n'est pas placé par rapport à lui du côté où il dirige sa main, ou comme un chien qui poursuit à terre l'ombre dansante d'un insecte – trompé par l'apparence du corps comme on l'est dans ce monde où nous ne percevons pas directement les âmes, je me jetai dans les bras de ma grand'mère et je suspendis mes lèvres à sa figure comme si j'accédais ainsi à ce coeur immense qu'elle m'ouvrait. Quand j'avais ainsi ma bouche collée à ses joues, à son front, j'y puisais quelque chose de si bienfaisant, de si nourricier, que je gardais l'immobilité, le sérieux, la tranquille avidité d'un enfant qui tette.

Je regardais ensuite sans me lasser son grand visage découpé comme un beau nuage ardent et calme, derrière lequel on sentait rayonner la tendresse. Et tout ce qui recevait encore, si faiblement que ce fût, un peu de ses sensations, tout ce qui pouvait ainsi être dit encore à elle, en était aussitôt si spiritualisé, si sanctifié que de mes paumes je lissais ses beaux cheveux à peine gris avec autant de respect, de précaution et de douceur que si j'y avais caressé sa bonté. Elle trouvait un tel plaisir dans toute peine qui m'en épargnait une, et, dans un moment d'immobilité et de calme pour mes membres fatigués quelque chose de si délicieux, que quand, ayant vu qu'elle voulait m'aider à me coucher et me déchausser, je fis le geste de l'en empêcher et de commencer à me déshabiller moi-même, elle arrêta d'un regard suppliant mes mains qui touchaient aux premiers boutons de ma veste et de mes bottines.

– Oh, je t'en prie, me dit-elle. C'est une telle joie pour ta grand'mère. Et surtout ne manque pas de frapper au mur si tu as besoin de quelque chose cette nuit, mon lit est adossé au tien, la cloison est très mince. D'ici un moment quand tu seras couché fais-le, pour voir si nous nous comprenons bien.

Et, en effet, ce soir-là, je frappai trois coups – que une semaine plus tard quand je fus souffrant je renouvelai pendant quelques jours tous les matins parce que ma grand'mère voulait me donner du lait de bonne heure. Alors quand je croyais entendre qu'elle était réveillée – pour qu'elle n'attendît pas et pût, tout de suite après, se rendormir – je risquais trois petits coups, timidement, faiblement, distinctement malgré tout, car si je craignais d'interrompre son sommeil dans le cas où je me serais trompé et où elle eût dormi, je n'aurais pas voulu non plus qu'elle continuât d'épier un appel qu'elle n'aurait pas distingué d'abord et que je n'oserais pas renouveler. Et à peine j'avais frappé mes coups que j'en entendais trois autres, d'une intonation différente de ceux-là, empreints d'une calme autorité, répétés à deux reprises pour plus de clarté et qui disaient : « Ne t'agite pas, j'ai entendu, dans quelques instants je serai là » ; et bientôt après ma grand'mère arrivait. Je lui disais que j'avais eu peur qu'elle ne m'entendît pas ou crût que c'était un voisin qui avait frappé ; elle riait :

– Confondre les coups de mon pauvre chou avec d'autres, mais entre mille sa grand'mère les reconnaîtrait ! Crois-tu donc qu'il y en ait d'autres au monde qui soient aussi bêtas, aussi fébriles, aussi partagés entre la crainte de me réveiller et de ne pas être compris. Mais quand même elle se contenterait d'un grattement on reconnaîtrait tout de suite sa petite souris, surtout quand elle est aussi unique et à plaindre que la mienne. Je l'entendais déjà depuis un moment qui hésitait, qui se remuait dans le lit, qui faisait tous ses manèges.
