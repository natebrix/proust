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
      "canonical_name": "Françoise",
      "surface_forms": [
        "Françoise"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Françoise",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.94,
      "evidence": "« ses larmes cessèrent aussitôt de couler… elle n’eut plus que des ronchonnements… d’affreux sarcasmes »; « des ruses si savantes et si impitoyables… presque tous les jours des asperges… crises d’asthme… elle fut obligée de finir par s’en aller »",
      "explanation": "The narrator exposes Françoise's calculated hardness towards the kitchen girl and her policy of excluding other servants, revealed through sarcasm and cruel ruses."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.94,
      "explanation": "She is strongly diminished locally by the exposure of her methodical cruelty and sarcasm."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-266-p-270"
}

### Candidate characters

[
  "Legrandin",
  "Mme de Cambremer",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Mais le jour où, pendant que le père du narrateur consultait le conseil de famille sur la rencontre de Legrandin, je descendis à la cuisine, était un de ceux où la Charité de Giotto, très malade de son accouchement récent, ne pouvait se lever ; Françoise, n'étant plus aidée, était en retard. Quand je fus en bas, elle était en train, dans l'arrière-cuisine qui donnait sur la basse-cour, de tuer un poulet qui, par sa résistance désespérée et bien naturelle, mais accompagnée par Françoise hors d'elle, tandis qu'elle cherchait à lui fendre le cou sous l'oreille, des cris de « sale bête ! sale bête ! », mettait la sainte douceur et l'onction de notre servante un peu moins en lumière qu'il n'eût fait, au dîner du lendemain, par sa peau brodée d'or comme une chasuble et son jus précieux égoutté d'un ciboire. Quand il fut mort, Françoise recueillit le sang qui coulait sans noyer sa rancune, eut encore un sursaut de colère, et regardant le cadavre de son ennemi, dit une dernière fois : « Sale bête ! » Je remontai tout tremblant ; j'aurais voulu qu'on mît Françoise tout de suite à la porte. Mais qui m'eût fait des boules aussi chaudes, du café aussi parfumé, et même... ces poulets ?... Et en réalité, ce lâche calcul, tout le monde avait eu à le faire comme moi. Car ma tante Léonie savait – ce que j'ignorais encore – que Françoise qui, pour sa fille, pour ses neveux, aurait donné sa vie sans une plainte, était pour d'autres êtres d'une dureté singulière. Malgré cela ma tante l'avait gardée, car si elle connaissait sa cruauté, elle appréciait son service. Je m'aperçus peu à peu que la douceur, la componction, les vertus de Françoise cachaient des tragédies d'arrière-cuisine, comme l'histoire découvre que le règne des Rois et des Reines qui sont représentés les mains jointes dans les vitraux des églises, furent marqués d'incidents sanglants. Je me rendis compte que, en dehors de ceux de sa parenté, les humains excitaient d'autant plus sa pitié par leurs malheurs, qu'ils vivaient plus éloignés d'elle. Les torrents de larmes qu'elle versait en lisant le journal sur les infortunes des inconnus se tarissaient vite si elle pouvait se représenter la personne qui en était l'objet d'une façon un peu précise. Une de ces nuits qui suivirent l'accouchement de la fille de cuisine, celle-ci fut prise d'atroces coliques : la mère du narrateur l'entendit se plaindre, se leva et réveilla Françoise qui, insensible, déclara que tous ces cris étaient une comédie, qu'elle voulait « faire la maîtresse ». Le médecin, qui craignait ces crises, avait mis un signet, dans un livre de médecine que nous avions, à la page où elles sont décrites et où il nous avait dit de nous reporter pour trouver l'indication des premiers soins à donner. Ma mère envoya Françoise chercher le livre en lui recommandant de ne pas laisser tomber le signet. Au bout d'une heure, Françoise n'était pas revenue ; la mère du narrateur indignée crut qu'elle s'était recouchée et me dit d'aller voir moi-même dans la bibliothèque. J'y trouvai Françoise qui, ayant voulu regarder ce que le signet marquait, lisait la description clinique de la crise et poussait des sanglots maintenant qu'il s'agissait d'une malade-type qu'elle ne connaissait pas. À chaque symptôme douloureux mentionné par l'auteur du traité, elle s'écriait : « Hé là ! Sainte Vierge, est-il possible que le bon Dieu veuille faire souffrir ainsi une malheureuse créature humaine ? Hé ! la pauvre ! »

### Passage

Mais dès que je l'eus appelée et qu'elle fut revenue près du lit de la Charité de Giotto, ses larmes cessèrent aussitôt de couler ; elle ne put reconnaître ni cette agréable sensation de pitié et d'attendrissement qu'elle connaissait bien et que la lecture des journaux lui avait souvent donnée, ni aucun plaisir de même famille ; dans l'ennui et dans l'irritation de s'être levée au milieu de la nuit pour la fille de cuisine, et à la vue des mêmes souffrances dont la description l'avait fait pleurer, elle n'eut plus que des ronchonnements de mauvaise humeur, même d'affreux sarcasmes, disant, quand elle crut que nous étions partis et ne pouvions plus l'entendre : « Elle n'avait qu'à ne pas faire ce qu'il faut pour ça ! ça lui a fait plaisir ! qu'elle ne fasse pas de manières maintenant. Faut-il tout de même qu'un garçon ait été abandonné du bon Dieu pour aller avec ça. Ah ! c'est bien comme on disait dans le patois de ma pauvre mère :

« Qui du cul d'un chien s'amourose

Il lui paraît une rose. »

Si, quand son petit-fils était un peu enrhumé du cerveau, elle partait la nuit, même malade, au lieu de se coucher, pour voir s'il n'avait besoin de rien, faisant quatre lieues à pied avant le jour afin d'être rentrée pour son travail, en revanche ce même amour des siens et son désir d'assurer la grandeur future de sa maison se traduisait dans sa politique à l'égard des autres domestiques par une maxime constante qui fut de n'en jamais laisser un seul s'implanter chez ma tante, qu'elle mettait d'ailleurs une sorte d'orgueil à ne laisser approcher par personne, préférant, quand elle-même était malade, se relever pour lui donner son eau de Vichy plutôt que de permettre l'accès de la chambre de sa maîtresse à la fille de cuisine. Et comme cet hyménoptère observé par Fabre, la guêpe fouisseuse, qui pour que ses petits après sa mort aient de la viande fraîche à manger, appelle l'anatomie au secours de sa cruauté et, ayant capturé des charançons et des araignées, leur perce avec un savoir et une adresse merveilleux le centre nerveux d'où dépend le mouvement des pattes, mais non les autres fonctions de la vie, de façon que l'insecte paralysé près duquel elle dépose ses oeufs, fournisse aux larves, quand elles écloront un gibier docile, inoffensif, incapable de fuite ou de résistance, mais nullement faisandé, Françoise trouvait pour servir sa volonté permanente de rendre la maison intenable à tout domestique, des ruses si savantes et si impitoyables que, bien des années plus tard, nous apprîmes que si cet été-là nous avions mangé presque tous les jours des asperges, c'était parce que leur odeur donnait à la pauvre fille de cuisine chargée de les éplucher des crises d'asthme d'une telle violence qu'elle fut obligée de finir par s'en aller.

Hélas ! nous devions définitivement changer d'opinion sur Legrandin. Un des dimanches qui suivit la rencontre sur le Pont-Vieux après laquelle mon père avait dû confesser son erreur, comme la messe finissait et qu'avec le soleil et le bruit du dehors quelque chose de si peu sacré entrait dans l'église que Mme Goupil, Mme Percepied (toutes les personnes qui tout à l'heure, à mon arrivée un peu en retard, étaient restées les yeux absorbés dans leur prière et que j'aurais même pu croire ne m'avoir pas vu entrer si, en même temps, leurs pieds n'avaient repoussé légèrement le petit banc qui m'empêchait de gagner ma chaise) commençaient à s'entretenir avec nous à haute voix de sujets tout temporels comme si nous étions déjà sur la place, nous vîmes sur le seuil brûlant du porche, dominant le tumulte bariolé du marché, Legrandin, que le mari de cette dame avec qui nous l'avions dernièrement rencontré était en train de présenter à la femme d'un autre gros propriétaire terrien des environs. La figure de Legrandin exprimait une animation, un zèle extraordinaires ; il fit un profond salut avec un renversement secondaire en arrière, qui ramena brusquement son dos au delà de la position de départ et qu'avait dû lui apprendre le mari de sa soeur, Mme de Cambremer. Ce redressement rapide fit refluer en une sorte d'onde fougueuse et musclée la croupe de Legrandin que je ne supposais pas si charnue ; et je ne sais pourquoi cette ondulation de pure matière, ce flot tout charnel, sans expression de spiritualité et qu'un empressement plein de bassesse fouettait en tempête, éveillèrent tout d'un coup dans mon esprit la possibilité d'un Legrandin tout différent de celui que nous connaissions. Cette dame le pria de dire quelque chose à son cocher, et tandis qu'il allait jusqu'à la voiture, l'empreinte de joie timide et dévouée que la présentation avait marquée sur son visage y persistait encore. Ravi dans une sorte de rêve, il souriait, puis il revint vers la dame en se hâtant et, comme il marchait plus vite qu'il n'en avait l'habitude, ses deux épaules oscillaient de droite et de gauche ridiculement, et il avait l'air tant il s'y abandonnait entièrement en n'ayant plus souci du reste, d'être le jouet inerte et mécanique du bonheur. Cependant, nous sortions du porche, nous allions passer à côté de lui, il était trop bien élevé pour détourner la tête, mais il fixa de son regard soudain chargé d'une rêverie profonde un point si éloigné de l'horizon qu'il ne put nous voir et n'eut pas à nous saluer. Son visage restait ingénu au-dessus d'un veston souple et droit qui avait l'air de se sentir fourvoyé malgré lui au milieu d'un luxe détesté. Et une lavallière à pois qu'agitait le vent de la Place continuait à flotter sur Legrandin comme l'étendard de son fier isolement et de sa noble indépendance. Au moment où nous arrivions à la maison, maman s'aperçut qu'on avait oublié le saint-honoré et demanda à mon père de retourner avec moi sur nos pas dire qu'on l'apportât tout de suite. Nous croisâmes près de l'église Legrandin qui venait en sens inverse conduisant la même dame à sa voiture. Il passa contre nous, ne s'interrompit pas de parler à sa voisine, et nous fit du coin de son oeil bleu un petit signe en quelque sorte intérieur aux paupières et qui, n'intéressant pas les muscles de son visage, put passer parfaitement inaperçu de son interlocutrice ; mais, cherchant à compenser par l'intensité du sentiment le champ un peu étroit où il en circonscrivait l'expression, dans ce coin d'azur qui nous était affecté il fit pétiller tout l'entrain de la bonne grâce qui dépassa l'enjouement, frisa la malice ; il subtilisa les finesses de l'amabilité jusqu'aux clignements de la connivence, aux demi-mots, aux sous-entendus, aux mystères de la complicité ; et finalement exalta les assurances d'amitié jusqu'aux protestations de tendresse, jusqu'à la déclaration d'amour, illuminant alors pour nous seuls, d'une langueur secrète et invisible à la châtelaine, une prunelle énamourée dans un visage de glace.
