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
      "confidence": 0.93,
      "evidence": "« sale bête ! »; « insensible, déclara que tous ces cris étaient une comédie »; « Je m’aperçus peu à peu que la douceur, la componction, les vertus de Françoise cachaient des tragédies d’arrière-cuisine »; « lisait la description clinique… et poussait des sanglots… d’une malade-type qu’elle ne connaissait pas ».",
      "explanation": "The passage reveals the cruelty and selective pity of Françoise, contradicting the virtuous image associated with her cooking talents. The narration insists on the gap between the displayed affection and the harshness towards close ones, which clearly diminishes her."
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
      "confidence": 0.93,
      "explanation": "Her local moral value is significantly lowered by the exposure of her hardness and her abstract empathy for strangers, opposed to her actual behavior towards her own."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-261-p-265"
}

### Candidate characters

[
  "Legrandin",
  "Swann",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

N'est-ce pas la fine notation de cette heure-ci ? Vous n'avez peut-être jamais lu Paul Desjardins. Lisez-le, mon enfant ; aujourd'hui il se mue, me dit-on, en frère prêcheur, mais ce fut longtemps un aquarelliste limpide...

### Passage

Les bois sont déjà noirs, le ciel est encor bleu...

Que le ciel reste toujours bleu pour vous, mon jeune ami ; et même à l'heure, qui vient pour moi maintenant, où les bois sont déjà noirs, où la nuit tombe vite, vous vous consolerez comme je fais en regardant du côté du ciel. » Il sortit de sa poche une cigarette, resta longtemps les yeux à l'horizon, « Adieu, les camarades », nous dit-il tout à coup, et il nous quitta.

À cette heure où je descendais apprendre le menu, le dîner était déjà commencé, et Françoise, commandant aux forces de la nature devenues ses aides, comme dans les féeries où les géants se font engager comme cuisiniers, frappait la houille, donnait à la vapeur des pommes de terre à étuver et faisait finir à point par le feu les chefs-d'oeuvre culinaires d'abord préparés dans des récipients de céramistes qui allaient des grandes cuves, marmites, chaudrons et poissonnières, aux terrines pour le gibier, moules à pâtisserie, et petits pots de crème en passant par une collection complète de casserole de toutes dimensions. Je m'arrêtais à voir sur la table, où la fille de cuisine venait de les écosser, les petits pois alignés et nombrés comme des billes vertes dans un jeu ; mais mon ravissement était devant les asperges, trempées d'outre-mer et de rose et dont l'épi, finement pignoché de mauve et d'azur, se dégrade insensiblement jusqu'au pied – encore souillé pourtant du sol de leur plant – par des irisations qui ne sont pas de la terre. Il me semblait que ces nuances célestes trahissaient les délicieuses créatures qui s'étaient amusées à se métamorphoser en légumes et qui, à travers le déguisement de leur chair comestible et ferme, laissaient apercevoir en ces couleurs naissantes d'aurore, en ces ébauches d'arc-en-ciel, en cette extinction de soirs bleus, cette essence précieuse que je reconnaissais encore quand, toute la nuit qui suivait un dîner où j'en avais mangé, elles jouaient, dans leurs farces poétiques et grossières comme une féerie de Shakespeare, à changer mon pot de chambre en un vase de parfum.

La pauvre Charité de Giotto, comme l'appelait Swann, chargée par Françoise de les « plumer », les avait près d'elle dans une corbeille, son air était douloureux, comme si elle ressentait tous les malheurs de la terre ; et les légères couronnes d'azur qui ceignaient les asperges au-dessus de leurs tuniques de rose étaient finement dessinées, étoile par étoile, comme le sont dans la fresque les fleurs bandées autour du front ou piquées dans la corbeille de la Vertu de Padoue. Et cependant, Françoise tournait à la broche un de ces poulets, comme elle seule savait en rôtir, qui avaient porté loin dans Combray l'odeur de ses mérites, et qui, pendant qu'elle nous les servait à table, faisaient prédominer la douceur dans ma conception spéciale de son caractère, l'arôme de cette chair qu'elle savait rendre si onctueuse et si tendre n'étant pour moi que le propre parfum d'une de ses vertus.

Mais le jour où, pendant que mon père consultait le conseil de famille sur la rencontre de Legrandin, je descendis à la cuisine, était un de ceux où la Charité de Giotto, très malade de son accouchement récent, ne pouvait se lever ; Françoise, n'étant plus aidée, était en retard. Quand je fus en bas, elle était en train, dans l'arrière-cuisine qui donnait sur la basse-cour, de tuer un poulet qui, par sa résistance désespérée et bien naturelle, mais accompagnée par Françoise hors d'elle, tandis qu'elle cherchait à lui fendre le cou sous l'oreille, des cris de « sale bête ! sale bête ! », mettait la sainte douceur et l'onction de notre servante un peu moins en lumière qu'il n'eût fait, au dîner du lendemain, par sa peau brodée d'or comme une chasuble et son jus précieux égoutté d'un ciboire. Quand il fut mort, Françoise recueillit le sang qui coulait sans noyer sa rancune, eut encore un sursaut de colère, et regardant le cadavre de son ennemi, dit une dernière fois : « Sale bête ! » Je remontai tout tremblant ; j'aurais voulu qu'on mît Françoise tout de suite à la porte. Mais qui m'eût fait des boules aussi chaudes, du café aussi parfumé, et même... ces poulets ?... Et en réalité, ce lâche calcul, tout le monde avait eu à le faire comme moi. Car ma tante Léonie savait – ce que j'ignorais encore – que Françoise qui, pour sa fille, pour ses neveux, aurait donné sa vie sans une plainte, était pour d'autres êtres d'une dureté singulière. Malgré cela ma tante l'avait gardée, car si elle connaissait sa cruauté, elle appréciait son service. Je m'aperçus peu à peu que la douceur, la componction, les vertus de Françoise cachaient des tragédies d'arrière-cuisine, comme l'histoire découvre que le règne des Rois et des Reines qui sont représentés les mains jointes dans les vitraux des églises, furent marqués d'incidents sanglants. Je me rendis compte que, en dehors de ceux de sa parenté, les humains excitaient d'autant plus sa pitié par leurs malheurs, qu'ils vivaient plus éloignés d'elle. Les torrents de larmes qu'elle versait en lisant le journal sur les infortunes des inconnus se tarissaient vite si elle pouvait se représenter la personne qui en était l'objet d'une façon un peu précise. Une de ces nuits qui suivirent l'accouchement de la fille de cuisine, celle-ci fut prise d'atroces coliques : maman l'entendit se plaindre, se leva et réveilla Françoise qui, insensible, déclara que tous ces cris étaient une comédie, qu'elle voulait « faire la maîtresse ». Le médecin, qui craignait ces crises, avait mis un signet, dans un livre de médecine que nous avions, à la page où elles sont décrites et où il nous avait dit de nous reporter pour trouver l'indication des premiers soins à donner. Ma mère envoya Françoise chercher le livre en lui recommandant de ne pas laisser tomber le signet. Au bout d'une heure, Françoise n'était pas revenue ; ma mère indignée crut qu'elle s'était recouchée et me dit d'aller voir moi-même dans la bibliothèque. J'y trouvai Françoise qui, ayant voulu regarder ce que le signet marquait, lisait la description clinique de la crise et poussait des sanglots maintenant qu'il s'agissait d'une malade-type qu'elle ne connaissait pas. À chaque symptôme douloureux mentionné par l'auteur du traité, elle s'écriait : « Hé là ! Sainte Vierge, est-il possible que le bon Dieu veuille faire souffrir ainsi une malheureuse créature humaine ? Hé ! la pauvre ! »
