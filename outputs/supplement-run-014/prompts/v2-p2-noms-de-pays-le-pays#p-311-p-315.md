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
      "canonical_name": "Albertine",
      "surface_forms": [
        "la petite Simonet",
        "Simonet"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.9
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "Albertine",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "« C'est une amie de la petite Simonet » ... On avait senti ... une curiosité de mieux regarder la personne favorisée ... Un privilège assurément qui ne paraissait pas donné à tout le monde. Car l'aristocratie est une chose relative.",
      "explanation": "Within the local beach microcosm, being 'the friend of the little Simonet' confers distinction, implying that Albertine herself occupies a small-scale aristocratic position."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.8,
      "explanation": "Her name functions as a prestige marker; association with her is treated as a privilege, which locally elevates her standing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-311-p-315"
}

### Candidate characters

[
  "Françoise",
  "la grand-mère",
  "le directeur",
  "le narrateur"
]

### Prior local context (optional)

D'ailleurs, il n'y avait même pas besoin pour rentrer de quitter la digue et de pénétrer dans l'hôtel par le hall, c'est-à-dire par derrière. En vertu d'une avance comparable à celle du samedi où à Combray on déjeunait une heure plus tôt, maintenant avec le plein de l'été les jours étaient devenus si longs que le soleil était encore haut dans le ciel, comme à une heure de goûter, quand on mettait le couvert pour le dîner au Grand-Hôtel de Balbec. Aussi les grandes fenêtres vitrées et à coulisses restaient-elles ouvertes de plain-pied avec la digue. Je n'avais qu'à enjamber un mince cadre de bois pour me trouver dans la salle à manger que je quittais aussitôt pour prendre l'ascenseur.

### Passage

En passant devant le bureau j'adressai un sourire au directeur, et sans l'ombre de dégoût, en recueillis un dans sa figure que, depuis que j'étais à Balbec, mon attention compréhensive injectait et transformait peu à peu comme une préparation d'histoire naturelle. Ses traits m'étaient devenus courants, chargés d'un sens médiocre, mais intelligible comme une écriture qu'on lit et ne ressemblaient plus en rien à ces caractères bizarres, intolérables que son visage m'avait présentés ce premier jour, où j'avais vu devant moi un personnage maintenant oublié, ou, si je parvenais à l'évoquer, méconnaissable, difficile à identifier avec la personnalité insignifiante et polie dont il n'était que la caricature, hideuse et sommaire. Sans la timidité ni la tristesse du soir de mon arrivée, je sonnai le lift qui ne restait plus silencieux pendant que je m'élevais à côté de lui dans l'ascenseur, comme dans une cage thoracique mobile qui se fût déplacée le long de la colonne montante, mais me répétait :

« Il n'y a plus autant de monde comme il y a un mois. On va commencer à s'en aller, les jours baissent. » Il disait cela, non que ce fût vrai, mais parce qu'ayant un engagement pour une partie plus chaude de la côte, il aurait voulu nous voir partir tous le plus tôt possible afin que l'hôtel fermât et qu'il eût quelques jours à lui, avant de « rentrer » dans sa nouvelle place. Rentrer et « nouvelle » n'étaient du reste pas des expressions contradictoires car, pour le lift, « rentrer » était la forme usuelle du verbe « entrer ». La seule chose qui m'étonnât était qu'il condescendît à dire « place », car il appartenait à ce prolétariat moderne qui désire effacer dans le langage la trace du régime de la domesticité. Du reste, au bout d'un instant, il m'apprit que dans la « situation » où il allait « rentrer », il aurait une plus jolie « tunique » et un meilleur « traitement » ; les mots « livrée » et « gages » lui paraissaient désuets et inconvenants. Et comme, par une contradiction absurde, le vocabulaire a, malgré tout, chez les « patrons », survécu à la conception de l'inégalité, je comprenais toujours mal ce que me disait le lift. Ainsi la seule chose qui m'intéressât était de savoir si ma grand'mère était à l'hôtel. Or, prévenant mes questions, le lift me disait : « Cette dame vient de sortir de chez vous. » J'y étais toujours pris, je croyais que c'était ma grand-mère. « Non, cette dame qui est je crois employée chez vous. » Comme dans l'ancien langage bourgeois, qui devrait bien être aboli, une cuisinière ne s'appelle pas une employée, je pensais un instant : « Mais il se trompe, nous ne possédons ni usine, ni employés. » Tout d'un coup, je me rappelais que le nom d'employé est comme le port de la moustache pour les garçons de café, une satisfaction d'amour-propre donnée aux domestiques et que cette dame qui venait de sortir était Françoise (probablement en visite à la caféterie ou en train de regarder coudre la femme de chambre de la dame belge), satisfaction qui ne suffisait pas encore au lift car il disait volontiers en s'apitoyant sur sa propre classe : « chez l'ouvrier » ou « chez le petit », se servant du même singulier que Racine quand il dit : « le pauvre... ». Mais d'habitude, car mon zèle et ma timidité du premier jour étaient loin, je ne parlais plus au lift. C'était lui maintenant qui restait sans recevoir de réponses dans la courte traversée dont il filait les noeuds à travers l'hôtel, évidé comme un jouet et qui déployait autour de nous, étage par étage, ses ramifications de couloirs dans les profondeurs desquels la lumière se veloutait, se dégradait, amincissait les portes de communication ou les degrés des escaliers intérieurs qu'elle convertissait en cette ambre dorée, inconsistante et mystérieuse comme un crépuscule, où Rembrandt découpe tantôt l'appui d'une fenêtre ou la manivelle d'un puits. Et à chaque étage une lueur d'or reflétée sur le tapis annonçait le coucher du soleil et la fenêtre des cabinets.

Je me demandais si les jeunes filles que je venais de voir habitaient Balbec et qui elles pouvaient être. Quand le désir est ainsi orienté vers une petite tribu humaine qu'il sélectionne, tout ce qui peut se rattacher à elle devient motif d'émotion, puis de rêverie. J'avais entendu une dame dire sur la digue : « C'est une amie de la petite Simonet » avec l'air de précision avantageuse de quelqu'un qui explique : « C'est le camarade inséparable du petit La Rochefoucauld. » Et aussitôt on avait senti sur la figure de la personne à qui on apprenait cela une curiosité de mieux regarder la personne favorisée qui était « amie de la petite Simonet ». Un privilège assurément qui ne paraissait pas donné à tout le monde. Car l'aristocratie est une chose relative. Et il y a des petits trous pas cher où le fils d'un marchand de meubles est prince des élégances et règne sur une cour comme un jeune prince de Galles. J'ai souvent cherché depuis à me rappeler comment avait résonné pour moi, sur la plage, ce nom de Simonet, encore incertain alors dans sa forme que j'avais mal distinguée, et aussi quant à sa signification, à la désignation par lui de telle personne, ou peut-être de telle autre ; en somme empreint de ce vague et de cette nouveauté si émouvants pour nous dans la suite, quand ce nom, dont les lettres sont à chaque seconde plus profondément gravées en nous par notre attention incessante, est devenu (ce qui ne devait arriver pour moi, à l'égard de la petite Simonet, que quelques années plus tard) le premier vocable que nous retrouvions, soit au moment du réveil, soit après un évanouissement, même avant la notion de l'heure qu'il est, du lieu où nous sommes, presque avant le mot « je », comme si l'être qu'il nomme était plus nous que nous-même, et comme si après quelques moments d'inconscience, la trêve qui expire avant toute autre est celle pendant laquelle on ne pensait pas à lui.

Je ne sais pourquoi je me dis dès le premier jour que le nom de Simonet devait être celui d'une des jeunes filles ; je ne cessai plus de me demander comment je pourrais connaître la famille Simonet ; et cela par des gens qu'elle jugeât supérieurs à elle-même, ce qui ne devait pas être difficile si ce n'étaient que de petites grues du peuple, pour qu'elle ne pût avoir une idée dédaigneuse de moi. Car on ne peut avoir de connaissance parfaite, on ne peut pratiquer l'absorption complète de qui vous dédaigne, tant qu'on n'a pas vaincu ce dédain. Or, chaque fois que l'image de femmes si différentes pénètre en nous, à moins que l'oubli ou la concurrence d'autres images ne l'élimine, nous n'avons de repos que nous n'ayons converti ces étrangères en quelque chose qui soit pareil à nous, notre âme étant à cet égard douée du même genre de réaction et d'activité que notre organisme physique, lequel ne peut tolérer l'immixtion dans son sein d'un corps étranger sans qu'il s'exerce aussitôt à digérer et assimiler l'intrus ; la petite Simonet devait être la plus jolie de toutes – celle, d'ailleurs, qui, me semblait-il, aurait pu devenir ma maîtresse, car elle était la seule qui à deux ou trois reprises, détournant à demi la tête, avait paru prendre conscience de mon fixe regard. Je demandai au lift s'il ne connaissait pas à Balbec des Simonet. N'aimant pas à dire qu'il ignorait quelque chose il répondit qu'il lui semblait avoir entendu causer de ce nom-là. Arrivé au dernier étage, je le priai de me faire apporter les dernières listes d'étrangers.

Je sortis de l'ascenseur, mais au lieu d'aller vers ma chambre je m'engageai plus avant dans le couloir, car à cette heure-là le valet de chambre de l'étage, quoiqu'il craignît les courants d'air, avait ouvert la fenêtre du bout, laquelle regardait, au lieu de la mer, le côté de la colline et de la vallée, mais ne les laissait jamais voir, car ses vitres, d'un verre opaque, étaient le plus souvent fermées. Je m'arrêtai devant elle en une courte station et le temps de faire mes dévotions à la « vue » que pour une fois elle découvrait au delà de la colline à laquelle était adossé l'hôtel et qui ne contenait qu'une maison posée à quelque distance, mais à laquelle la perspective et la lumière du soir en lui conservant son volume donnait une ciselure précieuse et un écrin de velours, comme à une de ces architectures en miniature, petit temple ou petite chapelle d'orfèvrerie et d'émaux qui servent de reliquaires et qu'on n'expose qu'à de rares jours à la vénération des fidèles. Mais cet instant d'adoration avait déjà trop duré, car le valet de chambre qui tenait d'une main un trousseau de clefs et de l'autre me saluait en touchant sa calotte de sacristain, mais sans la soulever à cause de l'air pur et frais du soir, venait refermer comme ceux d'une châsse les deux battants de la croisée et dérobait à mon adoration le monument réduit et la relique d'or. J'entrai dans ma chambre.
