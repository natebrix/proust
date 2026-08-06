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
      "canonical_name": "Bergotte",
      "surface_forms": [
        "Bergotte",
        "le Maître"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bergotte",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« le Bergotte était avant tout quelque élément précieux et vrai, caché au coeur de quelque chose, puis extrait d'elle par ce grand écrivain grâce à son génie » ; « la beauté de leurs phrases est imprévisible » ; « le jour où le jeune Bergotte put montrer ... le salon de mauvais goût ... il monta plus haut ... il les survolait. » Le débit jugé « prétentieux, emphatique et monotone » est expliqué comme le signe même du pouvoir esthétique qui, dans ses livres, « produisait ... la suite des images de l'harmonie. »",
      "explanation": "The narrator requalifies the tiring speech traits (by Norpois and 'one') as the expression of creative originality. He contrasts Bergotte with the imitators and more 'distinguished' socialites, affirming his unpredictable genius and artistic superiority."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Locally, Bergotte is strongly elevated: his singular way of speaking is interpreted as an indicator of genius, and he is said to surpass both the imitators and the more refined friends."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-206-p-210"
}

### Candidate characters

[
  "Brichot",
  "Norpois",
  "Odette",
  "Swann",
  "docteur Cottard",
  "le narrateur"
]

### Prior local context (optional)

Cependant on était passé à table. À côté de mon assiette je trouvai un oeillet dont la tige était enveloppée dans du papier d'argent. Il m'embarrassa moins que n'avait fait l'enveloppe remise dans l'antichambre et que j'avais complètement oubliée. L'usage, pourtant aussi nouveau pour moi, me parut plus intelligible quand je vis tous les convives masculins s'emparer d'un oeillet semblable qui accompagnait leur couvert et l'introduire dans la boutonnière de leur redingote. Je fis comme eux avec cet air naturel d'un libre penseur dans une église, lequel ne connaît pas la messe, mais se lève quand tout le monde se lève et se met à genoux un peu après que tout le monde s'est mis à genoux. Un autre usage inconnu et moins éphémère me déplut davantage. De l'autre côté de mon assiette il y en avait une plus petite remplie d'une matière noirâtre que je ne savais pas être du caviar. J'étais ignorant de ce qu'il fallait en faire, mais résolu à n'en pas manger.

### Passage

Bergotte n'était pas placé loin de moi, j'entendais parfaitement ses paroles. Je compris alors l'impression de Norpois. Il avait en effet un organe bizarre ; rien n'altère autant les qualités matérielles de la voix que de contenir de la pensée : la sonorité des diphtongues, l'énergie des labiales, en sont influencées. La diction l'est aussi. La sienne me semblait entièrement différente de sa manière d'écrire et même les choses qu'il disait de celles qui remplissent ses ouvrages. Mais la voix sort d'un masque sous lequel elle ne suffit pas à nous faire reconnaître d'abord un visage que nous avons vu à découvert dans le style. Dans certains passages de la conversation où Bergotte avait l'habitude de se mettre à parler d'une façon qui ne paraissait pas affectée et déplaisante qu'à Norpois, j'ai été long à découvrir une exacte correspondance avec les parties de ses livres où sa forme devenait si poétique et musicale. Alors il voyait dans ce qu'il disait une beauté plastique indépendante de la signification des phrases, et comme la parole humaine est en rapport avec l'âme, mais sans l'exprimer comme fait le style, Bergotte avait l'air de parler presque à contresens, psalmodiant certains mots et, s'il poursuivait au-dessous d'eux une seule image, les filant sans intervalle comme un même son, avec une fatigante monotonie. De sorte qu'un débit prétentieux, emphatique et monotone était le signe de la qualité esthétique de ses propos et l'effet, dans sa conversation, de ce même pouvoir qui produisait dans ses livres la suite des images de l'harmonie. J'avais eu d'autant plus de peine à m'en apercevoir d'abord que ce qu'il disait à ces moments-là, précisément parce que c'était vraiment de Bergotte, n'avait pas l'air d'être du Bergotte. C'était un foisonnement d'idées précises, non incluses dans ce « genre Bergotte » que beaucoup de chroniqueurs s'étaient approprié ; et cette dissemblance était probablement – vue d'une façon trouble à travers la conversation, comme une image derrière un verre fumé – un autre aspect de ce fait que quand on lisait une page de Bergotte, elle n'était jamais ce qu'aurait écrit n'importe lequel de ces plats imitateurs qui pourtant, dans le journal et dans le livre, ornaient leur prose de tant d'images et de pensées « à la Bergotte ». Cette différence dans le style venait de ce que « le Bergotte » était avant tout quelque élément précieux et vrai, caché au coeur de quelque chose, puis extrait d'elle par ce grand écrivain grâce à son génie, extraction qui était le but du doux Chantre et non pas de faire du Bergotte. À vrai dire il en faisait malgré lui puisqu'il était Bergotte, et qu'en ce sens chaque nouvelle beauté de son oeuvre était la petite quantité de Bergotte enfouie dans une chose et qu'il en avait tirée. Mais si par là chacune de ces beautés était apparentée avec les autres et reconnaissable, elle restait cependant particulière, comme la découverte qui l'avait mise à jour ; nouvelle, par conséquent différente de ce qu'on appelait le genre Bergotte qui était une vague synthèse des Bergotte déjà trouvés et rédigés par lui, lesquels ne permettaient nullement à des hommes sans génie d'augurer ce qu'il découvrirait ailleurs. Il en est ainsi pour tous les grands écrivains, la beauté de leurs phrases est imprévisible, comme est celle d'une femme qu'on ne connaît pas encore ; elle est création puisqu'elle s'applique à un objet extérieur auquel ils pensent – et non à soi – et qu'ils n'ont pas encore exprimé. Un auteur de Mémoires, d'aujourd'hui, voulant, sans trop en avoir l'air, faire du Saint-Simon, pourra à la rigueur écrire la première ligne du portrait de Villars : « C'était un assez grand homme brun... avec une physionomie vive, ouverte, sortante », mais quel déterminisme pourra lui faire trouver la seconde ligne qui commence par : « et véritablement un peu folle ». La vraie variété est dans cette plénitude d'éléments réels et inattendus, dans le rameau chargé de fleurs bleues qui s'élance, contre toute attente, de la haie printanière qui semblait déjà comble, tandis que l'imitation purement formelle de la variété (et on pourrait raisonner de même pour toutes les autres qualités du style) n'est que vide et uniformité, c'est-à-dire ce qui est le plus opposé à la variété, et ne peut chez les imitateurs en donner l'illusion et en rappeler le souvenir que pour celui qui ne l'a pas comprise chez les maîtres.

Aussi – de même que la diction de Bergotte eût sans doute charmé si lui-même n'avait été que quelque amateur récitant du prétendu Bergotte, au lieu qu'elle était liée à la pensée de Bergotte en travail et en action par des rapports vitaux que l'oreille ne dégageait pas immédiatement – de même c'était parce que Bergotte appliquait cette pensée avec précision à la réalité qui lui plaisait que son langage avait quelque chose de positif, de trop nourrissant, qui décevait ceux qui s'attendaient à l'entendre parler seulement de « l'éternel torrent des apparences » et des « mystérieux frissons de la beauté ». Enfin la qualité toujours rare et neuve de ce qu'il écrivait se traduisait dans sa conversation par une façon si subtile d'aborder une question, en négligeant tous ses aspects déjà connus, qu'il avait l'air de la prendre par un petit côté, d'être dans le faux, de faire du paradoxe, et qu'ainsi ses idées semblaient le plus souvent confuses, chacun appelant idées claires celles qui sont au même degré de confusion que les siennes propres. D'ailleurs toute nouveauté ayant pour condition l'élimination préalable du poncif auquel nous étions habitués et qui nous semblait la réalité même, toute conversation neuve, aussi bien que toute peinture, toute musique originale, paraîtra toujours alambiquée et fatigante. Elle repose sur des figures auxquelles nous ne sommes pas accoutumées, le causeur nous paraît ne parler que par métaphores, ce qui lasse et donne l'impression d'un manque de vérité. (Au fond les anciennes formes de langage avaient été elles aussi autrefois des images difficiles à suivre quand l'auditeur ne connaissait pas encore l'univers qu'elles peignaient. Mais depuis longtemps on se figure que c'était l'univers réel, on se repose sur lui.) Aussi quand Bergotte, ce qui semble pourtant bien simple aujourd'hui, disait de Cottard que c'était un ludion qui cherchait son équilibre, et de Brichot que « plus encore qu'à Odette le soin de sa coiffure lui donnait de la peine parce que doublement préoccupé de son profil et de sa réputation, il fallait à tout moment que l'ordonnance de la chevelure lui donnât l'air à la fois d'un lion et d'un philosophe », on éprouvait vite de la fatigue et on eût voulu reprendre pied sur quelque chose de plus concret, disait-on pour signifier de plus habituel. Les paroles méconnaissables sorties du masque que j'avais sous les yeux, c'était bien à l'écrivain que j'admirais qu'il fallait les rapporter, elles n'auraient pas su s'insérer dans ses livres à la façon d'un puzzle qui s'encadre entre d'autres, elles étaient dans un autre plan et nécessitaient une transposition moyennant laquelle un jour que je me répétais des phrases que j'avais entendu dire à Bergotte, j'y retrouvai toute l'armature de son style écrit, dont je pus reconnaître et nommer les différentes pièces dans ce discours parlé qui m'avait paru si différent.

À un point de vue plus accessoire, la façon spéciale, un peu trop minutieuse et intense, qu'il avait de prononcer certains mots, certains adjectifs qui revenaient souvent dans sa conversation et qu'il ne disait pas sans une certaine emphase, faisant ressortir toutes leurs syllabes et chanter la dernière (comme pour le mot « visage » qu'il substituait toujours au mot « figure » et à qui il ajoutait un grand nombre de v, d's, de g, qui semblaient tous exploser de sa main ouverte à ces moments) correspondait exactement à la belle place où dans sa prose il mettait ces mots aimés en lumière, précédés d'une sorte de marge et composés de telle façon, dans le nombre total de la phrase, qu'on était obligé, sous peine de faire une faute de mesure, d'y faire compter toute leur « quantité ». Pourtant, on ne retrouvait pas dans le langage de Bergotte certain éclairage qui dans ses livres comme dans ceux de quelques autres auteurs modifie souvent dans la phrase écrite l'apparence des mots. C'est sans doute qu'il vient de grandes profondeurs et n'amène pas ses rayons jusqu'à nos paroles dans les heures où, ouverts aux autres par la conversation, nous sommes dans une certaine mesure fermés à nous-même. À cet égard il y avait plus d'intonations, plus d'accent, dans ses livres que dans ses propos ; accent indépendant de la beauté du style, que l'auteur lui-même n'a pas perçu sans doute, car il n'est pas séparable de sa personnalité la plus intime. C'est cet accent qui, aux moments où, dans ses livres, Bergotte était entièrement naturel, rythmait les mots souvent alors fort insignifiants qu'il écrivait. Cet accent n'est pas noté dans le texte, rien ne l'y indique et pourtant il s'ajoute de lui-même aux phrases, on ne peut pas les dire autrement, il est ce qu'il y avait de plus éphémère et pourtant de plus profond chez l'écrivain, et c'est cela qui portera témoignage sur sa nature, qui dira si malgré toutes les duretés qu'il a exprimées il était doux, malgré toutes les sensualités, sentimental.

Certaines particularités d'élocution qui existaient à l'état de faibles traces dans la conversation de Bergotte ne lui appartenaient pas en propre, car quand j'ai connu plus tard ses frères et ses soeurs, je les ai retrouvées chez eux bien plus accentuées. C'était quelque chose de brusque et de rauque dans les derniers mots d'une phrase gaie, quelque chose d'affaibli et d'expirant à la fin d'une phrase triste. Swann, qui avait connu le Maître quand il était enfant, m'a dit qu'alors on entendait chez lui, tout autant que chez ses frères et soeurs, ces inflexions en quelque sorte familiales, tour à tour cris de violente gaieté, murmures d'une lente mélancolie, et que dans la salle où ils jouaient tous ensemble il faisait sa partie mieux qu'aucun, dans leurs concerts successivement assourdissants et languides. Si particulier qu'il soit, tout ce bruit qui s'échappe des êtres est fugitif et ne leur survit pas. Mais il n'en fut pas ainsi de la prononciation de la famille Bergotte. Car s'il est difficile de comprendre jamais, même dans les Maîtres Chanteurs, comment un artiste peut inventer la musique en écoutant gazouiller les oiseaux, pourtant Bergotte avait transposé et fixé dans sa prose cette façon de traîner sur des mots qui se répètent en clameurs de joie ou qui s'égouttent en tristes soupirs. Il y a dans ses livres telles terminaisons de phrases où l'accumulation des sonorités se prolonge, comme aux derniers accords d'une ouverture d'Opéra qui ne peut pas finir et redit plusieurs fois sa suprême cadence avant que le chef d'orchestre pose son bâton, dans lesquelles je retrouvai plus tard un équivalent musical de ces cuivres phonétiques de la famille Bergotte. Mais pour lui, à partir du moment où il les transporta dans ses livres, il cessa inconsciemment d'en user dans son discours. Du jour où il avait commencé d'écrire et, à plus forte raison, plus tard, quand je le connus, sa voix s'en était désorchestrée pour toujours.

Ces jeunes Bergotte – le futur écrivain et ses frères et soeurs – n'étaient sans doute pas supérieurs, au contraire, à des jeunes gens plus fins, plus spirituels qui trouvaient les Bergotte bien bruyants, voire un peu vulgaires, agaçants dans leurs plaisanteries qui caractérisaient le « genre » moitié prétentieux, moitié bêta, de la maison. Mais le génie, même le grand talent, vient moins d'éléments intellectuels et d'affinement spécial supérieurs à ceux d'autrui, que de la faculté de les transformer, de les transposer. Pour faire chauffer un liquide avec une lampe électrique, il ne s'agit pas d'avoir la plus forte lampe possible, mais une dont le courant puisse cesser d'éclairer, être dérivé et donner, au lieu de lumière, de la chaleur. Pour se promener dans les airs, il n'est pas nécessaire d'avoir l'automobile la plus puissante, mais une automobile qui ne continuant pas de courir à terre et coupant d'une verticale la ligne qu'elle suivait soit capable de convertir en force ascensionnelle sa vitesse horizontale. De même ceux qui produisent des oeuvres géniales ne sont pas ceux qui vivent dans le milieu le plus délicat, qui ont la conversation la plus brillante, la culture la plus étendue, mais ceux qui ont eu le pouvoir, cessant brusquement de vivre pour eux-mêmes, de rendre leur personnalité pareille à un miroir, de telle sorte que leur vie, si médiocre d'ailleurs qu'elle pouvait être mondainement et même, dans un certain sens, intellectuellement parlant, s'y reflète, le génie consistant dans le pouvoir réfléchissant et non dans la qualité intrinsèque du spectacle reflété. Le jour où le jeune Bergotte put montrer au monde de ses lecteurs le salon de mauvais goût où il avait passé son enfance et les causeries pas très drôles qu'il y tenait avec ses frères, ce jour-là il monta plus haut que les amis de sa famille, plus spirituels et plus distingués : ceux-ci dans leurs belles Rolls-Royce pourraient rentrer chez eux en témoignant un peu de mépris pour la vulgarité des Bergotte ; mais lui, de son modeste appareil qui venait enfin de « décoller », il les survolait.
