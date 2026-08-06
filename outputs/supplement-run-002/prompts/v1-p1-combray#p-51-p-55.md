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
  "appraisal_events": [],
  "status_effects": [],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-51-p-55"
}

### Candidate characters

[
  "Octave",
  "le grand-père du narrateur",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

Combray de loin, à dix lieues à la ronde, vu du chemin de fer quand nous y arrivions la dernière semaine avant Pâques, ce n'était qu'une église résumant la ville, la représentant, parlant d'elle et pour elle aux lointains, et, quand on approchait, tenant serrés autour de sa haute mante sombre, en plein champ, contre le vent, comme une pastoure ses brebis, les dos laineux et gris des maisons rassemblées qu'un reste de remparts du moyen âge cernait çà et là d'un trait aussi parfaitement circulaire qu'une petite ville dans un tableau de primitif. À l'habiter, Combray était un peu triste, comme ses rues dont les maisons construites en pierres noirâtres du pays, précédées de degrés extérieurs, coiffées de pignons qui rabattaient l'ombre devant elles, étaient assez obscures pour qu'il fallût dès que le jour commençait à tomber relever les rideaux dans les « salles » ; des rues aux graves noms de saints (desquels plusieurs se rattachaient à l'histoire des premiers seigneurs de Combray) : rue Saint-Hilaire, rue Saint-Jacques où était la maison de ma tante, rue Sainte-Hildegarde, où donnait la grille, et rue du Saint-Esprit sur laquelle s'ouvrait la petite porte latérale de son jardin ; et ces rues de Combray existent dans une partie de ma mémoire si reculée, peinte de couleurs si différentes de celles qui maintenant revêtent pour moi le monde, qu'en vérité elles me paraissent toutes, et l'église qui les dominait sur la Place, plus irréelles encore que les projections de la lanterne magique ; et qu'à certains moments, il me semble que pouvoir encore traverser la rue Saint-Hilaire, pouvoir louer une chambre rue de l'Oiseau – à la vieille hôtellerie de l'Oiseau Flesché, des soupiraux de laquelle montait une odeur de cuisine qui s'élève encore par moments en moi aussi intermittente et aussi chaude – serait une entrée en contact avec l'Au-delà plus merveilleusement surnaturelle que de faire la connaissance de Golo et de causer avec Geneviève.

### Passage

La cousine de mon grand-père – ma grand'tante – chez qui nous habitions, était la mère de cette tante Léonie qui, depuis la mort de son mari, mon oncle Octave, n'avait plus voulu quitter, d'abord Combray, puis à Combray sa maison, puis sa chambre, puis son lit et ne « descendait » plus, toujours couchée dans un état incertain de chagrin, de débilité physique, de maladie, d'idée fixe et de dévotion. Son appartement particulier donnait sur la rue Saint-Jacques qui aboutissait beaucoup plus loin au Grand-Pré (par opposition au Petit-Pré, verdoyant au milieu de la ville, entre trois rues), et qui, unie, grisâtre, avec les trois hautes marches de grès presque devant chaque porte, semblait comme un défilé pratiqué par un tailleur d'images gothiques à même la pierre où il eût sculpté une crèche ou un calvaire. Ma tante n'habitait plus effectivement que deux chambres contiguës, restant l'après-midi dans l'une pendant qu'on aérait l'autre. C'étaient de ces chambres de province qui – de même qu'en certains pays des parties entières de l'air ou de la mer sont illuminées ou parfumées par des myriades de protozoaires que nous ne voyons pas – nous enchantent des mille odeurs qu'y dégagent les vertus, la sagesse, les habitudes, toute une vie secrète, invisible, surabondante et morale que l'atmosphère y tient en suspens ; odeurs naturelles encore, certes, et couleur du temps comme celles de la campagne voisine, mais déjà casanières, humaines et renfermées, gelée exquise, industrieuse et limpide de tous les fruits de l'année qui ont quitté le verger pour l'armoire ; saisonnières, mais mobilières et domestiques, corrigeant le piquant de la gelée blanche par la douceur du pain chaud, oisives et ponctuelles comme une horloge de village, flâneuses et rangées, insoucieuses et prévoyantes, lingères, matinales, dévotes, heureuses d'une paix qui n'apporte qu'un surcroît d'anxiété et d'un prosaïsme qui sert de grand réservoir de poésie à celui qui la traverse sans y avoir vécu. L'air y était saturé de la fine fleur d'un silence si nourricier, si succulent que je ne m'y avançais qu'avec une sorte de gourmandise, surtout par ces premiers matins encore froids de la semaine de Pâques où je le goûtais mieux parce que je venais seulement d'arriver à Combray : avant que j'entrasse souhaiter le bonjour à ma tante on me faisait attendre un instant dans la première pièce où le soleil, d'hiver encore, était venu se mettre au chaud devant le feu, déjà allumé entre les deux briques et qui badigeonnait toute la chambre d'une odeur de suie, en faisait comme un de ces grands « devants de four » de campagne, ou de ces manteaux de cheminée de châteaux, sous lesquels on souhaite que se déclarent dehors la pluie, la neige, même quelque catastrophe diluvienne pour ajouter au confort de la réclusion la poésie de l'hivernage ; je faisais quelques pas du prie-Dieu aux fauteuils en velours frappé, toujours revêtus d'un appui-tête au crochet ; et le feu cuisant comme une pâte les appétissantes odeurs dont l'air de la chambre était tout grumeleux et qu'avait déjà fait travailler et « lever » la fraîcheur humide et ensoleillée du matin, il les feuilletait, les dorait, les godait, les boursouflait, en faisant un invisible et palpable gâteau provincial, un immense « chausson » où, à peine goûtés les arômes plus croustillants, plus fins, plus réputés, mais plus secs aussi du placard, de la commode, du papier à ramages, je revenais toujours avec une convoitise inavouée m'engluer dans l'odeur médiane, poisseuse, fade, indigeste et fruitée du couvre-lit à fleurs.

Dans la chambre voisine, j'entendais ma tante qui causait toute seule à mi-voix. Elle ne parlait jamais qu'assez bas parce qu'elle croyait avoir dans la tête quelque chose de cassé et de flottant qu'elle eût déplacé en parlant trop fort, mais elle ne restait jamais longtemps, même seule, sans dire quelque chose, parce qu'elle croyait que c'était salutaire pour sa gorge et qu'en empêchant le sang de s'y arrêter, cela rendrait moins fréquents les étouffements et les angoisses dont elle souffrait ; puis, dans l'inertie absolue où elle vivait, elle prêtait à ses moindres sensations une importance extraordinaire ; elle les douait d'une motilité qui lui rendait difficile de les garder pour elle, et à défaut de confident à qui les communiquer, elle se les annonçait à elle-même, en un perpétuel monologue qui était sa seule forme d'activité. Malheureusement, ayant pris l'habitude de penser tout haut, elle ne faisait pas toujours attention à ce qu'il n'y eût personne dans la chambre voisine, et je l'entendais souvent se dire à elle-même : « Il faut que je me rappelle bien que je n'ai pas dormi » (car ne jamais dormir était sa grande prétention dont notre langage à tous gardait le respect et la trace : le matin Françoise ne venait pas « l'éveiller », mais « entrait » chez elle ; quand ma tante voulait faire un somme dans la journée, on disait qu'elle voulait « réfléchir » ou « reposer » ; et quand il lui arrivait de s'oublier en causant jusqu'à dire : « ce qui m'a réveillée » ou « j'ai rêvé que », elle rougissait et se reprenait au plus vite).

Au bout d'un moment, j'entrais l'embrasser ; Françoise faisait infuser son thé ; ou, si ma tante se sentait agitée, elle demandait à la place sa tisane, et c'étais moi qui étais chargé de faire tomber du sac de pharmacie dans une assiette la quantité de tilleul qu'il fallait mettre ensuite dans l'eau bouillante. Le desséchement des tiges les avait incurvées en un capricieux treillage dans les entrelacs duquel s'ouvraient les fleurs pâles, comme si un peintre les eût arrangées, les eût fait poser de la façon la plus ornementale. Les feuilles, ayant perdu ou changé leur aspect, avaient l'air des choses les plus disparates, d'une aile transparente de mouche, de l'envers blanc d'une étiquette, d'un pétale de rose, mais qui eussent été empilées, concassées ou tressées comme dans la confection d'un nid. Mille petits détails inutiles – charmante prodigalité du pharmacien – qu'on eût supprimés dans une préparation factice, me donnaient, comme un livre où on s'émerveille de rencontrer le nom d'une personne de connaissance, le plaisir de comprendre que c'était bien des tiges de vrais tilleuls, comme ceux que je voyais avenue de la Gare, modifiées, justement parce que c'étaient non des doubles, mais elles-mêmes et qu'elles avaient vieilli. Et chaque caractère nouveau n'y étant que la métamorphose d'un caractère ancien, dans de petites boules grises je reconnaissais les boutons verts qui ne sont pas venus à terme ; mais surtout l'éclat rose, lunaire et doux qui faisait se détacher les fleurs dans la forêt fragile des tiges où elles étaient suspendues comme de petites roses d'or – signe, comme la lueur qui révèle encore sur une muraille la place d'une fresque effacée, de la différence entre les parties de l'arbre qui avaient été « en couleur » et celles qui ne l'avaient pas été – me montrait que ces pétales étaient bien ceux qui avant de fleurir le sac de pharmacie avaient embaumé les soirs de printemps. Cette flamme rose de cierge, c'était leur couleur encore, mais à demi éteinte et assoupie dans cette vie diminuée qu'était la leur maintenant et qui est comme le crépuscule des fleurs. Bientôt ma tante pouvait tremper dans l'infusion bouillante dont elle savourait le goût de feuille morte ou de fleur fanée une petite madeleine dont elle me tendait un morceau quand il était suffisamment amolli.

D'un côté de son lit était une grande commode jaune en bois de citronnier et une table qui tenait à la fois de l'officine et du maître-autel, où, au-dessus d'une statuette de la Vierge et d'une bouteille de Vichy-Célestins, on trouvait des livres de messe et des ordonnances de médicaments, tous ce qu'il fallait pour suivre de son lit les offices et son régime, pour ne manquer l'heure ni de la pepsine, ni des Vêpres. De l'autre côté, son lit longeait la fenêtre, elle avait la rue sous les yeux et y lisait du matin au soir, pour se désennuyer, à la façon des princes persans, la chronique quotidienne mais immémoriale de Combray, qu'elle commentait ensuite avec Françoise.

Je n'étais pas avec ma tante depuis cinq minutes, qu'elle me renvoyait par peur que je la fatigue. Elle tendait à mes lèvres son triste front pâle et fade sur lequel, à cette heure matinale, elle n'avait pas encore arrangé ses faux cheveux, et où les vertèbres transparaissaient comme les pointes d'une couronne d'épines ou les grains d'un rosaire, et elle me disait : « Allons, mon pauvre enfant, va-t'en, va te préparer pour la messe ; et si en bas tu rencontres Françoise, dis-lui de ne pas s'amuser trop longtemps avec vous, qu'elle monte bientôt voir si je n'ai besoin de rien. »
