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
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Robert de Saint-Loup",
        "Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Je m’étais absolument trompé… il faisait… des pieds et des mains pour qu’il [son engagement] le fût… plus profondément français… il obtint de servir comme officier d’infanterie… délicatesse morale qui empêche d’exprimer les sentiments trop profonds.",
      "explanation": "The narrator corrects his interpretation and presents Robert as courageous, patriotic, and reserved in expressing his feelings, which greatly enhances his local esteem."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "His moral and patriotic value is greatly enhanced by the narrator's rectification and evidence of his commitment."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p2-m-de-charlus-pendant-la-guerre#p-16-p-20"
}

### Candidate characters

[
  "Albertine",
  "Bloch",
  "Brichot",
  "Dreyfus",
  "Françoise",
  "Gilberte",
  "M. Verdurin",
  "Mme Verdurin",
  "Swann",
  "baron de Charlus",
  "docteur Cottard",
  "duc de Guermantes",
  "duchesse de Guermantes",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "marquis de Cambremer"
]

### Prior local context (optional)

Mme Verdurin disait : « C'est désolant, je vais téléphoner à Bontemps de faire le nécessaire pour demain, on a encore « caviardé » toute la fin de l'article de Norpois et simplement parce qu'il laissait entendre qu'on avait « limogé » Percin. » Car la bêtise courante faisait que chacun tirait sa gloire d'user des expressions courantes, et croyait montrer qu'elle était ainsi à la mode comme faisait une bourgeoise en disant, quand on parlait de M. de marquis de Bréauté ou de baron de Charlus : « Qui ? Bebel de marquis de Bréauté, Mémé de baron de Charlus ? » Les duchesses font de même, d'ailleurs, et avaient le même plaisir à dire « limoger » car, chez les duchesses, c'est, pour les roturiers un peu poètes, le nom qui diffère, mais elles s'expriment selon la catégorie d'esprit à laquelle elles appartiennent et où il y a aussi énormément de bourgeois. Les classes d'esprit n'ont pas égard à la naissance.

### Passage

Tous ces téléphonages de Mme Verdurin n'étaient pas, d'ailleurs, sans inconvénient. Quoique nous ayons oublié de le dire, le « salon » Verdurin, s'il continuait en esprit et en vérité, s'était transporté momentanément dans un des plus grands hôtels de Paris, le manque de charbon et de lumière rendant plus difficiles les réceptions des Verdurin dans l'ancien logis, fort humide, des Ambassadeurs de Venise. Le nouveau salon ne manquait pas, du reste, d'agrément. Comme à Venise la place, comptée à cause de l'eau, commande la forme des palais, comme un bout de jardin dans Paris ravit plus qu'un parc en province, l'étroite salle à manger qu'avait Mme Verdurin à l'hôtel faisait d'une sorte de losange aux murs éclatants de blancheur comme un écran sur lequel se détachaient à chaque mercredi, et presque tous les jours, tous les gens les plus intéressants, les plus variés, les femmes les plus élégantes de Paris, ravis de profiter du luxe des Verdurin qui, grâce à leur fortune, allait croissant à une époque où les plus riches se restreignaient faute de toucher leurs revenus. La forme donnée aux réceptions se trouvait modifiée sans qu'elles cessassent d'enchanter Brichot, qui, au fur et à mesure que les relations des Verdurin allaient s'étendant, y trouvait des plaisirs nouveaux et accumulés dans un petit espace comme des surprises dans un chausson de Noël. Enfin, certains jours, les dîneurs étaient si nombreux que la salle à manger de l'appartement privé était trop petite, on donnait le dîner dans la salle à manger immense d'en bas, où les fidèles, tout en feignant hypocritement de déplorer l'intimité d'en haut, étaient ravis au fond – en faisant bande à part comme jadis dans le petit chemin de fer – d'être un objet de spectacle et d'envie pour les tables voisines. Sans doute dans les temps habituels de la paix une note mondaine subrepticement envoyée au Figaro ou au Gaulois aurait fait savoir à plus de monde que n'en pouvait tenir la salle à manger du Majestic que Brichot avait dîné avec la duchesse de Duras. Mais depuis la guerre, les courriéristes mondains ayant supprimé ce genre d'informations (ils se rattrapaient sur les enterrements, les citations et les banquets franco-américains), la publicité ne pouvait plus exister que par ce moyen enfantin et restreint, digne des premiers âges, et antérieur à la découverte de Gutenberg, être vu à la table de Mme Verdurin. Après le dîner on montait dans les salons de la Patronne, puis les téléphonages commençaient. Mais beaucoup de grands hôtels étaient, à cette époque, peuplés d'espions qui notaient les nouvelles téléphonées par Bontemps avec une indiscrétion que corrigeait seulement par bonheur le manque de sûreté de ses informations, toujours démenties par l'événement.

Avant l'heure où les thés d'après-midi finissaient, à la tombée du jour, dans le ciel encore clair, on voyait de loin de petites taches brunes qu'on eût pu prendre, dans le soir bleu, pour des moucherons ou pour des oiseaux. Ainsi quand on voit de très loin une montagne on pourrait croire que c'est un nuage. Mais on est ému parce qu'on sait que ce nuage est immense, à l'état solide, et résistant. Ainsi étais-je ému parce que la tache brune dans le ciel d'été n'était ni un moucheron, ni un oiseau, mais un aéroplane monté par des hommes qui veillaient sur Paris. Le souvenir des aéroplanes que j'avais vus avec Albertine dans notre dernière promenade, près de Versailles, n'entrait pour rien dans cette émotion, car le souvenir de cette promenade m'était devenu indifférent.

À l'heure du dîner les restaurants étaient pleins et si, passant dans la rue, je voyais un pauvre permissionnaire, échappé pour six jours au risque permanent de la mort, et prêt à repartir pour les tranchées, arrêter un instant ses yeux devant les vitrines illuminées, je souffrais comme à l'hôtel de Balbec quand les pêcheurs nous regardaient dîner, mais je souffrais davantage parce que je savais que la misère du soldat est plus grande que celle du pauvre, les réunissant toutes, et plus touchante encore parce qu'elle est plus résignée, plus noble, et que c'est d'un hochement de tête philosophe, sans haine, que, prêt à repartir pour la guerre, il disait en voyant se bousculer les embusqués retenant leurs tables : « On ne dirait pas que c'est la guerre ici. » Puis à 9 h. ½, alors que personne n'avait encore eu le temps de finir de dîner, à cause des ordonnances de police on éteignait brusquement toutes les lumières et la nouvelle bousculade des embusqués arrachant leurs pardessus aux chasseurs du restaurant où j'avais dîné avec Saint-Loup un soir de perme avait lieu à 9 h. 35 dans une mystérieuse pénombre de chambre où l'on montre la lanterne magique, ou de salle de spectacle servant à exhiber les films d'un de ces cinémas vers lesquels allaient se précipiter dîneurs et dîneuses. Mais après cette heure-là, pour ceux qui, comme moi, le soir dont je parle, étaient restés à dîner chez eux, et sortaient pour aller voir des amis, Paris était, au moins dans certains quartiers, encore plus noir que n'était le Combray de mon enfance ; les visites qu'on se faisait prenaient un air de visites de voisins de campagne. Ah ! si Albertine avait vécu, qu'il eût été doux, les soirs où j'aurais dîné en ville, de lui donner rendez-vous dehors, sous les arcades. D'abord, je n'aurais rien vu, j'aurais eu l'émotion de croire qu'elle avait manqué au rendez-vous, quand tout à coup j'eusse vu se détacher du mur noir une de ses chères robes grises, ses yeux souriants qui m'auraient aperçu, et nous aurions pu nous promener enlacés sans que personne nous distinguât, nous dérangeât et rentrer ensuite à la maison. Hélas, j'étais seul et je me faisais l'effet d'aller faire une visite de voisin à la campagne, de ces visites comme Swann venait nous en faire après le dîner, sans rencontrer plus de passants dans l'obscurité de Tansonville, par ce petit chemin de halage, jusqu'à la rue du Saint-Esprit, que je n'en rencontrais maintenant dans les rues devenues de sinueux chemins rustiques de la rue Clotilde à la rue Bonaparte. D'ailleurs, comme ces fragments de paysage, que le temps qu'il fait modifie, n'étaient plus contrariés par un cadre devenu nuisible, les soirs où le vent chassait un grain glacial je me croyais bien plus au bord de la mer furieuse, dont j'avais jadis tant rêvé, que je ne m'y étais senti à Balbec ; et même d'autres éléments de nature qui n'existaient pas jusque-là à Paris faisaient croire qu'on venait, descendant du train, d'arriver pour les vacances, en pleine campagne : par exemple le contraste de lumière et d'ombre qu'on avait à côté de soi par terre les soirs de clair de lune. Celui-ci donnait de ces effets que les villes ne connaissent pas, même en plein hiver ; ses rayons s'étalaient sur la neige qu'aucun travailleur ne déblayait plus, boulevard Haussmann, comme ils eussent fait sur un glacier des Alpes. Les silhouettes des arbres se reflétaient nettes et pures sur cette neige d'or bleuté, avec la délicatesse qu'elles ont dans certaines peintures japonaises ou dans certains fonds de Raphaël ; elles étaient allongées à terre au pied de l'arbre lui-même, comme on les voit souvent dans la nature au soleil couchant, quand celui-ci inonde et rend réfléchissantes les prairies où des arbres s'élèvent à intervalles réguliers. Mais, par un raffinement d'une délicatesse délicieuse, la prairie sur laquelle se développaient ces ombres d'arbres, légères comme des âmes, était une prairie paradisiaque, non pas verte mais d'un blanc si éclatant, à cause du clair de lune qui rayonnait sur la neige de jade, qu'on aurait dit que cette prairie était tissée seulement avec des pétales de poiriers en fleurs. Et sur les places, les divinités des fontaines publiques tenant en main un jet de glace avaient l'air de statues d'une matière double pour l'exécution desquelles l'artiste avait voulu marier exclusivement le bronze au cristal. Par ces jours exceptionnels, toutes les maisons étaient noires. Mais au printemps, au contraire, parfois de temps à autre, bravant les règlements de la police, un hôtel particulier, ou seulement un étage d'un hôtel, ou même seulement une chambre d'un étage, n'ayant pas fermé ses volets apparaissait, ayant l'air de se soutenir toute seule sur d'impalpables ténèbres, comme une projection purement lumineuse, comme une apparition sans consistance. Et la femme qu'en levant les yeux bien haut on distinguait dans cette pénombre dorée prenait, dans cette nuit où l'on était perdu et où elle-même semblait recluse, le charme mystérieux et voilé d'une vision d'Orient. Puis on passait et rien n'interrompait plus l'hygiénique et monotone piétinement rythmique dans l'obscurité.

Je songeais que je n'avais revu depuis bien longtemps aucune des personnes dont il a été question dans cet ouvrage. En 1914, pendant les deux mois que j'avais passés à Paris, j'avais aperçu Charlus et vu Bloch et Saint-Loup, ce dernier seulement deux fois. La seconde fois était certainement celle où il s'était le plus montré lui-même ; il avait effacé toutes les impressions peu agréables de manque de sincérité qu'il m'avait produites pendant le séjour à Tansonville que je viens de rapporter et j'avais reconnu en lui toutes les belles qualités d'autrefois. La première fois que je l'avais vu après la déclaration de guerre, c'est-à-dire au début de la semaine qui suivit, tandis que Bloch faisait montre des sentiments les plus chauvins, Saint-Loup n'avait pas assez d'ironie pour lui-même qui ne reprenait pas de service et j'avais été presque choqué de la violence de son ton. Saint-Loup revenait de Balbec. « Non, s'écria-t-il avec force et gaîté, tous ceux qui ne se battent pas, quelque raison qu'ils donnent, c'est qu'ils n'ont pas envie d'être tués, c'est par peur. » Et avec le même geste d'affirmation plus énergique encore que celui avec lequel il avait souligné la peur des autres, il ajouta : « Et moi, si je ne reprends pas de service, c'est tout bonnement par peur, na. » J'avais déjà remarqué chez différentes personnes que l'affectation des sentiments louables n'est pas la seule couverture des mauvais, mais qu'une plus nouvelle est l'exhibition de ces mauvais, de sorte qu'on n'ait pas l'air au moins de s'en cacher. De plus, chez Saint-Loup cette tendance était fortifiée par son habitude, quand il avait commis une indiscrétion, fait une gaffe, et qu'on aurait pu les lui reprocher, de les proclamer en disant que c'était exprès. Habitude qui, je crois bien, devait lui venir de quelque professeur à l'École de Guerre dans l'intimité de qui il avait vécu et pour qui il professait une grande admiration. Je n'eus donc aucun embarras pour interpréter cette boutade comme la ratification verbale d'un sentiment que Saint-Loup aimait mieux proclamer, puisqu'il avait dicté sa conduite et son abstention dans la guerre qui commençait. « Est-ce que tu as entendu dire, demanda-t-il en me quittant, que ma tante Mme de Guermantes divorcerait ? Personnellement je n'en sais absolument rien. On dit cela de temps en temps et je l'ai entendu annoncer si souvent que j'attendrai que ce soit fait pour le croire. J'ajoute que ce serait très compréhensible ; mon oncle est un homme charmant, non seulement dans le monde, mais pour ses amis, pour ses parents. Même, d'une façon, il a beaucoup plus de coeur que ma tante qui est une sainte, mais qui le lui fait terriblement sentir. Seulement c'est un mari terrible, qui n'a jamais cessé de tromper sa femme, de l'insulter, de la brutaliser, de la priver d'argent. Ce serait si naturel qu'elle le quitte que c'est une raison pour que ce soit vrai, mais aussi pour que cela ne le soit pas parce que c'en est une pour qu'on en ait l'idée et qu'on le dise. Et puis du moment qu'elle l'a supporté si longtemps... Maintenant je sais bien qu'il y a tant de choses qu'on annonce à tort, qu'on dément, et puis qui plus tard deviennent vraies. » Cela me fit penser à lui demander s'il avait jamais été question, avant son mariage avec Gilberte, qu'il épousât Mlle de Guermantes. Il sursauta et m'assura que non, que ce n'était qu'un de ces bruits du monde, qui naissent de temps à autre on ne sait pourquoi, s'évanouissent de même et dont la fausseté ne rend pas ceux qui ont cru en eux plus prudents, dès que naît un bruit nouveau de fiançailles, de divorce, ou un bruit politique, pour y ajouter foi et le colporter.

Quarante-huit heures n'étaient pas passées que certains faits que j'appris me prouvèrent que je m'étais absolument trompé dans l'interprétation des paroles de Saint-Loup : « Tous ceux qui ne sont pas au front, c'est qu'ils ont peur. » Saint-Loup avait dit cela pour briller dans la conversation, pour faire de l'originalité psychologique, tant qu'il n'était pas sûr que son engagement serait accepté. Mais il faisait pendant ce temps-là des pieds et des mains pour qu'il le fût, étant en cela moins original, au sens qu'il croyait qu'il fallait donner à ce mot, mais plus profondément français de Saint-André-des-Champs, plus en conformité avec tout ce qu'il y avait à ce moment-là de meilleur chez les Français de Saint-André-des-Champs, seigneurs, bourgeois et serfs respectueux des seigneurs ou révoltés contre les seigneurs, deux divisions également françaises de la même famille, sous-embranchement Françoise et sous-embranchement Sauton, d'où deux flèches se dirigeaient à nouveau dans une même direction, qui était la frontière. Bloch avait été enchanté d'entendre l'aveu de la lâcheté d'un nationaliste (qui l'était d'ailleurs si peu) et, comme Saint-Loup avait demandé si lui-même devait partir, avait pris une figure de grand-prêtre pour répondre : « Myope. » Mais Bloch avait complètement changé d'avis sur la guerre quelques jours après où il vint me voir affolé. Quoique « myope », il avait été reconnu bon pour le service. Je le ramenais chez lui quand nous rencontrâmes Saint-Loup qui avait rendez-vous, pour être présenté au Ministère de la Guerre à un colonel, avec un ancien officier, « M. de Cambremer », me dit-il. « Ah ! c'est vrai, mais c'est d'une ancienne connaissance que je te parle. Tu connais aussi bien que moi Cancan. » Je lui répondis que je le connaissais en effet et sa femme aussi, que je ne les appréciais qu'à demi. Mais j'étais tellement habitué, depuis que je les avais vus pour la première fois, à considérer la femme comme une personne malgré tout remarquable, connaissant à fond Schopenhauer et ayant accès, en somme, dans un milieu intellectuel qui était fermé à son grossier époux, que je fus d'abord étonné d'entendre Saint-Loup répondre : « Sa femme est idiote, je te l'abandonne. Mais lui est un excellent homme qui était doué et qui est resté fort agréable. » Par l'« idiotie » de la femme, Saint-Loup entendait sans doute le désir éperdu de celle-ci de fréquenter le grand monde, ce que le grand monde juge le plus sévèrement. Par les qualités du mari, sans doute quelque chose de celles que lui reconnaissait sa nièce quand elle le trouvait le mieux de la famille. Lui, du moins, ne se souciait pas de duchesses, mais à vrai dire c'est là une « intelligence » qui diffère autant de celle qui caractérise les penseurs, que « l'intelligence » reconnue par le public à tel homme riche « d'avoir su faire sa fortune ». Mais les paroles de Saint-Loup ne me déplaisaient pas en ce qu'elles rappelaient que la prétention avoisine la bêtise et que la simplicité a un goût un peu caché mais agréable. Je n'avais pas eu, il est vrai, l'occasion de savourer celle de M. de Cambremer. Mais c'est justement ce qui fait qu'un être est tant d'êtres différents selon les personnes qui le jugent, en dehors même des différences de jugement. De Cambremer je n'avais connu que l'écorce. Et sa saveur, qui m'était attestée par d'autres, m'était inconnue. Bloch nous quitta devant sa porte, débordant d'amertume contre Saint-Loup, lui disant qu'eux autres, « beaux fils galonnés », paradant dans les États-Majors, ne risquaient rien, et que lui, simple soldat de 2e classe, n'avait pas envie de se faire « trouer la peau » pour Guillaume. « Il paraît qu'il est gravement malade, l'Empereur Guillaume », répondit Saint-Loup. Bloch qui, comme tous les gens qui tiennent de près à la Bourse, accueillait avec une facilité particulière les nouvelles sensationnelles, ajouta : « On dit même beaucoup qu'il est mort. » À la Bourse tout souverain malade, que ce soit Edouard VII ou Guillaume II, est mort, toute ville sur le point d'être assiégée est prise. « On ne le cache, ajouta Bloch, que pour ne pas déprimer l'opinion chez les Boches. Mais il est mort dans la nuit d'hier. Mon père le tient d'une source de tout premier ordre. » Les sources de tout premier ordre étaient les seules dont tînt compte Bloch le père, alors que, par la chance qu'il avait, grâce à de « hautes relations », d'être en communication avec elles, il en recevait la nouvelle encore secrète que l'Extérieure allait monter ou la de Beers fléchir. D'ailleurs, si à ce moment précis se produisait une hausse sur la de Beers, ou des « offres » sur l'Extérieure, si le marché de la première était « ferme » et « actif », celui de la seconde « hésitant », « faible », et qu'on s'y tînt « sur la réserve », la source de premier ordre n'en restait pas moins une source de premier ordre. Aussi Bloch nous annonça-t-il la mort du Kaiser d'un air mystérieux et important, mais aussi rageur. Il était surtout particulièrement exaspéré d'entendre Saint-Loup dire : « l'Empereur Guillaume ». Je crois que sous le couperet de la guillotine Saint-Loup et duc de Guermantes  n'auraient pas pu dire autrement. Deux hommes du monde restant seuls vivants dans une île déserte, où ils n'auraient à faire preuve de bonnes façons pour personne, se reconnaîtraient à ces traces d'éducation, comme deux latinistes citeraient correctement du Virgile. Saint-Loup n'eût jamais pu, même torturé par les Allemands, dire autrement que « l'Empereur Guillaume ». Et ce savoir-vivre est malgré tout l'indice de grandes entraves pour l'esprit. Celui qui ne sait pas les rejeter reste un homme du monde. Cette élégante médiocrité est d'ailleurs délicieuse – surtout avec tout ce qui s'y allie de générosité cachée et d'héroïsme inexprimé – à côté de la vulgarité de Bloch, à la fois pleutre et fanfaron, qui criait à Saint-Loup : « Tu ne pourrais pas dire « Guillaume » tout court ? C'est ça, tu as la frousse, déjà ici tu te mets à plat ventre devant lui ! Ah ! ça nous fera de beaux soldats à la frontière, ils lécheront les bottes des Boches. Vous êtes des galonnés qui savez parader dans un carrousel. Un point, c'est tout. » « Ce pauvre Bloch veut absolument que je ne fasse que parader », me dit Saint-Loup en souriant, quand nous eûmes quitté notre camarade. Et je sentais bien que parader n'était pas du tout ce que désirait Saint-Loup, bien que je ne me rendisse pas compte alors de ses intentions aussi exactement que je le fis plus tard quand, la cavalerie restant inactive, il obtint de servir comme officier d'infanterie, puis de chasseurs à pied, et enfin quand vint la suite qu'on lira plus loin. Mais du patriotisme de Saint-Loup, Bloch ne se rendit pas compte, simplement parce que Saint-Loup ne l'exprimait nullement. Si Bloch nous avait fait des professions de foi méchamment antimilitaristes une fois qu'il avait été reconnu « bon », il avait eu préalablement les déclarations les plus chauvines quand il se croyait réformé pour myopie. Mais ces déclarations, Saint-Loup eût été incapable de les faire ; d'abord par une espèce de délicatesse morale qui empêche d'exprimer les sentiments trop profonds et qu'on trouve tout naturels. Ma mère autrefois non seulement n'eût pas hésité une seconde à mourir pour ma grand'mère, mais aurait horriblement souffert si on l'avait empêchée de le faire. Néanmoins, il m'est impossible d'imaginer rétrospectivement dans sa bouche une phrase telle que : « Je donnerais ma vie pour ma mère. » Aussi tacite était, dans son amour de la France, Saint-Loup qu'en ce moment je trouvais beaucoup plus Saint-Loup (autant que je pouvais me représenter son père) que Guermantes. Il eût été préservé aussi d'exprimer ces sentiments-là par la qualité en quelque sorte morale de son intelligence. Il y a chez les travailleurs intelligents et vraiment sérieux une certaine aversion pour ceux qui mettent en littérature ce qu'ils font, le font valoir. Nous n'avions été ensemble ni au lycée, ni à la Sorbonne, mais nous avions séparément suivi certains cours des mêmes maîtres, et je me rappelle le sourire de Saint-Loup en parlant de ceux qui, tout en faisant un cours remarquable, voulaient se faire passer pour des hommes de génie en donnant un nom ambitieux à leurs théories. Pour peu que nous en parlions, Saint-Loup riait de bon coeur. Naturellement notre prédilection n'allait pas d'instinct aux Cottard ou aux Brichot, mais enfin nous avions une certaine considération pour les gens qui savaient à fond le grec ou la médecine et ne se croyaient pas autorisés pour cela à faire les charlatans. De même que toutes les actions de maman reposaient jadis sur le sentiment qu'elle eût donné sa vie pour sa mère, comme elle ne s'était jamais formulé ce sentiment à elle-même, en tout cas elle eût trouvé non pas seulement inutile et ridicule, mais choquant et honteux de l'exprimer aux autres ; de même il m'était impossible d'imaginer Saint-Loup (me parlant de son équipement, des courses qu'il avait à faire, de nos chances de victoire, du peu de valeur de l'armée russe, de ce que ferait l'Angleterre) prononçant une des phrases les plus éloquentes que peut dire le Ministre le plus sympathique aux députés debout et enthousiastes. Je ne peux cependant pas dire que, dans ce côté négatif qui l'empêchait d'exprimer les beaux sentiments qu'il ressentait, il n'y avait pas un effet de l'« esprit des Guermantes », comme on en a vu tant d'exemples chez Swann. Car si je le trouvais Saint-Loup surtout, il restait Guermantes aussi et par là, parmi les nombreux mobiles qui excitaient son courage, il y en avait qui n'étaient pas les mêmes que ceux de ses amis de Doncières, ces jeunes gens épris de leur métier avec qui j'avais dîné chaque soir et dont tant se firent tuer à la bataille de la Marne ou ailleurs en entraînant leurs hommes. Les jeunes socialistes qu'il pouvait y avoir à Doncières quand j'y étais, mais que je ne connaissais pas parce qu'ils ne fréquentaient pas le milieu de Saint-Loup, purent se rendre compte que les officiers de ce milieu n'étaient nullement des « aristos » dans l'acception hautainement fière et bassement jouisseuse que le « populo », les officiers sortis des rangs, les francs-maçons donnaient à ce surnom. Et pareillement d'ailleurs, ce même patriotisme, les officiers nobles le rencontrèrent pleinement chez les socialistes que je les avais entendu accuser, pendant que j'étais à Doncières, en pleine affaire Dreyfus, d'être des sans-patrie. Le patriotisme des militaires, aussi sincère, aussi profond, avait pris une forme définie qu'ils croyaient intangible et sur laquelle ils s'indignaient de voir jeter « l'opprobre », tandis que les patriotes en quelque sorte inconscients, indépendants, sans religion patriotique définie, qu'étaient les radicaux-socialistes, n'avaient pas su comprendre quelle réalité profonde vivait dans ce qu'ils croyaient de vaines et haineuses formules. Sans doute Saint-Loup comme eux s'était habitué à développer en lui, comme la partie la plus vraie de lui-même, la recherche et la conception des meilleures manoeuvres en vue des plus grands succès stratégiques et tactiques, de sorte que, pour lui comme pour eux, la vie de son corps était quelque chose de relativement peu important qui pouvait être facilement sacrifié à cette partie intérieure, véritable noyau vital chez eux, autour duquel l'existence personnelle n'avait de valeur que comme un épiderme protecteur.
