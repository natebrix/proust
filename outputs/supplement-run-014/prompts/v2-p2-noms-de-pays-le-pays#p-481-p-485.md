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
      "canonical_name": "le directeur",
      "surface_forms": [
        "le directeur",
        "directeur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "le directeur",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.92,
      "evidence": "Il « arpentait les corridors, vêtu d'une redingote neuve », changeant sans cesse de cravates — « ces élégances coûtent moins cher que d'assurer le chauffage » — et « avait l'air d'inspecter le néant », « le fantôme d'un souverain » dans un hôtel ruiné; il réclame des « moyens de commotion » et promet « quelle phalange je saurai réunir » malgré le déficit. Françoise le traite même de quelqu'un qui « mangeait de l'argent ».",
      "explanation": "The narrator long ridicules the director, highlighting his empty vanity, grandiose poses, and practical incompetence; the portrait clearly undermines his dignity and credibility."
    }
  ],
  "status_effects": [
    {
      "character": "le directeur",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.92,
      "explanation": "He emerges locally diminished and ridiculed by the ironic portrait and signs of mismanagement."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-481-p-485"
}

### Candidate characters

[
  "Albertine",
  "Françoise",
  "Norpois",
  "la Berma",
  "la grand-mère",
  "le narrateur"
]

### Prior local context (optional)

Sans doute leurs visages à toutes avaient bien changé pour moi de sens depuis que la façon dont il fallait les lire m'avait été dans une certaine mesure indiquée par leurs propos, propos auxquels je pouvais attribuer une valeur d'autant plus grande que par mes questions je les provoquais à mon gré, les faisais varier comme un expérimentateur qui demande à des contre-épreuves la vérification de ce qu'il a supposé. Et c'est en somme une façon comme une autre de résoudre le problème de l'existence, qu'approcher suffisamment les choses et les personnes qui nous ont paru de loin belles et mystérieuses, pour nous rendre compte qu'elles sont sans mystère et sans beauté ; c'est une des hygiènes entre lesquelles on peut opter, une hygiène qui n'est peut-être pas très recommandable, mais elle nous donne un certain calme pour passer la vie, et aussi – comme elle permet de ne rien regretter, en nous persuadant que nous avons atteint le meilleur, et que le meilleur n'était pas grand'chose – pour nous résigner à la mort.

### Passage

J'avais remplacé au fond du cerveau de ces jeunes filles le mépris de la chasteté, le souvenir de quotidiennes passades, par d'honnêtes principes capables peut-être de fléchir, mais ayant jusqu'ici préservé de tout écart celles qui les avaient reçus de leur milieu bourgeois. Or quand on s'est trompé dès le début, même pour les petites choses, quand une erreur de supposition ou de souvenirs vous fait chercher l'auteur d'un potin malveillant ou l'endroit où on a égaré un objet dans une fausse direction, il peut arriver qu'on ne découvre son erreur que pour lui substituer non pas la vérité, mais une autre erreur. Je tirais, en ce qui concernait leur manière de vivre et la conduite à tenir avec elles, toutes les conséquences du mot innocence que j'avais lu, en causant familièrement avec elles, sur leur visage. Mais peut-être l'avais-je lu étourdiment, dans le lapsus d'un déchiffrage trop rapide, et n'y était-il pas plus écrit que le nom de Jules Ferry sur le programme de la matinée où j'avais entendu pour la première fois la Berma, ce qui ne m'avait pas empêché de soutenir à Norpois que Jules Ferry, sans doute possible, écrivait des levers de rideau.

Pour n'importe laquelle de mes amies de la petite bande, comment le dernier visage que je lui avais vu n'eût-il pas été le seul que je me rappelasse, puisque, de nos souvenirs relatifs à une personne, l'intelligence élimine tout ce qui ne concourt pas à l'utilité immédiate de nos relations quotidiennes (même et surtout si ces relations sont imprégnées d'amour, lequel, toujours insatisfait, vit dans le moment qui va venir). Elle laisse filer la chaîne des jours passés, n'en garde fortement que le dernier bout souvent d'un tout autre métal que les chaînons disparus dans la nuit, et dans le voyage que nous faisons à travers la vie, ne tient pour réel que le pays où nous sommes présentement. Toutes mes premières impressions, déjà si lointaines, ne pouvaient pas trouver contre leur déformation journalière un recours dans ma mémoire ; pendant les longues heures que je passais à causer, à goûter, à jouer avec ces jeunes filles, je ne me souvenais même pas qu'elles étaient les mêmes vierges impitoyables et sensuelles que j'avais vues, comme dans une fresque, défiler devant la mer.

Les géographes, les archéologues nous conduisent bien dans l'île de Calypso, exhument bien le palais de Minos. Seulement Calypso n'est plus qu'une femme, Minos qu'un roi sans rien de divin. Même les qualités et les défauts que l'histoire nous enseigne alors avoir été l'apanage de ces personnes fort réelles diffèrent souvent beaucoup de ceux que nous avions prêtés aux êtres fabuleux qui portaient le même nom. Ainsi s'était dissipée toute la gracieuse mythologie océanique que j'avais composée les premiers jours. Mais il n'est pas tout à fait indifférent qu'il nous arrive au moins quelquefois de passer notre temps dans la familiarité de ce que nous avons cru inaccessible et que nous avons désiré. Dans le commerce des personnes que nous avons d'abord trouvées désagréables, persiste toujours, même au milieu du plaisir factice qu'on peut finir par goûter auprès d'elles, le goût frelaté des défauts qu'elles ont réussi à dissimuler. Mais dans des relations comme celles que j'avais avec Albertine et ses amies, le plaisir vrai qui est à leur origine laisse ce parfum qu'aucun artifice ne parvient à donner aux fruits forcés, aux raisins qui n'ont pas mûri au soleil. Les créatures surnaturelles qu'elles avaient été un instant pour moi mettaient encore, même à mon insu, quelque merveilleux, dans les rapports les plus banals que j'avais avec elles, ou plutôt préservaient ces rapports d'avoir jamais rien de banal. Mon désir avait cherché avec tant d'avidité la signification des yeux qui maintenant me connaissaient et me souriaient, mais qui, le premier jour, avaient croisé mes regards comme des rayons d'un autre univers, il avait distribué si largement et si minutieusement la couleur et le parfum sur les surfaces carnées de ces jeunes filles qui, étendues sur la falaise, me tendaient simplement des sandwiches ou jouaient aux devinettes, que souvent dans l'après-midi, pendant que j'étais allongé, comme ces peintres qui cherchant la grandeur de l'antique dans la vie moderne donnent à une femme qui se coupe un ongle de pied la noblesse du « Tireur d'épine » ou qui comme Rubens, font des déesses avec des femmes de leur connaissance pour composer une scène mythologique, ces beaux corps bruns et blonds, de types si opposés, répandus autour de moi dans l'herbe, je les regardais sans les vider peut-être de tout le médiocre contenu dont l'existence journalière les avait remplis, et pourtant sans me rappeler expressément leur céleste origine, comme si pareil à Hercule ou à Télémaque, j'avais été en train de jouer au milieu des nymphes.

Puis les concerts finirent, le mauvais temps arriva, mes amies quittèrent Balbec, non pas toutes ensemble, comme les hirondelles, mais dans la même semaine. Albertine s'en alla la première, brusquement, sans qu'aucune de ses amies eût pu comprendre, ni alors, ni plus tard, pourquoi elle était rentrée tout à coup à Paris, où ni travaux, ni distractions ne la rappelaient. « Elle n'a dit ni quoi ni qu'est-ce et puis elle est partie », grommelait Françoise qui aurait d'ailleurs voulu que nous en fissions autant. Elle nous trouvait indiscrets vis-à-vis des employés, pourtant déjà bien réduits en nombre, mais retenus par les rares clients qui restaient, vis-à-vis du directeur qui « mangeait de l'argent ». Il est vrai que depuis longtemps l'hôtel qui n'allait pas tarder à fermer avait vu partir presque tout le monde ; jamais il n'avait été aussi agréable. Ce n'était pas l'avis du directeur ; tout le long des salons où l'on gelait et à la porte desquels ne veillait plus aucun groom, il arpentait les corridors, vêtu d'une redingote neuve, si soigné par le coiffeur que sa figure fade avait l'air de consister en un mélange où pour une partie de chair il y en aurait eu trois de cosmétique, changeant sans cesse de cravates (ces élégances coûtent moins cher que d'assurer le chauffage et de garder le personnel, et tel qui ne peut plus envoyer dix mille francs à une oeuvre de bienfaisance fait encore sans peine le généreux en donnant cent sous de pourboire au télégraphiste qui lui apporte une dépêche). Il avait l'air d'inspecter le néant, de vouloir donner, grâce à sa bonne tenue personnelle, un air provisoire à la misère que l'on sentait dans cet hôtel où la saison n'avait pas été bonne, et paraissait comme le fantôme d'un souverain qui revient hanter les ruines de ce qui fut jadis son palais. Il fut surtout mécontent quand le chemin de fer d'intérêt local, qui n'avait plus assez de voyageurs, cessa de fonctionner pour jusqu'au printemps suivant. « Ce qui manque ici, disait le directeur, ce sont le moyens de commotion. » Malgré le déficit qu'il enregistrait, il faisait pour les années suivantes des projets grandioses. Et comme il était tout de même capable de retenir exactement de belles expressions, quand elles s'appliquaient à l'industrie hôtelière et avaient pour effet de la magnifier : « Je n'étais pas suffisamment secondé quoique à la salle à manger j'avais une bonne équipe, disait-il ; mais les chasseurs laissaient un peu à désirer ; vous verrez l'année prochaine quelle phalange je saurai réunir. » En attendant, l'interruption des services du B.C.B. l'obligeait à envoyer chercher les lettres et quelquefois conduire les voyageurs dans une carriole. Je demandais souvent à monter à côté du cocher et cela me fit faire des promenades par tous les temps, comme dans l'hiver que j'avais passé à Combray.

Parfois pourtant la pluie trop cinglante nous retenait, ma grand'mère et moi, le Casino étant fermé, dans des pièces presque complètement vides comme à fond de cale d'un bateau quand le vent souffle, et où chaque jour, comme au cours d'une traversée, une nouvelle personne d'entre celles près de qui nous avions passé trois mois sans les connaître, le premier président de Rennes, la bâtonnier de Caen, une dame américaine et ses filles, venaient à nous, entamaient la conversation, inventaient quelque manière de trouver les heures moins longues, révélaient un talent, nous enseignaient un jeu, nous invitaient à prendre le thé, ou à faire de la musique, à nous réunir à une certaine heure, à combiner ensemble de ces distractions qui possèdent le vrai secret de nous faire donner du plaisir, lequel est de n'y pas prétendre, mais seulement de nous aider à passer le temps de notre ennui, enfin nouaient avec nous sur la fin de notre séjour des amitiés que le lendemain leurs départs successifs venaient interrompre. Je fis même la connaissance du jeune homme riche, d'un de ses deux amis nobles et de l'actrice qui était revenue pour quelques jours ; mais la petite société ne se composait plus que de trois personnes, l'autre ami était rentré à Paris. Ils me demandèrent de venir dîner avec eux dans leur restaurant. Je crois qu'ils furent assez contents que je n'acceptasse pas. Mais ils avaient fait l'invitation le plus aimablement possible, et bien qu'elle vînt en réalité du jeune homme riche, puisque les autres personnes n'étaient que ses hôtes, comme l'ami qui l'accompagnait, le marquis Maurice de Vaudémont, était de très grande maison, instinctivement l'actrice, en me demandant si je ne voudrais pas venir, me dit pour me flatter :
