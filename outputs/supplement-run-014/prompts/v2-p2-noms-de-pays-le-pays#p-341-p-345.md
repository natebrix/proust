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
      "canonical_name": "jeune blonde de Rivebelle",
      "surface_forms": [
        "la jeune blonde de Rivebelle",
        "jeune blonde de Rivebelle"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "jeune blonde de Rivebelle",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« je me rappelai la jeune blonde de Rivebelle… elle venait seule de s’élever du fond de mon souvenir… j’étais prêt à tout pour cela, je ne pensais plus qu’à elle »",
      "explanation": "The narrator’s attention fixes exclusively on the young blonde from Rivebelle; he recalls her, assumes she noticed him, and declares he thinks only of her, signaling strong local elevation through desire."
    }
  ],
  "status_effects": [
    {
      "character": "jeune blonde de Rivebelle",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "She gains local esteem and desirability as the narrator’s singular focus of interest and attraction."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-341-p-345"
}

### Candidate characters

[
  "Robert de Saint-Loup",
  "le narrateur"
]

### Prior local context (optional)

Ce n'est pas assez dire que j'avais rejoint le calme et la santé, car c'était plus qu'une simple distance qui les avait la veille séparés de moi, j'avais eu toute la nuit à lutter contre un flot contraire, et puis je ne me retrouvais pas seulement auprès d'eux, ils étaient rentrés en moi. À des points précis et encore un peu douloureux de ma tête vide et qui serait un jour brisée, laissant mes idées s'échapper à jamais, celles-ci avaient une fois encore repris leur place, et retrouvé cette existence dont hélas ! jusqu'ici elles n'avaient pas su profiter.

### Passage

Une fois de plus j'avais échappé à l'impossibilité de dormir, au déluge, au naufrage des crises nerveuses. Je ne craignais plus du tout ce qui me menaçait la veille au soir quand j'étais démuni de repos. Une nouvelle vie s'ouvrait devant moi ; sans faire un seul mouvement, car j'étais encore brisé quoique déjà dispos, je goûtais ma fatigue avec allégresse ; elle avait isolé et rompu les os de mes jambes, de mes bras, que je sentais assemblés devant moi, prêts à se rejoindre, et que j'allais relever rien qu'en chantant comme l'architecte de la fable.

Tout à coup je me rappelai la jeune blonde à l'air triste que j'avais vue à Rivebelle et qui m'avait regardé un instant. Pendant toute la soirée, bien d'autres m'avaient semblé agréables, maintenant elle venait seule de s'élever du fond de mon souvenir. Il me semblait qu'elle m'avait remarqué, je m'attendais à ce qu'un des garçons de Rivebelle vînt me dire un mot de sa part. Saint-Loup ne la connaissait pas et croyait qu'elle était comme il faut. Il serait bien difficile de la voir, de la voir sans cesse. Mais j'étais prêt à tout pour cela, je ne pensais plus qu'à elle. La philosophie parle souvent d'actes libres et d'actes nécessaires. Peut-être n'en est-il pas de plus complètement subi par nous, que celui qui en vertu d'une force ascensionnelle comprimée pendant l'action, fait jusque-là, une fois notre pensée au repos, remonter ainsi un souvenir nivelé avec les autres par la force oppressive de la distraction, et s'élancer parce qu'à notre insu il contenait plus que les autres un charme dont nous ne nous apercevons que vingt-quatre heures après. Et peut-être n'y a-t-il pas non plus d'acte aussi libre, car il est encore dépourvu de l'habitude, de cette sorte de manie mentale qui, dans l'amour, favorise la renaissance exclusive de l'image d'une certaine personne.

Ce jour-là était justement le lendemain de celui où j'avais vu défiler devant la mer le beau cortège de jeunes filles. J'interrogeai à leur sujet plusieurs clients de l'hôtel, qui venaient presque tous les ans à Balbec. Ils ne purent me renseigner. Plus tard une photographie m'expliqua pourquoi. Qui eût pu reconnaître maintenant en elles, à peine mais déjà sorties d'un âge où on change si complètement, telle masse amorphe et délicieuse, encore tout enfantine, de petites filles que, quelques années seulement auparavant, on pouvait voir assises en cercle sur le sable, autour d'une tente : sorte de blanche et vague constellation où l'on n'eût distingué deux yeux plus brillants que les autres, un malicieux visage, des cheveux blonds, que pour les reperdre et les confondre bien vite au sein de la nébuleuse indistincte et lactée.

Sans doute en ces années-là encore si peu éloignées, ce n'était pas comme la veille dans leur première apparition devant moi, la vision du groupe, mais le groupe lui-même qui manquait de netteté. Alors, ces enfants trop jeunes étaient encore à ce degré élémentaire de formation où la personnalité n'a pas mis son sceau sur chaque visage. Comme ces organismes primitifs où l'individu n'existe guère par lui-même, est plutôt constitué par le polypier que par chacun des polypes qui le composent, elles restaient pressées les unes contre les autres. Parfois l'une faisait tomber sa voisine, et alors un fou rire qui semblait la seule manifestation de leur vie personnelle, les agitait toutes à la fois, effaçant, confondant ces visages indécis et grimaçants dans la gelée d'une seule grappe scintillatrice et tremblante. Dans une photographie ancienne qu'elles devaient me donner un jour, et que j'ai gardée, leur troupe enfantine offre déjà le même nombre de figurantes, que plus tard leur cortège féminin ; on y sent qu'elles devaient déjà faire sur la plage une tache singulière qui forçait à les regarder, mais on ne peut les y reconnaître individuellement que par le raisonnement, en laissant le champ libre à toutes les transformations possibles pendant la jeunesse jusqu'à la limite où ces formes reconstituées empiéteraient sur une autre individualité qu'il faut identifier aussi et dont le beau visage, à cause de la concomitance d'une grande taille et de cheveux frisés, a chance d'avoir été jadis ce ratatinement de grimace rabougrie présenté par la carte-album ; et la distance parcourue en peu de temps par les caractères physiques de chacune de ces jeunes filles faisant d'eux un critérium fort vague, et d'autre part ce qu'elles avaient de commun et comme de collectif étant dès lors marqué, il arrivait parfois à leurs meilleures amies de les prendre l'une pour l'autre sur cette photographie, si bien que le doute ne pouvait finalement être tranché que par tel accessoire de toilette que l'une était certaine d'avoir porté, à l'exclusion des autres. Depuis ces jours si différents de celui où je venais de les voir sur la digue, si différents et pourtant si proches, elles se laissaient encore aller au rire comme je m'en étais rendu compte la veille, mais à un rire qui n'était pas celui intermittent et presque automatique de l'enfance, détente spasmodique qui autrefois faisait à tous moments faire un plongeon à ces têtes comme les blocs de vairons dans la Vivonne se dispersaient et disparaissaient pour se reformer un instant après ; leurs physionomies maintenant étaient devenues maîtresses d'elles-mêmes, leurs yeux étaient fixés sur le but qu'ils poursuivaient ; et il avait fallu hier l'indécision et le tremblé de ma perception première pour confondre indistinctement, comme l'avaient fait l'hilarité ancienne et la vieille photographie, les sporades aujourd'hui individualisées et désunies du pâle madrépore.

Sans doute bien des fois, au passage de jolies jeunes filles, je m'étais fait la promesse de les revoir. D'habitude, elles ne reparaissent pas ; d'ailleurs la mémoire, qui oublie vite leur existence, retrouverait difficilement leurs traits ; nos yeux ne les reconnaîtraient peut-être pas, et déjà nous avons vu passer de nouvelles jeunes filles que nous ne reverrons pas non plus. Mais d'autres fois, et c'est ainsi que cela devait arriver pour la petite bande insolente, le hasard les ramène avec insistance devant nous. Il nous paraît alors beau, car nous discernons en lui comme un commencement d'organisation, d'effort, pour composer notre vie ; il nous rend facile, inévitable et quelquefois – après des interruptions qui ont pu faire espérer de cesser de nous souvenir – cruelle la fidélité des images à la possession desquelles nous nous croirons plus tard avoir été prédestinés, et que sans lui nous aurions pu, tout au début, oublier, comme tant d'autres, si aisément.
