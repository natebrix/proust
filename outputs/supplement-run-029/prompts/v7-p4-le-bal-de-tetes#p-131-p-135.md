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
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [],
  "status_effects": [],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-131-p-135"
}

### Candidate characters

[
  "Albertine",
  "Swann",
  "la grand-mère",
  "la mère du narrateur",
  "le directeur",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

Moi, c'était autre chose que les adieux d'un mourant à sa femme que j'avais à écrire, de plus long et à plus d'une personne. Long à écrire. Le jour, tout au plus pourrais-je essayer de dormir. Si je travaillais, ce ne serait que la nuit. Mais il me faudrait beaucoup de nuits, peut-être cent, peut-être mille. Et je vivrais dans l'anxiété de ne pas savoir si le Maître de ma destinée, moins indulgent que le sultan Sheriar, le matin, quand j'interromprais mon récit, voudrait bien surseoir à mon arrêt de mort et me permettrait de reprendre la suite le prochain soir. Non pas que je prétendisse refaire, en quoi que ce fût, les Mille et une Nuits, pas plus que les Mémoires de Saint-Simon, écrits eux aussi la nuit, pas plus qu'aucun des livres que j'avais tant aimés et desquels, dans ma naïveté d'enfant, superstitieusement attaché à eux comme à mes amours, je ne pouvais sans horreur imaginer une oeuvre qui serait différente. Mais, comme Elstir, comme Chardin, on ne peut refaire ce qu'on aime qu'en le renonçant. Sans doute mes livres, eux aussi, comme mon être de chair, finiraient un jour par mourir. Mais il faut se résigner à mourir. On accepte la pensée que dans dix ans soi-même, dans cent ans ses livres, ne seront plus. La durée éternelle n'est pas plus promise aux oeuvres qu'aux hommes.

### Passage

Ce serait un livre aussi long que les Mille et une Nuits peut-être, mais tout autre. Sans doute, quand on est amoureux d'une oeuvre, on voudrait faire quelque chose de tout pareil, mais il faut sacrifier son amour du moment et ne pas penser à son goût, mais à une vérité qui ne nous demande pas nos préférences et nous défend d'y songer. Et c'est seulement si on la suit qu'on se trouve parfois rencontrer ce qu'on a abandonné, et avoir écrit, en les oubliant, les Contes arabes ou les Mémoires de Saint-Simon d'une autre époque. Mais était-il encore temps pour moi ? n'était-il pas trop tard ?

En tout cas, si j'avais encore la force d'accomplir mon oeuvre, je sentais que la nature des circonstances qui m'avaient, aujourd'hui même, au cours de cette matinée chez la princesse de Guermantes, donné à la fois l'idée de mon oeuvre et la crainte de ne pouvoir la réaliser, marquerait certainement avant tout, dans celle-ci, la forme que j'avais pressentie autrefois dans l'église de Combray, au cours de certains jours qui avaient tant influé sur moi – et qui nous reste habituellement invisible – la forme du Temps. Cette dimension du Temps, que j'avais jadis pressentie dans l'église de Combray, je tâcherais de la rendre continuellement sensible dans une transcription du monde qui serait forcément bien différente de celle que nous donnent nos sens si mensongers. Certes, il est bien d'autres erreurs de nos sens – on a vu que divers épisodes de ce récit me l'avaient prouvé – qui faussent pour nous l'aspect réel de ce monde. Mais enfin, je pourrais, à la rigueur, dans la transcription plus exacte que je m'efforcerais de donner, ne pas changer la place des sons, m'abstenir de les détacher de leur cause, à côté de laquelle l'intelligence les situe après coup, bien que faire chanter la pluie au milieu de la chambre et tomber en déluge dans la cour l'ébullition de notre tisane ne doit pas être, en somme, plus déconcertant que ce qu'ont fait si souvent les peintres quand ils peignent, très près ou très loin de nous, selon que les lois de la perspective, l'intensité des couleurs et la première illusion du regard nous les font apparaître, une voile ou un pic que le raisonnement déplacera ensuite de distances quelquefois énormes.

Je pourrais, bien que l'erreur soit plus grave, continuer, comme on fait, à mettre des traits dans le visage d'une passante, alors qu'à la place du nez, des joues et du menton, il ne devrait y avoir qu'un espace vide sur lequel jouerait tout au plus le reflet de nos désirs. Et même, si je n'avais pas le loisir de préparer, chose déjà bien plus importante, les cent masques qu'il convient d'attacher à un même visage, ne fût-ce que selon les yeux qui le voient et le sens où ils en lisent les traits et, pour les mêmes yeux, selon l'espérance ou la crainte, ou au contraire l'amour et l'habitude qui cachent pendant tant d'années les changements de l'âge, même enfin si je n'entreprenais pas, ce dont ma liaison avec Albertine suffisait pourtant à me montrer que sans cela tout est factice et mensonger, de représenter certaines personnes non pas au dehors, mais en dedans de nous où leurs moindres actes peuvent amener des troubles mortels, et de faire varier aussi la lumière du ciel moral selon les différences de pression de notre sensibilité ou selon la sérénité de notre certitude, sous laquelle un objet est si petit alors qu'un simple nuage de risque en multiplie en un moment la grandeur, si je ne pouvais apporter ces changements et bien d'autres (dont la nécessité, si on veut peindre le réel, a pu apparaître au cours de ce récit) dans la transcription d'un univers qui était à redessiner tout entier, du moins ne manquerais-je pas avant toute chose d'y décrire l'homme comme ayant la longueur non de son corps mais de ses années, comme devant, tâche de plus en plus énorme et qui finit par le vaincre, les traîner avec lui quand il se déplace. D'ailleurs, que nous occupions une place sans cesse accrue dans le Temps, tout le monde le sent, et cette universalité ne pouvait que me réjouir puisque c'est la vérité, la vérité soupçonnée par chacun, que je devais chercher à élucider. Non seulement tout le monde sent que nous occupons une place dans le Temps, mais, cette place, le plus simple la mesure approximativement comme il mesurerait celle que nous occupons dans l'espace. Sans doute, on se trompe souvent dans cette évaluation, mais qu'on ait cru pouvoir la faire signifie qu'on concevait l'âge comme quelque chose de mesurable.

Je me disais aussi : « Non seulement est-il encore temps, mais suis-je en état d'accomplir mon oeuvre ? » La maladie qui, en me faisant, comme un rude directeur de conscience, mourir au monde, m'avait rendu service (car si le grain de froment ne meurt après qu'on l'a semé, il restera seul, mais s'il meurt, il portera beaucoup de fruits), la maladie qui, après que la paresse m'avait protégé contre la facilité, allait peut-être me garder contre la paresse, la maladie avait usé mes forces et, comme je l'avais remarqué depuis longtemps, au moment où j'avais cessé d'aimer Albertine, les forces de ma mémoire. Or la recréation par la mémoire d'impressions qu'il fallait ensuite approfondir, éclairer, transformer en équivalents d'intelligence, n'était-elle pas une des conditions, presque l'essence même de l'oeuvre d'art telle que je l'avais conçue tout à l'heure dans la bibliothèque ? Ah ! si j'avais encore eu les forces qui étaient intactes dans la soirée que j'avais alors évoquée en apercevant François le Champi ? C'était de cette soirée, où ma mère avait abdiqué, que datait, avec la mort lente de ma grand'mère, le déclin de ma volonté, de ma santé. Tout s'était décidé au moment où, ne pouvant plus supporter d'attendre au lendemain pour poser mes lèvres sur le visage de ma mère, j'avais pris ma résolution, j'avais sauté du lit et étais allé, en chemise de nuit, m'installer à la fenêtre par où entrait le clair de lune jusqu'à ce que j'eusse entendu partir Swann. Mes parents l'avaient accompagné, j'avais entendu la porte s'ouvrir, sonner, se refermer. À ce moment même, dans l'hôtel du prince de Guermantes, ce bruit de pas de mes parents reconduisant Swann, ce tintement rebondissant, ferrugineux, interminable, criard et frais de la petite sonnette, qui m'annonçait qu'enfin Swann était parti et que maman allait monter, je les entendais encore, je les entendais eux-mêmes, eux situés pourtant si loin dans le passé. Alors, en pensant à tous les événements qui se plaçaient forcément entre l'instant où je les avais entendus et la matinée Guermantes, je fus effrayé de penser que c'était bien cette sonnette qui tintait encore en moi, sans que je pusse rien changer aux criaillements de son grelot, puisque, ne me rappelant plus bien comment ils s'éteignaient, pour le réapprendre, pour bien l'écouter, je dus m'efforcer de ne plus entendre le son des conversations que les masques tenaient autour de moi. Pour tâcher de l'entendre de plus près, c'est en moi-même que j'étais obligé de redescendre. C'est donc que ce tintement y était toujours, et aussi, entre lui et l'instant présent, tout ce passé indéfiniment déroulé que je ne savais pas que je portais. Quand il avait tinté j'existais déjà et, depuis, pour que j'entendisse encore ce tintement, il fallait qu'il n'y eût pas eu discontinuité, que je n'eusse pas un instant pris de repos, cessé d'exister, de penser, d'avoir conscience de moi, puisque cet instant ancien tenait encore à moi, que je pouvais encore le retrouver, retourner jusqu'à lui, rien qu'en descendant plus profondément en moi. C'était cette notion du temps incorporé, des années passées non séparées de nous, que j'avais maintenant l'intention de mettre si fort en relief dans mon oeuvre. Et c'est parce qu'ils contiennent ainsi les heures du passé que les corps humains peuvent faire tant de mal à ceux qui les aiment, parce qu'ils contiennent tant de souvenirs, de joies et de désirs déjà effacés pour eux, mais si cruels pour celui qui contemple et prolonge dans l'ordre du temps le corps chéri dont il est jaloux, jaloux jusqu'à en souhaiter la destruction. Car après la mort le Temps se retire du corps et les souvenirs – si indifférents, si pâlis – sont effacés de celle qui n'est plus et le seront bientôt de celui qu'ils torturent encore, eux qui finiront par périr quand le désir d'un corps vivant ne les entretiendra plus.

J'éprouvais un sentiment de fatigue profonde à sentir que tout ce temps si long non seulement avait sans une interruption été vécu, pensé, sécrété par moi, qu'il était ma vie, qu'il était moi-même, mais encore que j'avais à toute minute à le maintenir attaché à moi, qu'il me supportait, que j'étais juché à son sommet vertigineux, que je ne pouvais me mouvoir sans le déplacer avec moi.
