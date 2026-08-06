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
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Bergotte",
      "surface_forms": [
        "Bergotte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.78,
      "evidence": "« le sentiment de Gilberte pour moi, trop ancien déjà pour pouvoir changer, c'était l'indifférence ; que dans mon amitié avec Gilberte, c'est moi seul qui aimais. »",
      "explanation": "The narrator, guided by the « ouvrière invisible » who reorders the facts, concludes that Gilberte is indifferent and does not share his love. This clarification locally diminishes Gilberte in the affective economy of the scene (non-lover, detached)."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "She is posited as indifferent and not invested, which locally diminishes her as an affective partner."
    }
  ],
  "ambiguities": [
    "Le constat d’indifférence provient d’une déduction intérieure du narrateur (l’« ouvrière invisible ») plutôt que d’un aveu explicite de Gilberte, même si le texte présente cette conclusion comme convaincante.",
    "Le narrateur loue brièvement la bonté de Gilberte (pour la page de Bergotte), ce qui nuance la pure diminution, mais la dynamique dominante reste la reconnaissance de son indifférence."
  ],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-35-p-40"
}

### Candidate characters

[
  "Françoise",
  "Odette",
  "Swann",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Je me redisais en étouffant mes sanglots les mots où Gilberte avait laissé éclater sa joie de ne pas venir de longtemps aux Champs-Élysées. Mais déjà le charme dont, par son simple fonctionnement, se remplissait mon esprit dès qu'il songeait à elle, la position particulière, unique – fût elle affligeante – où me plaçait inévitablement, par rapport à Gilberte, la contrainte interne d'un pli mental, avaient commencé à ajouter, même à cette marque d'indifférence, quelque chose de romanesque, et au milieu de mes larmes se formait un sourire qui n'était que l'ébauche timide d'un baiser. Et quand vint l'heure du courrier, je me dis ce soir-là comme tous les autres : « Je vais recevoir une lettre de Gilberte, elle va me dire enfin qu'elle n'a jamais cessé de m'aimer, et m'expliquera la raison mystérieuse pour laquelle elle a été forcée de me le cacher jusqu'ici, de faire semblant de pouvoir être heureuse sans me voir, la raison pour laquelle elle a pris l'apparence de la Gilberte simple camarade. »

### Passage

Tous les soirs je me plaisais à imaginer cette lettre, je croyais la lire, je m'en récitais chaque phrase. Tout d'un coup je m'arrêtais effrayé. Je comprenais que si je devais recevoir une lettre de Gilberte, ce ne pourrait pas en tous cas être celle-là, puisque c'était moi qui venais de la composer. Et dès lors, je m'efforçais de détourner ma pensée des mots que j'aurais aimé qu'elle m'écrivît, par peur, en les énonçant, d'exclure justement ceux-là – les plus chers, les plus désirés – du champ des réalisations possibles. Même si par une invraisemblable coïncidence, c'eût été justement la lettre que j'avais inventée que de son côté m'eût adressée Gilberte, y reconnaissant mon oeuvre, je n'eusse pas eu l'impression de recevoir quelque chose qui ne vînt pas de moi, quelque chose de réel, de nouveau, un bonheur extérieur à mon esprit, indépendant de ma volonté, vraiment donné par l'amour.

En attendant je relisais une page que ne m'avait pas écrite Gilberte, mais qui du moins me venait d'elle, cette page de Bergotte sur la beauté des vieux mythes dont s'est inspiré Racine, et que, à côté de la bille d'agathe, je gardais toujours auprès de moi. J'étais attendri par la bonté de mon amie qui me l'avait fait rechercher ; et comme chacun a besoin de trouver des raisons à sa passion, jusqu'à être heureux de reconnaître dans l'être qu'il aime des qualités que la littérature ou la conversation lui ont appris être de celles qui sont dignes d'exciter l'amour, jusqu'à les assimiler par imitation et en faire des raisons nouvelles de son amour, ces qualités fussent-elles les plus oppressées à celles que cet amour eût recherchées tant qu'il était spontané – comme Swann autrefois le caractère esthétique de la beauté d'Odette – moi, qui avais d'abord aimé Gilberte, dès Combray, à cause de tout l'inconnu de sa vie, dans lequel j'aurais voulu me précipiter, m'incarner, en délaissant la mienne qui ne m'était plus rien, je pensais maintenant comme à un inestimable avantage, que de cette mienne vie trop connue, dédaignée, Gilberte pourrait devenir un jour l'humble servante, la commode et confortable collaboratrice, qui le soir, m'aidant dans mes travaux, collationnerait pour moi des brochures. Quant à Bergotte, ce vieillard infiniment sage et presque divin à cause de qui j'avais d'abord aimé Gilberte, avant même de l'avoir vue, maintenant c'était surtout à cause de Gilberte que je l'aimais. Avec autant de plaisir que les pages qu'il avait écrites sur Racine, je regardais le papier fermé de grands cachets de cire blancs et noué d'un flot de rubans mauves dans lequel elle me les avait apportées. Je baisai la bille d'agate qui était la meilleure part du coeur de mon amie, la part qui n'était pas frivole, mais fidèle, et qui bien que parée du charme mystérieux de la vie de Gilberte demeurait près de moi, habitait ma chambre, couchait dans mon lit. Mais la beauté de cette pierre, et la beauté aussi de ces pages de Bergotte, que j'étais heureux d'associer à l'idée de mon amour pour Gilberte comme si dans les moments où celui-ci ne m'apparaissait plus que comme un néant, elles lui donnaient une sorte de consistance, je m'apercevais qu'elles étaient antérieures à cet amour, qu'elles ne lui ressemblaient pas, que leurs éléments avaient été fixés par le talent ou par les lois minéralogiques avant que Gilberte ne me connût, que rien dans le livre ni dans la pierre n'eût été autre si Gilberte ne m'avait pas aimé, et que rien par conséquent ne m'autorisait à lire en eux un message de bonheur. Et tandis que mon amour, attendant sans cesse du lendemain l'aveu de celui de Gilberte, annulait, défaisait chaque soir le travail mal fait de la journée, dans l'ombre de moi-même une ouvrière inconnue ne laissait pas au rebut les fils arrachés, et les disposait, sans souci de me plaire et de travailler à mon bonheur, dans un ordre différent qu'elle donnait à tous ses ouvrages. Ne portant aucun intérêt particulier à mon amour, ne commençant pas par décider que j'étais aimé, elle recueillait les actions de Gilberte qui m'avaient semblé inexplicables et ses fautes que j'avais excusées. Alors les unes et les autres prenaient un sens. Il semblait dire, cet ordre nouveau, qu'en voyant Gilberte, au lieu qu'elle vînt aux Champs-Élysées, aller à une matinée, faire des courses avec son institutrice et se préparer à une absence pour les vacances du jour de l'an, j'avais tort de penser, de me dire : « c'est qu'elle est frivole ou docile. » Car elle eût cessé d'être l'un ou l'autre si elle m'avait aimé, et si elle avait été forcée d'obéir, c'eût été avec le même désespoir que j'avais les jours où je ne la voyais pas. Il disait encore, cet ordre nouveau, que je devais pourtant savoir ce que c'était qu'aimer puisque j'aimais Gilberte ; il me faisait remarquer le souci perpétuel que j'avais de me faire valoir à ses yeux, à cause duquel j'essayais de persuader à ma mère d'acheter à Françoise un caoutchouc et un chapeau avec un plumet bleu, ou plutôt de ne plus m'envoyer aux Champs-Élysées avec cette bonne dont je rougissais (à quoi ma mère répondait que j'étais injuste pour Françoise, que c'était une brave femme qui nous était dévouée), et aussi ce besoin unique de voir Gilberte qui faisait que des mois d'avance je ne pensais qu'à tâcher d'apprendre à quelle époque elle quitterait Paris et où elle irait, trouvant le pays le plus agréable un lieu d'exil si elle ne devait pas y être, et ne désirant que rester toujours à Paris tant que je pourrais la voir aux Champs-Élysées ; et il n'avait pas de peine à me montrer que ce souci-là, ni ce besoin, je ne les trouverais sous les actions de Gilberte. Elle au contraire appréciait son institutrice, sans s'inquiéter de ce que j'en pensais. Elle trouvait naturel de ne pas venir aux Champs-Élysées, si c'était pour aller faire des emplettes avec Mademoiselle, agréable si c'était pour sortir avec sa mère. Et à supposer même qu'elle m'eût permis d'aller passer les vacances au même endroit qu'elle, du moins pour choisir cet endroit elle s'occupait du désir de ses parents, de mille amusements dont on lui avait parlé et nullement que ce fût celui où ma famille avait l'intention de m'envoyer. Quand elle m'assurait parfois qu'elle m'aimait moins qu'un de ses amis, moins qu'elle ne m'aimait la veille, parce que je lui avais fait perdre sa partie par une négligence, je lui demandais pardon, je lui demandais ce qu'il fallait faire pour qu'elle recommençât à m'aimer autant, pour qu'elle m'aimât plus que les autres ; je voulais qu'elle me dît que c'était déjà fait, je l'en suppliais comme si elle avait pu modifier son affection pour moi à son gré, au mien, pour me faire plaisir, rien que par les mots qu'elle dirait, selon ma bonne ou ma mauvaise conduite. Ne savais-je donc pas que ce que j'éprouvais, moi, pour elle, ne dépendait ni de ses actions, ni de ma volonté ?

Il disait enfin, l'ordre nouveau dessiné par l'ouvrière invisible, que si nous pouvons désirer que les actions d'une personne qui nous a peinés jusqu'ici n'aient pas été sincères, il y a dans leur suite une clarté contre quoi notre désir ne peut rien et à laquelle, plutôt qu'à lui, nous devons demander quelles seront ses actions de demain.

Ces paroles nouvelles, mon amour les entendait ; elles le persuadaient que le lendemain ne serait pas différent de ce qu'avaient été tous les autres jours ; que le sentiment de Gilberte pour moi, trop ancien déjà pour pouvoir changer, c'était l'indifférence ; que dans mon amitié avec Gilberte, c'est moi seul qui aimais. « C'est vrai, répondait mon amour, il n'y a plus rien à faire de cette amitié-là, elle ne changera pas. » Alors dès le lendemain (ou attendant une fête s'il y en avait une prochaine, un anniversaire, le nouvel an peut-être, un de ces jours qui ne sont pas pareils aux autres, où le temps recommence sur de nouveaux frais en rejetant l'héritage du passé, en n'acceptant pas le legs de ses tristesses) je demandais à Gilberte de renoncer à notre amitié ancienne et de jeter les bases d'une nouvelle amitié.

J'avais toujours à portée de ma main un plan de Paris qui, parce qu'on pouvait y distinguer la rue où habitaient M. et Odette, me semblait contenir un trésor. Et par plaisir, par une sorte de fidélité chevaleresque aussi, à propos de n'importe quoi, je disais le nom de cette rue, si bien que mon père me demandait, n'étant pas comme ma mère et ma grand'mère au courant de mon amour :

– Mais pourquoi parles-tu tout le temps de cette rue, elle n'a rien d'extraordinaire, elle est très agréable à habiter parce qu'elle est à deux pas du Bois, mais il y en a dix autres dans le même cas.
