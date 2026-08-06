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
  },
  "Norpois": {
    "aliases": [
      "Norpois",
      "M. de Norpois",
      "le marquis de Norpois"
    ]
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "l'Ambassadeur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
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
      "source": "Norpois",
      "target": "Françoise",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "ironized",
      "confidence": 0.83,
      "evidence": "Françoise ... ne pouvait pas « tenir son sérieux » si on lui rappelait qu'elle avait été traitée par l'Ambassadeur de « chef de premier ordre », ce que la mère du narrateur était allée lui transmettre comme un ministre de la guerre les félicitations d'un souverain de passage après « la Revue ».",
      "explanation": "Norpois's high-status compliment publicly recognizes Françoise’s culinary art. The narrator frames the scene with gentle irony (ceremonial delivery of the praise), but the compliment itself stands and is locally consequential."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Being hailed as a 'chef de premier ordre' by an ambassador and having the praise formally relayed elevates her standing in the household and affirms her professional pride."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-89-p-98"
}

### Candidate characters

[
  "Odette",
  "Swann",
  "la Berma",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

J'avais ajouté ces derniers mots par scrupule et pour ne pas avoir l'air de m'être vanté d'une relation que je n'avais pas. Mais en les prononçant, je sentais qu'ils étaient déjà devenus inutiles, car dès le début de mon remerciement, d'une ardeur réfrigérante, j'avais vu passer sur le visage de l'Ambassadeur une expression d'hésitation et de mécontentement, et dans ses yeux ce regard vertical, étroit et oblique (comme, dans le dessin en perspective d'un solide, la ligne fuyante d'une de ses faces), regard qui s'adresse à cet interlocuteur invisible qu'on a en soi-même, au moment où on lui dit quelque chose que l'autre interlocuteur, le Monsieur avec qui on parlait jusqu'ici – moi dans la circonstance – ne doit pas entendre. Je me rendis compte aussitôt que ces phrases que j'avais prononcées et qui, faibles encore auprès de l'effusion reconnaissante dont j'étais envahi, m'avaient paru devoir toucher Norpois et achever de le décider à une intervention qui lui eût donné si peu de peine, et à moi tant de joie, étaient peut-être (entre toutes celles qu'eussent pu chercher diaboliquement des personnes qui m'eussent voulu du mal) les seules qui pussent avoir pour résultat de l'y faire renoncer. En les entendant en effet, de même qu'au moment où un inconnu, avec qui nous venions d'échanger agréablement des impressions que nous avions pu croire semblables sur des passants que nous nous accordions à trouver vulgaires, nous montre tout à coup l'abîme pathologique qui le sépare de nous en ajoutant négligemment tout en tâtant sa poche : « C'est malheureux que je n'aie pas mon revolver, il n'en serait pas resté un seul », Norpois qui savait que rien n'était moins précieux ni plus aisé que d'être recommandé à Odette et introduit chez elle, et qui vit que pour moi, au contraire, cela présentait un tel prix, par conséquent, sans doute, une grande difficulté, pensa que le désir, normal en apparence, que j'avais exprimé, devait dissimuler quelque pensée différente, quelque visée suspecte, quelque faute antérieure, à cause de quoi, dans la certitude de déplaire à Odette, personne n'avait jusqu'ici voulu se charger de lui transmettre une commission de ma part. Et je compris que cette commission, il ne la ferait jamais, qu'il pourrait voir Odette quotidiennement pendant des années, sans pour cela lui parler une seule fois de moi. Il lui demanda cependant quelques jours plus tard un renseignement que je désirais et chargea mon père de me le transmettre. Mais il n'avait pas cru devoir dire pour qui il le demandait. Elle n'apprendrait donc pas que je connaissais Norpois et que je souhaitais tant d'aller chez elle ; et ce fut peut-être un malheur moins grand que je ne croyais. Car la seconde de ces nouvelles n'eût probablement pas beaucoup ajouté à l'efficacité, d'ailleurs incertaine, de la première. Pour Odette, l'idée de sa propre vie et de sa demeure n'éveillant aucun trouble mystérieux, une personne qui la connaissait, qui allait chez elle, ne lui semblait pas un être fabuleux comme il le paraissait à moi qui aurais jeté dans les fenêtres de Swann une pierre si j'avais pu écrire sur elle que je connaissais Norpois : j'étais persuadé qu'un tel message, même transmis d'une façon aussi brutale, m'eût donné beaucoup plus de prestige aux yeux de la maîtresse de la maison qu'il ne l'eût indisposée contre moi. Mais, même si j'avais pu me rendre compte que la mission dont ne s'acquitta pas Norpois fût restée sans utilité, bien plus, qu'elle eût pu me nuire auprès des Swann, je n'aurais pas eu le courage, s'il s'était montré consentant, d'en décharger l'Ambassadeur et de renoncer à la volupté, si funestes qu'en pussent être les suites, que mon nom et ma personne se trouvassent ainsi un moment auprès de Gilberte, dans sa maison et sa vie inconnues.

### Passage

Quand Norpois fut parti, mon père jeta un coup d'oeil sur le journal du soir ; je songeais de nouveau à la Berma. Le plaisir que j'avais eu à l'entendre exigeait d'autant plus d'être complété qu'il était loin d'égaler celui que je m'étais promis ; aussi s'assimilait-il immédiatement tout ce qui était susceptible de le nourrir, par exemple ces mérites que Norpois avait reconnus à la Berma et que mon esprit avait bus d'un seul trait comme un pré trop sec sur qui on verse de l'eau. Or mon père me passa le journal en me désignant un entrefilet conçu en ces termes : « La représentation de Phèdre qui a été donnée devant une salle enthousiaste où on remarquait les principales notabilités du monde des arts et de la critique a été pour Mme Berma, qui jouait le rôle de Phèdre, l'occasion d'un triomphe comme elle en a rarement connu de plus éclatant au cours de sa prestigieuse carrière. Nous reviendrons plus longuement sur cette représentation qui constitue un véritable événement théâtral ; disons seulement que les juges les plus autorisés s'accordaient à déclarer qu'une telle interprétation renouvelait entièrement le rôle de Phèdre, qui est un des plus beaux et des plus fouillés de Racine, et constituait la plus pure et la plus haute manifestation d'art à laquelle de notre temps il ait été donné d'assister. » Dès que mon esprit eut conçu cette idée nouvelle de « la plus pure et haute manifestation d'art », celle-ci se rapprocha du plaisir imparfait que j'avais éprouvé au théâtre, lui ajouta un peu de ce qui lui manquait et leur réunion forma quelque chose de si exaltant que je m'écriai : « Quelle grande artiste ! » Sans doute on peut trouver que je n'étais pas absolument sincère. Mais qu'on songe plutôt à tant d'écrivains qui, mécontents du morceau qu'ils viennent d'écrire, s'ils lisent un éloge du génie de Chateaubriand, ou évoquant tel grand artiste dont ils ont souhaité d'être l'égal, fredonnant par exemple en eux-mêmes telle phrase de Beethoven de laquelle ils comparent la tristesse à celle qu'ils ont voulu mettre dans leur prose, se remplissent tellement de cette idée de génie qu'ils l'ajoutent à leurs propres productions en repensant à elles, ne les voient plus telles qu'elles leur étaient apparues d'abord, et risquant un acte de foi dans la valeur de leur oeuvre se disent : « Après tout ! » sans se rendre compte que, dans le total qui détermine leur satisfaction finale, ils font entrer le souvenir de merveilleuses pages de Chateaubriand qu'ils assimilent aux leurs, mais enfin qu'ils n'ont point écrites ; qu'on se rappelle tant d'hommes qui croient en l'amour d'une maîtresse de qui ils ne connaissent que les trahisons ; tous ceux aussi qui espèrent alternativement soit une survie incompréhensible dès qu'ils pensent, maris inconsolables, à une femme qu'ils ont perdue et qu'ils aiment encore, artistes, à la gloire future de laquelle ils pourront jouir, soit un néant rassurant quand leur intelligence se reporte au contraire aux fautes que sans lui ils auraient à expier après leur mort ; qu'on pense encore aux touristes qu'exalte la beauté d'ensemble d'un voyage dont jour par jour ils n'ont éprouvé que de l'ennui, et qu'on dise, si dans la vie en commun que mènent les idées au sein de notre esprit, il est une seule de celles qui nous rendent le plus heureux qui n'ait été d'abord en véritable parasite demander à une idée étrangère et voisine le meilleur de la force qui lui manquait.

Ma mère ne parut pas très satisfaite que mon père ne songeât plus pour moi à la « carrière ». Je crois que, soucieuse avant tout qu'une règle d'existence disciplinât les caprices de mes nerfs, ce qu'elle regrettait, c'était moins de me voir renoncer à la diplomatie que m'adonner à la littérature. « Mais laisse donc, s'écria mon père, il faut avant tout prendre du plaisir à ce qu'on fait. Or, il n'est plus un enfant. Il sait bien maintenant ce qu'il aime, il est peu probable qu'il change, et il est capable de se rendre compte de ce qui le rendra heureux dans l'existence. » En attendant que, grâce à la liberté qu'elles m'octroyaient, je fusse, ou non, heureux dans l'existence, les paroles de mon père me firent ce soir-là bien de la peine. De tout temps ses gentillesses imprévues m'avaient, quand elles se produisaient, donné une telle envie d'embrasser au-dessus de sa barbe ses joues colorées que si je n'y cédais pas, c'était seulement par peur de lui déplaire. Aujourd'hui, comme un auteur s'effraye de voir ses propres rêveries qui lui paraissent sans grande valeur parce qu'il ne les sépare pas de lui-même, obliger un éditeur à choisir un papier, à employer des caractères peut-être trop beaux pour elles, je me demandais si mon désir d'écrire était quelque chose d'assez important pour que mon père dépensât à cause de cela tant de bonté. Mais surtout en parlant de mes goûts qui ne changeraient plus, de ce qui était destiné à rendre mon existence heureuse, il insinuait en moi deux terribles soupçons. Le premier, c'était que (alors que chaque jour je me considérais comme sur le seuil de ma vie encore intacte et qui ne débuterait que le lendemain matin) mon existence était déjà commencée, bien plus, que ce qui allait en suivre ne serait pas très différent de ce qui avait précédé. Le second soupçon, qui n'était à vrai dire qu'une autre forme du premier, c'est que je n'étais pas situé en dehors du Temps, mais soumis à ses lois, tout comme ces personnages de roman qui, à cause de cela, me jetaient dans une telle tristesse, quand je lisais leur vie, à Combray, au fond de ma guérite d'osier. Théoriquement on sait que la terre tourne, mais en fait on ne s'en aperçoit pas, le sol sur lequel on marche semble ne pas bouger et on vit tranquille. Il en est ainsi du Temps dans la vie. Et pour rendre sa fuite sensible, les romanciers sont obligés, en accélérant follement les battements de l'aiguille, de faire franchir au lecteur dix, vingt, trente ans, en deux minutes. Au haut d'une page on a quitté un amant plein d'espoir, au bas de la suivante on le retrouve octogénaire, accomplissant péniblement dans le préau d'un hospice sa promenade quotidienne, répondant à peine aux paroles qu'on lui adresse, ayant oublié le passé. En disant de moi : « Ce n'est plus un enfant, ses goûts ne changeront plus, etc. », mon père venait tout d'un coup de me faire apparaître à moi-même dans le Temps, et me causait le même genre de tristesse que si j'avais été non pas encore l'hospitalisé ramolli, mais ces héros dont l'auteur, sur un ton indifférent qui est particulièrement cruel, nous dit à la fin d'un livre : « Il quitte de moins en moins la campagne. Il a fini par s'y fixer définitivement, etc. »

Cependant, mon père, pour aller au-devant des critiques que nous aurions pu faire sur notre invité, dit à maman :

– J'avoue que le père Norpois a été un peu « poncif » comme vous dites. Quand il a dit qu'il aurait été « peu séant » de poser une question au comte de Paris, j'ai eu peur que vous ne vous mettiez à rire.

– Mais pas du tout, répondit ma mère, j'aime beaucoup qu'un homme de cette valeur et de cet âge ait gardé cette sorte de naïveté qui ne prouve qu'un fond d'honnêteté et de bonne éducation.

– Je crois bien ! Cela ne l'empêche pas d'être fin et intelligent, je le sais moi qui le vois à la Commission tout autre qu'il n'est ici, s'écria mon père, heureux de voir que maman appréciait Norpois, et voulant lui persuader qu'il était encore supérieur à ce qu'elle croyait, parce que la cordialité surfait avec autant de plaisir qu'en prend la taquinerie à déprécier. Comment a-t-il donc dit... « avec les princes on ne sait jamais... »

– Mais oui, comme tu dis là. J'avais remarqué, c'est très fin. On voit qu'il a une profonde expérience de la vie.

– C'est extraordinaire qu'il ait dîné chez les Swann et qu'il y ait trouvé en somme des gens réguliers, des fonctionnaires... Où est-ce que Odette a pu aller pêcher tout ce monde-là ?

– As-tu remarqué avec quelle malice il a fait cette réflexion : « C'est une maison où il va surtout des hommes ! »

Et tous deux cherchaient à reproduire la manière dont Norpois avait dit cette phrase, comme ils auraient fait pour quelque intonation de Bressant ou de Thiron dans l'Aventurière ou dans le Gendre de M. Poirier. Mais de tous ses mots, le plus goûté le fut par Françoise qui, encore plusieurs années après, ne pouvait pas « tenir son sérieux » si on lui rappelait qu'elle avait été traitée par l'Ambassadeur de « chef de premier ordre », ce que ma mère était allée lui transmettre comme un ministre de la guerre les félicitations d'un souverain de passage après « la Revue ». Je l'avais d'ailleurs précédée à la cuisine. Car j'avais fait promettre à Françoise, pacifiste mais cruelle, qu'elle ne ferait pas trop souffrir le lapin qu'elle avait à tuer et je n'avais pas eu de nouvelles de cette mort ; Françoise m'assura qu'elle s'était passée le mieux du monde et très rapidement : « J'ai jamais vu une bête comme ça ; elle est morte sans dire seulement une parole, vous auriez dit qu'elle était muette. » Peu au courant du langage des bêtes, j'alléguai que le lapin ne criait peut-être pas comme le poulet. « Attendez un peu voir, me dit Françoise indignée de mon ignorance, si les lapins ne crient pas autant comme les poulets. Ils ont même la voix bien plus forte. » Françoise accepta les compliments de Norpois avec la fière simplicité, le regard joyeux et – fût-ce momentanément – intelligent, d'un artiste à qui on parle de son art. Ma mère l'avait envoyée autrefois dans certains grands restaurants voir comment on y faisait la cuisine. J'eus ce soir-là à l'entendre traiter les plus célèbres de gargotes le même plaisir qu'autrefois à apprendre, pour les artistes dramatiques, que la hiérarchie de leurs mérites n'était pas la même que celle de leurs réputations. « L'Ambassadeur, lui dit ma mère, assure que nulle part on ne mange de boeuf froid et de soufflés comme les vôtres. » Françoise, avec un air de modestie et de rendre hommage à la vérité, l'accorda, sans être, d'ailleurs, impressionnée par le titre d'ambassadeur ; elle disait de Norpois, avec l'amabilité due à quelqu'un qui l'avait prise pour un « chef » : « C'est un bon vieux comme moi. » Elle avait bien cherché à l'apercevoir quand il était arrivé, mais sachant que maman détestait qu'on fût derrière les portes ou aux fenêtres et pensant qu'elle saurait par les autres domestiques ou par les concierges qu'elle avait fait le guet (car Françoise ne voyait partout que « jalousies » et « racontages » qui jouaient dans son imagination le même rôle permanent et funeste que, pour telles autres personnes, les intrigues des jésuites ou des juifs), elle s'était contentée de regarder par la croisée de la cuisine, « pour ne pas avoir des raisons avec Madame », et sous l'aspect sommaire de Norpois elle avait « cru voir Monsieur Legrand », à cause de son agileté, et bien qu'il n'y eût pas un trait commun entre eux. « Mais enfin, lui demanda ma mère, comment expliquez-vous que personne ne fasse la gelée aussi bien que vous (quand vous le voulez) ? – Je ne sais pas d'où ce que ça devient », répondit Françoise (qui n'établissait pas une démarcation bien nette entre le verbe venir, au moins pris dans certaines acceptions, et le verbe devenir). Elle disait vrai du reste, en partie, et n'était pas beaucoup plus capable – ou désireuse – de dévoiler le mystère qui faisait la supériorité de ses gelées ou de ses crèmes, qu'une grande élégante pour ses toilettes, ou une grande cantatrice pour son chant. Leurs explications ne nous disent pas grand'chose ; il en était de même des recettes de notre cuisinière. « Ils font cuire trop à la va-vite, répondit-elle en parlant des grands restaurateurs, et puis pas tout ensemble. Il faut que le boeuf, il devienne comme une éponge, alors il boit tout le jus jusqu'au fond. Pourtant il y avait un de ces Cafés où il me semble qu'on savait bien un peu faire la cuisine. Je ne dis pas que c'était tout à fait ma gelée, mais c'était fait bien doucement et les soufflés ils avaient bien de la crème. – Est-ce Henry ? demanda mon père qui nous avait rejoints et appréciait beaucoup le restaurant de la place Gaillon où il avait à dates fixes des repas de corps. – Oh non ! dit Françoise avec une douceur qui cachait un profond dédain, je parlais d'un petit restaurant. Chez cet Henry c'est très bon bien sûr, mais c'est pas un restaurant, c'est plutôt... un bouillon ! – Weber ? – Ah ! non, Monsieur, je voulais dire un bon restaurant. Weber c'est dans la rue Royale, ce n'est pas un restaurant, c'est une brasserie. Je ne sais pas si ce qu'ils vous donnent est servi. Je crois qu'ils n'ont même pas de nappe, ils posent cela comme cela sur la table, va comme je te pousse. – Cirro ? » Françoise sourit : « Oh ! là je crois qu'en fait de cuisine il y a surtout des dames du monde. (Monde signifiait pour Françoise demi-monde.) Dame, il faut ça pour la jeunesse. » Nous nous apercevions qu'avec son air de simplicité Françoise était pour les cuisiniers célèbres une plus terrible « camarade » que ne peut l'être l'actrice la plus envieuse et la plus infatuée. Nous sentîmes pourtant qu'elle avait un sentiment juste de son art et le respect des traditions, car elle ajouta : « Non, je veux dire un restaurant où c'est qu'il y avait l'air d'avoir une bien bonne petite cuisine bourgeoise. C'est une maison encore assez conséquente. Ça travaillait beaucoup. Ah ! on en ramassait des sous là dedans (Françoise, économe, comptait par sous, non par louis comme les décavés). Madame connaît bien, là-bas, à droite, sur les grands boulevards, un peu en arrière... » Le restaurant dont elle parlait avec cette équité mêlée d'orgueil et de bonhomie, c'était... le Café Anglais.
