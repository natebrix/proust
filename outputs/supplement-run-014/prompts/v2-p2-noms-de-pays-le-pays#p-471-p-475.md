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
        "Albertine"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Albertine",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.84,
      "evidence": "« ...cette possession était impossible »; « elle me dit avec froideur: ‘Je vous pardonne... mais ne recommencez jamais’ »; « peu à peu se détacha d’elle mon désir... Mes rêves l’abandonnèrent... se reportèrent... sur Andrée ».",
      "explanation": "After Albertine's firm refusal and her reprimand, the narrator withdraws his desire and plans for her, which significantly diminishes her emotional position with him."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Albertine",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.8,
      "evidence": "« toujours... ont plu davantage »; « elle était pourtant invitée non seulement à dîner, mais à demeure »; « passée... dans la famille d’un régent de la Banque de France »; « reçue comme ‘l’enfant de la maison’ »; « ces commissions... n’avaient aucun succès ».",
      "explanation": "The narrator emphasizes Albertine's 'vogue' and her sought-after inclusion in brighter circles; attempts at gossip fail, which locally reinforces her social consideration."
    }
  ],
  "status_effects": [
    {
      "character": "Albertine",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.78,
      "explanation": "Despite her poverty, she is sought after and welcomed in prominent houses, which elevates her local social position."
    },
    {
      "character": "Albertine",
      "dimension": "emotional_position",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "She loses the narrator's romantic attention, who diverts his dreams to others, initially Andrée."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-471-p-475"
}

### Candidate characters

[
  "Andrée",
  "Bloch",
  "Norpois",
  "Robert de Saint-Loup",
  "duchesse de Guermantes",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

J'allai dîner avec la grand-mère, je sentais en moi un secret qu'elle ne connaissait pas. De même, pour Albertine, demain ses amies seraient avec elle, sans savoir ce qu'il y avait de nouveau entre nous, et quand elle embrasserait sa nièce sur le front, Mme Bontemps ignorerait que j'étais entre elles deux, dans cet arrangement de cheveux qui avait pour but, caché à tous, de me plaire, à moi, à moi qui avais jusque-là tant envié Mme Bontemps, parce qu'apparentée aux mêmes personnes que sa nièce, elle avait les mêmes deuils à porter, les mêmes visites de famille à faire ; or je me trouvais être pour Albertine plus que n'était sa tante elle-même. Auprès de sa tante, c'est à moi qu'elle penserait. Qu'allait-il se passer tout à l'heure, je ne le savais pas trop. En tous cas le Grand-Hôtel, la soirée, ne me sembleraient plus vides ; ils contenaient mon bonheur. Je sonnai le lift pour monter à la chambre qu'Albertine avait prise, du côté de la vallée. Les moindres mouvements, comme m'asseoir sur la banquette de l'ascenseur, m'étaient doux, parce qu'ils étaient en relation immédiate avec mon coeur ; je ne voyais dans les cordes à l'aide desquelles l'appareil s'élevait, dans les quelques marches qui me restaient à monter, que les rouages, que les degrés matérialisés de ma joie. Je n'avais plus que deux ou trois pas à faire dans le couloir avant d'arriver à cette chambre où était renfermée la substance précieuse de ce corps rose – cette chambre qui, même s'il devait s'y dérouler des actes délicieux, garderait cette permanence, cet air d'être, pour un passant non informé, semblable à toutes les autres, qui font des choses les témoins obstinément muets, les scrupuleux confidents, les inviolables dépositaires du plaisir. Ces quelques pas du palier à la chambre d'Albertine, ces quelques pas que personne ne pouvait plus arrêter, je les fis avec délices, avec prudence, comme plongé dans un élément nouveau, comme si en avançant j'avais lentement déplacé du bonheur, et en même temps avec un sentiment inconnu de toute puissance, et d'entrer enfin dans un héritage qui m'eût de tout temps appartenu. Puis tout d'un coup je pensai que j'avais tort d'avoir des doutes, elle m'avait dit de venir quand elle serait couchée. C'était clair, je trépignais de joie, je renversai à demi Françoise qui était sur mon chemin, je courais, les yeux étincelants, vers la chambre de mon amie. Je trouvai Albertine dans son lit. Dégageant son cou, sa chemise blanche changeait les proportions de son visage, qui, congestionné par le lit, ou le rhume, ou le dîner, semblait plus rose ; je pensai aux couleurs que j'avais eues quelques heures auparavant à côté de moi, sur la digue, et desquelles j'allais enfin savoir le goût ; sa joue était traversée du haut en bas par une de ses longues tresses noires et bouclées que pour me plaire elle avait défaites entièrement. Elle me regardait en souriant. À côté d'elle, dans la fenêtre, la vallée était éclairée par le clair de lune. La vue du cou nu d'Albertine, de ces joues trop roses, m'avait jeté dans une telle ivresse, c'est-à-dire avait pour moi la réalité du monde non plus dans la nature, mais dans le torrent des sensations que j'avais peine à contenir, que cette vue avait rompu l'équilibre entre la vie immense, indestructible qui roulait dans mon être, et la vie de l'univers, si chétive en comparaison. La mer, que j'apercevais à côté de la vallée dans la fenêtre, les seins bombés des premières falaises de Maineville, le ciel où la lune n'était pas encore montée au zénith, tout cela semblait plus léger à porter que des plumes pour les globes de mes prunelles qu'entre mes paupières je sentais dilatés, résistants, prêts à soulever bien d'autres fardeaux, toutes les montagnes du monde, sur leur surface délicate. Leur orbe ne se trouvait plus suffisamment rempli par la sphère même de l'horizon. Et tout ce que la nature eût pu m'apporter de vie m'eût semblé bien mince, les souffles de la mer m'eussent paru bien courts pour l'immense aspiration qui soulevait ma poitrine. La mort eût dû me frapper en ce moment que cela m'eût paru indifférent ou plutôt impossible, car la vie n'était pas hors de moi, elle était en moi ; j'aurais souri de pitié si un philosophe eût émis l'idée qu'un jour même éloigné, j'aurais à mourir, que les forces éternelles de la nature me survivraient, les forces de cette nature sous les pieds divins de qui je n'étais qu'un grain de poussière ; qu'après moi il y aurait encore ces falaises arrondies et bombées, cette mer, ce clair de lune, ce ciel ! Comment cela eût-il été possible, comment le monde eût-il pu durer plus que moi, puisque je n'étais pas perdu en lui, puisque c'était lui qui était enclos en moi, en moi qu'il était bien loin de remplir, en moi, où, en sentant la place d'y entasser tant d'autres trésors, je jetais dédaigneusement dans un coin ciel, mer et falaises. « Finissez ou je sonne », s'écria Albertine voyant que je me jetais sur elle pour l'embrasser. Mais je me disais que ce n'était pas pour rien faire qu'une jeune fille fait venir un jeune homme en cachette, en s'arrangeant pour que sa tante ne le sache pas, que d'ailleurs l'audace réussit à ceux qui savent profiter des occasions ; dans l'état d'exaltation où j'étais, le visage rond d'Albertine, éclairé d'un feu intérieur comme par une veilleuse, prenait pour moi un tel relief qu'imitant la rotation d'une sphère ardente, il me semblait tourner, telles ces figures de Michel Ange qu'emporte un immobile et vertigineux tourbillon. J'allais savoir l'odeur, le goût, qu'avait ce fruit rose inconnu. J'entendis un son précipité, prolongé et criard. Albertine avait sonné de toutes ses forces.

### Passage

J'avais cru que l'amour que j'avais pour Albertine n'était pas fondé sur l'espoir de la possession physique. Pourtant quand il m'eut paru résulter de l'expérience de ce soir-là que cette possession était impossible et qu'après n'avoir pas douté le premier jour, sur la plage, qu'Albertine ne fût dévergondée, puis être passé par des suppositions intermédiaires, il me sembla acquis d'une manière définitive qu'elle était absolument vertueuse ; quand, à son retour de chez sa tante, huit jours plus tard, elle me dit avec froideur : « Je vous pardonne, je regrette même de vous avoir fait de la peine, mais ne recommencez jamais », au contraire de ce qui s'était produit quand Bloch m'avait dit qu'on pouvait avoir toutes les femmes, et comme si, au lieu d'une jeune fille réelle, j'avais connu une poupée de cire, il arriva que peu à peu se détacha d'elle mon désir de pénétrer dans sa vie, de la suivre dans les pays où elle avait passé son enfance, d'être initié par elle à une vie de sport ; ma curiosité intellectuelle de ce qu'elle pensait sur tel ou tel sujet ne survécut pas à la croyance que je pourrais l'embrasser. Mes rêves l'abandonnèrent dès qu'ils cessèrent d'être alimentés par l'espoir d'une possession dont je les avais crus indépendants. Dès lors ils se retrouvèrent libres de se reporter – selon le charme que je lui avais trouvé un certain jour, surtout selon la possibilité et les chances que j'entrevoyais d'être aimé par elle – sur telle ou telle des amies d'Albertine et d'abord sur Andrée. Pourtant si Albertine n'avait pas existé, peut-être n'aurais-je pas eu le plaisir que je commençai à prendre de plus en plus, les jours qui suivirent, à la gentillesse que me témoignait Andrée. Albertine ne raconta à personne l'échec que j'avais essuyé auprès d'elle. Elle était une de ces jolies filles qui dès leur extrême jeunesse, par leur beauté, mais surtout par un agrément, un charme qui restent assez mystérieux, et qui ont leur source peut-être dans des réserves de vitalité où de moins favorisés par la nature viennent se désaltérer, toujours, dans leur famille, au milieu de leurs amies, dans le monde, ont plu davantage que de plus belles, de plus riches ; elle était de ces êtres à qui, avant l'âge de l'amour et bien plus encore quand il est venu, on demande plus qu'eux ne demandent et même qu'ils ne peuvent donner. Dès son enfance Albertine avait toujours eu en admiration devant elle quatre ou cinq petites camarades, parmi lesquelles se trouvait Andrée qui lui était si supérieure et le savait (et peut-être cette attraction qu'Albertine exerçait bien involontairement avait-elle été à l'origine, avait-elle servi à la fondation de la petite bande). Cette attraction s'exerçait même assez loin dans les milieux relativement plus brillants, où, s'il y avait une pavane à danser, on demandait Albertine plutôt qu'une jeune fille mieux née. La conséquence était que, n'ayant pas un sou de dot, vivant assez mal, d'ailleurs, à la charge de M. Bontemps qu'on disait véreux et qui souhaitait se débarrasser d'elle, elle était pourtant invitée non seulement à dîner, mais à demeure, chez des personnes qui aux yeux de Saint-Loup n'eussent eu aucune élégance, mais qui pour la mère de Rosemonde ou pour la mère d'Andrée, femmes très riches mais qui ne connaissaient pas ces personnes, représentaient quelque chose d'énorme. Ainsi Albertine passait tous les ans quelques semaines dans la famille d'un régent de la Banque de France, président du Conseil d'administration d'une grande Compagnie de Chemins de fer. La femme de ce financier recevait des personnages importants et n'avait jamais dit son « jour » à la mère d'Andrée, laquelle trouvait cette dame impolie, mais n'en était pas moins prodigieusement intéressée par tout ce qui se passait chez elle. Aussi exhortait-elle tous les ans Andrée à inviter Albertine, dans leur villa, parce que, disait-elle, c'était une bonne oeuvre d'offrir un séjour à la mer à une fille qui n'avait pas elle-même les moyens de voyager et dont la tante ne s'occupait guère ; la mère d'Andrée n'était probablement pas mue par l'espoir que le régent de la Banque et sa femme, apprenant qu'Albertine était choyée par elle et sa fille, concevraient d'elles deux une bonne opinion ; à plus forte raison n'espérait-elle pas qu'Albertine, pourtant si bonne et adroite, saurait la faire inviter, ou tout au moins faire inviter Andrée aux garden-parties du financier. Mais chaque soir à dîner, tout en prenant un air dédaigneux et indifférent, elle était enchantée d'entendre Albertine lui raconter ce qui s'était passé au château pendant qu'elle y était, les gens qui y avaient été reçus et qu'elle connaissait presque tous de vue ou de nom. Même la pensée qu'elle ne les connaissait que de cette façon, c'est-à-dire ne les connaissait pas (elle appelait cela connaître les gens « de tout temps »), donnait à la mère d'Andrée une pointe de mélancolie tandis qu'elle posait à Albertine des questions sur eux d'un air hautain et distrait, du bout des lèvres, et eût pu la laisser incertaine et inquiète sur l'importance de sa propre situation si elle ne s'était rassurée elle-même et replacée dans la « réalité de la vie » en disant au maître d'hôtel : « Vous direz au chef que ses petits pois ne sont pas assez fondants. » Elle retrouvait alors sa sérénité. Elle était bien décidée à ce qu'Andrée n'épousât qu'un homme d'excellente famille naturellement, mais assez riche pour qu'elle pût elle aussi avoir un chef et deux cochers. C'était cela le positif, la vérité effective d'une situation. Mais qu'Albertine eût dîné au château du régent de la Banque avec telle ou telle dame, que cette dame l'eût même invitée pour l'hiver suivant, cela n'en donnait pas moins à la jeune fille, pour la mère d'Andrée, une sorte de considération particulière qui s'alliait très bien à la pitié et même au mépris excités par son infortune, mépris augmenté par le fait que M. Bontemps eût trahi son drapeau et se fût – même vaguement panamiste, disait-on – rallié au gouvernement. Ce qui n'empêchait pas, d'ailleurs, la mère d'Andrée, par amour de la vérité, de foudroyer de son dédain les gens qui avaient l'air de croire qu'Albertine était d'une basse extraction. « Comment, c'est tout ce qu'il y a de mieux, ce sont des Simonet, avec un seul n. » Certes, à cause du milieu où tout cela évoluait, où l'argent joue un tel rôle, et où l'élégance vous fait inviter mais non épouser, aucun mariage « potable » ne semblait pouvoir être pour Albertine la conséquence utile de la considération si distinguée dont elle jouissait et qu'on n'eût pas trouvée compensatrice de sa pauvreté. Mais même à eux seuls, et n'apportant pas l'espoir d'une conséquence matrimoniale, ces « succès » excitaient l'envie de certaines mères méchantes, furieuses de voir Albertine être reçue comme « l'enfant de la maison » par la femme du régent de la Banque, même par la mère d'Andrée, qu'elles connaissaient à peine. Aussi disaient-elles à des amis communs d'elles et de ces deux dames que celles-ci seraient indignées si elles savaient la vérité, c'est-à-dire qu'Albertine racontait chez l'une (et « vice versa ») tout ce que l'intimité où on l'admettait imprudemment lui permettait de découvrir chez l'autre, mille petits secrets qu'il eût été infiniment désagréable à l'intéressée de voir dévoilés. Ces femmes envieuses disaient cela pour que cela fût répété et pour brouiller Albertine avec ses protectrices. Mais ces commissions comme il arrive souvent n'avaient aucun succès. On sentait trop la méchanceté qui les dictait et cela ne faisait que faire mépriser un peu plus celles qui en avaient pris l'initiative. La mère d'Andrée était trop fixée sur le compte d'Albertine pour changer d'opinion à son égard. Elle la considérait comme une « malheureuse », mais d'une nature excellente et qui ne savait qu'inventer pour faire plaisir.

Si cette sorte de vogue qu'avait obtenue Albertine ne paraissait devoir comporter aucun résultat pratique, elle avait imprimé à l'amie d'Andrée le caractère distinctif des êtres qui toujours recherchés n'ont jamais besoin de s'offrir (caractère qui se retrouve aussi, pour des raisons analogues, à une autre extrémité de la société, chez des femmes d'une grande élégance) et qui est de ne pas faire montre des succès qu'ils ont, de les cacher plutôt. Elle ne disait jamais de quelqu'un : « Il a envie de me voir », parlait de tous avec une grande bienveillance, et comme si ce fût elle qui eût couru après, recherché les autres. Si on parlait d'un jeune homme qui quelques minutes auparavant venait de lui faire en tête à tête les plus sanglants reproches parce qu'elle lui avait refusé un rendez-vous, bien loin de s'en vanter publiquement, ou de lui en vouloir à lui, elle faisait son éloge : « C'est un si gentil garçon ! » Elle était même ennuyée de tellement plaire, parce que cela l'obligeait à faire de la peine, tandis que, par nature, elle aimait à faire plaisir. Elle aimait même à faire plaisir au point d'en être arrivée à pratiquer un mensonge spécial à certaines personnes utilitaires, à certains hommes arrivés. Existant d'ailleurs à l'état embryonnaire chez un nombre énorme de personnes, ce genre d'insincérité consiste à ne pas savoir se contenter pour un seul acte, de faire, grâce à lui, plaisir à une seule personne. Par exemple, si la tante d'Albertine désirait que sa nièce l'accompagnât à une matinée peu amusante, Albertine en s'y rendant aurait pu trouver suffisant d'en tirer le profit moral d'avoir fait plaisir à sa tante. Mais accueillie gentiment par les maîtres de la maison, elle aimait mieux leur dire qu'elle désirait depuis si longtemps les voir qu'elle avait choisi cette occasion et sollicité la permission de sa tante. Cela ne suffisait pas encore : à cette matinée se trouvait une des amies d'Albertine qui avait un gros chagrin. Albertine lui disait : « Je n'ai pas voulu te laisser seule, j'ai pensé que ça te ferait du bien de m'avoir près de toi. Si tu veux que nous laissions la matinée, que nous allions ailleurs, je ferai ce que tu voudras, je désire avant tout te voir moins triste » (ce qui était vrai aussi du reste). Parfois il arrivait pourtant que le but fictif détruisait le but réel. Ainsi Albertine ayant un service à demander pour une de ses amies allait pour cela voir une certaine dame. Mais arrivée chez cette dame bonne et sympathique, la jeune fille obéissant à son insu au principe de l'utilisation multiple d'une seule action, trouvait plus affectueux d'avoir l'air d'être venue seulement à cause du plaisir qu'elle avait senti qu'elle éprouverait à revoir cette dame. Celle-ci était infiniment touchée qu'Albertine eût accompli un long trajet par pure amitié. En voyant la dame presque émue, Albertine l'aimait encore davantage. Seulement il arrivait ceci : elle éprouvait si vivement le plaisir d'amitié pour lequel elle avait prétendu mensongèrement être venue, qu'elle craignait de faire douter la dame de sentiments en réalité sincères, si elle lui demandait le service pour l'amie. La dame croirait qu'Albertine était venue pour cela, ce qui était vrai, mais elle conclurait qu'Albertine n'avait pas de plaisir désintéressé à la voir, ce qui était faux. De sorte qu'Albertine repartait sans avoir demandé le service, comme les hommes qui ont été si bons avec une femme dans l'espoir d'obtenir ses faveurs, qu'ils ne font pas leur déclaration pour garder à cette bonté un caractère de noblesse. Dans d'autres cas on ne peut pas dire que le véritable but fût sacrifié au but accessoire et imaginé après coup, mais le premier était tellement opposé au second, que si la personne qu'Albertine attendrissait en lui déclarant l'un avait appris l'autre, son plaisir se serait aussitôt changé en la peine la plus profonde. La suite du récit fera, beaucoup plus loin, mieux comprendre ce genre de contradictions. Disons par un exemple emprunté à un ordre de faits tout différents qu'elles sont très fréquentes dans les situations les plus diverses que présente la vie. Un mari a installé sa maîtresse dans la ville où il est en garnison. Sa femme restée à Paris, et à demi au courant de la vérité, se désole, écrit à son mari des lettres de jalousie. Or, la maîtresse est obligée de venir passer un jour à Paris. Le mari ne peut résister à ses prières de l'accompagner et obtient une permission de vingt-quatre heures. Mais comme il est bon et souffre de faire de la peine à sa femme, il arrive chez celle-ci, lui dit, en versant quelques larmes sincères, qu'affolé par ses lettres il a trouvé le moyen de s'échapper pour venir la consoler et l'embrasser. Il a trouvé ainsi le moyen de donner par un seul voyage une preuve d'amour à la fois à sa maîtresse et à sa femme. Mais si cette dernière apprenait pour quelle raison il est venu à Paris, sa joie se changerait sans doute en douleur, à moins que voir l'ingrat ne la rendît malgré tout plus heureuse qu'il ne la fait souffrir par ses mensonges. Parmi les hommes qui m'ont paru pratiquer avec le plus de suite le système des fins multiples se trouve Norpois. Il acceptait quelquefois de s'entremettre entre deux amis brouillés, et cela faisait qu'on l'appelait le plus obligeant des hommes. Mais il ne lui suffisait pas d'avoir l'air de rendre service à celui qui était venu le solliciter, il présentait à l'autre la démarche qu'il faisait auprès de lui comme entreprise non à la requête du premier, mais dans l'intérêt du second, ce qu'il persuadait facilement à un interlocuteur suggestionné d'avance par l'idée qu'il avait devant lui « le plus serviable des hommes ». De cette façon, jouant sur les deux tableaux, faisant ce qu'on appelle en termes de coulisse de la contre-partie, il ne laissait jamais courir aucun risque à son influence, et les services qu'il rendait ne constituaient pas une aliénation, mais une fructification d'une partie de son crédit. D'autre part, chaque service, semblant doublement rendu, augmentait d'autant plus sa réputation d'ami serviable, et encore d'ami serviable avec efficacité, qui ne donne pas des coups d'épée dans l'eau, dont toutes les démarches portent, ce que démontrait la reconnaissance des deux intéressés. Cette duplicité dans l'obligeance était, et avec des démentis comme en toute créature humaine, une partie importante du caractère de Norpois. Et souvent au ministère, il se servit de mon père, lequel était assez naïf, en lui faisant croire qu'il le servait.

Plaisant plus qu'elle ne voulait et n'ayant pas besoin de claironner ses succès, Albertine garda le silence sur la scène qu'elle avait eue avec moi auprès de son lit, et qu'une laide aurait voulu faire connaître à l'univers. D'ailleurs son attitude dans cette scène, je ne parvenais pas à me l'expliquer. Pour ce qui concerne l'hypothèse d'une vertu absolue (hypothèse à laquelle j'avais d'abord attribué la violence avec laquelle Albertine avait refusé de se laisser embrasser et prendre par moi, et qui n'était du reste nullement indispensable à ma conception de la bonté, de l'honnêteté foncière de mon amie) je ne laissai pas de la remanier à plusieurs reprises. Cette hypothèse était tellement le contraire de celle que j'avais bâtie le premier jour où j'avais vu Albertine. Puis tant d'actes différents, tous de gentillesse pour moi (une gentillesse caressante, parfois inquiète, alarmée, jalouse de ma prédilection pour Andrée) baignaient de tous côtés le geste de rudesse par lequel, pour m'échapper, elle avait tiré sur la sonnette. Pourquoi donc m'avait-elle demandé de venir passer la soirée près de son lit ? Pourquoi parlait-elle tout le temps le langage de la tendresse ? Sur quoi repose le désir de voir un ami, de craindre qu'il vous préfère votre amie, de chercher à lui faire plaisir, de lui dire romanesquement que les autres ne sauront pas qu'il a passé la soirée auprès de vous, si vous lui refusez un plaisir aussi simple et si ce n'est pas un plaisir pour vous ? Je ne pouvais croire tout de même que la vertu d'Albertine allât jusque-là et j'en arrivais à me demander s'il n'y avait pas eu à sa violence une raison de coquetterie, par exemple une odeur désagréable qu'elle aurait cru avoir sur elle et par laquelle elle eût craint de me déplaire, ou de pusillanimité, si par exemple elle croyait, dans son ignorance des réalités de l'amour, que mon état de faiblesse nerveuse pouvait avoir quelque chose de contagieux par le baiser.

Elle fut certainement désolée de n'avoir pu me faire plaisir et me donna un petit crayon d'or, par cette vertueuse perversité des gens qui, attendris par votre gentillesse et ne souscrivant pas à vous accorder ce qu'elle réclame, veulent cependant faire en votre faveur autre chose : le critique dont l'article flatterait le romancier l'invite à la place à dîner, la duchesse n'emmène pas le snob avec elle au théâtre, mais lui envoie sa loge pour un soir où elle ne l'occupera pas. Tant ceux qui font le moins et pourraient ne rien faire sont poussés par le scrupule à faire quelque chose. Je dis à Albertine qu'en me donnant ce crayon, elle me faisait un grand plaisir, moins grand pourtant que celui que j'aurais eu si le soir où elle était venue coucher à l'hôtel elle m'avait permis de l'embrasser. « Cela m'aurait rendu si heureux ! qu'est-ce que cela pouvait vous faire ? je suis étonné que vous me l'ayez refusé. – Ce qui m'étonne, me répondit-elle, c'est que vous trouviez cela étonnant. Je me demande quelles jeunes filles vous avez pu connaître pour que ma conduite vous ait surpris. – Je suis désolé de vous avoir fâchée, mais, même maintenant je ne peux pas vous dire que je trouve que j'ai eu tort. Mon avis est que ce sont des choses qui n'ont aucune importance, et je ne comprends pas qu'une jeune fille qui peut si facilement faire plaisir, n'y consente pas. Entendons-nous, ajoutai-je pour donner une demi-satisfaction à ses idées morales, en me rappelant comment elle et ses amies avaient flétri l'amie de l'actrice Léa, je ne veux pas dire qu'une jeune fille puisse tout faire et qu'il n'y ait rien d'immoral. Ainsi, tenez, ces relations dont vous parliez l'autre jour à propos d'une petite qui habite Balbec et qui existeraient entre elle et une actrice, je trouve cela ignoble, tellement ignoble que je pense que ce sont des ennemis de la jeune fille qui auront inventé cela et que ce n'est pas vrai. Cela me semble improbable, impossible. Mais se laisser embrasser et même plus par un ami, puisque vous dites que je suis votre ami... – Vous l'êtes, mais j'en ai eu d'autres avant vous, j'ai connu des jeunes gens qui, je vous assure, avaient pour moi tout autant d'amitié. Hé bien, il n'y en a pas un qui aurait osé une chose pareille. Ils savaient la paire de calottes qu'ils auraient reçue. D'ailleurs ils n'y songeaient même pas, on se serrait la main bien franchement, bien amicalement, en bons camarades, jamais on n'aurait parlé de s'embrasser et on n'en était pas moins amis pour cela. Allez, si vous tenez à mon amitié, vous pouvez être content, car il faut que je vous aime joliment pour vous pardonner. Mais je suis sûre que vous vous fichez bien de moi. Avouez que c'est Andrée qui vous plaît. Au fond, vous avez raison, elle est beaucoup plus gentille que moi, et elle est ravissante ! Ah ! les hommes ! » Malgré ma déception récente, ces paroles si franches, en me donnant une grande estime pour Albertine, me causaient une impression très douce. Et peut-être cette impression eut-elle plus tard pour moi de grandes et fâcheuses conséquences, car ce fut par elle que commença à se former ce sentiment presque familial, ce noyau moral qui devait toujours subsister au milieu de mon amour pour Albertine. Un tel sentiment peut être la cause des plus grandes peines. Car pour souffrir vraiment par une femme, il faut avoir cru complètement en elle. Pour le moment, cet embryon d'estime morale, d'amitié, restait au milieu de mon âme comme une pierre d'attente. Il n'eût rien pu, à lui seul, contre mon bonheur s'il fût demeuré ainsi sans s'accroître, dans une inertie qu'il devait garder l'année suivante et à plus forte raison pendant ces dernières semaines de mon premier séjour à Balbec. Il était en moi comme un de ces hôtes qu'il serait malgré tout plus prudent qu'on expulsât, mais qu'on laisse à leur place sans les inquiéter, tant les rendent provisoirement inoffensifs leur faiblesse et leur isolement au milieu d'une âme étrangère.

Mes rêves se retrouvaient libres maintenant de se reporter sur telle ou telle des amies d'Albertine et d'abord sur Andrée, dont les gentillesses m'eussent peut-être moins touché si je n'avais été certain qu'elles seraient connues d'Albertine. Certes la préférence que depuis longtemps j'avais feinte pour Andrée m'avait fourni – en habitudes de causeries, de déclarations de tendresse – comme la matière d'un amour tout prêt pour elle, auquel il n'avait jusqu'ici manqué qu'un sentiment sincère qui s'y ajoutât et que maintenant mon coeur redevenu libre aurait pu fournir. Mais pour que j'aimasse vraiment Andrée, elle était trop intellectuelle, trop nerveuse, trop maladive, trop semblable à moi. Si Albertine me semblait maintenant vide, Andrée était remplie de quelque chose que je connaissais trop. J'avais cru le premier jour voir sur la plage une maîtresse de coureur, enivrée de l'amour des sports, et Andrée me disait que si elle s'était mise à en faire, c'était sur l'ordre de son médecin pour soigner sa neurasthénie et ses troubles de nutrition, mais que ses meilleures heures étaient celles où elle traduisait un roman de George Eliot. Ma déception, suite d'une erreur initiale sur ce qu'était Andrée, n'eut, en fait, aucune importance pour moi. Mais l'erreur était du genre de celles qui, si elles permettent à l'amour de naître et ne sont reconnues pour des erreurs que lorsqu'il n'est plus modifiable, deviennent une cause de souffrances. Ces erreurs – qui peuvent être différentes de celle que je commis pour Andrée et même inverses – tiennent souvent, dans le cas d'Andrée en particulier, à ce qu'on prend suffisamment l'aspect, les façons de ce qu'on n'est pas mais qu'on voudrait être, pour faire illusion au premier abord. À l'apparence extérieure, l'affectation, l'imitation, le désir d'être admiré, soit des bons, soit des méchants, ajoutent les faux semblants des paroles, des gestes. Il y a des cynismes, des cruautés qui ne résistent pas plus à l'épreuve que certaines bontés, certaines générosités. De même qu'on découvre souvent un avare vaniteux dans un homme connu pour ses charités, sa forfanterie de vice nous fait supposer une Messaline dans une honnête fille pleine de préjugés. J'avais cru trouver en Andrée une créature saine et primitive, alors qu'elle n'était qu'un être cherchant la santé, comme étaient peut-être beaucoup de ceux en qui elle avait cru la trouver et qui n'en avaient pas plus la réalité qu'un gros arthritique à figure rouge et en veste de flanelle blanche n'est forcément un Hercule. Or, il est telles circonstances où il n'est pas indifférent pour le bonheur que la personne qu'on a aimée pour ce qu'elle paraissait avoir de sain ne fût en réalité qu'une de ces malades qui ne reçoivent leur santé que d'autres, comme les planètes empruntent leur lumière, comme certains corps ne font que laisser passer l'électricité.
