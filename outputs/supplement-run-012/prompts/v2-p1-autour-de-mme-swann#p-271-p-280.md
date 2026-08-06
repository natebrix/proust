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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette",
        "Mme Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    },
    {
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.85,
      "evidence": "« comme si les chefs-d'oeuvre se faisaient par ‘relations’ »; son “Gilberte!” retient sa fille et « avait … accéléré l’évolution … qui détachait peu à peu de moi mon amie »; puis elle « se mit à parler anglais à sa fille » plaçant un mur entre moi et Gilberte.",
      "explanation": "The narrator presents Odette as confusing artistic success and social connections, and as intervening inopportunely (reminder of Gilberte, an aside in English) in a way that harms the situation and excludes the narrator."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Gilberte",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "uncertain",
      "confidence": 0.66,
      "evidence": "« Plusieurs fois je sentis que Gilberte désirait éloigner mes visites »; haussement d’épaules, froideur, rires (« je me fiche de vous » selon le narrateur), refus d’expliquer (« je ne peux pas vous expliquer »).",
      "explanation": "The narrator reports signs of distancing and disdain from Gilberte toward him, read as a snub; he hesitates, however, about the exact interpretation, which leaves an element of uncertainty."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "Locally, Odette is depicted as naïve and socially clumsy, her interventions working against the situation."
    },
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.66,
      "explanation": "Her attitude is presented as cold and unfriendly, amounting to a snub toward the narrator."
    }
  ],
  "ambiguities": [
    "Le narrateur oscille dans la lecture du rire et des paroles de Gilberte, ce qui rend incertaine la force du ‘snub’.",
    "Le passage en anglais entre Odette et Gilberte peut être une pratique anodine, mais le narrateur l’interprète comme une exclusion intentionnelle."
  ],
  "unit_id": "v2-p1-autour-de-mme-swann#p-271-p-280"
}

### Candidate characters

[
  "Bergotte",
  "Swann",
  "le narrateur"
]

### Prior local context (optional)

D'ailleurs, me disais-je, en passant ma vie chez les Swann ne fais-je pas comme Bergotte ? À mes parents il semblait presque que, tout en étant paresseux, je menais, puisque c'était dans le même salon qu'un grand écrivain, la vie la plus favorable au talent. Et pourtant, que quelqu'un puisse être dispensé de faire ce talent soi-même, par le dedans, et le reçoive d'autrui, est aussi impossible que se faire une bonne santé (malgré qu'on manque à toutes les règles de l'hygiène et qu'on commette les pires excès) rien qu'en dînant souvent en ville avec un médecin. La personne du reste qui était le plus complètement dupe de l'illusion qui m'abusait ainsi que mes parents, c'était Odette. Quand je lui disais que je ne pouvais pas venir, qu'il fallait que je restasse à travailler, elle avait l'air de trouver que je faisais bien des embarras, qu'il y avait un peu de sottise et de prétention dans mes paroles :

### Passage

– Mais Bergotte vient bien, lui ? Est-ce que vous trouvez que ce qu'il écrit n'est pas bien. Cela sera même mieux bientôt, ajoutait-elle, car il est plus aigu, plus concentré dans le journal que dans le livre où il délaie un peu. J'ai obtenu qu'il fasse désormais le « leader article » dans le Figaro. Ce sera tout à fait « the right man in the right place. »

Et elle ajoutait :

– Venez, il vous dira mieux que personne ce qu'il faut faire.

Et c'était comme on invite un engagé volontaire avec son colonel, c'était dans l'intérêt de ma carrière, et comme si les chefs-d'oeuvre se faisaient par « relations », qu'elle me disait de ne pas manquer de venir le lendemain dîner chez elle avec Bergotte.

Ainsi pas plus du côté des Swann que du côté de mes parents, c'est-à-dire de ceux qui, à des moments différents, avaient semblé devoir y mettre obstacle, aucune opposition n'était plus faite à cette douce vie où je pouvais voir Gilberte comme je voulais, avec ravissement, sinon avec calme. Il ne peut pas y en avoir dans l'amour, puisque ce qu'on a obtenu n'est jamais qu'un nouveau point de départ pour désirer davantage. Tant que je n'avais pu aller chez elle, les yeux fixés vers cet inaccessible bonheur, je ne pouvais même pas imaginer les causes nouvelles de trouble qui m'y attendaient. Une fois la résistance de ses parents brisée, et le problème enfin résolu, il recommença à se poser, chaque fois dans d'autres termes. En ce sens c'était bien en effet chaque jour une nouvelle amitié qui commençait. Chaque soir en rentrant je me rendais compte que j'avais à dire à Gilberte des choses capitales, desquelles notre amitié dépendait, et ces choses n'étaient jamais les mêmes. Mais enfin j'étais heureux et aucune menace ne s'élevait plus contre mon bonheur. Il allait en venir hélas, d'un côté où je n'avais jamais aperçu aucun péril, du côté de Gilberte et de moi-même. J'aurais pourtant dû être tourmenté par ce qui, au contraire, me rassurait, par ce que je croyais du bonheur. C'est, dans l'amour, un état anormal, capable de donner tout de suite, à l'accident le plus simple en apparence et qui peut toujours survenir, une gravité que par lui-même cet accident ne comporterait pas. Ce qui rend si heureux, c'est la présence dans le coeur de quelque chose d'instable, qu'on s'arrange perpétuellement à maintenir et dont on ne s'aperçoit presque plus tant qu'il n'est pas déplacé. En réalité, dans l'amour il y a une souffrance permanente, que la joie neutralise, rend virtuelle, ajourne, mais qui peut à tout moment devenir ce qu'elle serait depuis longtemps si l'on n'avait pas obtenu ce qu'on souhaitait, atroce.

Plusieurs fois je sentis que Gilberte désirait éloigner mes visites. Il est vrai que quand je tenais trop à la voir je n'avais qu'à me faire inviter par ses parents qui étaient de plus en plus persuadés de mon excellente influence sur elle. Grâce à eux, pensais-je, mon amour ne court aucun risque ; du moment que je les ai pour moi, je peux être tranquille puisqu'ils ont toute autorité sur Gilberte. Malheureusement à certains signes d'impatience que celle-ci laissait échapper quand son père me faisait venir en quelque sorte malgré elle, je me demandai si ce que j'avais considéré comme une protection pour mon bonheur n'était pas au contraire la raison secrète pour laquelle il ne pourrait durer.

La dernière fois que je vins voir Gilberte, il pleuvait ; elle était invitée à une leçon de danses chez des gens qu'elle connaissait trop peu pour pouvoir m'emmener avec elle. J'avais pris à cause de l'humidité plus de caféine que d'habitude. Peut-être à cause du mauvais temps, peut-être ayant quelque prévention contre la maison où cette matinée devait avoir lieu, Odette, au moment où sa fille allait partir, la rappela avec une extrême vivacité : « Gilberte ! » et me désigna pour signifier que j'étais venu pour la voir, qu'elle devait rester avec moi. Ce « Gilberte » avait été prononcé, crié plutôt, dans une bonne intention pour moi, mais au haussement d'épaules que fit Gilberte en ôtant ses affaires, je compris que sa mère avait involontairement accéléré l'évolution, peut-être jusque-là possible encore à arrêter, qui détachait peu à peu de moi mon amie. « On n'est pas obligé d'aller danser tous les jours », dit Odette à sa fille, avec une sagesse sans doute apprise autrefois de Swann. Puis, redevenant Odette, elle se mit à parler anglais à sa fille. Aussitôt ce fut comme si un mur m'avait caché une partie de la vie de Gilberte, comme si un génie malfaisant avait emmené loin de moi mon amie. Dans une langue que nous savons, nous avons substitué à l'opacité des sons la transparence des idées. Mais une langue que nous ne savons pas est un palais clos dans lequel celle que nous aimons peut nous tromper, sans que, restés au dehors et désespérément crispés dans notre impuissance, nous parvenions à rien voir, à rien empêcher. Telle cette conversation en anglais dont je n'eusse que souri un mois auparavant et au milieu de laquelle quelques noms propres français ne laissaient pas d'accroître et d'orienter mes inquiétudes, avait, tenue à deux pas de moi par deux personnes immobiles, la même cruauté, me faisait aussi délaissé et seul qu'un enlèvement. Enfin Odette nous quitta. Ce jour-là, peut-être par rancune contre moi, cause involontaire qu'elle n'allât pas s'amuser, peut-être aussi parce que la devinant fâchée j'étais préventivement plus froid que d'habitude, le visage de Gilberte, dépouillé de toute joie, nu, saccagé, sembla tout l'après-midi vouer un regret mélancolique au pas-de-quatre que ma présence l'empêchait d'aller danser, et défier toutes les créatures, à commencer par moi, de comprendre les raisons subtiles qui avaient déterminé chez elle une inclination sentimentale pour le boston. Elle se borna à échanger, par moments, avec moi, sur le temps qu'il faisait, la recrudescence de la pluie, l'avance de la pendule, une conversation ponctuée de silences et de monosyllabes où je m'entêtais moi-même, avec une sorte de rage désespérée, à détruire les instants que nous aurions pu donner à l'amitié et au bonheur. Et à tous nos propos une sorte de dureté suprême était conférée par le paroxysme de leur insignifiance paradoxale, lequel me consolait pourtant, car il empêchait Gilberte d'être dupe de la banalité de mes réflexions et de l'indifférence de mon accent. C'est en vain que je disais : « Il me semble que l'autre jour la pendule retardait plutôt », elle traduisait évidemment : « Comme vous êtes méchante ! » J'avais beau m'obstiner à prolonger, tout le long de ce jour pluvieux, ces paroles sans éclaircies, je savais que ma froideur n'était pas quelque chose d'aussi définitivement figé que je le feignais, et que Gilberte devait bien sentir que si, après le lui avoir déjà dit trois fois, je m'étais hasardé une quatrième à lui répéter que les jours diminuaient, j'aurais eu de la peine à me retenir de fondre en larmes. Quand elle était ainsi, quand un sourire ne remplissait pas ses yeux et ne découvrait pas son visage, on ne peut dire de quelle désolante monotonie étaient empreints ses yeux tristes et ses traits maussades. Sa figure, devenue presque livide, ressemblait alors à ces plages ennuyeuses où la mer retirée très loin vous fatigue d'un reflet toujours pareil que cerne un horizon immuable et borné. À la fin, ne voyant pas se produire de la part de Gilberte le changement heureux que j'attendais depuis plusieurs heures, je lui dis qu'elle n'était pas gentille : « C'est vous qui n'êtes pas gentil », me répondit-elle. « Mais si ! » Je me demandai ce que j'avais fait, et ne le trouvant pas, le lui demandai à elle-même : « Naturellement, vous vous trouvez gentil ! » me dit-elle en riant longuement. Alors je sentis ce qu'il y avait de douloureux pour moi à ne pouvoir atteindre cet autre plan, plus insaisissable, de sa pensée, que décrivait son rire. Ce rire avait l'air de signifier : « Non, non, je ne me laisse pas prendre à tout ce que vous me dites, je sais que vous êtes fou de moi, mais cela ne me fait ni chaud ni froid, car je me fiche de vous. » Mais je me disais qu'après tout le rire n'est pas un langage assez déterminé pour que je pusse être assuré de bien comprendre celui-là. Et les paroles de Gilberte étaient affectueuses. « Mais en quoi ne suis-je pas gentil, lui demandai-je, dites-le moi, je ferai tout ce que vous voudrez. – Non, cela ne servirait à rien, je ne peux pas vous expliquer. » Un instant j'eus peur qu'elle crût que je ne l'aimasse pas, et ce fut pour moi une autre souffrance, non moins vive, mais qui réclamait une dialectique différente. « Si vous saviez le chagrin que vous me faites, vous me le diriez. » Mais ce chagrin qui, si elle avait douté de mon amour, eût dû la réjouir, l'irrita au contraire. Alors, comprenant mon erreur, décidé à ne plus tenir compte de ses paroles, la laissant, sans la croire, me dire : « Je vous aimais vraiment, vous verrez cela un jour » (ce jour où les coupables assurent que leur innocence sera reconnue et qui, pour des raisons mystérieuses, n'est jamais celui où on les interroge), j'eus le courage de prendre subitement la résolution de ne plus la voir, et sans le lui annoncer encore, parce qu'elle ne m'aurait pas cru.

Un chagrin causé par une personne qu'on aime peut être amer, même quand il est inséré au milieu de préoccupations, de joies, qui n'ont pas cet être pour objet et desquelles notre attention ne se détourne que de temps en temps pour revenir à lui. Mais quand un tel chagrin naît – comme c'était le cas pour celui-ci – à un moment où le bonheur de voir cette personne nous remplit tout entiers, la brusque dépression qui se produit alors dans notre âme jusque-là ensoleillée, soutenue et calme, détermine en nous une tempête furieuse contre laquelle nous ne savons pas si nous serons capables de lutter jusqu'au bout. Celle qui soufflait sur mon coeur était si violente que je revins vers la maison, bousculé, meurtri, sentant que je ne pourrais retrouver la respiration qu'en rebroussant chemin, qu'en retournant sous un prétexte quelconque auprès de Gilberte. Mais elle se serait dit : « Encore lui ! Décidément je peux tout me permettre, il reviendra chaque fois d'autant plus docile qu'il m'aura quittée plus malheureux. » Puis j'étais irrésistiblement ramené vers elle par ma pensée, et ces orientations alternatives, cet affolement de la boussole intérieure persistèrent quand je fus rentré, et se traduisirent par les brouillons de lettres contradictoires que j'écrivis à Gilberte.

J'allais passer par une de ces conjonctures difficiles en face desquelles il arrive généralement qu'on se trouve à plusieurs reprises dans la vie et auxquelles, bien qu'on n'ait pas changé de caractère, de nature – notre nature qui crée elle-même nos amours, et presque les femmes que nous aimons, et jusqu'à leurs fautes – on ne fait pas face de la même manière à chaque fois, c'est-à-dire à tout âge. À ces moments-là notre vie est divisée, et comme distribuée dans une balance, en deux plateaux opposés où elle tient tout entière. Dans l'un, il y a notre désir de ne pas déplaire, de ne pas paraître trop humble à l'être que nous aimons sans parvenir à le comprendre, mais que nous trouvons plus habile de laisser un peu de côté pour qu'il n'ait pas ce sentiment de se croire indispensable qui le détournerait de nous ; de l'autre côté, il y a une souffrance – non pas une souffrance localisée et partielle – qui ne pourrait au contraire être apaisée que si renonçant à plaire à cette femme et à lui faire croire que nous pouvons nous passer d'elle, nous allions la retrouver. Quand on retire du plateau où est la fierté une petite quantité de volonté qu'on a eu la faiblesse de laisser s'user avec l'âge, qu'on ajoute dans le plateau où est le chagrin une souffrance physique acquise et à qui on a permis de s'aggraver, et au lieu de la solution courageuse qui l'aurait emporté à vingt ans, c'est l'autre, devenue trop lourde et sans assez de contre-poids, qui nous abaisse à cinquante. D'autant plus que les situations tout en se répétant changent, et qu'il y a chance pour qu'au milieu ou à la fin de la vie on ait eu pour soi-même la funeste complaisance de compliquer l'amour d'une part d'habitude que l'adolescence, retenue par d'autres devoirs, moins libre de soi-même, ne connaît pas.

Je venais d'écrire à Gilberte une lettre où je laissais tonner ma fureur, non sans pourtant jeter la bouée de quelques mots placés comme au hasard, et où mon amie pourrait accrocher une réconciliation ; un instant après, le vent ayant tourné, c'était des phrases tendres que je lui adressais pour la douceur de certaines expressions désolées, de tels « jamais plus », si attendrissants pour ceux qui les emploient, si fastidieux pour celle qui les lira, soit qu'elle les croie mensongers et traduise « jamais plus » par « ce soir même, si vous voulez bien de moi » ou qu'elle les croie vrais et lui annonçant alors une de ces séparations définitives qui nous sont si parfaitement égales dans la vie quand il s'agit d'êtres dont nous ne sommes pas épris. Mais puisque nous sommes incapables tandis que nous aimons d'agir en dignes prédécesseurs de l'être prochain que nous serons et qui n'aimera plus, comment pourrions-nous tout à fait imaginer l'état d'esprit d'une femme à qui, même si nous savions que nous lui sommes indifférents, nous avons perpétuellement fait tenir dans nos rêveries, pour nous bercer d'un beau songe ou nous consoler d'un gros chagrin, les mêmes propos que si elle nous aimait. Devant les pensées, les actions d'une femme que nous aimons, nous sommes aussi désorientés que le pouvaient être devant les phénomènes de la nature, les premiers physiciens (avant que la science fût constituée et eût mis un peu de lumière dans l'inconnu). Ou pis encore, comme un être pour l'esprit de qui le principe de causalité existerait à peine, un être qui ne serait pas capable d'établir un lien entre un phénomène et un autre et devant qui le spectacle du monde serait incertain comme un rêve. Certes je m'efforçais de sortir de cette incohérence, de trouver des causes. Je tâchais même d'être « objectif » et pour cela de bien tenir compte de la disproportion qui existait entre l'importance qu'avait pour moi Gilberte et celle non seulement que j'avais pour elle, mais qu'elle-même avait pour les autres êtres que moi, disproportion qui, si je l'eusse omise, eût risqué de me faire prendre une simple amabilité de mon amie pour un aveu passionné, une démarche grotesque et avilissante de ma part pour le simple et gracieux mouvement qui vous dirige vers de beaux yeux. Mais je craignais aussi de tomber dans l'excès contraire, où j'aurais vu dans l'arrivée inexacte de Gilberte à un rendez-vous un mouvement de mauvaise humeur, une hostilité irrémédiable. Je tâchais de trouver entre ces deux optiques également déformantes celle qui me donnerait la vision juste des choses ; les calculs qu'il me fallait faire pour cela me distrayaient un peu de ma souffrance ; et soit par obéissance à la réponse des nombres, soit que je leur eusse fait dire ce que je désirais, je me décidai le lendemain à aller chez les Swann, heureux, mais de la même façon que ceux qui, s'étant tourmentés longtemps à cause d'un voyage qu'ils ne voulaient pas faire, ne vont pas plus loin que la gare, et rentrent chez eux défaire leur malle. Et comme, pendant qu'on hésite, la seule idée d'une résolution possible (à moins d'avoir rendu cette idée inerte en décidant qu'on ne prendra pas la résolution) développe, comme une graine vivace, les linéaments, tout le détail des émotions qui naîtraient de l'acte exécuté, je me dis que j'avais été bien absurde de me faire, en projetant de ne plus voir Gilberte, autant de mal que si j'eusse dû réaliser ce projet et que, puisque au contraire c'était pour finir par retourner chez elle, j'aurais pu faire l'économie de tant de velléités et d'acceptations douloureuses. Mais cette reprise des relations d'amitié ne dura que le temps d'aller jusqu'à chez les Swann, non pas parce que leur maître d'hôtel, lequel m'aimait beaucoup, me dit que Gilberte était sortie (je sus en effet, dès le soir même, que c'était vrai, par des gens qui l'avaient rencontrée), mais à cause de la façon dont il me le dit : « Monsieur, Mademoiselle est sortie, je peux affirmer à Monsieur que je ne mens pas. Si Monsieur veut se renseigner, je peux faire venir la femme de chambre. Monsieur pense bien que je ferais tout ce que je pourrais pour lui faire plaisir et que si Mademoiselle était là, je mènerais tout de suite Monsieur auprès d'elle. » Ces paroles, de la sorte qui est la seule importante, involontaires, nous donnant la radiographie au moins sommaire de la réalité insoupçonnable que cacherait un discours étudié, prouvaient que dans l'entourage de Gilberte on avait l'impression que je lui étais importun ; aussi, à peine le maître d'hôtel les eut-il prononcées, qu'elles engendrèrent chez moi de la haine à laquelle je préférai donner comme objet, au lieu de Gilberte, le maître d'hôtel ; il concentra sur lui tous les sentiments de colère que j'avais pu avoir pour mon amie ; débarrassé d'eux grâce à ces paroles, mon amour subsista seul ; mais elles m'avaient montré en même temps que je devais pendant quelque temps ne pas chercher à voir Gilberte. Elle allait certainement m'écrire pour s'excuser. Malgré cela, je ne retournerais pas tout de suite la voir, afin de lui prouver que je pouvais vivre sans elle. D'ailleurs, une fois que j'aurais reçu sa lettre, fréquenter Gilberte serait une chose dont je pourrais plus aisément me priver pendant quelque temps, parce que je serais sûr de la retrouver dès que je le voudrais. Ce qu'il me fallait pour supporter moins tristement l'absence volontaire, c'était sentir mon coeur débarrassé de la terrible incertitude de savoir si nous n'étions pas brouillés pour toujours, si elle n'était pas fiancée, partie, enlevée. Les jours qui suivirent ressemblèrent à ceux de cette ancienne semaine du jour de l'an que j'avais dû passer sans Gilberte. Mais cette semaine-là finie, jadis, d'une part mon amie reviendrait aux Champs-Élysées, je la reverrais comme auparavant, j'en étais sûr ; et, d'autre part, je savais avec non moins de certitude que tant que dureraient les vacances du jour de l'an, ce n'était pas la peine d'aller aux Champs-Élysées. De sorte que, durant cette triste semaine déjà lointaine, j'avais supporté ma tristesse avec calme parce qu'elle n'était mêlée ni de crainte ni d'espérance. Maintenant, au contraire, c'était ce dernier sentiment qui presque autant que la crainte rendait ma souffrance intolérable. N'ayant pas eu de lettre de Gilberte le soir même, j'avais fait la part de sa négligence, de ses occupations, je ne doutais pas d'en trouver une d'elle dans le courrier du matin. Il fut attendu par moi, chaque jour, avec des palpitations de coeur auxquelles succédait un état d'abattement quand je n'y avais trouvé que des lettres de personnes qui n'étaient pas Gilberte ou bien rien, ce qui n'était pas pire, les preuves d'amitié d'une autre me rendant plus cruelles celles de son indifférence. Je me remettais à espérer pour le courrier de l'après-midi. Même entre les heures des levées des lettres je n'osais pas sortir, car elle eût pu faire porter la sienne. Puis le moment finissait par arriver où, ni facteur ni valet de pied des Swann ne pouvant plus venir, il fallait remettre au lendemain matin l'espoir d'être rassuré, et ainsi, parce que je croyais que ma souffrance ne durerait pas, j'étais obligé pour ainsi dire de la renouveler sans cesse. Le chagrin était peut-être le même, mais au lieu de ne faire, comme autrefois, que prolonger uniformément une émotion initiale, recommençait plusieurs fois par jour en débutant par une émotion si fréquemment renouvelée qu'elle finissait – elle, état tout physique, si momentané – par se stabiliser, si bien que les troubles causés par l'attente ayant à peine le temps de se calmer avant qu'une nouvelle raison d'attendre survînt, il n'y avait plus une seule minute par jour où je ne fusse dans cette anxiété qu'il est pourtant si difficile de supporter pendant une heure. Ainsi ma souffrance était infiniment plus cruelle qu'au temps de cet ancien 1er janvier, parce que cette fois il y avait en moi, au lieu de l'acceptation pure et simple de cette souffrance, l'espoir, à chaque instant, de la voir cesser.
