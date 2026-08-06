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
        "M. de Norpois"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Norpois",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.82,
      "evidence": "Après avoir cité: « Les chiens aboient, la caravane passe », Norpois s’arrête pour juger de l’effet. Le narrateur note que « le proverbe nous était connu », que ces maximes remplacent d’autres par cycles (« culture alternée, et généralement triennale »), et que des formules comme « Le Cabinet de Saint-James… » suffisent à faire reconnaître le diplomate; sa « réputation de grand lettré » tient à « l’emploi raisonné de citations ».",
      "explanation": "The narrator reveals the routine and fashionable character of Norpois's quotations and formulas, suggesting that his literary renown rests on clichés rather than on genuine intellectual substance."
    }
  ],
  "status_effects": [
    {
      "character": "Norpois",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "Locally, Norpois is lowered: his authority as a man of letters appears fabricated by the use of commonplaces and proverbs for effect."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-37-p-44"
}

### Candidate characters

[
  "M. de Vaugoubert",
  "baron de Charlus",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Ma mère comptait beaucoup sur la salade d'ananas et de truffes. Mais l'Ambassadeur après avoir exercé un instant sur le mets la pénétration de son regard d'observateur la mangea en restant entouré de discrétion diplomatique et ne nous livra pas sa pensée. Ma mère insista pour qu'il en reprit, ce que fit Norpois, mais en disant seulement au lieu du compliment qu'on espérait : « J'obéis, Madame, puisque je vois que c'est là de votre part un véritable oukase. »

### Passage

– Nous avons lu dans les « feuilles » que vous vous étiez entretenu longuement avec le roi Théodose, lui dit mon père.

– En effet, le roi, qui a une rare mémoire des physionomies, a eu la bonté de se souvenir en m'apercevant à l'orchestre que j'avais eu l'honneur de le voir pendant plusieurs jours à la cour de Bavière, quand il ne songeait pas à son trône oriental (vous savez qu'il y a été appelé par un congrès européen, et il a même fort hésité à l'accepter, jugeant cette souveraineté un peu inégale à sa race, la plus noble, héraldiquement parlant, de toute l'Europe). Un aide de camp est venu me dire d'aller saluer Sa Majesté, à l'ordre de qui je me suis naturellement empressé de déférer.

– Avez-vous été content des résultats de son séjour ?

– Enchanté ! Il était permis de concevoir quelque appréhension sur la façon dont un monarque encore si jeune se tirerait de ce pas difficile, surtout dans des conjonctures aussi délicates. Pour ma part je faisais pleine confiance au sens politique du souverain. Mais j'avoue que mes espérances ont été dépassées. Le toast qu'il a prononcé à l'Élysée, et qui, d'après des renseignements qui me viennent de source tout à fait autorisée, avait été composé par lui du premier mot jusqu'au dernier, était entièrement digne de l'intérêt qu'il a excité partout. C'est tout simplement un coup de maître ; un peu hardi je le veux bien, mais d'une audace qu'en somme l'événement a pleinement justifiée. Les traditions diplomatiques ont certainement du bon, mais dans l'espèce elles avaient fini par faire vivre son pays et le nôtre dans une atmosphère de renfermé qui n'était plus respirable. Eh bien ! une des manières de renouveler l'air, évidemment une de celles qu'on ne peut pas recommander mais que le roi Théodose pouvait se permettre, c'est de casser les vitres. Et il l'a fait avec une belle humeur qui a ravi tout le monde, et aussi une justesse dans les termes où on a reconnu tout de suite la race de princes lettrés à laquelle il appartient par sa mère. Il est certain que quand il a parlé des « affinités » qui unissent son pays à la France, l'expression, pour peu usitée qu'elle puisse être dans le vocabulaire des chancelleries, était singulièrement heureuse. Vous voyez que la littérature ne nuit pas, même dans la diplomatie, même sur un trône, ajouta-t-il en s'adressant à moi. La chose était constatée depuis longtemps, je le veux bien, et les rapports entre les deux puissances étaient devenus excellents. Encore fallait-il qu'elle fût dite. Le mot était attendu, il a été choisi à merveille, vous avez vu comme il a porté. Pour ma part j'y applaudis des deux mains.

– Votre ami, M. de Vaugoubert, qui préparait le rapprochement depuis des années, a dû être content.

– D'autant plus que Sa Majesté qui est assez coutumière du fait avait tenu à lui en faire la surprise. Cette surprise a été complète du reste pour tout le monde, à commencer par le Ministre des Affaires étrangères, qui, à ce qu'on m'a dit, ne l'a pas trouvée à son goût. À quelqu'un qui lui en parlait, il aurait répondu très nettement, assez haut pour être entendu des personnes voisines : « Je n'ai été ni consulté, ni prévenu », indiquant clairement par là qu'il déclinait toute responsabilité dans l'événement. Il faut avouer que celui-ci a fait un beau tapage et je n'oserais pas affirmer, ajouta-t-il avec un sourire malicieux, que tels de mes collègues pour qui la loi suprême semble être celle du moindre effort n'en ont pas été troublés dans leur quiétude.

Quant à Vaugoubert, vous savez qu'il avait été fort attaqué pour sa politique de rapprochement avec la France, et il avait dû d'autant plus en souffrir, que c'est un sensible, un coeur exquis. J'en puis d'autant mieux témoigner que, bien qu'il soit mon cadet et de beaucoup, je l'ai fort pratiqué, nous sommes amis de longue date, et je le connais bien. D'ailleurs qui ne le connaîtrait ? C'est une âme de cristal. C'est même le seul défaut qu'on pourrait lui reprocher, il n'est pas nécessaire que le coeur d'un diplomate soit aussi transparent que le sien. Cela n'empêche pas qu'on parle de l'envoyer à Rome, ce qui est un bel avancement, mais un bien gros morceau. Entre nous, je crois que Vaugoubert, si dénué qu'il soit d'ambition, en serait fort content et ne demande nullement qu'on éloigne de lui ce calice. Il fera peut-être merveille là-bas ; il est le candidat de la Consulta, et pour ma part, je le vois très bien, lui artiste, dans le cadre du palais Farnèse et la galerie des Carraches. Il semble qu'au moins personne ne devrait pouvoir le haïr ; mais il y a autour du roi Théodose toute une camarilla plus ou moins inféodée à la Wilhelmstrasse dont elle suit docilement les inspirations et qui a cherché de toutes façons à lui tailler des croupières. Vaugoubert n'a pas eu à faire face seulement aux intrigues de couloirs mais aux injures de folliculaires à gages qui plus tard, lâches comme l'est tout journaliste stipendié, ont été des premiers à demander l'aman, mais qui en attendant n'ont pas reculé à faire état, contre notre représentant, des ineptes accusations de gens sans aveu. Pendant plus d'un mois les ennemis de Vaugoubert ont dansé autour de lui la danse du scalp, dit Norpois, en détachant avec force ce dernier mot. Mais un bon averti en vaut deux ; ces injures il les a repoussées du pied, ajouta-t-il plus énergiquement encore, et avec un regard si farouche que nous cessâmes un instant de manger. Comme dit un beau proverbe arabe : « Les chiens aboient, la caravane passe. » Après avoir jeté cette citation, Norpois s'arrêta pour nous regarder et juger de l'effet qu'elle avait produit sur nous. Il fut grand, le proverbe nous était connu. Il avait remplacé cette année-là chez les hommes de haute valeur cet autre : « Qui sème le vent récolte la tempête », lequel avait besoin de repos, n'étant pas infatigable et vivace comme : « Travailler pour le roi de Prusse. » Car la culture de ces gens éminents était une culture alternée, et généralement triennale. Certes les citations de ce genre, et desquelles Norpois excellait à émailler ses articles de la Revue, n'étaient point nécessaires pour que ceux-ci parussent solides et bien informés. Même dépourvus de l'ornement qu'elles apportaient, il suffisait que Norpois écrivît à point nommé – ce qu'il ne manquait pas de faire – : « Le Cabinet de Saint-James ne fut pas le dernier à sentir le péril » ou bien : « L'émotion fut grande au Pont-aux-Chantres où l'on suivait d'un oeil inquiet la politique égoïste mais habile de la monarchie bicéphale », ou : « Un cri d'alarme partit de Montecitorio », ou encore : « Cet éternel double jeu qui est bien dans la manière du Ballplatz ». À ces expressions le lecteur profane avait aussitôt reconnu et salué le diplomate de carrière. Mais ce qui avait fait dire qu'il était plus que cela, qu'il possédait une culture supérieure, cela avait été l'emploi raisonné de citations dont le modèle achevé restait alors : « Faites-moi de bonne politique et je vous ferai de bonnes finances, comme avait coutume de dire le baron Louis. » (On n'avait pas encore importé d'Orient : « La Victoire est à celui des deux adversaires qui sait souffrir un quart d'heure de plus que l'autre, comme disent les Japonais. ») Cette réputation de grand lettré, jointe à un véritable génie d'intrigue caché sous le masque de l'indifférence, avait fait entrer Norpois à l'Académie des Sciences Morales. Et quelques personnes pensèrent même qu'il ne serait pas déplacé à l'Académie française, le jour où, voulant indiquer que c'est en resserrant l'alliance russe que nous pourrions arriver à une entente avec l'Angleterre, il n'hésita pas à écrire : « Qu'on le sache bien au quai d'Orsay, qu'on l'enseigne désormais dans tous les manuels de géographie qui se montrent incomplets à cet égard, qu'on refuse impitoyablement au baccalauréat tout candidat qui ne saura pas le dire : « Si tous les chemins mènent à Rome, en revanche la route qui va de Paris à Londres passe nécessairement par Pétersbourg. »

– Somme toute, continua Norpois en s'adressant à mon père, Vaugoubert s'est taillé là un beau succès et qui dépasse même celui qu'il avait escompté. Il s'attendait en effet à un toast correct (ce qui après les nuages des dernières années était déjà fort beau) mais à rien de plus. Plusieurs personnes qui étaient au nombre des assistants m'ont assuré qu'on ne peut pas en lisant ce toast se rendre compte de l'effet qu'il a produit, prononcé et détaillé à merveille par le roi qui est maître en l'art de dire et qui soulignait au passage toutes les intentions, toutes les finesses. Je me suis laissé raconter à ce propos un fait assez piquant et qui met en relief une fois de plus chez le roi Théodose cette bonne grâce juvénile qui lui gagne si bien les coeurs. On m'a affirmé que précisément à ce mot d'« affinités » qui était en somme la grosse innovation du discours, et qui défraiera, encore longtemps vous verrez, les commentaires des chancelleries, Sa Majesté, prévoyant la joie de notre ambassadeur, qui allait trouver là le juste couronnement de ses efforts, de son rêve pourrait-on dire et, somme toute, son bâton de maréchal, se tourna à demi vers Vaugoubert et fixant sur lui ce regard si prenant des Oettingen, détacha ce mot si bien choisi d'« affinités », ce mot qui était une véritable trouvaille sur un ton qui faisait savoir à tous qu'il était employé à bon escient et en pleine connaissance de cause. Il paraît que Vaugoubert avait peine à maîtriser son émotion et, dans une certaine mesure, j'avoue que je le comprends. Une personne digne de toute créance m'a même confié que le roi se serait approché de Vaugoubert après le dîner, quand Sa Majesté a tenu cercle, et lui aurait dit à mi-voix : « Êtes-vous content de votre élève, mon cher marquis ? »
