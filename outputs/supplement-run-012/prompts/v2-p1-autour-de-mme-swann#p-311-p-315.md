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
      "canonical_name": "Gilberte",
      "surface_forms": [
        "Gilberte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.78,
      "evidence": "« Je ne vous verrai probablement plus » ... « à la prochaine demande de rendez-vous qu'elle me ferait adresser, j'aurais encore ... le courage de ne pas céder et, de refus en refus, j'arriverais ... à ne désirerais pas la voir. »",
      "explanation": "The narrator frames his letter and planned repeated refusals as moving away from meetings with Gilberte, a prospective exclusionary stance toward her."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "inclusion_exclusion",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.75,
      "explanation": "She is locally placed at a distance by the narrator's expressed intention to refuse meetings and his letter signaling probable separation."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-311-p-315"
}

### Candidate characters

[
  "M. Verdurin",
  "Mme Cottard",
  "Odette",
  "Swann",
  "le peintre",
  "le narrateur"
]

### Prior local context (optional)

D'ailleurs, j'aurais eu beau parler à Gilberte, elle ne m'aurait pas entendu. Nous nous imaginons toujours, quand nous parlons, que ce sont nos oreilles, notre esprit qui écoutent. Mes paroles ne seraient parvenues à Gilberte que déviées, comme si elles avaient eu à traverser le rideau mouvant d'une cataracte avant d'arriver à mon amie, méconnaissables, rendant un son ridicule, n'ayant plus aucune espèce de sens. La vérité qu'on met dans les mots ne se fraye pas son chemin directement, n'est pas douée d'une évidence irrésistible. Il faut qu'assez de temps passe pour qu'une vérité de même ordre ait pu se former en eux. Alors l'adversaire politique qui, malgré tous les raisonnements et toutes les preuves, tenait le sectateur de la doctrine opposée pour un traître, partage lui-même la conviction détestée à laquelle celui qui cherchait inutilement à la répandre ne tient plus. Alors, le chef-d'oeuvre qui pour les admirateurs qui le lisaient haut semblait montrer en soi les preuves de son excellence et n'offrait à ceux qui écoutaient qu'une image insane ou médiocre, sera par eux proclamé chef-d'oeuvre trop tard pour que l'auteur puisse l'apprendre. Pareillement en amour les barrières, quoi qu'on fasse, ne peuvent être brisées du dehors par celui qu'elles désespèrent ; et c'est quand il ne se souciera plus d'elles que, tout à coup, par l'effet du travail venu d'un autre côté, accompli à l'intérieur de celle qui n'aimait pas, ces barrières, attaquées jadis sans succès, tomberont sans utilité. Si j'étais venu annoncer à Gilberte mon indifférence future et le moyen de la prévenir, elle aurait induit de cette démarche que mon amour pour elle, le besoin que j'avais d'elle, étaient encore plus grands qu'elle n'avait cru, et son ennui de me voir en eût été augmenté. Et il est bien vrai, du reste, que c'est cet amour qui m'aidait, par les états d'esprit disparates qu'il faisait se succéder en moi, à prévoir, mieux qu'elle, la fin de cet amour. Pourtant, un tel avertissement, je l'eusse peut-être adressé, par lettre ou de vive voix, à Gilberte, quand assez de temps eût passé, me la rendant ainsi, il est vrai, moins indispensable, mais aussi ayant pu lui prouver qu'elle ne me l'était pas. Malheureusement, certaines personnes bien ou mal intentionnées lui parlèrent de moi d'une façon qui dut lui laisser croire qu'elles le faisaient à ma prière. Chaque fois que j'appris ainsi que docteur Cottard, la mère du narrateur elle-même, et jusqu'à Norpois avaient, par de maladroites paroles, rendu inutile tout le sacrifice que je venais d'accomplir, gâché tout le résultat de ma réserve en me donnant faussement l'air d'en être sorti, j'avais un double ennui. D'abord je ne pouvais plus faire dater que de ce jour-là ma pénible et fructueuse abstention que les fâcheux avaient à mon insu interrompue et, par conséquent, annihilée. Mais, de plus, j'eusse eu moins de plaisir à voir Gilberte qui me croyait maintenant non plus dignement résigné, mais manoeuvrant dans l'ombre pour une entrevue qu'elle avait dédaigné d'accorder. Je maudissais ces vains bavardages de gens qui souvent, sans même l'intention de nuire ou de rendre service, pour rien, pour parler, quelquefois parce que nous n'avons pas pu nous empêcher de le faire devant eux et qu'ils sont indiscrets (comme nous), nous causent, à point nommé, tant de mal. Il est vrai que dans la funeste besogne accomplie pour la destruction de notre amour, ils sont loin de jouer un rôle égal à deux personnes qui ont pour habitude, l'une par excès de bonté et l'autre de méchanceté, de tout défaire au moment que tout allait s'arranger. Mais ces deux personnes-là, nous ne leur en voulons pas comme aux inopportuns docteur Cottard, car la dernière, c'est la personne que nous aimons, et la première, c'est nous-même.

### Passage

Cependant, comme presque chaque fois que j'allais la voir, Odette m'invitait à venir goûter avec sa fille et me disait de répondre directement à celle-ci, j'écrivais souvent à Gilberte, et dans cette correspondance je ne choisissais pas les phrases qui eussent pu, me semblait-il, la persuader, je cherchais seulement à frayer le lit le plus doux au ruissellement de mes pleurs. Car le regret comme le désir ne cherche pas à s'analyser, mais à se satisfaire ; quand on commence d'aimer, on passe le temps non à savoir ce qu'est son amour, mais à préparer les possibilités des rendez-vous du lendemain. Quand on renonce, on cherche non à connaître son chagrin, mais à offrir de lui à celle qui le cause l'expression qui nous paraît la plus tendre. On dit les choses qu'on éprouve le besoin de dire et que l'autre ne comprendra pas, on ne parle que pour soi-même. J'écrivais : « J'avais cru que ce ne serait pas possible. Hélas, je vois que ce n'est pas si difficile. » Je disais aussi : « Je ne vous verrai probablement plus », je le disais en continuant à me garder d'une froideur qu'elle eût pu croire affectée, et ces mots, en les écrivant, me faisaient pleurer, parce que je sentais qu'ils exprimaient non ce que j'aurais voulu croire, mais ce qui arriverait en réalité. Car à la prochaine demande de rendez-vous qu'elle me ferait adresser, j'aurais encore comme cette fois le courage de ne pas céder et, de refus en refus, j'arriverais peu à peu au moment où à force de ne plus l'avoir vue je ne désirerais pas la voir. Je pleurais mais je trouvais le courage, je connaissais la douceur, de sacrifier le bonheur d'être auprès d'elle à la possibilité de lui paraître agréable un jour, un jour où, hélas ! lui paraître agréable me serait indifférent. L'hypothèse même, pourtant si peu vraisemblable, qu'en ce moment, comme elle l'avait prétendu pendant la dernière visite que je lui avais faite, elle m'aimât, que ce que je prenais pour l'ennui qu'on éprouve auprès de quelqu'un dont on est las ne fût dû qu'à une susceptibilité jalouse, à une feinte d'indifférence analogue à la mienne, ne faisait que rendre ma résolution moins cruelle. Il me semblait alors que dans quelques années, après que nous nous serions oubliés l'un l'autre, quand je pourrais rétrospectivement lui dire que cette lettre qu'en ce moment j'étais en train de lui écrire n'avait été nullement sincère, elle me répondrait : « Comment, vous, vous m'aimiez ? Si vous saviez comme je l'attendais, cette lettre, comme j'espérais un rendez-vous, comme elle me fit pleurer. » La pensée, pendant que je lui écrivais, aussitôt rentré de chez sa mère, que j'étais peut-être en train de consommer précisément ce malentendu-là, cette pensée par sa tristesse même, par le plaisir d'imaginer que j'étais aimé de Gilberte, me poussait à continuer ma lettre.

Si, au moment de quitter Odette quand son « thé » finissait, je pensais à ce que j'allais écrire à sa fille, Mme Cottard elle, en s'en allant, avait eu des pensées d'un caractère tout différent. Faisant sa « petite inspection », elle n'avait pas manqué de féliciter Odette sur les meubles nouveaux, les récentes « acquisitions » remarquées dans le salon. Elle pouvait d'ailleurs y retrouver, quoique en bien petit nombre, quelques-uns des objets qu'Odette avait autrefois dans l'hôtel de la rue Lapérouse, notamment ses animaux en matières précieuses, ses fétiches.

Mais Odette ayant appris d'un ami qu'elle vénérait le mot « tocard » – lequel avait ouvert de nouveaux horizons parce qu'il désignait précisément les choses que quelques années auparavant elle avait trouvées « chic » – toutes ces choses-là successivement avaient suivi dans leur retraite le treillage doré qui servait d'appui aux chrysanthèmes, mainte bonbonnière de chez Giroux et le papier à lettres à couronne (pour ne pas parler des louis en carton semés sur les cheminées et que, bien avant qu'elle connût Swann, un homme de goût lui avait conseillé de sacrifier). D'ailleurs dans le désordre artiste, dans le pêle-mêle d'atelier, des pièces aux murs encore peints de couleurs sombres qui les faisaient aussi différentes que possible des salons blancs que Odette eut un peu plus tard, l'Extrême-Orient reculait de plus en plus devant l'invasion du XVIIIe siècle ; et les coussins que, afin que je fusse plus « confortable », Odette entassait et pétrissait derrière mon dos étaient semés de bouquets Louis XV, et non plus comme autrefois de dragons chinois. Dans la chambre où on la trouvait le plus souvent et dont elle disait : « Oui, je l'aime assez, je m'y tiens beaucoup ; je ne pourrais pas vivre au milieu de choses hostiles et pompier ; c'est ici que je travaille » (sans d'ailleurs préciser si c'était à un tableau, peut-être à un livre, le goût d'en écrire commençait à venir aux femmes qui aiment à faire quelque chose et à ne pas être inutiles), elle était entourée de Saxe (aimant cette dernière sorte de porcelaine, dont elle prononçait le nom avec un accent anglais, jusqu'à dire à propos de tout : C'est joli, cela ressemble à des fleurs de Saxe), elle redoutait pour eux, plus encore que jadis pour ses magots et ses potiches, le toucher ignorant des domestiques auxquels elle faisait expier les transes qu'ils lui avaient données par des emportements auxquels Swann, maître si poli et doux, assistait sans en être choqué. La vue lucide de certaines infériorités n'ôte d'ailleurs rien à la tendresse ; celle-ci les fait au contraire trouver charmantes. Maintenant c'était plus rarement dans des robes de chambre japonaises qu'Odette recevait ses intimes, mais plutôt dans les soies claires et mousseuses de peignoirs Watteau desquelles elle faisait le geste de caresser sur ses seins l'écume fleurie, et dans lesquelles elle se baignait, se prélassait, s'ébattait, avec un tel air de bien-être, de rafraîchissement de la peau, et des respirations si profondes, qu'elle semblait les considérer non pas comme décoratives à la façon d'un cadre, mais comme nécessaires de la même manière que le « tub » et le « footing », pour contenter les exigences de sa physionomie et les raffinements de son hygiène. Elle avait l'habitude de dire qu'elle se passerait plus aisément de pain que d'art et de propreté, et qu'elle eût été plus triste de voir brûler la Joconde que des « foultitudes » de personnes qu'elle connaissait. Théories qui semblaient paradoxales à ses amies, mais la faisaient passer pour une femme supérieure auprès d'elles et lui valaient une fois par semaine la visite du ministre de Belgique, de sorte que dans le petit monde dont elle était le soleil, chacun eût été bien étonné si l'on avait appris qu'ailleurs, chez les Verdurin par exemple, elle passât pour bête. À cause de cette vivacité d'esprit, Odette préférait la société des hommes à celle des femmes. Mais quand elle critiquait celles-ci c'était toujours en cocotte, signalant en elles les défauts qui pouvaient leur nuire auprès des hommes, de grosses attaches, un vilain teint, pas d'orthographe, des poils aux jambes, une odeur pestilentielle, de faux sourcils. Pour telle au contraire qui lui avait jadis montré de l'indulgence et de l'amabilité, elle était plus tendre, surtout si celle-là était malheureuse. Elle la défendait avec adresse et disait : « On est injuste pour elle, car c'est une gentille femme, je vous assure. »

Ce n'était pas seulement l'ameublement du salon d'Odette, c'était Odette elle-même que Mme Cottard et tous ceux qui avaient fréquenté Mme de Crécy auraient eu peine s'ils ne l'avaient pas vue depuis longtemps à reconnaître. Elle semblait avoir tant d'années de moins qu'autrefois. Sans doute, cela tenait en partie à ce qu'elle avait engraissé, et, devenue mieux portante, avait l'air plus calme, frais, reposé, et d'autre part à ce que les coiffures nouvelles, aux cheveux lissés, donnaient plus d'extension à son visage qu'une poudre rose animait, et où ses yeux et son profil, jadis trop saillants, semblaient maintenant résorbés. Mais une autre raison de ce changement consistait en ceci que, arrivée au milieu de la vie, Odette s'était enfin découvert, ou inventé, une physionomie personnelle, un « caractère » immuable, un « genre de beauté », et sur ses traits décousus – qui pendant si longtemps, livrés aux caprices hasardeux et impuissants de la chair, prenant à la moindre fatigue pour un instant des années, une sorte de vieillesse passagère, lui avaient composé tant bien que mal, selon son humeur et selon sa mine, un visage épars, journalier, informe et charmant – avait appliqué ce type fixe, comme une jeunesse immortelle.

Swann avait dans sa chambre, au lieu des belles photographies qu'on faisait maintenant de sa femme, et où la même expression énigmatique et victorieuse laissait reconnaître, quels que fussent la robe et le chapeau, sa silhouette et son visage triomphants, un petit daguerréotype ancien tout simple, antérieur à ce type, et duquel la jeunesse et la beauté d'Odette, non encore trouvées par elle, semblaient absentes. Mais sans doute Swann, fidèle ou revenu à une conception différente, goûtait-il dans la jeune femme grêle aux yeux pensifs, aux traits las, à l'attitude suspendue entre la marche et l'immobilité, une grâce plus botticellienne. Il aimait encore en effet à voir en sa femme un Botticelli. Odette qui au contraire cherchait non à faire ressortir, mais à compenser, à dissimuler ce qui, en elle-même, ne lui plaisait pas, ce qui était peut-être, pour un artiste, son « caractère », mais que, comme femme, elle trouvait des défauts, ne voulait pas entendre parler de ce peintre. Swann possédait une merveilleuse écharpe orientale, bleue et rose, qu'il avait achetée parce que c'était exactement celle de la Vierge du Magnificat. Mais Odette ne voulait pas la porter. Une fois seulement elle laissa son mari lui commander une toilette toute criblée de pâquerettes, de bluets, de myosotis et de campanules d'après la Primavera du Printemps. Parfois, le soir, quand elle était fatiguée, il me faisait remarquer tout bas comme elle donnait sans s'en rendre compte à ses mains pensives le mouvement délié, un peu tourmenté de la Vierge qui trempe sa plume dans l'encrier que lui tend l'ange, avant d'écrire sur le livre saint où est déjà tracé le mot Magnificat. Mais il ajoutait : « Surtout ne le lui dites pas, il suffirait qu'elle le sût pour qu'elle fît autrement. »
