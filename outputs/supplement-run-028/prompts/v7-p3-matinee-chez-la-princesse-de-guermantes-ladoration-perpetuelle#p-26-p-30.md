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
      "canonical_name": "Bergotte",
      "surface_forms": [
        "Bergotte"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Bergotte",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.86,
      "evidence": "« On préférait à Bergotte ... des écrivains qui semblaient plus profonds simplement parce qu'ils écrivaient moins bien. » et « dont les plus jolies phrases avaient exigé en réalité un bien plus profond repli sur soi-même »",
      "explanation": "The narrator defends the artistic value of Bergotte, asserting that his complexity rests on a depth of attention to true impression, against the critical fashions that disdain him."
    }
  ],
  "status_effects": [
    {
      "character": "Bergotte",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.84,
      "explanation": "Bergotte is locally elevated by the authority of the narrator, despite the existence of a social preference for other writers."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle#p-26-p-30"
}

### Candidate characters

[
  "Bloch",
  "Norpois",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

L'idée d'un art populaire comme d'un art patriotique, si même elle n'avait pas été dangereuse, me semblait ridicule. S'il s'agissait de le rendre accessible au peuple, on sacrifiait les raffinements de la forme « bons pour des oisifs » ; or, j'avais assez fréquenté de gens du monde pour savoir que ce sont eux les véritables illettrés, et non les ouvriers électriciens. À cet égard, un art, populaire par la forme, eût été destiné plutôt aux membres du Jockey qu'à ceux de la Confédération générale du travail ; quant aux sujets, les romans populaires enivrent autant les gens du peuple que les enfants ces livres qui sont écrits pour eux. On cherche à se dépayser en lisant, et les ouvriers sont aussi curieux des princes que les princes des ouvriers. Dès le début de la guerre, M. Barrès avait dit que l'artiste (en l'espèce le Titien) doit avant tout servir la gloire de sa patrie. Mais il ne peut la servir qu'en étant artiste, c'est-à-dire qu'à condition, au moment où il étudie les lois de l'Art, institue ses expériences et fait ses découvertes, aussi délicates que celles de la Science, de ne pas penser à autre chose – fût-ce à la patrie – qu'à la vérité qui est devant lui. N'imitons pas les révolutionnaires qui par « civisme » méprisaient, s'ils ne les détruisaient pas, les oeuvres de Watteau et de La Tour, peintres qui honoraient davantage la France que tous ceux de la Révolution. L'anatomie n'est peut-être pas ce que choisirait un coeur tendre, si l'on avait le choix. Ce n'est pas la bonté de son coeur vertueux, laquelle était fort grande, qui a fait écrire à Choderlos de Laclos les Liaisons Dangereuses, ni son goût pour la bourgeoisie, petite ou grande, qui a fait choisir à Flaubert comme sujets ceux de Madame Bovary et de l'Éducation Sentimentale. Certains disaient que l'art d'une époque de hâte serait bref, comme ceux qui prédisaient avant la guerre qu'elle serait courte. Le chemin de fer devait aussi tuer la contemplation, il était vain de regretter le temps des diligences, mais l'automobile remplit leur fonction et arrête à nouveau les touristes vers les églises abandonnées.

### Passage

Une image offerte par la vie nous apporte en réalité, à ce moment-là, des sensations multiples et différentes. La vue, par exemple, de la couverture d'un livre déjà lu a tissé dans les caractères de son titre les rayons de lune d'une lointaine nuit d'été. Le goût du café au lait matinal nous apporte cette vague espérance d'un beau temps qui jadis si souvent, pendant que nous le buvions dans un bol de porcelaine blanche, crémeuse et plissée, qui semblait du lait durci, se mit à nous sourire dans la claire incertitude du petit jour. Une heure n'est pas qu'une heure, c'est un vase rempli de parfums, de sons, de projets et de climats. Ce que nous appelons la réalité est un certain rapport entre ces sensations et ces souvenirs qui nous entourent simultanément – rapport que supprime une simple vision cinématographique, laquelle s'éloigne par là d'autant plus du vrai qu'elle prétend se borner à lui – rapport unique que l'écrivain doit retrouver pour en enchaîner à jamais dans sa phrase les deux termes différents. On peut faire se succéder indéfiniment dans une description les objets qui figuraient dans le lieu décrit, la vérité ne commencera qu'au moment où l'écrivain prendra deux objets différents, posera leur rapport, analogue dans le monde de l'art à celui qu'est le rapport unique de la loi causale dans le monde de la science, et les enfermera dans les anneaux nécessaires d'un beau style, ou même, ainsi que la vie, quand, en rapprochant une qualité commune à deux sensations, il dégagera leur essence en les réunissant l'une et l'autre, pour les soustraire aux contingences du temps, dans une métaphore, et les enchaînera par le lien indescriptible d'une alliance de mots. La nature elle-même, à ce point de vue, ne m'avait-elle pas mis sur la voie de l'art, n'était-elle pas commencement d'art, elle qui souvent ne m'avait permis de connaître la beauté d'une chose que longtemps après, dans une autre, midi à Combray que dans le bruit de ses cloches, les matinées de Doncières que dans les hoquets de notre calorifère à eau ? Le rapport peut être peu intéressant, les objets médiocres, le style mauvais, mais tant qu'il n'y a pas eu cela il n'y a rien eu. La littérature qui se contente de « décrire les choses », de donner un misérable relevé de leurs lignes et de leur surface, est, malgré sa prétention réaliste, la plus éloignée de la réalité, celle qui nous appauvrit et nous attriste le plus, ne parlât-elle que de gloire et de grandeurs, car elle coupe brusquement toute communication de notre moi présent avec le passé, dont les choses gardent l'essence, et l'avenir, où elles nous incitent à le goûter encore. Mais il y avait plus. Si la réalité était cette espèce de déchet de l'expérience, à peu près identique pour chacun, parce que, quand nous disons : un mauvais temps, une guerre, une station de voitures, un restaurant éclairé, un jardin en fleurs, tout le monde sait ce que nous voulons dire ; si la réalité était cela, sans doute une sorte de film cinématographique de ces choses suffirait et le « style », la « littérature » qui s'écarteraient de leur simple donnée seraient un hors-d'oeuvre artificiel. Mais était-ce bien cela la réalité ? Si j'essayais de me rendre compte de ce qui se passe, en effet, en nous au moment où une chose nous fait une certaine impression, soit que, comme ce jour où, en passant sur le pont de la Vivonne, l'ombre d'un nuage sur l'eau m'eût fait crier « zut alors ! » en sautant de joie ; soit qu'écoutant une phrase de Bergotte tout ce que j'eusse vu de mon impression c'est ceci qui ne lui convenait pas spécialement : « C'est admirable » ; soit qu'irrité d'un mauvais procédé, Bloch prononçât ces mots qui ne convenaient pas du tout à une aventure si vulgaire : « Qu'on agisse ainsi, je trouve cela même fantastique » ; soit quand, flatté d'être bien reçu chez les Guermantes, et d'ailleurs un peu grisé par leurs vins, je n'aie pu m'empêcher de dire à mi-voix, seul, en les quittant : « Ce sont tout de même des êtres exquis avec qui il serait doux de passer la vie », je m'apercevais que, pour exprimer ces impressions, pour écrire ce livre essentiel, le seul livre vrai, un grand écrivain n'a pas, dans le sens courant, à l'inventer puisqu'il existe déjà en chacun de nous, mais à le traduire. Le devoir et la tâche d'un écrivain sont ceux d'un traducteur.

Or si, quand il s'agit du langage inexact de l'amour-propre par exemple, le redressement de l'oblique discours intérieur (qui va s'éloignant de plus en plus de l'impression première et cérébrale) jusqu'à ce qu'il se confonde avec la droite qui aurait dû partir de l'impression, si ce redressement est chose malaisée contre quoi boude notre paresse, il est d'autres cas, celui où il s'agit de l'amour, par exemple, où ce même redressement devient douloureux. Toutes nos feintes indifférences, toute notre indignation contre ses mensonges si naturels, si semblables à ceux que nous pratiquons nous-mêmes, en un mot tout ce que nous n'avons cessé, chaque fois que nous étions malheureux ou trahis, non seulement de dire à l'être aimé, mais même, en attendant de le voir, de nous dire sans fin à nous-mêmes, quelquefois à haute voix, dans le silence de notre chambre troublé par quelques : « non, vraiment, de tels procédés sont intolérables » et « j'ai voulu te recevoir une dernière fois et ne nierai pas que cela me fasse de la peine », ramener tout cela à la vérité ressentie dont cela s'était tant écarté, c'est abolir tout ce à quoi nous tenions le plus, ce qui, seul à seul avec nous-mêmes, dans des projets fiévreux de lettres et de démarches, fut notre entretien passionné avec nous-mêmes.

Même dans les joies artistiques, qu'on recherche pourtant en vue de l'impression qu'elles donnent, nous nous arrangeons le plus vite possible à laisser de côté comme inexprimable ce qui est précisément cette impression même, et à nous attacher à ce qui nous permet d'en éprouver le plaisir sans le connaître, jusqu'au fond et de croire le communiquer à d'autres amateurs avec qui la conversation sera possible, parce que nous leur parlerons d'une chose qui est la même pour eux et pour nous, la racine personnelle de notre propre impression étant supprimée. Dans les moments mêmes où nous sommes les spectateurs les plus désintéressés de la nature, de la société, de l'amour, de l'art lui-même, comme toute impression est double, à demi engainée dans l'objet, prolongée en nous-mêmes par une autre moitié que seuls nous pourrions connaître, nous nous empressons de négliger celle-là, c'est-à-dire la seule à laquelle nous devrions nous attacher, et nous ne tenons compte que de l'autre moitié qui, ne pouvant pas être approfondie parce qu'elle est extérieure, ne sera cause pour nous d'aucune fatigue : le petit sillon qu'une phrase musicale ou la vue d'une église a creusé en nous, nous trouvons trop difficile de tâcher de l'apercevoir. Mais nous rejouons la symphonie, nous retournons voir l'église jusqu'à ce que – dans cette fuite loin de notre propre vie que nous n'avons pas le courage de regarder, et qui s'appelle l'érudition – nous les connaissions aussi bien, de la même manière, que le plus savant amateur de musique ou d'archéologie. Aussi combien s'en tiennent là qui n'extraient rien de leur impression, vieillissent inutiles et insatisfaits, comme des célibataires de l'art. Ils ont les chagrins qu'ont les vierges et les paresseux, et que la fécondité dans le travail guérirait. Ils sont plus exaltés à propos des oeuvres d'art que les véritables artistes, car leur exaltation n'étant pas pour eux l'objet d'un dur labeur d'approfondissement, elle se répand au dehors, échauffe leurs conversations, empourpre leur visage ; ils croient accomplir un acte en hurlant à se casser la voix : « Bravo, bravo » après l'exécution d'une oeuvre qu'ils aiment. Mais ces manifestations ne les forcent pas à éclaircir la nature de leur amour, ils ne la connaissent pas. Cependant celui-ci, inutilisé, reflue même sur leurs conversations les plus calmes, leur fait faire de grands gestes, des grimaces, des hochements de tête quand ils parlent d'art. « J'ai été à un concert où on jouait une musique qui, je vous avouerai, ne m'emballait pas. On commence alors le quatuor. Ah ! mais, nom d'une pipe ! ça change (la figure de l'amateur à ce moment-là exprime une inquiétude anxieuse comme s'il pensait : « Mais je vois des étincelles, ça sent le roussi, il y a le feu »). Tonnerre de Dieu, ce que j'entends là c'est exaspérant, c'est mal écrit, mais c'est épastrouillant, ce n'est pas l'oeuvre de tout le monde. » Encore, si risibles que soient ces amateurs, ils ne sont pas tout à fait à dédaigner. Ils sont les premiers essais de la nature qui veut créer l'artiste, aussi informes, aussi peu viables que ces premiers animaux qui précédèrent les espèces actuelles et qui n'étaient pas constitués pour durer. Ces amateurs velléitaires et stériles doivent nous toucher comme ces premiers appareils qui ne purent quitter la terre mais où résidait, non encore le moyen secret et qui restait à découvrir, mais le désir du vol. « Et, mon vieux, ajoute l'amateur en vous prenant par le bras, moi c'est la huitième fois que je l'entends, et je vous jure bien que ce n'est pas la dernière. » Et, en effet, comme ils n'assimilent pas ce qui dans l'art est vraiment nourricier, ils ont tout le temps besoin de joies artistiques, en proie à une boulimie qui ne les rassasie jamais. Ils vont donc applaudir longtemps de suite la même oeuvre, croyant, de plus, que leur présence réalise un devoir, un acte, comme d'autres personnes la leur à une séance d'un Conseil d'administration, à un enterrement. Puis viennent des oeuvres autres, même opposées, que ce soit en littérature, en peinture ou en musique. Car la faculté de lancer des idées, des systèmes, et surtout de se les assimiler, a toujours été beaucoup plus fréquente, même chez ceux qui produisent, que le véritable goût, mais prend une extension plus considérable depuis que les revues, les journaux littéraires se sont multipliés (et avec eux les vocations factices d'écrivains et d'artistes). Ainsi la meilleure partie de la jeunesse, la plus intelligente, la plus intéressée, n'aimait-elle plus que les oeuvres ayant une haute portée morale et sociologique, même religieuse. Elle s'imaginait que c'était là le critérium de la valeur d'une oeuvre, renouvelant ainsi l'erreur des David, des Chenavard, des Brunetière, etc. On préférait à Bergotte, dont les plus jolies phrases avaient exigé en réalité un bien plus profond repli sur soi-même, des écrivains qui semblaient plus profonds simplement parce qu'ils écrivaient moins bien. La complication de son écriture n'était faite que pour des gens du monde, disaient des démocrates, qui faisaient ainsi aux gens du monde un honneur immérité. Mais dès que l'intelligence raisonneuse veut se mettre à juger des oeuvres d'art, il n'y a plus rien de fixe, de certain : on peut démontrer tout ce qu'on veut. Alors que la réalité du talent est un bien, une acquisition universelle, dont on doit avant tout constater la présence sous les modes apparentes de la pensée et du style, c'est sur ces dernières que la critique s'arrête pour classer les auteurs. Elle sacre prophète à cause de son ton péremptoire, de son mépris affiché pour l'école qui l'a précédé, un écrivain qui n'apporte nul message nouveau. Cette constante aberration de la critique est telle qu'un écrivain devrait presque préférer être jugé par le grand public (si celui-ci n'était incapable de se rendre compte même de ce qu'un artiste a tenté dans un ordre de recherches qui lui est inconnu). Car il y a plus d'analogie entre la vie instinctive du public et le talent d'un grand écrivain, qui n'est qu'un instinct religieusement écouté au milieu du silence, imposé à tout le reste, un instinct perfectionné et compris, qu'avec le verbiage superficiel et les critères changeants des juges attitrés. Leur logomachie se renouvelle de dix ans en dix ans (car le kaléidoscope n'est pas composé seulement par les groupes mondains, mais par les idées sociales, politiques, religieuses qui prennent une ampleur momentanée grâce à leur réfraction dans les masses étendues, mais restent limitées malgré cela à la courte vie des idées dont la nouveauté n'a pu séduire que des esprits peu exigeants en fait de preuves). Ainsi s'étaient succédé les partis et les écoles, faisant se prendre à eux toujours les mêmes esprits, hommes d'une intelligence relative, toujours voués aux engouements dont s'abstiennent des esprits plus scrupuleux et plus difficiles en fait de preuves. Malheureusement, justement parce que les autres ne sont que de demi-esprits, ils ont besoin de se compléter dans l'action, ils agissent ainsi plus que les esprits supérieurs, attirent à eux la foule et créent autour d'eux non seulement les réputations surfaites et les dédains injustifiés mais les guerres civiles et les guerres extérieures, dont un peu de critique point royaliste sur soi-même devrait préserver. Et quant à la jouissance que donne à un esprit parfaitement juste, à un coeur vraiment vivant, la belle pensée d'un maître, elle est sans doute entièrement saine, mais, si précieux que soient les hommes qui la goûtent vraiment (combien y en a-t-il en vingt ans), elle les réduit tout de même à n'être que la pleine conscience d'un autre. Qu'un homme ait tout fait pour être aimé d'une femme qui n'eût pu que le rendre malheureux, mais n'ait même pas réussi, malgré ses efforts redoublés pendant des années, à obtenir un rendez-vous de cette femme, au lieu de chercher à exprimer ses souffrances et le péril auquel il a échappé, il relit sans cesse, en mettant sous elle « un million de mots » et les souvenirs les plus émouvants de sa propre vie, cette pensée de La Bruyère : « Les hommes souvent veulent aimer et ne sauraient y réussir, ils cherchent leur défaite sans pouvoir la rencontrer, et, si j'ose ainsi parler, ils sont contraints de demeurer libres. » Que ce soit ce sens ou non qu'ait eu cette pensée pour celui qui l'écrivit (pour qu'elle l'eût, et ce serait plus beau, il faudrait « être aimés » au lieu d'« aimer »), il est certain qu'en lui ce lettré sensible la vivifie, la gonfle de signification jusqu'à la faire éclater, il ne peut la redire qu'en débordant de joie tant il la trouve vraie et belle, mais il n'y a malgré tout rien ajouté, et il reste seulement la pensée de La Bruyère.

Comment la littérature de notations aurait-elle une valeur quelconque, puisque c'est sous de petites choses comme celles qu'elle note que la réalité est contenue (la grandeur dans le bruit lointain d'un aéroplane, dans la ligne du clocher de Saint-Hilaire, le passé dans la saveur d'une madeleine, etc.) et qu'elles sont sans signification par elles-mêmes si on ne l'en dégage pas ?

Peu à peu conservée par la mémoire, c'est la chaîne de toutes les impressions inexactes, où ne reste rien de ce que nous avons réellement éprouvé, qui constitue pour nous notre pensée, notre vie, la réalité, et c'est ce mensonge-là que ne ferait que reproduire un art soi-disant « vécu », simple comme la vie, sans beauté, double emploi si ennuyeux et si vain de ce que nos yeux voient et de ce que notre intelligence constate, qu'on se demande où celui qui s'y livre trouve l'étincelle joyeuse et motrice, capable de le mettre en train et de le faire avancer dans sa besogne. La grandeur de l'art véritable, au contraire, de celui que Norpois eût appelé un jeu de dilettante, c'était de retrouver, de ressaisir, de nous faire connaître cette réalité loin de laquelle nous vivons, de laquelle nous nous écartons de plus en plus au fur et à mesure que prend plus d'épaisseur et d'imperméabilité la connaissance conventionnelle que nous lui substituons, cette réalité que nous risquerions fort de mourir sans l'avoir connue, et qui est tout simplement notre vie, la vraie vie, la vie enfin découverte et éclaircie, la seule vie, par conséquent, réellement vécue, cette vie qui, en un sens, habite à chaque instant chez tous les hommes aussi bien que chez l'artiste. Mais ils ne la voient pas, parce qu'ils ne cherchent pas à l'éclaircir. Et ainsi leur passé est encombré d'innombrables clichés qui restent inutiles parce que l'intelligence ne les a pas « développés ». Ressaisir notre vie ; et aussi la vie des autres ; car le style, pour l'écrivain aussi bien que pour le peintre, est une question non de technique, mais de vision. Il est la révélation, qui serait impossible par des moyens directs et conscients, de la différence qualitative qu'il y a dans la façon dont nous apparaît le monde, différence qui, s'il n'y avait pas l'art, resterait le secret éternel de chacun. Par l'art seulement, nous pouvons sortir de nous, savoir ce que voit un autre de cet univers qui n'est pas le même que le nôtre et dont les paysages nous seraient restés aussi inconnus que ceux qu'il peut y avoir dans la lune. Grâce à l'art, au lieu de voir un seul monde, le nôtre, nous le voyons se multiplier, et autant qu'il y a d'artistes originaux, autant nous avons de mondes à notre disposition, plus différents les uns des autres que ceux qui roulent dans l'infini, et qui bien des siècles après qu'est éteint le foyer dont ils émanaient, qu'il s'appelât Rembrandt ou Ver Meer, nous envoient leur rayon spécial.
