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
      "canonical_name": "la grand-mère",
      "surface_forms": [
        "la grand-mère"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.97
    }
  ],
  "appraisal_events": [],
  "status_effects": [],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-141-p-145"
}

### Candidate characters

[
  "Françoise",
  "le narrateur"
]

### Prior local context (optional)

Cette obscure fraîcheur de ma chambre était au plein soleil de la rue ce que l'ombre est au rayon, c'est-à-dire aussi lumineuse que lui et offrait à mon imagination le spectacle total de l'été dont mes sens, si j'avais été en promenade, n'auraient pu jouir que par morceaux ; et ainsi elle s'accordait bien à mon repos qui (grâce aux aventures racontées par mes livres et qui venaient l'émouvoir) supportait pareil au repos d'une main immobile au milieu d'une eau courante, le choc et l'animation d'un torrent d'activité.

### Passage

Mais ma grand'mère, même si le temps trop chaud s'était gâté, si un orage ou seulement un grain était survenu, venait me supplier de sortir. Et ne voulant pas renoncer à ma lecture, j'allais du moins la continuer au jardin, sous le marronnier, dans une petite guérite en sparterie et en toile au fond de laquelle j'étais assis et me croyais caché aux yeux des personnes qui pourraient venir faire visite à mes parents.

Et ma pensée n'était-elle pas aussi comme une autre crèche au fond de laquelle je sentais que je restais enfoncé, même pour regarder ce qui se passait au dehors ? Quand je voyais un objet extérieur, la conscience que je le voyais restait entre moi et lui, le bordait d'un mince liseré spirituel qui m'empêchait de jamais toucher directement sa matière ; elle se volatilisait en quelque sorte avant que je prisse contact avec elle, comme un corps incandescent qu'on approche d'un objet mouillé ne touche pas son humidité parce qu'il se fait toujours précéder d'une zone d'évaporation. Dans l'espèce d'écran diapré d'états différents que, tandis que je lisais, déployait simultanément ma conscience, et qui allaient des aspirations les plus profondément cachées en moi-même jusqu'à la vision tout extérieure de l'horizon que j'avais, au bout du jardin, sous les yeux, ce qu'il y avait d'abord en moi de plus intime, la poignée sans cesse en mouvement qui gouvernait le reste, c'était ma croyance en la richesse philosophique, en la beauté du livre que je lisais, et mon désir de me les approprier, quel que fût ce livre. Car, même si je l'avais acheté à Combray, en l'apercevant devant l'épicerie Borange, trop distante de la maison pour que Françoise pût s'y fournir comme chez Camus, mais mieux achalandée comme papeterie et librairie, retenu par des ficelles dans la mosaïque des brochures et des livraisons qui revêtaient les deux vantaux de sa porte plus mystérieuse, plus semée de pensées qu'une porte de cathédrale, c'est que je l'avais reconnu pour m'avoir été cité comme un ouvrage remarquable par le professeur ou le camarade qui me paraissait à cette époque détenir le secret de la vérité et de la beauté à demi pressenties, à demi incompréhensibles, dont la connaissance était le but vague mais permanent de ma pensée.

Après cette croyance centrale qui, pendant ma lecture, exécutait d'incessants mouvements du dedans au dehors, vers la découverte de la vérité, venaient les émotions que me donnait l'action à laquelle je prenais part, car ces après-midi-là étaient plus remplis d'événements dramatiques que ne l'est souvent toute une vie. C'était les événements qui survenaient dans le livre que je lisais ; il est vrai que les personnages qu'ils affectaient n'étaient pas « réels », comme disait Françoise. Mais tous les sentiments que nous font éprouver la joie ou l'infortune d'un personnage réel ne se produisent en nous que par l'intermédiaire d'une image de cette joie ou de cette infortune ; l'ingéniosité du premier romancier consista à comprendre que dans l'appareil de nos émotions, l'image étant le seul élément essentiel, la simplification qui consisterait à supprimer purement et simplement les personnages réels serait un perfectionnement décisif. Un être réel, si profondément que nous sympathisions avec lui, pour une grande part est perçu par nos sens, c'est-à-dire nous reste opaque, offre un poids mort que notre sensibilité ne peut soulever. Qu'un malheur le frappe, ce n'est qu'en une petite partie de la notion totale que nous avons de lui que nous pourrons en être émus ; bien plus, ce n'est qu'en une partie de la notion totale qu'il a de soi qu'il pourra l'être lui-même. La trouvaille du romancier a été d'avoir l'idée de remplacer ces parties impénétrables à l'âme par une quantité égale de parties immatérielles, c'est-à-dire que notre âme peut s'assimiler. Qu'importe dès lors que les actions, les émotions de ces êtres d'un nouveau genre nous apparaissent comme vraies, puisque nous les avons faites nôtres, puisque c'est en nous qu'elles se produisent, qu'elles tiennent sous leur dépendance, tandis que nous tournons fiévreusement les pages du livre, la rapidité de notre respiration et l'intensité de notre regard. Et une fois que le romancier nous a mis dans cet état, où comme dans tous les états purement intérieurs toute émotion est décuplée, où son livre va nous troubler à la façon d'un rêve mais d'un rêve plus clair que ceux que nous avons en dormant et dont le souvenir durera davantage, alors, voici qu'il déchaîne en nous pendant une heure tous les bonheurs et tous les malheurs possibles dont nous mettrions dans la vie des années à connaître quelques-uns, et dont les plus intenses ne nous seraient jamais révélés parce que la lenteur avec laquelle ils se produisent nous en ôte la perception ; (ainsi notre coeur change, dans la vie, et c'est la pire douleur ; mais nous ne la connaissons que dans la lecture, en imagination : dans la réalité il change, comme certains phénomènes de la nature se produisent assez lentement pour que, si nous pouvons constater successivement chacun de ses états différents, en revanche, la sensation même du changement nous soit épargnée).

Déjà moins intérieur à mon corps que cette vie des personnages, venait ensuite, à demi projeté devant moi, le paysage où se déroulait l'action et qui exerçait sur ma pensée une bien plus grande influence que l'autre, que celui que j'avais sous les yeux quand je les levais du livre. C'est ainsi que pendant deux étés, dans la chaleur du jardin de Combray, j'ai eu, à cause du livre que je lisais alors, la nostalgie d'un pays montueux et fluviatile, où je verrais beaucoup de scieries et où, au fond de l'eau claire, des morceaux de bois pourrissaient sous des touffes de cresson : non loin montaient le long de murs bas des grappes de fleurs violettes et rougeâtres. Et comme le rêve d'une femme qui m'aurait aimé était toujours présent à ma pensée, ces étés-là ce rêve fut imprégné de la fraîcheur des eaux courantes ; et quelle que fût la femme que j'évoquais, des grappes de fleurs violettes et rougeâtres s'élevaient aussitôt de chaque côté d'elle comme des couleurs complémentaires.

Ce n'était pas seulement parce qu'une image dont nous rêvons reste toujours marquée, s'embellit et bénéficie du reflet des couleurs étrangères qui par hasard l'entourent dans notre rêverie ; car ces paysages des livres que je lisais n'étaient pas pour moi que des paysages plus vivement représentés à mon imagination que ceux que Combray mettait sous mes yeux, mais qui eussent été analogues. Par le choix qu'en avait fait l'auteur, par la foi avec laquelle ma pensée allait au-devant de sa parole comme d'une révélation, ils me semblaient être – impression que ne me donnait guère le pays où je me trouvais, et surtout notre jardin, produit sans prestige de la correcte fantaisie du jardinier que méprisait ma grand'mère – une part véritable de la Nature elle-même, digne d'être étudiée et approfondie.
