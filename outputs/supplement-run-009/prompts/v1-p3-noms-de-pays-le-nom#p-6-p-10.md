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
      "canonical_name": "docteur Cottard",
      "surface_forms": [
        "docteur Cottard",
        "le docteur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "docteur Cottard",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "neutral_report",
      "confidence": 0.82,
      "evidence": "« docteur Cottard déclara qu'il fallait renoncer ... »; « il défendit aussi d'une façon absolue qu'on me laissât aller au théâtre entendre la Berma »",
      "explanation": "By issuing decisive medical prohibitions that others obey, Cottard is positioned as the authoritative voice whose judgment overrides prior plans."
    }
  ],
  "status_effects": [
    {
      "character": "docteur Cottard",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "His authoritative declarations determine the course of action, showing others' deference to his judgment."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-6-p-10"
}

### Candidate characters

[
  "Bergotte",
  "Françoise",
  "la Berma",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Mais si ces noms absorbèrent à tout jamais l'image que j'avais de ces villes, ce ne fut qu'en la transformant, qu'en soumettant sa réapparition en moi à leurs lois propres ; ils eurent ainsi pour conséquence de la rendre plus belle, mais aussi plus différente de ce que les villes de Normandie ou de Toscane pouvaient être en réalité, et, en accroissant les joies arbitraires de mon imagination, d'aggraver la déception future de mes voyages. Ils exaltèrent l'idée que je me faisais de certains lieux de la terre, en les faisant plus particuliers, par conséquent plus réels. Je ne me représentais pas alors les villes, les paysages, les monuments, comme des tableaux plus ou moins agréables, découpés çà et là dans une même matière, mais chacun d'eux comme un inconnu, essentiellement différent des autres, dont mon âme avait soif et qu'elle aurait profit à connaître. Combien ils prirent quelque chose de plus individuel encore, d'être désignés par des noms, des noms qui n'étaient que pour eux, des noms comme en ont les personnes. Les mots nous présentent des choses une petite image claire et usuelle comme celles que l'on suspend aux murs des écoles pour donner aux enfants l'exemple de ce qu'est un établi, un oiseau, une fourmilière, choses conçues comme pareilles à toutes celles de même sorte. Mais les noms présentent des personnes – et des villes qu'ils nous habituent à croire individuelles, uniques comme des personnes – une image confuse qui tire d'eux, de leur sonorité éclatante ou sombre, la couleur dont elle est peinte uniformément comme une de ces affiches, entièrement bleues ou entièrement rouges, dans lesquelles, à cause des limites du procédé employé ou par un caprice du décorateur, sont bleus ou rouges, non seulement le ciel et la mer, mais les barques, l'église, les passants.

### Passage

Le nom de Parme, une des villes où je désirais le plus aller, depuis que j'avais lu la Chartreuse, m'apparaissant compact, lisse, mauve et doux ; si on me parlait d'une maison quelconque de Parme dans laquelle je serais reçu, on me causait le plaisir de penser que j'habiterais une demeure lisse, compacte, mauve et douce, qui n'avait de rapport avec les demeures d'aucune ville d'Italie, puisque je l'imaginais seulement à l'aide de cette syllabe lourde du nom de Parme, où ne circule aucun air, et de tout ce que je lui avais fait absorber de douceur stendhalienne et du reflet des violettes. Et quand je pensais à Florence, c'était comme à une ville miraculeusement embaumée et semblable à une corolle, parce qu'elle s'appelait la cité des lys et sa cathédrale, Sainte-Marie-des-Fleurs. Quant à Balbec, c'était un de ces noms où comme sur une vieille poterie normande qui garde la couleur de la terre d'où elle fut tirée, on voit se peindre encore la représentation de quelque usage aboli, de quelque droit féodal, d'un état ancien de lieux, d'une manière désuète de prononcer qui en avait formé les syllabes hétéroclites et que je ne doutais pas de retrouver jusque chez l'aubergiste qui me servirait du café au lait à mon arrivée, me menant voir la mer déchaînée devant l'église et auquel je prêtais l'aspect disputeur, solennel et médiéval d'un personnage de fabliau.

Si ma santé s'affermissait et que mes parents me permissent, sinon d'aller séjourner à Balbec, du moins de prendre une fois, pour faire connaissance avec l'architecture et les paysages de la Normandie ou de la Bretagne, ce train d'une heure vingt-deux dans lequel j'étais monté tant de fois en imagination, j'aurais voulu m'arrêter de préférence dans les villes les plus belles ; mais j'avais beau les comparer, comment choisir plus qu'entre des êtres individuels, qui ne sont pas interchangeables, entre Bayeux si haute dans sa noble dentelle rougeâtre et dont le faîte était illuminé par le vieil or de sa dernière syllabe ; Vitré dont l'accent aigu losangeait de bois noir le vitrage ancien ; le doux Lamballe qui, dans son blanc, va du jaune coquille d'oeuf au gris perle ; Coutances, cathédrale normande, que sa diphtongue finale, grasse et jaunissante, couronne par une tour de beurre ; Lannion avec le bruit, dans son silence villageois, du coche suivi de la mouche ; Questambert, Pontorson, risibles et naïfs, plumes blanches et becs jaunes éparpillés sur la route de ces lieux fluviatiles et poétiques ; Benodet, nom à peine amarré que semble vouloir entraîner la rivière au milieu de ses algues ; Pont-Aven, envolée blanche et rose de l'aile d'une coiffe légère qui se reflète en tremblant dans une eau verdie de canal ; Quimperlé, lui, mieux attaché et, depuis le moyen âge, entre les ruisseaux dont il gazouille et s'emperle en une grisaille pareille à celle que dessinent, à travers les toiles d'araignées d'une verrière, les rayons de soleil changés en pointes émoussées d'argent bruni.

Ces images étaient fausses pour une autre raison encore ; c'est qu'elles étaient forcément très simplifiées ; sans doute ce à quoi aspirait mon imagination et que mes sens ne percevaient qu'incomplètement et sans plaisir dans le présent, je l'avais enfermé dans le refuge des noms ; sans doute, parce que j'y avais accumulé du rêve, ils aimantaient maintenant mes désirs ; mais les noms ne sont pas très vastes ; c'est tout au plus si je pouvais y faire entrer deux ou trois des « curiosités » principales de la ville et elles s'y juxtaposaient sans intermédiaires ; dans le nom de Balbec, comme dans le verre grossissant de ces porte-plume qu'on achète aux bains de mer, j'apercevais des vagues soulevées autour d'une église de style persan. Peut-être même la simplification de ces images fut-elle une des causes de l'empire qu'elles prirent sur moi. Quand mon père eut décidé, une année, que nous irions passer les vacances de Pâques à Florence et à Venise, n'ayant pas la place de faire entrer dans le nom de Florence les éléments qui composent d'habitude les villes, je fus contraint à faire sortir une cité surnaturelle de la fécondation, par certains parfums printaniers, de ce que je croyais être, en son essence, le génie de Giotto. Tout au plus – et parce qu'on ne peut pas faire tenir dans un nom beaucoup plus de durée que d'espace – comme certains tableaux de Giotto eux-mêmes qui montrent à deux moments différents de l'action un même personnage, ici couché dans son lit, là s'apprêtant à monter à cheval, le nom de Florence était-il divisé en deux compartiments. Dans l'un, sous un dais architectural, je contemplais une fresque à laquelle était partiellement superposé un rideau de soleil matinal, poudreux, oblique et progressif ; dans l'autre (car ne pensant pas aux noms comme à un idéal inaccessible, mais comme à une ambiance réelle dans laquelle j'irais me plonger, la vie non vécue encore, la vie intacte et pure que j'y enfermais donnait aux plaisirs les plus matériels, aux scènes les plus simples, cet attrait qu'ils ont dans les oeuvres des primitifs), je traversais rapidement – pour trouver plus vite le déjeuner qui m'attendait avec des fruits et du vin de Chianti – le Ponte-Vecchio encombré de jonquilles, de narcisses et d'anémones. Voilà (bien que je fusse à Paris) ce que je voyais et non ce qui était autour de moi. Même à un simple point de vue réaliste, les pays que nous désirons tiennent à chaque moment beaucoup plus de place dans notre vie véritable, que le pays où nous nous trouvons effectivement. Sans doute si alors j'avais fait moi-même plus attention à ce qu'il y avait dans ma pensée quand je prononçais les mots « aller à Florence, à Parme, à Pise, à Venise », je me serais rendu compte que ce que je voyais n'était nullement une ville, mais quelque chose d'aussi différent de tout ce que je connaissais, d'aussi délicieux, que pourrait être pour une humanité dont la vie se serait toujours écoulée dans des fins d'après-midi d'hiver, cette merveille inconnue : une matinée de printemps. Ces images irréelles, fixes, toujours pareilles, remplissant mes nuits et mes jours, différencièrent cette époque de ma vie de celles qui l'avaient précédée (et qui auraient pu se confondre avec elle aux yeux d'un observateur qui ne voit les choses que du dehors, c'est-à-dire qui ne voit rien), comme dans un opéra un motif mélodique introduit une nouveauté qu'on ne pourrait pas soupçonner si on ne faisait que lire le livret, moins encore si on restait en dehors du théâtre à compter seulement les quarts d'heure qui s'écoulent. Et encore, même à ce point de vue de simple quantité, dans notre vie les jours ne sont pas égaux. Pour parcourir les jours, les natures un peu nerveuses, comme était la mienne, disposent, comme les voitures automobiles, de « vitesses » différentes. Il y a des jours montueux et malaisés qu'on met un temps infini à gravir et des jours en pente qui se laissent descendre à fond de train en chantant. Pendant ce mois – où je ressassai comme une mélodie, sans pouvoir m'en rassasier, ces images de Florence, de Venise et de Pise, desquelles le désir qu'elles excitaient en moi gardait quelque chose d'aussi profondément individuel que si ç'avait été un amour, un amour pour une personne – je ne cessai pas de croire qu'elles correspondaient à une réalité indépendante de moi, et elles me firent connaître une aussi belle espérance que pouvait en nourrir un chrétien des premiers âges à la veille d'entrer dans le paradis. Aussi sans que je me souciasse de la contradiction qu'il y avait à vouloir regarder et toucher avec les organes des sens ce qui avait été élaboré par la rêverie et non perçu par eux – et d'autant plus tentant pour eux, plus différent de ce qu'ils connaissaient – c'est ce qui me rappelait la réalité de ces images, qui enflammait le plus mon désir, parce que c'était comme une promesse qu'il serait contenté. Et, bien que mon exaltation eût pour motif un désir de jouissances artistiques, les guides l'entretenaient encore plus que les livres d'esthétique et, plus que les guides, l'indicateur des chemins de fer. Ce qui m'émouvait, c'était de penser que cette Florence que je voyais proche mais inaccessible dans mon imagination, si le trajet qui la séparait de moi, en moi-même, n'était pas viable, je pourrais l'atteindre par un biais, par un détour, en prenant la « voie de terre ». Certes, quand je me répétais, donnant ainsi tant de valeur à ce que j'allais voir, que Venise était « l'école de Giorgione, la demeure du Titien, le plus complet musée de l'architecture domestique au moyen âge », je me sentais heureux. Je l'étais pourtant davantage quand, sorti pour une course, marchant vite à cause du temps qui, après quelques jours de printemps précoce était redevenu un temps d'hiver (comme celui que nous trouvions d'habitude à Combray, la Semaine Sainte) – voyant sur les boulevards les marronniers qui, plongés dans un air glacial et liquide comme de l'eau, n'en commençaient pas moins, invités exacts, déjà en tenue, et qui ne se sont pas laissé décourager, à arrondir et à ciseler, en leurs blocs congelés, l'irrésistible verdure dont la puissance abortive du froid contrariait mais ne parvenait pas à réfréner la progressive poussée – je pensais que déjà le Ponte-Vecchio était jonché à foison de jacinthes et d'anémones et que le soleil du printemps teignait déjà les flots du Grand Canal d'un si sombre azur et de si nobles émeraudes qu'en venant se briser aux pieds des peintures du Titien, ils pouvaient rivaliser de riche coloris avec elles. Je ne pus plus contenir ma joie quand mon père, tout en consultant le baromètre et en déplorant le froid, commença à chercher quels seraient les meilleurs trains, et quand je compris qu'en pénétrant après le déjeuner dans le laboratoire charbonneux, dans la chambre magique qui se chargeait d'opérer la transmutation tout autour d'elle, on pouvait s'éveiller le lendemain dans la cité de marbre et d'or « rehaussée de jaspe et pavée d'émeraudes ». Ainsi elle et la Cité des lys n'étaient pas seulement des tableaux fictifs qu'on mettait à volonté devant son imagination, mais existaient à une certaine distance de Paris qu'il fallait absolument franchir si l'on voulait les voir, à une certaine place déterminée de la terre, et à aucune autre, en un mot étaient bien réelles. Elles le devinrent encore plus pour moi, quand mon père en disant : « En somme, vous pourriez rester à Venise du 20 avril au 29 et arriver à Florence dès le matin de Pâques », les fit sortir toutes deux non plus seulement de l'Espace abstrait, mais de ce Temps imaginaire où nous situons non pas un seul voyage à la fois, mais d'autres, simultanés et sans trop d'émotion puisqu'ils ne sont que possibles – ce Temps qui se refabrique si bien qu'on peut encore le passer dans une ville après qu'on l'a passé dans une autre – et leur consacra de ces jours particuliers qui sont le certificat d'authenticité des objets auxquels on les emploie, car ces jours uniques, ils se consument par l'usage, ils ne reviennent pas, on ne peut plus les vivre ici quand on les a vécus là ; je sentis que c'était vers la semaine qui commençait le lundi où la blanchisseuse devait rapporter le gilet blanc que j'avais couvert d'encre, que se dirigeaient pour s'y absorber au sortir du temps idéal où elles n'existaient pas encore, les deux cités Reines dont j'allais avoir, par la plus émouvante des géométries, à inscrire les dômes et les tours dans le plan de ma propre vie.

Mais je n'étais encore qu'en chemin vers le dernier degré de l'allégresse ; je l'atteignis enfin (ayant seulement alors la révélation que sur les rues clapotantes, rougies du reflet des fresques de Giorgione, ce n'était pas, comme j'avais, malgré tant d'avertissements, continué à l'imaginer, les hommes « majestueux et terribles comme la mer, portant leur armure aux reflets de bronze sous les plis de leur manteau sanglant » qui se promèneraient dans Venise la semaine prochaine, la veille de Pâques, mais que ce pourrait être moi, le personnage minuscule que, dans une grande photographie de Saint-Marc qu'on m'avait prêtée, l'illustrateur avait représenté, en chapeau melon, devant les proches), quand j'entendis mon père me dire : « Il doit faire encore froid sur le Grand Canal, tu ferais bien de mettre à tout hasard dans ta malle ton pardessus d'hiver et ton gros veston. » À ces mots je m'élevai à une sorte d'extase ; ce que j'avais cru jusque-là impossible, je me sentis vraiment pénétrer entre ces « rochers d'améthyste pareils à un récif de la mer des Indes » ; par une gymnastique suprême et au-dessus de mes forces, me dévêtant comme d'une carapace sans objet de l'air de ma chambre, qui m'entourait, je le remplaçai par des parties égales d'air vénitien, cette atmosphère marine, indicible et particulière comme celle des rêves que mon imagination avait enfermée dans le nom de Venise, je sentis s'opérer en moi une miraculeuse désincarnation ; elle se doubla aussitôt de la vague envie de vomir qu'on éprouve quand on vient de prendre un gros mal de gorge, et on dut me mettre au lit avec une fièvre si tenace, que le docteur déclara qu'il fallait renoncer non seulement à me laisser partir maintenant à Florence et à Venise mais, même quand je serais entièrement rétabli, m'éviter, d'ici au moins un an, tout projet de voyage et toute cause d'agitation.

Et hélas, il défendit aussi d'une façon absolue qu'on me laissât aller au théâtre entendre la Berma ; l'artiste sublime, à laquelle Bergotte trouvait du génie, m'aurait, en me faisant connaître quelque chose qui était peut-être aussi important et aussi beau, consolé de n'avoir pas été à Florence et à Venise, de n'aller pas à Balbec. On devait se contenter de m'envoyer chaque jour aux Champs-Élysées, sous la surveillance d'une personne qui m'empêcherait de me fatiguer et qui fut Françoise, entrée à notre service après la mort de ma tante Léonie. Aller aux Champs-Élysées me fut insupportable. Si seulement Bergotte les eût décrits dans un de ses livres, sans doute j'aurais désiré de les connaître, comme toutes les choses dont on avait commencé par mettre le « double » dans mon imagination. Elle les réchauffait, les faisait vivre, leur donnait une personnalité, et je voulais les retrouver dans la réalité ; mais dans ce jardin public rien ne se rattachait à mes rêves.
