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
      "source": "narrator",
      "target": "Françoise",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« Pendant les quinze jours que dura la dernière maladie de ma tante, Françoise ne la quitta pas un instant… »; « Alors nous comprîmes que … c’était de la vénération et de l’amour. »",
      "explanation": "The narrator corrects an earlier misreading and elevates Françoise by revealing her steadfast devotion and love during the aunt's final illness."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Françoise",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.72,
      "evidence": "« un démon me poussait à souhaiter qu’elle fût en colère… »; « Je ne sais pas m’esprimer », je triomphais…; « j’adoptais … le point de vue mesquin »",
      "explanation": "The narrator admits to provoking Françoise and belittling her language and views on mourning; the passage frames this as petty, but it is still a local snub directed at her."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "She is locally raised in esteem as loving, devoted, and misjudged earlier by the household."
    },
    {
      "character": "Françoise",
      "dimension": "rhetorical_position",
      "delta": -1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.7,
      "explanation": "In the narrator’s recounting, she is put at a disadvantage in argument and mocked for her speech, despite the narrator’s self-critique."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-316-p-320"
}

### Candidate characters

[
  "le narrateur"
]

### Prior local context (optional)

Souvent le soleil se cachait derrière une nuée qui déformait son ovale et dont il jaunissait la bordure. L'éclat, mais non la clarté, était enlevé à la campagne où toute vie semblait suspendue, tandis que le petit village de Roussainville sculptait sur le ciel le relief de ses arêtes blanches avec une précision et un fini accablants. Un peu de vent faisait envoler un corbeau qui retombait dans le lointain, et, contre le ciel blanchissant, le lointain des bois paraissait plus bleu, comme peint dans ces camaïeux qui décorent les trumeaux des anciennes demeures.

### Passage

Mais d'autres fois se mettait à tomber la pluie dont nous avait menacés le capucin que l'opticien avait à sa devanture ; les gouttes d'eau, comme des oiseaux migrateurs qui prennent leur vol tous ensemble, descendaient à rangs pressés du ciel. Elles ne se séparent point, elles ne vont pas à l'aventure pendant la rapide traversée, mais chacune tenant sa place attire à elle celle qui la suit et le ciel en est plus obscurci qu'au départ des hirondelles. Nous nous réfugiions dans le bois. Quand leur voyage semblait fini, quelques-unes, plus débiles, plus lentes, arrivaient encore. Mais nous ressortions de notre abri, car les gouttes se plaisent aux feuillages, et la terre était déjà presque séchée que plus d'une s'attardait à jouer sur les nervures d'une feuille, et suspendue à la pointe, reposée, brillant au soleil, tout d'un coup se laissait glisser de toute la hauteur de la branche et nous tombait sur le nez.

Souvent aussi nous allions nous abriter, pêle-mêle avec les saints et les patriarches de pierre sous le porche de Saint-André-des-Champs. Que cette église était française ! Au-dessus de la porte, les saints, les rois-chevaliers une fleur de lys à la main, des scènes de noces et de funérailles, étaient représentés comme ils pouvaient l'être dans l'âme de Françoise. Le sculpteur avait aussi narré certaines anecdotes relatives à Aristote et à Virgile de la même façon que Françoise à la cuisine parlait volontiers de saint Louis comme si elle l'avait personnellement connu, et généralement pour faire honte par la comparaison à mes grands-parents moins « justes ». On sentait que les notions que l'artiste médiéval et la paysanne médiévale (survivant au XIXe siècle) avaient de l'histoire ancienne ou chrétienne, et qui se distinguaient par autant d'inexactitude que de bonhomie, ils les tenaient non des livres, mais d'une tradition à la fois antique et directe, ininterrompue, orale, déformée, méconnaissable et vivante. Une autre personnalité de Combray que je reconnaissais aussi, virtuelle et prophétisée, dans la sculpture gothique de Saint-André-des-Champs c'était le jeune Théodore, le garçon de chez Camus. Françoise sentait d'ailleurs si bien en lui un pays et un contemporain que, quand ma tante Léonie était trop malade pour que Françoise pût suffire à la retourner dans son lit, à la porter dans son fauteuil, plutôt que de laisser la fille de cuisine monter se faire « bien voir » de ma tante, elle appelait Théodore. Or ce garçon, qui passait et avec raison pour si mauvais sujet, était tellement rempli de l'âme qui avait décoré Saint-André-des-Champs et notamment des sentiments de respect que Françoise trouvait dus aux « pauvres malades », à « sa pauvre maîtresse », qu'il avait pour soulever la tête de ma tante sur son oreiller la mine naïve et zélée des petits anges des bas-reliefs, s'empressant, un cierge à la main, autour de la Vierge défaillante, comme si les visages de pierre sculptée, grisâtres et nus, ainsi que sont les bois en hiver, n'étaient qu'un ensommeillement, qu'une réserve, prête à refleurir dans la vie en innombrables visages populaires, révérends et futés comme celui de Théodore, enluminés de la rougeur d'une pomme mûre. Non plus appliquée à la pierre comme ces petits anges, mais détachée du porche, d'une stature plus qu'humaine, debout sur un socle comme sur un tabouret qui lui évitât de poser ses pieds sur le sol humide, une sainte avait les joues pleines, le sein ferme et qui gonflait la draperie comme une grappe mûre dans un sac de crin, le front étroit, le nez court et mutin, les prunelles enfoncées, l'air valide, insensible et courageux des paysannes de la contrée. Cette ressemblance, qui insinuait dans la statue une douceur que je n'y avais pas cherchée, était souvent certifiée par quelque fille des champs, venue comme nous se mettre à couvert, et dont la présence, pareille à celle de ces feuillages pariétaires qui ont poussé à côté des feuillages sculptés, semblait destinée à permettre, par une confrontation avec la nature, de juger de la vérité de l'oeuvre d'art. Devant nous, dans le lointain, terre promise ou maudite, Roussainville, dans les murs duquel je n'ai jamais pénétré, Roussainville, tantôt, quand la pluie avait déjà cessé pour nous, continuait à être châtié comme un village de la Bible par toutes les lances de l'orage qui flagellaient obliquement les demeures de ses habitants, ou bien était déjà pardonné par Dieu le Père qui faisait descendre vers lui, inégalement longues, comme les rayons d'un ostensoir d'autel, les tiges d'or effrangées de son soleil reparu.

Quelquefois le temps était tout à fait gâté, il fallait rentrer et rester enfermé dans la maison. Çà et là au loin dans la campagne que l'obscurité et l'humidité faisaient ressembler à la mer, des maisons isolées, accrochées au flanc d'une colline plongée dans la nuit et dans l'eau, brillaient comme des petits bateaux qui ont replié leurs voiles et sont immobiles au large pour toute la nuit. Mais qu'importait la pluie, qu'importait l'orage ! L'été, le mauvais temps n'est qu'une humeur passagère, superficielle, du beau temps sous-jacent et fixe, bien différent du beau temps instable et fluide de l'hiver et qui, au contraire, installé sur la terre où il s'est solidifié en denses feuillages sur lesquels la pluie peut s'égoutter sans compromettre la résistance de leur permanente joie, a hissé pour toute la saison, jusque dans les rues du village, aux murs des maisons et des jardins, ses pavillons de soie violette ou blanche. Assis dans le petit salon, où j'attendais l'heure du dîner en lisant, j'entendais l'eau dégoutter de nos marronniers, mais je savais que l'averse ne faisait que vernir leurs feuilles et qu'ils promettaient de demeurer là, comme des gages de l'été, toute la nuit pluvieuse, à assurer la continuité du beau temps ; qu'il avait beau pleuvoir, demain, au-dessus de la barrière blanche de Tansonville, onduleraient, aussi nombreuses, de petites feuilles en forme de coeur ; et c'est sans tristesse que j'apercevais le peuplier de la rue des Perchamps adresser à l'orage des supplications et des salutations désespérées ; c'est sans tristesse que j'entendais au fond du jardin les derniers roulements du tonnerre roucouler dans les lilas.

Si le temps était mauvais dès le matin, mes parents renonçaient à la promenade et je ne sortais pas. Mais je pris ensuite l'habitude d'aller, ces jours-là, marcher seul du côté de Méséglise-la-Vineuse, dans l'automne où nous dûmes venir à Combray pour la succession de ma tante Léonie, car elle était enfin morte, faisant triompher à la fois ceux qui prétendaient que son régime affaiblissant finirait par la tuer, et non moins les autres qui avaient toujours soutenu qu'elle souffrait d'une maladie non pas imaginaire mais organique, à l'évidence de laquelle les sceptiques seraient bien obligés de se rendre quand elle y aurait succombé ; et ne causant par sa mort de grande douleur qu'à un seul être, mais à celui-là, sauvage. Pendant les quinze jours que dura la dernière maladie de ma tante, Françoise ne la quitta pas un instant, ne se déshabilla pas, ne laissa personne lui donner aucun soin, et ne quitta son corps que quand il fut enterré. Alors nous comprîmes que cette sorte de crainte où Françoise avait vécu des mauvaises paroles, des soupçons, des colères de ma tante avait développé chez elle un sentiment que nous avions pris pour de la haine et qui était de la vénération et de l'amour. Sa véritable maîtresse, aux décisions impossibles à prévoir, aux ruses difficiles à déjouer, au bon coeur facile à fléchir, sa souveraine, son mystérieux et tout-puissant monarque n'était plus. À côté d'elle nous comptions pour bien peu de chose. Il était loin le temps où, quand nous avions commencé à venir passer nos vacances à Combray, nous possédions autant de prestige que ma tante aux yeux de Françoise. Cet automne-là, tout occupés des formalités à remplir, des entretiens avec les notaires et avec les fermiers, mes parents, n'ayant guère de loisir pour faire des sorties que le temps d'ailleurs contrariait, prirent l'habitude de me laisser aller me promener sans eux du côté de Méséglise, enveloppé dans un grand plaid qui me protégeait contre la pluie et que je jetais d'autant plus volontiers sur mes épaules que je sentais que ses rayures écossaises scandalisaient Françoise, dans l'esprit de qui on n'aurait pu faire entrer l'idée que la couleur des vêtements n'a rien à faire avec le deuil et à qui d'ailleurs le chagrin que nous avions de la mort de ma tante plaisait peu, parce que nous n'avions pas donné de grand repas funèbre, que nous ne prenions pas un son de voix spécial pour parler d'elle, que même parfois je chantonnais. Je suis sûr que dans un livre – et en cela j'étais bien moi-même comme Françoise – cette conception du deuil d'après la Chanson de Roland et le portail de Saint-André-des-Champs m'eût été sympathique. Mais dès que Françoise était auprès de moi, un démon me poussait à souhaiter qu'elle fût en colère, je saisissais le moindre prétexte pour lui dire que je regrettais ma tante parce que c'était une bonne femme, malgré ses ridicules, mais nullement parce que c'était ma tante, qu'elle eût pu être ma tante et me sembler odieuse, et sa mort ne me faire aucune peine, propos qui m'eussent semblé ineptes dans un livre.

Si alors Françoise, remplie comme un poète d'un flot de pensées confuses sur le chagrin, sur les souvenirs de famille, s'excusait de ne pas savoir répondre à mes théories et disait : « Je ne sais pas m'esprimer », je triomphais de cet aveu avec un bon sens ironique et brutal digne du docteur Percepied ; et si elle ajoutait : « Elle était tout de même de la parentèse, il reste toujours le respect qu'on doit à la parentèse », je haussais les épaules et je me disais : « Je suis bien bon de discuter avec une illettrée qui fait des cuirs pareils », adoptant ainsi pour juger Françoise le point de vue mesquin d'hommes dont ceux qui les méprisent le plus dans l'impartialité de la méditation, sont fort capables de tenir le rôle, quand ils jouent une des scènes vulgaires de la vie.
