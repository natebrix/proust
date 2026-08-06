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
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Swann",
      "type": "narrated_diminishment",
      "polarity": "mixed",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "« à l'affaiblissement de son amour correspondait simultanément un affaiblissement du désir de rester amoureux »; « il s'aperçut qu'il n'en ressentait aucune douleur, que l'amour était loin maintenant »; regret de n'avoir pu « faire ses adieux » à l'Odette qui lui inspirait amour et jalousie.",
      "explanation": "The narrator insists on the gradual extinction of Swann's love and jealousy, confirmed when the proof of Forcheville as a lover no longer causes him pain. The passage remains ambivalent, mixing relief, numbness, and regret."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.86,
      "explanation": "Swann is locally diminished emotionally: his attachment is extinguished, he is numb and regrets not having acknowledged his former feeling, despite some quickly dissipated dreamlike resurges."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-586-p-590"
}

### Candidate characters

[
  "Legrandin",
  "Mme Verdurin",
  "Mme de Cambremer",
  "Napoléon III",
  "Odette",
  "baron de Charlus",
  "comte de Forcheville",
  "docteur Cottard",
  "général de Froberville",
  "le grand-père du narrateur",
  "le peintre",
  "marquise de Saint-Euverte",
  "le narrateur"
]

### Prior local context (optional)

Pour faire concurrence aux sentiments maladifs que Swann avait pour Odette, Mme Cottard, meilleur thérapeute que n'eût été son mari, avait greffé à côté d'eux d'autres sentiments, normaux ceux-là, de gratitude, d'amitié, des sentiments qui dans l'esprit de Swann rendraient Odette plus humaine (plus semblable aux autres femmes, parce que d'autres femmes aussi pouvaient les lui inspirer), hâteraient sa transformation définitive en cette Odette aimée d'affection paisible, qui l'avait ramené un soir après une fête chez le peintre boire un verre d'orangeade avec comte de Forcheville et près de qui Swann avait entrevu qu'il pourrait vivre heureux.

### Passage

Jadis ayant souvent pensé avec terreur qu'un jour il cesserait d'être épris d'Odette, il s'était promis d'être vigilant, et dès qu'il sentirait que son amour commencerait à le quitter, de s'accrocher à lui, de le retenir. Mais voici qu'à l'affaiblissement de son amour correspondait simultanément un affaiblissement du désir de rester amoureux. Car on ne peut pas changer, c'est-à-dire devenir une autre personne, tout en continuant à obéir aux sentiments de celle qu'on n'est plus. Parfois le nom, aperçu dans un journal, d'un des hommes qu'il supposait avoir pu être les amants d'Odette, lui redonnait de la jalousie. Mais elle était bien légère et comme elle lui prouvait qu'il n'était pas encore complètement sorti de ce temps où il avait tant souffert – mais aussi où il avait connu une manière de sentir si voluptueuse – et que les hasards de la route lui permettraient peut-être d'en apercevoir encore furtivement et de loin les beautés, cette jalousie lui procurait plutôt une excitation agréable comme au morne Parisien qui quitte Venise pour retrouver la France, un dernier moustique prouve que l'Italie et l'été ne sont pas encore bien loin. Mais le plus souvent le temps si particulier de sa vie d'où il sortait, quand il faisait effort sinon pour y rester, du moins pour en avoir une vision claire pendant qu'il le pouvait encore, il s'apercevait qu'il ne le pouvait déjà plus ; il aurait voulu apercevoir comme un paysage qui allait disparaître cet amour qu'il venait de quitter ; mais il est si difficile d'être double et de se donner le spectacle véridique d'un sentiment qu'on a cessé de posséder, que bientôt l'obscurité se faisant dans son cerveau, il ne voyait plus rien, renonçait à regarder, retirait son lorgnon, en essuyait les verres ; et il se disait qu'il valait mieux se reposer un peu, qu'il serait encore temps tout à l'heure, et se rencognait, avec l'incuriosité, dans l'engourdissement du voyageur ensommeillé qui rabat son chapeau sur ses yeux pour dormir dans le wagon qu'il sent l'entraîner de plus en plus vite, loin du pays où il a si longtemps vécu et qu'il s'était promis de ne pas laisser fuir sans lui donner un dernier adieu. Même, comme ce voyageur s'il se réveille seulement en France, quand Swann ramassa par hasard près de lui la preuve que Forcheville avait été l'amant d'Odette, il s'aperçut qu'il n'en ressentait aucune douleur, que l'amour était loin maintenant et regretta de n'avoir pas été averti du moment où il le quittait pour toujours. Et de même qu'avant d'embrasser Odette pour la première fois il avait cherché à imprimer dans sa mémoire le visage qu'elle avait eu si longtemps pour lui et qu'allait transformer le souvenir de ce baiser, de même il eût voulu, en pensée au moins, avoir pu faire ses adieux, pendant qu'elle existait encore, à cette Odette lui inspirant de l'amour, de la jalousie, à cette Odette lui causant des souffrances et que maintenant il ne reverrait jamais.

Il se trompait. Il devait la revoir une fois encore, quelques semaines plus tard. Ce fut en dormant, dans le crépuscule d'un rêve. Il se promenait avec Mme Verdurin, le docteur Cottard, un jeune homme en fez qu'il ne pouvait identifier, le peintre, Odette, Napoléon III et mon grand-père, sur un chemin qui suivait la mer et la surplombait à pic tantôt de très haut, tantôt de quelques mètres seulement, de sorte qu'on montait et redescendait constamment ; ceux des promeneurs qui redescendaient déjà n'étaient plus visibles à ceux qui montaient encore, le peu de jour qui restât faiblissait et il semblait alors qu'une nuit noire allait s'étendre immédiatement. Par moment les vagues sautaient jusqu'au bord, et Swann, sentait sur sa joue des éclaboussures glacées. Odette lui disait de les essuyer, il ne pouvait pas et en était confus vis-à-vis d'elle, ainsi que d'être en chemise de nuit. Il espérait qu'à cause de l'obscurité on ne s'en rendait pas compte, mais cependant Mme Verdurin le fixa d'un regard étonné durant un long moment pendant lequel il vit sa figure se déformer, son nez s'allonger et qu'elle avait de grandes moustaches. Il se détourna pour regarder Odette, ses joues étaient pâles, avec des petits points rouges, ses traits tirés, cernés, mais elle le regardait avec des yeux pleins de tendresse prêts à se détacher comme des larmes pour tomber sur lui, et il se sentait l'aimer tellement qu'il aurait voulu l'emmener tout de suite. Tout d'un coup Odette tourna son poignet, regarda une petite montre et dit : « Il faut que je m'en aille », elle prenait congé de tout le monde, de la même façon, sans prendre à part Swann, sans lui dire où elle le reverrait le soir ou un autre jour. Il n'osa pas le lui demander, il aurait voulu la suivre et était obligé, sans se retourner vers elle, de répondre en souriant à une question de Mme Verdurin, mais son coeur battait horriblement, il éprouvait de la haine pour Odette, il aurait voulu crever ses yeux qu'il aimait tant tout à l'heure, écraser ses joues sans fraîcheur. Il continuait à monter avec Mme Verdurin, c'est-à-dire à s'éloigner à chaque pas d'Odette, qui descendait en sens inverse. Au bout d'une seconde il y eut beaucoup d'heures qu'elle était partie. Le peintre fit remarquer à Swann que Napoléon III s'était éclipsé un instant après elle. « C'était certainement entendu entre eux, ajouta-t-il, ils ont dû se rejoindre en bas de la côte, mais n'ont pas voulu dire adieu ensemble à cause des convenances. Elle est sa maîtresse. » Le jeune homme inconnu se mit à pleurer. Swann essaya de le consoler. « Après tout elle a raison, lui dit-il en lui essuyant les yeux et en lui ôtant son fez pour qu'il fût plus à son aise. Je le lui ai conseillé dix fois. Pourquoi en être triste ? C'était bien l'homme qui pouvait la comprendre. » Ainsi Swann se parlait-il à lui-même, car le jeune homme qu'il n'avait pu identifier d'abord était aussi lui ; comme certains romanciers, il avait distribué sa personnalité à deux personnages, celui qui faisait le rêve, et un qu'il voyait devant lui coiffé d'un fez.

Quant à Napoléon III, c'est à Forcheville que quelque vague association d'idées, puis une certaine modification dans la physionomie habituelle du baron, enfin le grand cordon de la Légion d'honneur en sautoir, lui avaient fait donner ce nom ; mais en réalité, et pour tout ce que le personnage présent dans le rêve lui représentait et lui rappelait, c'était bien Forcheville. Car d'images incomplètes et changeantes Swann endormi tirait des déductions fausses, ayant d'ailleurs momentanément un tel pouvoir créateur qu'il se reproduisait par simple division comme certains organismes inférieurs ; avec la chaleur sentie de sa propre paume il modelait le creux d'une main étrangère qu'il croyait serrer, et de sentiments et d'impressions dont il n'avait pas conscience encore, faisait naître comme des péripéties qui, par leur enchaînement logique, amèneraient à point nommé dans le sommeil de Swann le personnage nécessaire pour recevoir son amour ou provoquer son réveil. Une nuit noire se fit tout d'un coup, un tocsin sonna, des habitants passèrent en courant, se sauvant des maisons en flammes ; Swann entendait le bruit des vagues qui sautaient et son coeur qui, avec la même violence, battait d'anxiété dans sa poitrine. Tout d'un coup ses palpitations de coeur redoublèrent de vitesse, il éprouva une souffrance, une nausée inexplicables ; un paysan couvert de brûlures lui jetait en passant : « Venez demander à Charlus où Odette est allée finir la soirée avec son camarade, il a été avec elle autrefois et elle lui dit tout. C'est eux qui ont mis le feu. » C'était son valet de chambre qui venait l'éveiller et lui disait :

– Monsieur, il est huit heures et le coiffeur est là, je lui ai dit de repasser dans une heure.

Mais ces paroles, en pénétrant dans les ondes du sommeil où Swann était plongé, n'étaient arrivées jusqu'à sa conscience qu'en subissant cette déviation qui fait qu'au fond de l'eau un rayon paraît un soleil, de même qu'un moment auparavant le bruit de la sonnette prenant au fond de ces abîmes une sonorité de tocsin avait enfanté l'épisode de l'incendie. Cependant le décor qu'il avait sous les yeux vola en poussière, il ouvrit les yeux, entendit une dernière fois le bruit d'une des vagues de la mer qui s'éloignait. Il toucha sa joue. Elle était sèche. Et pourtant il se rappelait la sensation de l'eau froide et le goût du sel. Il se leva, s'habilla. Il avait fait venir le coiffeur de bonne heure parce qu'il avait écrit la veille à mon grand-père qu'il irait dans l'après-midi à Combray, ayant appris que Mme de Cambremer – Mlle Legrandin – devait y passer quelques jours. Associant dans son souvenir au charme de ce jeune visage celui d'une campagne où il n'était pas allé depuis si longtemps, ils lui offraient ensemble un attrait qui l'avait décidé à quitter enfin Paris pour quelques jours. Comme les différents hasards qui nous mettent en présence de certaines personnes ne coïncident pas avec le temps où nous les aimons, mais, le dépassant, peuvent se produire avant qu'il commence et se répéter après qu'il a fini, les premières apparitions que fait dans notre vie un être destiné plus tard à nous plaire, prennent rétrospectivement à nos yeux une valeur d'avertissement, de présage. C'est de cette façon que Swann s'était souvent reporté à l'image d'Odette rencontrée au théâtre, ce premier soir où il ne songeait pas à la revoir jamais – et qu'il se rappelait maintenant la soirée de Mme de Saint-Euverte où il avait présenté le général de Froberville à Mme de Cambremer. Les intérêts de notre vie sont si multiples qu'il n'est pas rare que dans une même circonstance les jalons d'un bonheur qui n'existe pas encore soient posés à côté de l'aggravation d'un chagrin dont nous souffrons. Et sans doute cela aurait pu arriver à Swann ailleurs que chez Mme de Saint-Euverte. Qui sait même, dans le cas où, ce soir-là, il se fût trouvé ailleurs, si d'autres bonheurs, d'autres chagrins ne lui seraient pas arrivés, et qui ensuite lui eussent paru avoir été inévitables ? Mais ce qui lui semblait l'avoir été, c'était ce qui avait eu lieu, et il n'était pas loin de voir quelque chose de providentiel dans ce fait qu'il se fût décidé à aller à la soirée de Mme de Saint-Euverte, parce que son esprit désireux d'admirer la richesse d'invention de la vie et incapable de se poser longtemps une question difficile, comme de savoir ce qui eût été le plus à souhaiter, considérait dans les souffrances qu'il avait éprouvées ce soir-là et les plaisirs encore insoupçonnés qui germaient déjà – et entre lesquels la balance était trop difficile à établir – une sorte d'enchaînement nécessaire.
