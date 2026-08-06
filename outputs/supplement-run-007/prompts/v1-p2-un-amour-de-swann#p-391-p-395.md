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
    },
    {
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Odette",
      "target": "Swann",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.89,
      "evidence": "« elle lui écrivit que les M. Verdurin et leurs amis avaient manifesté le désir d’assister à ces représentations de Wagner, et que, s’il voulait bien lui envoyer cet argent, elle aurait enfin ... le plaisir de les inviter à son tour. De lui, elle ne disait pas un mot, il était sous-entendu que leur présence excluait la sienne. »",
      "explanation": "Through her letter, Odette solicits money from Swann to invite the Verdurins and Forcheville while implicitly excluding him from the project, which constitutes a concrete social slight."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "social_status",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "She positions herself as a hostess towards the Verdurins and Forcheville with Swann's funds, consolidating her place in this environment."
    },
    {
      "character": "Swann",
      "dimension": "inclusion_exclusion",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "He is explicitly excluded from a project he had contemplated, while being solicited as a payer, which places him outside the valued circle."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-391-p-395"
}

### Candidate characters

[
  "M. Verdurin",
  "Saniette",
  "comte de Forcheville"
]

### Prior local context (optional)

Alors à ces moments-là, pendant qu'elle leur faisait de l'orangeade, tout d'un coup, comme quand un réflecteur mal réglé d'abord promène autour d'un objet, sur la muraille, de grandes ombres fantastiques, qui viennent ensuite se replier et s'anéantir en lui, toutes les idées terribles et mouvantes qu'il se faisait d'Odette s'évanouissaient, rejoignaient le corps charmant que Swann avait devant lui. Il avait le brusque soupçon que cette heure passée chez Odette, sous la lampe, n'était peut-être pas une heure factice, à son usage à lui (destinée à masquer cette chose effrayante et délicieuse à laquelle il pensait sans cesse sans pouvoir bien se la représenter, une heure de la vraie vie d'Odette, de la vie d'Odette quand lui n'était pas là), avec des accessoires de théâtre et des fruits de carton, mais était peut-être une heure pour de bon de la vie d'Odette ; que s'il n'avait pas été là, elle eût avancé à comte de Forcheville le même fauteuil et lui eût versé non un breuvage inconnu, mais précisément cette orangeade ; que le monde habité par Odette n'était pas cet autre monde effroyable et surnaturel où il passait son temps à la situer et qui n'existait peut-être que dans son imagination, mais l'univers réel, ne dégageant aucune tristesse spéciale, comprenant cette table où il allait pouvoir écrire et cette boisson à laquelle il lui serait permis de goûter ; tous ces objets qu'il contemplait avec autant de curiosité et d'admiration que de gratitude, car si en absorbant ses rêves ils l'en avaient délivré, eux en revanche, s'en étaient enrichis, ils lui en montraient la réalisation palpable, et ils intéressaient son esprit, ils prenaient du relief devant ses regards, en même temps qu'ils tranquillisaient son coeur. Ah ! si le destin avait permis qu'il pût n'avoir qu'une seule demeure avec Odette et que chez elle il fût chez lui, si en demandant au domestique ce qu'il y avait à déjeuner, c'eût été le menu d'Odette qu'il avait appris en réponse, si quand Odette voulait aller le matin se promener avenue du Bois-de-Boulogne, son devoir de bon mari l'avait obligé, n'eût-il pas envie de sortir, à l'accompagner, portant son manteau quand elle avait trop chaud, et le soir après le dîner si elle avait envie de rester chez elle en déshabillé, s'il avait été forcé de rester là près d'elle, à faire ce qu'elle voudrait ; alors combien tous les riens de la vie de Swann qui lui semblaient si tristes, au contraire parce qu'ils auraient en même temps fait partie de la vie d'Odette auraient pris, même les plus familiers – et comme cette lampe, cette orangeade, ce fauteuil qui contenaient tant de rêve, qui matérialisaient tant de désir – une sorte de douceur surabondante et de densité mystérieuse.

### Passage

Pourtant il se doutait bien que ce qu'il regrettait ainsi, c'était un calme, une paix qui n'auraient pas été pour son amour une atmosphère favorable. Quand Odette cesserait d'être pour lui une créature toujours absente, regrettée, imaginaire ; quand le sentiment qu'il aurait pour elle ne serait plus ce même trouble mystérieux que lui causait la phrase de la sonate, mais de l'affection, de la reconnaissance ; quand s'établiraient entre eux des rapports normaux qui mettraient fin à sa folie et à sa tristesse, alors sans doute les actes de la vie d'Odette lui paraîtraient peu intéressants en eux-mêmes – comme il avait déjà eu plusieurs fois le soupçon qu'ils étaient, par exemple le jour où il avait lu à travers l'enveloppe la lettre adressée à Forcheville. Considérant son mal avec autant de sagacité que s'il se l'était inoculé pour en faire l'étude, il se disait que, quand il serait guéri, ce que pourrait faire Odette lui serait indifférent. Mais du sein de son état morbide, à vrai dire, il redoutait à l'égal de la mort une telle guérison, qui eût été en effet la mort de tout ce qu'il était actuellement.

Après ces tranquilles soirées, les soupçons de Swann étaient calmés ; il bénissait Odette et le lendemain, dès le matin, il faisait envoyer chez elle les plus beaux bijoux, parce que ces bontés de la veille avaient excité ou sa gratitude, ou le désir de les voir se renouveler, ou un paroxysme d'amour qui avait besoin de se dépenser.

Mais, à d'autres moments, sa douleur le reprenait, il s'imaginait qu'Odette était la maîtresse de Forcheville et que quand tous deux l'avaient vu, du fond du landau des Verdurin, au Bois, la veille de la fête de Chatou, où il n'avait pas été invité, la prier vainement, avec cet air de désespoir qu'avait remarqué jusqu'à son cocher, de revenir avec lui, puis s'en retourner de son côté, seul et vaincu, elle avait dû avoir pour le désigner à Forcheville et lui dire : « Hein ! ce qu'il rage ! » les mêmes regards brillants, malicieux, abaissés et sournois, que le jour où celui-ci avait chassé Saniette de chez les Verdurin.

Alors Swann la détestait. « Mais aussi, je suis trop bête, se disait-il, je paie avec mon argent le plaisir des autres. Elle fera tout de même bien de faire attention et de ne pas trop tirer sur la corde, car je pourrais bien ne plus rien donner du tout. En tous cas, renonçons provisoirement aux gentillesses supplémentaires ! Penser que pas plus tard qu'hier, comme elle disait avoir envie d'assister à la saison de Bayreuth, j'ai eu la bêtise de lui proposer de louer un des jolis châteaux du roi de Bavière pour nous deux dans les environs. Et d'ailleurs elle n'a pas paru plus ravie que cela, elle n'a encore dit ni oui ni non ; espérons qu'elle refusera, grand Dieu ! Entendre du Wagner pendant quinze jours avec elle qui s'en soucie comme un poisson d'une pomme, ce serait gai ! » Et sa haine, tout comme son amour, ayant besoin de se manifester et d'agir, il se plaisait à pousser de plus en plus loin ses imaginations mauvaises, parce que, grâce aux perfidies qu'il prêtait à Odette, il la détestait davantage et pourrait si – ce qu'il cherchait à se figurer – elles se trouvaient être vraies, avoir une occasion de la punir et d'assouvir sur elle sa rage grandissante. Il alla ainsi jusqu'à supposer qu'il allait recevoir une lettre d'elle où elle lui demanderait de l'argent pour louer ce château près de Bayreuth, mais en le prévenant qu'il n'y pourrait pas venir, parce qu'elle avait promis à Forcheville et aux Verdurin de les inviter. Ah ! comme il eût aimé qu'elle pût avoir cette audace. Quelle joie il aurait à refuser, à rédiger la réponse vengeresse dont il se complaisait à choisir, à énoncer tout haut les termes, comme s'il avait reçu la lettre en réalité !

Or, c'est ce qui arriva le lendemain même. Elle lui écrivit que les Verdurin et leurs amis avaient manifesté le désir d'assister à ces représentations de Wagner, et que, s'il voulait bien lui envoyer cet argent, elle aurait enfin, après avoir été si souvent reçue chez eux, le plaisir de les inviter à son tour. De lui, elle ne disait pas un mot, il était sous-entendu que leur présence excluait la sienne.
