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
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Gilberte",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.78,
      "evidence": "« une femme par toute nouvelle souffrance qu'elle nous inflige ... augmente son pouvoir sur nous »; « Ce n'était pas mon cas à l'égard de Gilberte. »",
      "explanation": "The narrator reflects that the pain Gilberte has caused heightens her power over him while reducing his ability to impose conditions, locally elevating her emotional leverage."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Gilberte",
      "type": "blame",
      "polarity": "negative",
      "narrative_stance": "ironized",
      "confidence": 0.65,
      "evidence": "« un de mes amis ... agissait envers moi avec la plus grande fausseté »; « La mienne m'apprit que ... la personne dont la fausseté récente me faisait encore mal était Gilberte. »; « je n'en reçus qu'une seule [lettre hostile], de Gilberte »",
      "explanation": "Through a dream the narrator imputes recent falseness and hostility to Gilberte. The passage presents this as his subjective inference, tinged with interpretive irony."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": -1,
      "based_on_events": [
        "E2"
      ],
      "confidence": 0.65,
      "explanation": "She is locally discredited by the narrator's accusation of recent falseness and a hostile letter, though this emerges via a dream-based, possibly ironic reading."
    },
    {
      "character": "Gilberte",
      "dimension": "emotional_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.78,
      "explanation": "She gains immediate emotional leverage over the narrator because his suffering increases his dependence on her."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-321-p-325"
}

### Candidate characters

[
  "Albertine",
  "Odette",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Ce qui m'aida à patienter tout l'espace d'une journée fut un projet que je fis. Du moment que tout était oublié, que j'étais réconcilié avec Gilberte, je ne voulais plus la voir qu'en amoureux. Tous les jours elle recevrait de moi les plus belles fleurs qui fussent. Et si Odette, bien qu'elle n'eût pas le droit d'être une mère trop sévère, ne me permettait pas des envois de fleurs quotidiens, je trouverais des cadeaux plus précieux et moins fréquents. Mes parents ne me donnaient pas assez d'argent pour acheter des choses chères. Je songeai à une grande potiche de vieux Chine qui me venait de ma tante Léonie et dont la mère du narrateur prédisait chaque jour que Françoise allait venir en lui disant : « A s'est décollée » et qu'il n'en resterait rien. Dans ces conditions n'était-il pas plus sage de la vendre, de la vendre pour pouvoir faire tout le plaisir que je voudrais à Gilberte. Il me semblait que je pourrais bien en tirer mille francs. Je la fis envelopper, l'habitude m'avait empêché de jamais la voir ; m'en séparer eut au moins un avantage qui fut de me faire faire sa connaissance. Je l'emportai avec moi avant d'aller chez les Swann, et en donnant leur adresse au cocher, je lui dis de prendre par les Champs-Élysées, au coin desquels était le magasin d'un grand marchand de chinoiseries que connaissait le père du narrateur. À ma grande surprise, il m'offrit séance tenante de la potiche non pas mille, mais dix mille francs. Je pris ces billets avec ravissement ; pendant toute une année, je pourrais combler chaque jour Gilberte de roses et de lilas. Quand je fus remonté dans la voiture en quittant le marchand, le cocher, tout naturellement, comme les Swann demeuraient près du Bois, se trouva, au lieu du chemin habituel, descendre l'avenue des Champs-Élysées. Il avait déjà dépassé le coin de la rue de Berri, quand, dans le crépuscule, je crus reconnaître, très près de la maison des Swann mais allant dans la direction inverse et s'en éloignant, Gilberte qui marchait lentement, quoique d'un pas délibéré, à côté d'un jeune homme avec qui elle causait et duquel je ne pus distinguer le visage. Je me soulevai dans la voiture, voulant faire arrêter, puis j'hésitai. Les deux promeneurs étaient déjà un peu loin et les deux lignes douces et parallèles que traçait leur lente promenade allaient s'estompant dans l'ombre élyséenne. Bientôt j'arrivai devant la maison de Gilberte. Je fus reçu par Odette : « Oh ! elle va être désolée, me dit-elle, je ne sais pas comment elle n'est pas là. Elle a eu très chaud tantôt à un cours, elle m'a dit qu'elle voulait aller prendre un peu l'air avec une de ses amies. – Je crois que je l'ai aperçue avenue des Champs-Élysées. – Je ne pense pas que ce fût elle. En tous cas ne le dites pas à son père, il n'aime pas qu'elle sorte à ces heures-là. Good evening. » Je partis, dis au cocher de reprendre le même chemin, mais ne retrouvai pas les deux promeneurs. Où avaient-ils été ? Que se disaient-ils dans le soir, de cet air confidentiel ?

### Passage

Je rentrai, tenant avec désespoir les dix mille francs inespérés qui avaient dû me permettre de faire tant de petits plaisirs à cette Gilberte que, maintenant, j'étais décidé à ne plus revoir. Sans doute, cet arrêt chez le marchand de chinoiseries m'avait réjoui en me faisant espérer que je ne verrais plus jamais mon amie que contente de moi et reconnaissante. Mais si je n'avais pas fait cet arrêt, si la voiture n'avait pas pris par l'avenue des Champs-Élysées, je n'eusse pas rencontré Gilberte et ce jeune homme. Ainsi un même fait porte des rameaux opposites et le malheur qu'il engendre annule le bonheur qu'il avait causé. Il m'était arrivé le contraire de ce qui se produit si fréquemment. On désire une joie, et le moyen matériel de l'atteindre fait défaut. « Il est triste, a dit La Bruyère, d'aimer sans une grande fortune. » Il ne reste plus qu'à essayer d'anéantir peu à peu le désir de cette joie. Pour moi, au contraire, le moyen matériel avait été obtenu, mais, au même moment, sinon par un effet logique, du moins par une conséquence fortuite de cette réussite première, la joie avait été dérobée. Il semble, d'ailleurs, qu'elle doive nous l'être toujours. D'ordinaire, il est vrai, pas dans la même soirée où nous avons acquis ce qui la rend possible. Le plus souvent nous continuons de nous évertuer et d'espérer quelque temps. Mais le bonheur ne peut jamais avoir lieu. Si les circonstances arrivent à être surmontées, la nature transporte la lutte du dehors au dedans et fait peu à peu changer assez notre coeur pour qu'il désire autre chose que ce qu'il va posséder. Et si la péripétie a été si rapide que notre coeur n'a pas eu le temps de changer, la nature ne désespère pas pour cela de nous vaincre, d'une manière plus tardive il est vrai, plus subtile, mais aussi efficace. C'est alors à la dernière seconde que la possession du bonheur nous est enlevée, ou plutôt c'est cette possession même que par une ruse diabolique la nature charge de détruire le bonheur. Ayant échoué dans tout ce qui était du domaine des faits et de la vie, c'est une impossibilité dernière, l'impossibilité psychologique du bonheur que la nature crée. Le phénomène du bonheur ne se produit pas ou donne lieu aux réactions les plus amères.

Je serrai les dix mille francs. Mais ils ne me servaient plus à rien. Je les dépensai du reste encore plus vite que si j'eusse envoyé tous les jours des fleurs à Gilberte, car quand le soir venait, j'étais si malheureux que je ne pouvais rester chez moi et allais pleurer dans les bras de femmes que je n'aimais pas. Quant à chercher à faire un plaisir quelconque à Gilberte, je ne le souhaitais plus ; maintenant retourner dans la maison de Gilberte n'eût pu que me faire souffrir. Même revoir Gilberte qui m'eût été si délicieux la veille ne m'eût plus suffi. Car j'aurais été inquiet tout le temps où je n'aurais pas été près d'elle. C'est ce qui fait qu'une femme par toute nouvelle souffrance qu'elle nous inflige, souvent sans le savoir, augmente son pouvoir sur nous, mais aussi nos exigences envers elle. Par ce mal qu'elle nous a fait, la femme nous cerne de plus en plus, redouble nos chaînes, mais aussi celles dont il nous aurait jusque-là semblé suffisant de la garrotter pour que nous nous sentions tranquilles. La veille encore, si je n'avais pas cru ennuyer Gilberte, je me serais contenté de réclamer de rares entrevues, lesquelles maintenant ne m'eussent plus contenté et que j'eusse remplacées par bien d'autres conditions. Car en amour, au contraire de ce qui se passe après les combats, on les fait plus dures, on ne cesse de les aggraver, plus on est vaincu, si toutefois on est en situation de les imposer. Ce n'était pas mon cas à l'égard de Gilberte. Aussi je préférai d'abord ne pas retourner chez sa mère. Je continuais bien à me dire que Gilberte ne m'aimait pas, que je le savais depuis assez longtemps, que je pouvais la revoir si je voulais, et, si je ne le voulais pas, l'oublier à la longue. Mais ces idées, comme un remède qui n'agit pas contre certaines affections, étaient sans aucune espèce de pouvoir efficace contre ces deux lignes parallèles que je revoyais de temps à autre, de Gilberte et du jeune homme s'enfonçant à petits pas dans l'avenue des Champs-Élysées. C'était un mal nouveau, qui lui aussi finirait par s'user, c'était une image qui un jour se présenterait à mon esprit entièrement décantée de tout ce qu'elle contenait de nocif, comme ces poisons mortels qu'on manie sans danger, comme un peu de dynamite à quoi on peut allumer sa cigarette sans crainte d'explosion. En attendant, il y avait en moi une autre force qui luttait de toute sa puissance contre cette force malsaine qui me représentait sans changement la promenade de Gilberte dans le crépuscule et qui, pour briser les assauts renouvelés de ma mémoire, travaillait utilement en sens inverse mon imagination. La première de ces deux forces, certes, continuait à me montrer ces deux promeneurs de l'avenue des Champs-Élysées, et m'offrait d'autres images désagréables, tirées du passé, par exemple Gilberte haussant les épaules quand sa mère lui demandait de rester avec moi. Mais la seconde force, travaillant sur le canevas de mes espérances, dessinait un avenir bien plus complaisamment développé que ce pauvre passé en somme si restreint. Pour une minute où je revoyais Gilberte maussade, combien n'y en avait-il pas où je combinais une démarche qu'elle ferait faire pour notre réconciliation, pour nos fiançailles peut-être. Il est vrai que cette force que l'imagination dirigeait vers l'avenir, elle la puisait malgré tout dans le passé. Au fur et à mesure que s'effacerait mon ennui que Gilberte eût haussé les épaules, diminuerait aussi le souvenir de son charme, souvenir qui me faisait souhaiter qu'elle revînt vers moi. Mais j'étais encore bien loin de cette mort du passé. J'aimais toujours celle qu'il est vrai que je croyais détester. Mais chaque fois qu'on me trouvait bien coiffé, ayant bonne mine, j'aurais voulu qu'elle fût là. J'étais irrité du désir que beaucoup de gens manifestèrent à cette époque de me recevoir et chez lesquels je refusai d'aller. Il y eut une scène à la maison parce que je n'accompagnai pas mon père à un dîner officiel où il devait y avoir les Bontemps avec leur nièce Albertine, petite jeune fille, presque encore enfant. Les différentes périodes de notre vie se chevauchent ainsi l'une l'autre. On refuse dédaigneusement, à cause de ce qu'on aime et qui vous sera un jour si égal, de voir ce qui vous est égal aujourd'hui, qu'on aimera demain, qu'on aurait peut-être pu, si on avait consenti à le voir, aimer plus tôt, et qui eût ainsi abrégé vos souffrances actuelles, pour les remplacer, il est vrai, par d'autres. Les miennes allaient se modifiant. J'avais l'étonnement d'apercevoir au fond de moi-même, un jour un sentiment, le jour suivant un autre, généralement inspirés par telle espérance ou telle crainte relatives à Gilberte, à la Gilberte que je portais en moi. J'aurais dû me dire que l'autre, la réelle, était peut-être entièrement différente de celle-là, ignorait tous les regrets que je lui prêtais, pensait probablement beaucoup moins à moi non seulement que moi à elle, mais que je ne la faisais elle-même penser à moi quand j'étais seul en tête à tête avec ma Gilberte fictive, cherchais quelles pouvaient être ses vraies intentions à mon égard et l'imaginais ainsi, son attention toujours tournée vers moi.

Pendant ces périodes où, tout en s'affaiblissant, persiste le chagrin, il faut distinguer entre celui que nous cause la pensée constante de la personne elle-même, et celui que raniment certains souvenirs, telle phrase méchante dite, tel verbe employé dans une lettre qu'on a reçue. En réservant de décrire à l'occasion d'un amour ultérieur les formes diverses du chagrin, disons que de ces deux-là la première est infiniment moins cruelle que la seconde. Cela tient à ce que notre notion de la personne, vivant toujours en nous, y est embellie de l'auréole que nous ne tardons pas à lui rendre, et s'empreint sinon des douceurs fréquentes de l'espoir, tout au moins du calme d'une tristesse permanente. (D'ailleurs, il est à remarquer que l'image d'une personne qui nous fait souffrir tient peu de place dans ces complications qui aggravent un chagrin d'amour, le prolongent et l'empêchent de guérir, comme dans certaines maladies la cause est hors de proportions avec la fièvre consécutive et la lenteur à entrer en convalescence.) Mais si l'idée de la personne que nous aimons reçoit le reflet d'une intelligence généralement optimiste, il n'en est pas de même de ces souvenirs particuliers, de ces propos méchants, de cette lettre hostile (je n'en reçus qu'une seule qui le fût, de Gilberte), on dirait que la personne elle-même réside dans ces fragments pourtant si restreints, et portée à une puissance qu'elle est bien loin d'avoir dans l'idée habituelle que nous nous formons d'elle tout entière. C'est que la lettre nous ne l'avons pas, comme l'image de l'être aimé, contemplée dans le calme mélancolique du regret ; nous l'avons lue, dévorée, dans l'angoisse affreuse dont nous étreignait un malheur inattendu. La formation de cette sorte de chagrins est autre ; ils nous viennent du dehors, et c'est par le chemin de la plus cruelle souffrance qu'ils sont allés jusqu'à notre coeur. L'image de notre amie, que nous croyons ancienne, authentique, a été en réalité refaite par nous bien des fois. Le souvenir cruel, lui, n'est pas contemporain de cette image restaurée, il est d'un autre âge, il est un des rares témoins d'un monstrueux passé. Mais comme ce passé continue à exister, sauf en nous à qui il a plu de lui substituer un merveilleux âge d'or, un paradis où tout le monde sera réconcilié, ces souvenirs, ces lettres, sont un rappel à la réalité et devraient nous faire sentir par le brusque mal qu'ils nous font combien nous nous sommes éloignés d'elle dans les folles espérances de notre attente quotidienne. Ce n'est pas que cette réalité doive toujours rester la même bien que cela arrive parfois. Il y a dans notre vie bien des femmes que nous n'avons jamais cherché à revoir et qui ont tout naturellement répondu à notre silence nullement voulu par un silence pareil. Seulement celles-là, comme nous ne les aimions pas, nous n'avons pas compté les années passées loin d'elles, et cet exemple, qui l'infirmerait, est négligé par nous quand nous raisonnons sur l'efficacité de l'isolement, comme le sont, par ceux qui croient aux pressentiments, tous les cas où les leurs ne furent pas vérifiés.

Mais enfin l'éloignement peut être efficace. Le désir, l'appétit de nous revoir, finissent par renaître dans le coeur qui actuellement nous méconnaît. Seulement il y faut du temps. Or, nos exigences en ce qui concerne le temps ne sont pas moins exorbitantes que celles réclamées par le coeur pour changer. D'abord, c'est précisément ce que nous accordons le moins aisément, car notre souffrance est cruelle et nous sommes pressés de la voir finir. Ensuite, ce temps dont l'autre coeur aura besoin pour changer, le nôtre s'en servira pour changer lui aussi, de sorte que quand le but que nous nous proposions deviendra accessible, il aura cessé d'être un but pour nous. D'ailleurs, l'idée même qu'il sera accessible, qu'il n'est pas de bonheur que, lorsqu'il ne sera plus un bonheur pour nous, nous ne finissions par atteindre, cette idée comporte une part, mais une part seulement, de vérité. Il nous échoit quand nous y sommes devenus indifférents. Mais précisément cette indifférence nous a rendus moins exigeants et nous permet de croire rétrospectivement qu'il nous eût ravis à une époque où il nous eût peut-être semblé fort incomplet. On n'est pas très difficile ni très bon juge sur ce dont on ne se soucie point. L'amabilité d'un être que nous n'aimons plus et qui semble encore excessive à notre indifférence eût peut-être été bien loin de suffire à notre amour. Ces tendres paroles, cette offre d'un rendez-vous, nous pensons au plaisir qu'elles nous auraient causé, non à toutes celles dont nous les aurions voulu voir immédiatement suivies et que par cette avidité nous aurions peut-être empêché de se produire. De sorte qu'il n'est pas certain que le bonheur survenu trop tard, quand on ne peut plus en jouir, quand on n'aime plus, soit tout à fait ce même bonheur dont le manque nous rendit jadis si malheureux. Une seule personne pourrait en décider, notre moi d'alors ; il n'est plus là ; et sans doute suffirait-il qu'il revînt, pour que, identique ou non, le bonheur s'évanouît.

En attendant ces réalisations après coup d'un rêve auquel je ne tiendrais plus, à force d'inventer, comme au temps où je connaissais à peine Gilberte, des paroles, des lettres, où elle implorait mon pardon, avouait n'avoir jamais aimé que moi et demandait à m'épouser, une série de douces images incessamment recréées, finirent par prendre plus de place dans mon esprit que la vision de Gilberte et du jeune homme, laquelle n'était plus alimentée par rien. Je serais peut-être dès lors retourné chez Odette sans un rêve que je fis et où un de mes amis, lequel n'était pourtant pas de ceux que je me connaissais, agissait envers moi avec la plus grande fausseté et croyait à la mienne. Brusquement réveillé par la souffrance que venait de me causer ce rêve et voyant qu'elle persistait, je repensai à lui, cherchai à me rappeler quel était l'ami que j'avais vu en dormant et dont le nom espagnol n'était déjà plus distinct. À la fois Joseph et Pharaon, je me mis à interpréter mon rêve. Je savais que dans beaucoup d'entre eux il ne faut tenir compte ni de l'apparence des personnes, lesquelles peuvent être déguisées et avoir interchangé leurs visages, comme ces saints mutilés des cathédrales que des archéologues ignorants ont refaits, en mettant sur le corps de l'un la tête de l'autre, et en mêlant les attributs et les noms. Ceux que les êtres portent dans un rêve peuvent nous abuser. La personne que nous aimons doit y être reconnue seulement à la force de la douleur éprouvée. La mienne m'apprit que, devenue pendant mon sommeil un jeune homme, la personne dont la fausseté récente me faisait encore mal était Gilberte. Je me rappelai alors que la dernière fois que je l'avais vue, le jour où sa mère l'avait empêchée d'aller à une matinée de danse, elle avait soit sincèrement, soit en le feignant, refusé tout en riant d'une façon étrange, de croire à mes bonnes intentions pour elle.
