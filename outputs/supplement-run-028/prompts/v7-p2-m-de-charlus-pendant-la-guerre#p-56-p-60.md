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
      "canonical_name": "baron de Charlus",
      "surface_forms": [
        "baron de Charlus",
        "le baron"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "baron de Charlus",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.91,
      "evidence": "Lettre posthume: « il ne serait pas sorti de chez moi vivant. J’étais décidé à le tuer. » Le narrateur ajoute: « Alors je compris la peur de Morel... l’aveu était vrai. »",
      "explanation": "Charlus's letter admitting his intention to kill Morel confirms Morel's fear and reveals a \"nearly mad side\" more serious than mere outbursts. The narrator endorses this reading."
    }
  ],
  "status_effects": [
    {
      "character": "baron de Charlus",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.91,
      "explanation": "Revealed as dangerous and ready for murder, he is locally lowered beyond a mere worldly outburst."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p2-m-de-charlus-pendant-la-guerre#p-56-p-60"
}

### Candidate characters

[
  "Brichot",
  "Morel",
  "Norpois",
  "Robert de Saint-Loup",
  "docteur Cottard",
  "duchesse de Guermantes",
  "le narrateur",
  "le peintre",
  "princesse de Guermantes"
]

### Prior local context (optional)

Malheureusement, dès le lendemain, disons-le tout de suite, baron de Charlus se trouva dans la rue face à face avec Morel ; celui-ci, pour exciter sa jalousie, le prit par le bras, lui raconta des histoires plus ou moins vraies et quand baron de Charlus éperdu, ayant besoin que Morel restât cette soirée auprès de lui, le supplia de ne pas aller ailleurs, l'autre, apercevant un camarade, dit adieu à baron de Charlus qui, de colère, espérant que cette menace que, bien entendu, il semblait ne devoir exécuter jamais, ferait rester Morel, lui dit : « Prends garde, je me vengerai », et Morel, riant, partit en tapotant sur le cou et en enlaçant par la taille son camarade étonné.

### Passage

À l'accent soudain tremblant avec lequel Charlus avait, en me parlant de Morel, scandé ses paroles, au regard trouble qui vacillait au fond de ses yeux, j'eus l'impression qu'il y avait autre chose qu'une banale insistance. Je ne me trompais pas et je dirai tout de suite les deux faits qui me le prouvèrent rétrospectivement (j'anticipe de beaucoup d'années pour le second de ces faits, postérieur à la mort de Charlus. Or elle ne devait se produire que bien plus tard, et nous aurons l'occasion de le revoir plusieurs fois, bien différent de ce que nous l'avons connu, et en particulier la dernière fois, à une époque où il avait entièrement oublié Morel). Quant au premier de ces faits, il se produisit deux ans seulement après le soir où je descendais ainsi les boulevards avec Charlus. Donc environ deux ans après cette soirée, je rencontrai Morel. Je pensai aussitôt à Charlus, au plaisir qu'il aurait à revoir le violoniste, et j'insistai auprès de lui pour qu'il allât le voir, fût-ce une fois. « Il a été bon pour vous, dis-je à Morel. Il est déjà vieux, il peut mourir, il faut liquider les vieilles querelles et effacer les traces de la brouille. » Morel parut entièrement de mon avis quant à un apaisement désirable, mais il n'en refusa pas moins catégoriquement de faire même une seule visite à Charlus. « Vous avez tort, lui dis-je. Est-ce par entêtement, par paresse, par méchanceté, par amour-propre mal placé, par vertu (soyez sûr qu'elle ne sera pas attaquée), par coquetterie ? » Alors le violoniste, tordant son visage pour un aveu qui lui coûtait sans doute extrêmement, me répondit en frissonnant : « Non, ce n'est pour rien de tout cela, la vertu je m'en fous ; la méchanceté, au contraire je commence à le plaindre ; ce n'est pas par coquetterie, elle serait inutile ; ce n'est pas par paresse, il y a des journées entières où je reste à me tourner les pouces, non, ce n'est à cause de rien de tout cela ; c'est, ne le dites jamais à personne et je suis fou de vous le dire, c'est, c'est... c'est... par peur ! » Il se mit à trembler de tous ses membres. Je lui avouai que je ne le comprenais pas. « Non, ne me demandez pas, n'en parlons plus, vous ne le connaissez pas comme moi, je peux dire que vous ne le connaissez pas du tout. – Mais quel tort peut-il vous faire ? il cherchera, d'ailleurs, d'autant moins à vous en faire qu'il n'y aura plus de rancune entre vous. Et puis, au fond, vous savez qu'il est très bon. – Parbleu si, je le sais qu'il est bon ! Et la délicatesse et la droiture. Mais laissez-moi, ne m'en parlez plus, je vous en supplie, c'est honteux à dire, j'ai peur ! » Le second fait date d'après la mort de Charlus. On m'apporta quelques souvenirs qu'il m'avait laissés et une lettre à triple enveloppe, écrite au moins dix ans avant sa mort. Mais il avait été gravement malade, avait pris ses dispositions, puis s'était rétabli avant de tomber plus tard dans l'état où nous le verrons le jour d'une matinée chez la princesse de Guermantes – et la lettre, restée dans un coffre avec les objets qu'il léguait à quelques amis, était restée là sept ans, sept ans pendant lesquels il avait entièrement oublié Morel. La lettre, tracée d'une écriture fine et ferme, était ainsi conçue : « Mon cher ami, les voies de la Providence sont inconnues. Parfois c'est du défaut d'un être médiocre qu'elle use pour empêcher de faillir la suréminence d'un juste. Vous connaissez Morel, d'où il est sorti, à quel faîte j'ai voulu l'élever, autant dire à mon niveau. Vous savez qu'il a préféré retourner non pas à la poussière et à la cendre d'où tout homme, c'est-à-dire le véritable phoenix, peut renaître, mais à la boue où rampe la vipère. Il s'est laissé choir, ce qui m'a préservé de déchoir. Vous savez que mes armes contiennent la devise même de Notre-Seigneur : « Inculcabis super leonem et aspidem » avec un homme représenté comme ayant à la plante de ses pieds, comme support héraldique, un lion et un serpent. Or si j'ai pu fouler ainsi le propre lion que je suis, c'est grâce au serpent et à sa prudence, qu'on appelle trop légèrement parfois un défaut, car la profonde sagesse de l'Évangile en fait une vertu, au moins une vertu pour les autres. Notre serpent aux sifflements jadis harmonieusement modulés, quand il avait un charmeur – fort charmé, du reste – n'était pas seulement musical et reptile, il avait jusqu'à la lâcheté cette vertu que je tiens maintenant pour divine, la Prudence. C'est cette divine prudence qui l'a fait résister aux appels que je lui ai fait transmettre de revenir me voir, et je n'aurai de paix en ce monde et d'espoir de pardon dans l'autre que si je vous en fais l'aveu. C'est lui qui a été en cela l'instrument de la Sagesse divine, car, je l'avais résolu, il ne serait pas sorti de chez moi vivant. Il fallait que l'un de nous deux disparût. J'étais décidé à le tuer. Dieu lui a conseillé la prudence pour me préserver d'un crime. Je ne doute pas que l'intercession de l'Archange Michel, mon saint patron, n'ait joué là un grand rôle et je le prie de me pardonner de l'avoir tant négligé pendant plusieurs années et d'avoir si mal répondu aux innombrables bontés qu'il m'a témoignées, tout spécialement dans ma lutte contre le mal. Je dois à ce serviteur, je le dis dans la plénitude de ma foi et de mon intelligence, que le Père céleste ait inspiré à Morel de ne pas venir. Aussi, c'est moi maintenant qui me meurs. Votre fidèlement dévoué, Semper idem, P. G. Charlus. » Alors je compris la peur de Morel ; certes il y avait dans cette lettre bien de l'orgueil et de la littérature. Mais l'aveu était vrai. Et Morel savait mieux que moi que le « côté presque fou » que Mme de Guermantes trouvait chez son beau-frère ne se bornait pas, comme je l'avais cru jusque-là, à ces dehors momentanés de rage superficielle et inopérante.

Mais il faut revenir en arrière. Je descends les boulevards à côté de Charlus, lequel vient de me prendre comme vague intermédiaire pour des ouvertures de paix entre lui et Morel. Voyant que je ne lui répondais pas, il continua ainsi : « Je ne sais pas, du reste, pourquoi il ne joue pas, on ne fait plus de musique sous prétexte que c'est la guerre, mais on danse, on dîne en ville. Les fêtes remplissent ce qui sera peut-être, si les Allemands avancent encore, les derniers jours de notre Pompéi. Pour peu que la lave de quelque Vésuve allemand (leurs pièces de marine ne sont pas moins terribles qu'un volcan) vienne les surprendre à leur toilette et éternise leur geste en l'interrompant, les enfants s'instruiront plus tard en regardant dans les livres de classes illustrés Mme Molé qui allait mettre une dernière couche de fard avant d'aller dîner chez une belle-soeur, ou Sosthène de Guermantes finissant de peindre ses faux sourcils ; ce sera matière à cours pour les Brichot de l'avenir ; la frivolité d'une époque quand dix siècles ont passé sur elle est digne de la plus grave érudition, surtout si elle a été conservée intacte par une éruption volcanique ou des matières analogues à la lave projetées par bombardement. Quels documents pour l'histoire future, quand les gaz asphyxiants analogues à ceux qu'émettait le Vésuve et des écroulements comme ceux qui ensevelirent Pompéi garderont intactes toutes les dernières imprudentes qui n'ont pas fait encore filer pour Bayonne leurs tableaux et leurs statues. D'ailleurs, n'est-ce pas déjà, depuis un an, Pompéi par fragments, chaque soir, que ces gens se sauvant dans les caves, non pas pour en rapporter quelque vieille bouteille de Mouton Rothschild ou de Saint-Émilion, mais pour cacher avec eux ce qu'ils ont de plus précieux, comme les prêtres d'Herculanum surpris par la mort au moment où ils emportaient les vases sacrés. C'est toujours l'attachement à l'objet qui amène la mort du possesseur. Paris, lui, ne fut pas, comme Herculanum, fondé par Hercule. Mais que de ressemblances s'imposent ! et cette lucidité qui nous est donnée n'est pas que de notre époque, chacune l'a possédée. Si je pense que nous pouvons avoir demain le sort des villes du Vésuve, celles-ci sentaient qu'elles étaient menacées du sort des villes maudites de la Bible. On a retrouvé sur les murs d'une des maisons de Pompéi cette inscription révélatrice : « Sodoma, Gomora. » Je ne sais si ce fut ce nom de Sodome et les idées qu'il éveilla en lui, soit celle du bombardement, qui firent que Charlus leva un instant les yeux au ciel, mais il les ramena bientôt sur la terre. « J'admire tous les héros de cette guerre, dit-il. Tenez, mon cher, les soldats anglais que j'ai un peu légèrement considérés au début de la guerre comme de simples joueurs de football assez présomptueux pour se mesurer avec des professionnels – et quels professionnels ! – hé bien, rien qu'esthétiquement ce sont des athlètes de la Grèce, vous entendez bien, de la Grèce, mon cher, ce sont les jeunes gens de Platon, ou plutôt des Spartiates. J'ai un ami qui est allé à Rouen où ils ont leur camp, il a vu des merveilles, de pures merveilles dont on n'a pas idée. Ce n'est plus Rouen, c'est une autre ville. Évidemment il y a aussi l'ancien Rouen, avec les Saints émaciés de la cathédrale. Bien entendu, c'est beau aussi, mais c'est autre chose. Et nos poilus ! je ne peux pas vous dire quelle saveur je trouve en nos poilus, aux petits Parigots, tenez, comme celui qui passe là, avec son air dessalé, sa mine éveillée et drôle. Il m'arrive souvent de les arrêter, de faire un brin de causette avec eux, quelle finesse, quel bon sens ! et les gars de province, comme ils sont amusants et gentils avec leur roulement d'r et leur jargon patoiseur !... Moi, j'ai toujours beaucoup vécu à la campagne, couché dans les fermes, je sais leur parler, mais notre admiration pour les Français ne doit pas nous faire déprécier nos ennemis, ce serait nous diminuer nous-mêmes. Et vous ne savez pas quel soldat est le soldat allemand, vous ne l'avez pas vu comme moi défiler au pas de parade, au pas de l'oie, « unter den Linden ». En revenant à l'idéal de virilité qu'il m'avait esquissé à Balbec et qui avec le temps avait pris chez lui une forme philosophique, usant, d'ailleurs, de raisonnements absurdes, qui par moments, même quand il venait d'être supérieur, laissaient voir la trame trop mince du simple homme du monde, bien qu'homme du monde intelligent : « Voyez-vous, me dit-il, le superbe gaillard qu'est le soldat boche est un être fort, sain, ne pensant qu'à la grandeur de son pays, « Deutschland über alles », ce qui n'est pas si bête, et tandis qu'ils se préparaient virilement, nous nous sommes abîmés dans le dilettantisme. » Ce mot signifiait probablement pour Charlus quelque chose d'analogue à la littérature, car aussitôt se rappelant sans doute que j'aimais les lettres et avais eu un moment l'intention de m'y adonner, il me tapa sur l'épaule (profitant du geste pour s'y appuyer jusqu'à me faire aussi mal qu'autrefois, quand je faisais mon service militaire, le recul contre l'omoplate du « 76 »), il me dit comme pour adoucir le reproche : « Oui, nous nous sommes abîmés dans le dilettantisme, nous tous, vous aussi, rappelez-vous, vous pouvez faire comme moi votre mea culpa, nous avons été trop dilettantes. » Par surprise du reproche, manque d'esprit de repartie, déférence envers mon interlocuteur et attendrissement pour son amicale bonté, je répondis comme si, ainsi qu'il m'y invitait, j'avais aussi à me frapper la poitrine, ce qui était parfaitement stupide car je n'avais pas l'ombre de dilettantisme à me reprocher. « Allons, me dit-il, je vous quitte (le groupe qui l'avait escorté de loin ayant fini par nous abandonner). Je m'en vais me coucher comme un très vieux Monsieur, d'autant plus qu'il paraît que la guerre a changé toutes nos habitudes, un de ces aphorismes qu'affectionne Norpois. » Je savais, du reste, qu'en rentrant chez lui Charlus ne cessait pas pour cela d'être au milieu des soldats, car il avait transformé son hôtel en hôpital militaire, cédant du reste, je le crois, aux besoins bien moins de son imagination que de son bon coeur.

Il faisait une nuit transparente et sans un souffle. J'imaginais que la Seine coulant entre ses ponts circulaires, faits de leur plateau et de son reflet, devait ressembler au Bosphore. Et symbole soit de cette invasion que prédisait le défaitisme de Charlus, soit de la coopération de nos frères musulmans avec les armées de la France, la lune étroite et recourbée comme un sequin semblait mettre le ciel parisien sous le signe oriental du croissant. Pour un instant encore il resta en arrêt devant un Sénégalais en me disant adieu et en me serrant la main à me la broyer, ce qui est une particularité allemande chez les gens qui sentent comme le baron, et en continuant pendant quelque temps à me la malaxer, eût dit jadis Cottard, comme si Charlus avait voulu rendre à mes articulations une souplesse qu'elles n'avaient point perdue. Chez certains aveugles, le toucher supplée dans une certaine mesure à la vue. Je ne sais trop de quel sens il prenait la place ici. Il croyait peut-être seulement me serrer la main comme il crut sans doute ne faire que voir le Sénégalais qui passait dans l'ombre et ne daigna pas s'apercevoir qu'il était admiré. Mais, dans ces deux cas, le baron se trompait, il péchait par excès de contact et de regards. « Est-ce que tout l'Orient de Decamps, de Fromentin, d'Ingres, de Delacroix n'est pas là dedans ? me dit-il, encore immobilisé par le passage du Sénégalais. Vous savez, moi, je ne m'intéresse jamais aux choses et aux êtres qu'en peintre, en philosophe. D'ailleurs je suis trop vieux. Mais quel malheur, pour compléter le tableau, que l'un de nous deux ne soit pas une odalisque. »

Ce ne fut pas l'Orient de Decamps, ni même de Delacroix qui commença de hanter mon imagination quand le baron m'eut quitté, mais le vieil Orient de ces Mille et une Nuits que j'avais tant aimées, et, me perdant peu à peu dans le lacis de ces rues noires, je pensais au calife Haroun Al Raschid en quête d'aventures dans les quartiers perdus de Bagdad. D'autre part, la chaleur du temps et de la marche m'avait donné soif, mais depuis longtemps tous les bars étaient fermés, et à cause de la pénurie d'essence les rares taxis que je rencontrais, conduits par des Levantins ou des Nègres, ne prenaient même pas la peine de répondre à mes signes. Le seul endroit où j'aurais pu me faire servir à boire et reprendre des forces pour rentrer chez moi eût été un hôtel. Mais dans la rue assez éloignée du centre où j'étais parvenu, tous, depuis que sur Paris les gothas lançaient leurs bombes, avaient fermé. Il en était de même de presque toutes les boutiques de commerçants, lesquels, faute d'employés ou eux-mêmes pris de peur, avaient fui à la campagne et laissé sur la porte un avertissement habituel écrit à la main et annonçant leur réouverture pour une époque éloignée et, d'ailleurs, problématique. Les autres établissements qui avaient pu survivre encore annonçaient de la même manière qu'ils n'ouvraient que deux fois par semaine. On sentait que la misère, l'abandon, la peur habitaient tout ce quartier. Je n'en fus que plus surpris de voir qu'entre ces maisons délaissées il y en avait une où la vie au contraire semblait avoir vaincu l'effroi, la faillite, et entretenait l'activité et la richesse. Derrière les volets clos de chaque fenêtre la lumière, tamisée à cause des ordonnances de police, décelait pourtant un insouci complet de l'économie. Et à tout instant la porte s'ouvrait pour laisser entrer ou sortir quelque visiteur nouveau. C'était un hôtel par qui la jalousie de tous les commerçants voisins (à cause de l'argent que ses propriétaires devaient gagner) devait être excitée ; et ma curiosité le fut aussi quand je vis sortir rapidement, à une quinzaine de mètres de moi, c'est-à-dire trop loin pour que dans l'obscurité profonde je pusse le reconnaître, un officier.

Quelque chose pourtant me frappa qui n'était pas sa figure que je ne voyais pas, ni son uniforme dissimulé dans une grande houppelande, mais la disproportion extraordinaire entre le nombre de points différents par où passa son corps et le petit nombre de secondes pendant lesquelles cette sortie, qui avait l'air de la sortie tentée par un assiégé, s'exécuta. De sorte que je pensai, si je ne le reconnus pas formellement – je ne dirai pas même à la tournure ni à la sveltesse, ni à l'allure, ni à la vélocité de Saint-Loup – mais à l'espèce d'ubiquité qui lui était si spéciale. Le militaire capable d'occuper en si peu de temps tant de positions différentes dans l'espace avait disparu, sans m'avoir aperçu, dans une rue de traverse, et je restais à me demander si je devais ou non entrer dans cet hôtel dont l'apparence modeste me fit fortement douter que ce fût Saint-Loup qui en fût sorti. Je me rappelai involontairement que Saint-Loup avait été injustement mêlé à une affaire d'espionnage parce qu'on avait trouvé son nom dans les lettres saisies sur un officier allemand. Pleine justice lui avait d'ailleurs été rendue par l'autorité militaire. Mais malgré moi je rapprochai ce fait de ce que je voyais. Cet hôtel servait-il de lieu de rendez-vous à des espions ? L'officier avait depuis un moment disparu quand je vis entrer de simples soldats de plusieurs armes, ce qui ajouta encore à la force de ma supposition. J'avais, d'autre part, extrêmement soif. « Il est probable que je pourrai trouver à boire ici », me dis-je, et j'en profitai pour tâcher d'assouvir, malgré l'inquiétude qui s'y mêlait, ma curiosité. Je ne pense donc pas que ce fut la curiosité de cette rencontre qui me décida à monter le petit escalier de quelques marches au bout duquel la porte d'une espèce de vestibule était ouverte, sans doute à cause de la chaleur. Je crus d'abord que, cette curiosité, je ne pourrais la satisfaire, car je vis plusieurs personnes venir demander une chambre, à qui on répondit qu'il n'y en avait plus une seule. Mais je compris ensuite qu'elles n'avaient évidemment contre elles que de ne pas faire partie du nid d'espionnage, car un simple marin s'étant présenté un moment après on se hâta de lui donner le n° 28. Je pus apercevoir sans être vu, grâce à l'obscurité, quelques militaires et deux ouvriers qui causaient tranquillement dans une petite pièce étouffée, prétentieusement ornée de portraits en couleurs de femmes découpés dans des magazines et des revues illustrées. Ces gens causaient tranquillement, en train d'exposer des idées patriotiques : « Qu'est-ce que tu veux, on fera comme les camarades », disait l'un. « Ah ! pour sûr que je pense bien ne pas être tué », répondait à un voeu que je n'avais pas entendu, un autre qui, à ce que je compris, repartait le lendemain pour un poste dangereux. « Par exemple, à vingt-deux ans, en n'ayant encore fait que six mois, ce serait fort », criait-il avec un ton où perçait encore plus que le désir de vivre longtemps la conscience de raisonner juste, et comme si le fait de n'avoir que vingt-deux ans devait lui donner plus de chances de ne pas être tué, et que ce dût être une chose impossible qu'il le fût. « À Paris c'est épatant, disait un autre ; on ne dirait pas qu'il y a la guerre. Et toi, Julot, tu t'engages toujours ? – Pour sûr que je m'engage, j'ai envie d'aller y taper un peu dans le tas à tous ces sales Boches. – Mais Joffre, c'est un homme qui couche avec les femmes des Ministres, c'est pas un homme qui a fait quelque chose. – C'est malheureux d'entendre des choses pareilles, dit un aviateur un peu plus âgé en se tournant vers l'ouvrier qui venait de faire entendre cette proposition ; je vous conseillerais pas de causer comme ça en première ligne, les poilus vous auraient vite expédié. » La banalité de ces conversations ne me donnait pas grande envie d'en entendre davantage, et j'allais entrer ou redescendre quand je fus tiré de mon indifférence en entendant ces phrases qui me firent frémir : « C'est épatant, le patron qui ne revient pas, dame, à cette heure-ci je ne sais pas trop où il trouvera des chaînes. – Mais puisque l'autre est déjà attaché. – Il est attaché bien sûr, il est attaché et il ne l'est pas, moi je serais attaché comme ça que je pourrais me détacher. – Mais le cadenas est fermé. – C'est entendu qu'il est fermé, mais ça peut s'ouvrir à la rigueur. Ce qu'il y a, c'est que les chaînes ne sont pas assez longues. Tu vas pas m'expliquer à moi ce que c'est, j'y ai tapé dessus hier pendant toute la nuit que le sang m'en coulait sur les mains. – C'est toi qui taperas ce soir. – Non, c'est pas moi, c'est Maurice. Mais ça sera moi dimanche, le patron me l'a promis. » Je compris maintenant pourquoi on avait eu besoin des bras solides du marin. Si on avait éloigné de paisibles bourgeois, ce n'était donc pas qu'un nid d'espions que cet hôtel. Un crime atroce allait y être consommé, si on n'arrivait pas à temps pour le découvrir et faire arrêter les coupables. Tout cela pourtant, dans cette nuit paisible et menacée, gardait une apparence de rêve, de conte, et c'est à la fois avec une fierté de justicier et une volupté de poète que j'entrai délibérément dans l'hôtel. Je touchai légèrement mon chapeau et les personnes présentes, sans se déranger, répondirent plus ou moins poliment à mon salut. « Est-ce que vous pourriez me dire à qui il faut m'adresser ? Je voudrais avoir une chambre et qu'on m'y monte à boire. – Attendez une minute, le patron est sorti. – Mais il y a le chef là-haut, insinua un des causeurs. – Mais tu sais bien qu'on ne peut pas le déranger. – Croyez-vous qu'on me donnera une chambre ? – J'crois. – Le 43 doit être libre », dit le jeune homme qui était sûr de ne pas être tué parce qu'il avait vingt-deux ans. Et il se poussa légèrement sur le sofa pour me faire place. « Si on ouvrait un peu la fenêtre, il y a une fumée ici », dit l'aviateur ; et en effet chacun avait sa pipe ou sa cigarette. « Oui, mais alors, fermez d'abord les volets, vous savez bien qu'il est défendu d'avoir de la lumière à cause des Zeppelins. – Il n'en viendra plus de Zeppelins. Les journaux ont même fait allusion sur ce qu'ils avaient été tous descendus. – Il n'en viendra plus, il n'en viendra plus, qu'est-ce que tu en sais ? Quand tu auras comme moi quinze mois de front et que tu auras abattu ton cinquième avion boche, tu pourras en causer. Faut pas croire les journaux. Ils sont allés hier sur Compiègne, ils ont tué une mère de famille avec ses deux enfants. – Une mère de famille avec ses deux enfants », dit avec des yeux ardents et un air de profonde pitié le jeune homme qui espérait bien ne pas être tué et qui avait, du reste, une figure énergique, ouverte et des plus sympathiques. « On n'a pas de nouvelles du grand Julot. Sa marraine n'a pas reçu de lettre de lui depuis huit jours et c'est la première fois qu'il reste si longtemps sans lui en donner. – Qui est sa marraine ? – C'est la dame qui tient le chalet de nécessité un peu plus bas que l'Olympia. – Ils couchent ensemble ? – Qu'est-ce que tu dis là ; c'est une femme mariée, tout ce qu'il y a de sérieuse. Elle lui envoie de l'argent toutes les semaines parce qu'elle a bon coeur. Ah ! c'est une chic femme. – Alors tu le connais, le grand Julot ? – Si je le connais ! reprit avec chaleur le jeune homme de vingt-deux ans. C'est un de mes meilleurs amis intimes. Il n'y en a pas beaucoup que j'estime comme lui, et bon camarade, toujours prêt à rendre service, ah ! tu parles que ce serait un rude malheur s'il lui était arrivé quelque chose. » Quelqu'un proposa une partie de dés et à la hâte fébrile avec laquelle le jeune homme de vingt-deux ans retournait les dés et criait les résultats, les yeux hors de la tête, il était aisé de voir qu'il avait un tempérament de joueur. Je ne saisis pas bien ce que quelqu'un lui dit ensuite, mais il s'écria d'un ton de profonde pitié : « Julot, un maquereau ! C'est-à-dire qu'il dit qu'il est un maquereau. Mais il n'est pas foutu de l'être. Moi je l'ai vu payer sa femme, oui, la payer. C'est-à-dire que je ne dis pas que Jeanne l'Algérienne ne lui donnait pas quelque chose, mais elle ne lui donnait pas plus de cinq francs, une femme qui était en maison, qui gagnait plus de cinquante francs par jour. Se faire donner que cinq francs ! il faut qu'un homme soit trop bête. Et maintenant qu'elle est sur le front, elle a une vie dure, je veux bien, mais elle gagne ce qu'elle veut ; eh bien, elle ne lui envoie rien. Ah ! un maquereau, Julot ? Il y en a beaucoup qui pourraient se dire maquereaux à ce compte-là. Non seulement ce n'est pas un maquereau, mais à mon avis c'est même un imbécile. » Le plus vieux de la bande, et que le patron avait sans doute, à cause de son âge, chargé de lui faire garder une certaine tenue, n'entendit, étant allé un moment jusqu'aux cabinets, que la fin de la conversation. Mais il ne put s'empêcher de me regarder et parut visiblement contrarié de l'effet qu'elle avait dû produire sur moi. Sans s'adresser spécialement au jeune homme de vingt-deux ans qui venait pourtant d'exposer cette théorie de l'amour vénal, il dit, d'une façon générale : « Vous causez trop et trop fort, la fenêtre est ouverte, il y a des gens qui dorment à cette heure-ci. Vous savez que si le patron rentrait et vous entendait causer comme ça, il ne serait pas content. » Précisément en ce moment on entendit la porte s'ouvrir et tout le monde se tut croyant que c'était le patron, mais ce n'était qu'un chauffeur d'auto étranger auquel tout le monde fit grand accueil. Mais en voyant une chaîne de montre superbe qui s'étalait sur la veste du chauffeur, le jeune homme de vingt-deux ans lui lança un coup d'oeil interrogatif et rieur, suivi d'un froncement de sourcil et d'un clignement d'oeil sévère dirigé de mon côté. Et je compris que le premier regard voulait dire : « Qu'est-ce que ça ? tu l'as volée ? Toutes mes félicitations. » Et le second : « Ne dis rien à cause de ce type que nous ne connaissons pas. »
