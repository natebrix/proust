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
      "canonical_name": "Legrandin",
      "surface_forms": [
        "Legrandin"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Legrandin",
      "type": "narrated_diminishment",
      "polarity": "negative",
      "narrative_stance": "endorsed",
      "confidence": 0.95,
      "evidence": "À l’évocation des Guermantes, ses yeux semblent « percés par une pointe », il répond avec une emphase gênée (« Non, je ne les connais pas… je n’ai jamais voulu… je suis une tête jacobine »). Le narrateur conclut : « il était snob », et compare Legrandin à « un saint Sébastien du snobisme » dont le corps trahit la vérité avant le discours.",
      "explanation": "The narrator unveils and locally condemns Legrandin's snobbery, highlighting the gap between his proud words of independence and the bodily/tonal signs of desire for aristocracy."
    }
  ],
  "status_effects": [
    {
      "character": "Legrandin",
      "dimension": "general_appraisal",
      "delta": -2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.94,
      "explanation": "His image is clearly lowered by the narrator’s exposition of his snobbery and insincerity."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-271-p-275"
}

### Candidate characters

[
  "duchesse de Guermantes",
  "la grand-mère",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

Hélas ! nous devions définitivement changer d'opinion sur Legrandin. Un des dimanches qui suivit la rencontre sur le Pont-Vieux après laquelle le père du narrateur avait dû confesser son erreur, comme la messe finissait et qu'avec le soleil et le bruit du dehors quelque chose de si peu sacré entrait dans l'église que Mme Goupil, Mme Percepied (toutes les personnes qui tout à l'heure, à mon arrivée un peu en retard, étaient restées les yeux absorbés dans leur prière et que j'aurais même pu croire ne m'avoir pas vu entrer si, en même temps, leurs pieds n'avaient repoussé légèrement le petit banc qui m'empêchait de gagner ma chaise) commençaient à s'entretenir avec nous à haute voix de sujets tout temporels comme si nous étions déjà sur la place, nous vîmes sur le seuil brûlant du porche, dominant le tumulte bariolé du marché, Legrandin, que le mari de cette dame avec qui nous l'avions dernièrement rencontré était en train de présenter à la femme d'un autre gros propriétaire terrien des environs. La figure de Legrandin exprimait une animation, un zèle extraordinaires ; il fit un profond salut avec un renversement secondaire en arrière, qui ramena brusquement son dos au delà de la position de départ et qu'avait dû lui apprendre le mari de sa soeur, Mme de Cambremer. Ce redressement rapide fit refluer en une sorte d'onde fougueuse et musclée la croupe de Legrandin que je ne supposais pas si charnue ; et je ne sais pourquoi cette ondulation de pure matière, ce flot tout charnel, sans expression de spiritualité et qu'un empressement plein de bassesse fouettait en tempête, éveillèrent tout d'un coup dans mon esprit la possibilité d'un Legrandin tout différent de celui que nous connaissions. Cette dame le pria de dire quelque chose à son cocher, et tandis qu'il allait jusqu'à la voiture, l'empreinte de joie timide et dévouée que la présentation avait marquée sur son visage y persistait encore. Ravi dans une sorte de rêve, il souriait, puis il revint vers la dame en se hâtant et, comme il marchait plus vite qu'il n'en avait l'habitude, ses deux épaules oscillaient de droite et de gauche ridiculement, et il avait l'air tant il s'y abandonnait entièrement en n'ayant plus souci du reste, d'être le jouet inerte et mécanique du bonheur. Cependant, nous sortions du porche, nous allions passer à côté de lui, il était trop bien élevé pour détourner la tête, mais il fixa de son regard soudain chargé d'une rêverie profonde un point si éloigné de l'horizon qu'il ne put nous voir et n'eut pas à nous saluer. Son visage restait ingénu au-dessus d'un veston souple et droit qui avait l'air de se sentir fourvoyé malgré lui au milieu d'un luxe détesté. Et une lavallière à pois qu'agitait le vent de la Place continuait à flotter sur Legrandin comme l'étendard de son fier isolement et de sa noble indépendance. Au moment où nous arrivions à la maison, la mère du narrateur s'aperçut qu'on avait oublié le saint-honoré et demanda à le père du narrateur de retourner avec moi sur nos pas dire qu'on l'apportât tout de suite. Nous croisâmes près de l'église Legrandin qui venait en sens inverse conduisant la même dame à sa voiture. Il passa contre nous, ne s'interrompit pas de parler à sa voisine, et nous fit du coin de son oeil bleu un petit signe en quelque sorte intérieur aux paupières et qui, n'intéressant pas les muscles de son visage, put passer parfaitement inaperçu de son interlocutrice ; mais, cherchant à compenser par l'intensité du sentiment le champ un peu étroit où il en circonscrivait l'expression, dans ce coin d'azur qui nous était affecté il fit pétiller tout l'entrain de la bonne grâce qui dépassa l'enjouement, frisa la malice ; il subtilisa les finesses de l'amabilité jusqu'aux clignements de la connivence, aux demi-mots, aux sous-entendus, aux mystères de la complicité ; et finalement exalta les assurances d'amitié jusqu'aux protestations de tendresse, jusqu'à la déclaration d'amour, illuminant alors pour nous seuls, d'une langueur secrète et invisible à la châtelaine, une prunelle énamourée dans un visage de glace.

### Passage

Il avait précisément demandé la veille à mes parents de m'envoyer dîner ce soir-là avec lui : « Venez tenir compagnie à votre vieil ami, m'avait-il dit. Comme le bouquet qu'un voyageur nous envoie d'un pays où nous ne retournerons plus, faites-moi respirer du lointain de votre adolescence ces fleurs des printemps que j'ai traversés moi aussi il y a bien des années. Venez avec la primevère, la barbe de chanoine, le bassin d'or, venez avec le sédum dont est fait le bouquet de dilection de la flore balzacienne, avec la fleur du jour de la Résurrection, la pâquerette et la boule de neige des jardins qui commence à embaumer dans les allées de votre grand'tante, quand ne sont pas encore fondues les dernières boules de neige des giboulées de Pâques. Venez avec la glorieuse vêture de soie du lis digne de Salomon, et l'émail polychrome des pensées, mais venez surtout avec la brise fraîche encore des dernières gelées et qui va entr'ouvrir, pour les deux papillons qui depuis ce matin attendent à la porte, la première rose de Jérusalem. »

On se demandait à la maison si on devait m'envoyer tout de même dîner avec Legrandin. Mais ma grand'mère refusa de croire qu'il eût été impoli. « Vous reconnaissez vous-même qu'il vient là avec sa tenue toute simple qui n'est guère celle d'un mondain. » Elle déclarait qu'en tous cas, et à tout mettre au pis, s'il l'avait été, mieux valait ne pas avoir l'air de s'en être aperçu. À vrai dire mon père lui-même, qui était pourtant le plus irrité contre l'attitude qu'avait eue Legrandin, gardait peut-être un dernier doute sur le sens qu'elle comportait. Elle était comme toute attitude ou action où se révèle le caractère profond et caché de quelqu'un : elle ne se relie pas à ses paroles antérieures, nous ne pouvons pas la faire confirmer par le témoignage du coupable qui n'avouera pas ; nous en sommes réduits à celui de nos sens dont nous nous demandons, devant ce souvenir isolé et incohérent, s'ils n'ont pas été le jouet d'une illusion ; de sorte que de telles attitudes, les seules qui aient de l'importance, nous laissent souvent quelques doutes.

Je dînai avec Legrandin sur sa terrasse ; il faisait clair de lune : « Il y a une jolie qualité de silence, n'est-ce pas, me dit-il ; aux coeurs blessés comme l'est le mien, un romancier que vous lirez plus tard prétend que conviennent seulement l'ombre et le silence. Et voyez-vous, mon enfant, il vient dans la vie une heure dont vous êtes bien loin encore où les yeux las ne tolèrent plus qu'une lumière, celle qu'une belle nuit comme celle-ci prépare et distille avec l'obscurité, où les oreilles ne peuvent plus écouter de musique que celle que joue le clair de lune sur la flûte du silence. » J'écoutais les paroles de Legrandin qui me paraissaient toujours si agréables ; mais troublé par le souvenir d'une femme que j'avais aperçue dernièrement pour la première fois, et pensant, maintenant que je savais que Legrandin était lié avec plusieurs personnalités aristocratiques des environs, que peut-être il connaissait celle-ci, prenant mon courage, je lui dis : « Est-ce que vous connaissez, monsieur, la... les châtelaines de Guermantes ? », heureux aussi en prononçant ce nom de prendre sur lui une sorte de pouvoir, par le seul fait de le tirer de mon rêve et de lui donner une existence objective et sonore.

Mais à ce nom de Guermantes, je vis au milieu des yeux bleus de notre ami se ficher une petite encoche brune comme s'ils venaient d'être percés par une pointe invisible, tandis que le reste de la prunelle réagissait en sécrétant des flots d'azur. Le cerne de sa paupière noircit, s'abaissa. Et sa bouche marquée d'un pli amer se ressaissant plus vite sourit, tandis que le regard restait douloureux, comme celui d'un beau martyr dont le corps est hérissé de flèches : « Non, je ne les connais pas », dit-il, mais au lieu de donner à un renseignement aussi simple, à une réponse aussi peu surprenante le ton naturel et courant qui convenait, il le débita en appuyant sur les mots, en s'inclinant, en saluant de la tête, à la fois avec l'insistance qu'on apporte, pour être cru, à une affirmation invraisemblable – comme si ce fait qu'il ne connût pas les Guermantes ne pouvait être l'effet que d'un hasard singulier – et aussi avec l'emphase de quelqu'un qui, ne pouvant pas taire une situation qui lui est pénible, préfère la proclamer pour donner aux autres l'idée que l'aveu qu'il fait ne lui cause aucun embarras, est facile, agréable, spontané, que la situation elle-même – l'absence de relations avec les Guermantes – pourrait bien avoir été non pas subie, mais voulue par lui, résulter de quelque tradition de famille, principe de morale ou voeu mystique lui interdisant nommément la fréquentation des Guermantes. « Non, reprit-il, expliquant par ses paroles sa propre intonation, non, je ne les connais pas, je n'ai jamais voulu, j'ai toujours tenu à sauvegarder ma pleine indépendance ; au fond je suis une tête jacobine, vous le savez. Beaucoup de gens sont venus à la rescousse, on me disait que j'avais tort de ne pas aller à Guermantes, que je me donnais l'air d'un malotru, d'un vieil ours. Mais voilà une réputation qui n'est pas pour m'effrayer, elle est si vraie ! Au fond, je n'aime plus au monde que quelques églises, deux ou trois livres, à peine davantage de tableaux, et le clair de lune quand la brise de votre jeunesse apporte jusqu'à moi l'odeur des parterres que mes vieilles prunelles ne distinguent plus. » Je ne comprenais pas bien que, pour ne pas aller chez des gens qu'on ne connaît pas, il fût nécessaire de tenir à son indépendance, et en quoi cela pouvait vous donner l'air d'un sauvage ou d'un ours. Mais ce que je comprenais, c'est que Legrandin n'était pas tout à fait véridique quand il disait n'aimer que les églises, le clair de lune et la jeunesse ; il aimait beaucoup les gens des châteaux et se trouvait pris devant eux d'une si grande peur de leur déplaire qu'il n'osait pas leur laisser voir qu'il avait pour amis des bourgeois, des fils de notaires ou d'agents de change, préférant, si la vérité devait se découvrir, que ce fût en son absence, loin de lui et « par défaut » ; il était snob. Sans doute il ne disait jamais rien de tout cela dans le langage que mes parents et moi-même nous aimions tant. Et si je demandais : « Connaissez-vous les Guermantes ? », Legrandin le causeur répondait : « Non, je n'ai jamais voulu les connaître. » Malheureusement il ne le répondait qu'en second, car un autre Legrandin qu'il cachait soigneusement au fond de lui, qu'il ne montrait pas, parce que ce Legrandin-là savait sur le nôtre, sur son snobisme, des histoires compromettantes, un autre Legrandin avait déjà répondu par la blessure du regard, par le rictus de la bouche, par la gravité excessive du ton de la réponse, par les mille flèches dont notre Legrandin s'était trouvé en un instant lardé et alangui, comme un saint Sébastien du snobisme : « Hélas ! que vous me faites mal, non je ne connais pas les Guermantes, ne réveillez pas la grande douleur de ma vie. » Et comme ce Legrandin enfant terrible, ce Legrandin maître chanteur, s'il n'avait pas le joli langage de l'autre, avait le verbe infiniment plus prompt, composé de ce qu'on appelle « réflexes », quand Legrandin le causeur voulait lui imposer silence, l'autre avait déjà parlé et notre ami avait beau se désoler de la mauvaise impression que les révélations de son alter ego avaient dû produire, il ne pouvait qu'entreprendre de la pallier.

Et certes cela ne veut pas dire que Legrandin ne fût pas sincère quand il tonnait contre les snobs. Il ne pouvait pas savoir, au moins par lui-même, qu'il le fût, puisque nous ne connaissons jamais que les passions des autres, et que ce que nous arrivons à savoir des nôtres, ce n'est que d'eux que nous avons pu l'apprendre. Sur nous, elles n'agissent que d'une façon seconde, par l'imagination qui substitue aux premiers mobiles des mobiles de relais qui sont plus décents. Jamais le snobisme de Legrandin ne lui conseillait d'aller voir souvent une duchesse. Il chargeait l'imagination de Legrandin de lui faire apparaître cette duchesse comme parée de toutes les grâces. Legrandin se rapprochait de la duchesse, s'estimant de céder à cet attrait de l'esprit et de la vertu qu'ignorent les infâmes snobs. Seuls les autres savaient qu'il en était un ; car, grâce à l'incapacité où ils étaient de comprendre le travail intermédiaire de son imagination, ils voyaient en face l'une de l'autre l'activité mondaine de Legrandin et sa cause première.
