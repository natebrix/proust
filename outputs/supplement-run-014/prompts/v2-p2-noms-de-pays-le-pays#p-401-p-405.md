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
      "canonical_name": "Elstir",
      "surface_forms": [
        "Elstir"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Elstir",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "Depuis que j'en avais vu dans des aquarelles d'Elstir, je cherchais à retrouver dans la réalité... j'essayais de trouver la beauté là où je ne m'étais jamais figuré qu'elle fût, dans les choses les plus usuelles, dans la vie profonde des « natures mortes ».",
      "explanation": "The narrator credits Elstir’s watercolors with teaching him to perceive poetic beauty in ordinary objects, signaling strong aesthetic admiration and authority."
    }
  ],
  "status_effects": [
    {
      "character": "Elstir",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.9,
      "explanation": "Elstir’s standing rises as the narrator portrays his art as reshaping perception and revealing beauty in the everyday."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-401-p-405"
}

### Candidate characters

[
  "Albertine",
  "Robert de Saint-Loup",
  "le narrateur"
]

### Prior local context (optional)

Au fond cette lettre ressemblait beaucoup par sa tendresse à celles que, quand je ne connaissais pas encore Robert de Saint-Loup, je m'étais imaginé qu'il m'écrirait, dans ces songeries d'où la froideur de son premier accueil m'avait tiré en me mettant en présence d'une réalité glaciale qui ne devait pas être définitive. Une fois que je l'eus reçue, chaque fois qu'à l'heure du déjeuner on apportait le courrier, je reconnaissais tout de suite quand c'était de lui que venait une lettre, car elle avait toujours ce second visage qu'un être montre quand il est absent et dans les traits duquel (les caractères de l'écriture) il n'y a aucune raison pour que nous ne croyions pas saisir une âme individuelle aussi bien que dans la ligne du nez ou les inflexions de la voix.

### Passage

Je restais maintenant volontiers à table pendant qu'on desservait, et si ce n'était pas un moment où les jeunes filles de la petite bande pouvaient passer, ce n'était plus uniquement du côté de la mer que je regardais. Depuis que j'en avais vu dans des aquarelles d'Elstir, je cherchais à retrouver dans la réalité, j'aimais comme quelque chose de poétique, le geste interrompu des couteaux encore de travers, la rondeur bombée d'une serviette défaite où le soleil intercale un morceau de velours jaune, le verre à demi vidé qui montre mieux ainsi le noble évasement de ses formes, et au fond de son vitrage translucide et pareil à une condensation du jour, un reste de vin sombre, mais scintillant de lumières, le déplacement des volumes, la transmutation des liquides par l'éclairage, l'altération des prunes qui passent du vert au bleu et du bleu à l'or dans le compotier déjà à demi dépouillé, la promenade des chaises vieillottes qui deux fois par jour viennent s'installer autour de la nappe dressée sur la table ainsi que sur un autel où sont célébrées les fêtes de la gourmandise, et sur laquelle au fond des huîtres quelques gouttes d'eau lustrale restent comme dans de petits bénitiers de pierre ; j'essayais de trouver la beauté là où je ne m'étais jamais figuré qu'elle fût, dans les choses les plus usuelles, dans la vie profonde des « natures mortes ».

Quand quelques jours après le départ de Saint-Loup, j'eus réussi à ce qu'Elstir donnât une petite matinée où je rencontrerais Albertine, le charme et l'élégance tout momentanés qu'on me trouva au moment où je sortais du Grand-Hôtel (et qui était dus à un repos prolongé, à des frais de toilette spéciaux), je regrettai de ne pas pouvoir les réserver (et aussi le crédit d'Elstir) pour la conquête de quelque autre personne plus intéressante, je regrettai de consommer tout cela pour le simple plaisir de faire la connaissance d'Albertine. Mon intelligence jugeait ce plaisir fort peu précieux, depuis qu'il était assuré. Mais en moi la volonté ne partagea pas un instant cette illusion, la volonté qui est le serviteur, persévérant et immuable, de nos personnalités successives ; cachée dans l'ombre, dédaignée, inlassablement fidèle, travaillant sans cesse, et sans se soucier des variations de notre moi, à ce qu'il ne manque jamais du nécessaire. Pendant qu'au moment où va se réaliser un voyage désiré, l'intelligence et la sensibilité commencent à se demander s'il vaut vraiment la peine d'être entrepris, la volonté qui sait que ces maîtres oisifs recommenceraient immédiatement à trouver merveilleux ce voyage, si celui-ci ne pouvait avoir lieu, la volonté les laisse disserter devant la gare, multiplier les hésitations ; mais elle s'occupe de prendre les billets et de nous mettre en wagon pour l'heure du départ. Elle est aussi invariable que l'intelligence et la sensibilité sont changeantes, mais comme elle est silencieuse, ne donne pas ses raisons, elle semble presque inexistante ; c'est sa ferme détermination que suivent les autres parties de notre moi, mais sans l'apercevoir, tandis qu'elles distinguent nettement leurs propres incertitudes. Ma sensibilité et mon intelligence instituèrent donc une discussion sur la valeur du plaisir qu'il y aurait à connaître Albertine tandis que je regardais dans la glace de vains et fragiles agréments qu'elles eussent voulu garder intacts pour une autre occasion. Mais ma volonté ne laissa pas passer l'heure où il fallait partir, et ce fut l'adresse d'Elstir qu'elle donna au cocher. Mon intelligence et ma sensibilité eurent le loisir, puisque le sort en était jeté, de trouver que c'était dommage. Si ma volonté avait donné une autre adresse, elles eussent été bien attrapées.

Quand j'arrivai chez Elstir, un peu plus tard, je crus d'abord que Mlle Simonet n'était pas dans l'atelier. Il y avait bien une jeune fille assise, en robe de soie, nu-tête, mais de laquelle je ne connaissais pas la magnifique chevelure, ni le nez, ni ce teint, et où je ne retrouvais pas l'entité que j'avais extraite d'une jeune cycliste se promenant coiffée d'un polo, le long de la mer. C'était pourtant Albertine. Mais même quand je le sus, je ne m'occupai pas d'elle. En entrant dans toute réunion mondaine, quand on est jeune, on meurt à soi-même, on devient un homme différent, tout salon étant un nouvel univers où, subissant la loi d'une autre perspective morale, on darde son attention, comme si elles devaient nous importer à jamais, sur des personnes, des danses, des parties de cartes, que l'on aura oubliées le lendemain. Obligé de suivre, pour me diriger vers une causerie avec Albertine, un chemin nullement tracé par moi et qui s'arrêtait d'abord devant Elstir, passait par d'autres groupes d'invités à qui on me nommait, puis le long du buffet, où m'étaient offertes, et où je mangeais, des tartes aux fraises, cependant que j'écoutais, immobile, une musique qu'on commençait d'exécuter, je me trouvais donner à ces divers épisodes la même importance qu'à ma présentation à Mlle Simonet, présentation qui n'était plus que l'un d'entre eux et que j'avais entièrement oublié d'avoir été, quelques minutes auparavant, le but unique de ma venue. D'ailleurs n'en est-il pas ainsi, dans la vie active, de nos vrais bonheurs, de nos grands malheurs ? Au milieu d'autres personnes, nous recevons de celle que nous aimons la réponse favorable ou mortelle que nous attendions depuis une année. Mais il faut continuer à causer, les idées s'ajoutent les unes aux autres, développant une surface sous laquelle c'est à peine si de temps à autre vient sourdement affleurer le souvenir autrement profond, mais fort étroit, que le malheur est venu pour nous. Si, au lieu du malheur, c'est le bonheur, il peut arriver que ce ne soit que plusieurs années après que nous nous rappelons que le plus grand événement de notre vie sentimentale s'est produit, sans que nous eussions le temps de lui accorder une longue attention, presque d'en prendre conscience, dans une réunion mondaine par exemple, et où nous ne nous étions rendus que dans l'attente de cet événement.

Au moment où Elstir me demanda de venir pour qu'il me présentât à Albertine, assise un peu plus loin, je finis d'abord de manger un éclair au café et demandai avec intérêt à un vieux monsieur dont je venais de faire la connaissance et auquel je crus pouvoir offrir la rose qu'il admirait à ma boutonnière, de me donner des détails sur certaines foires normandes. Ce n'est pas à dire que la présentation qui suivit ne me causa aucun plaisir et n'offrit pas, à mes yeux, une certaine gravité. Pour le plaisir, je ne le connus naturellement qu'un peu plus tard, quand, rentré à l'hôtel, resté seul, je fus redevenu moi-même. Il en est des plaisirs comme des photographies. Ce qu'on prend en présence de l'être aimé n'est qu'un cliché négatif, on le développe plus tard, une fois chez soi, quand on a retrouvé à sa disposition cette chambre noire intérieure dont l'entrée est « condamnée » tant qu'on voit du monde.

Si la connaissance du plaisir fut ainsi retardée pour moi de quelques heures, en revanche la gravité de cette présentation, je la ressentis tout de suite. Au moment de la présentation, nous avons beau nous sentir tout à coup gratifiés et porteurs d'un « bon », valable pour des plaisirs futurs, après lequel nous courions depuis des semaines, nous comprenons bien que son obtention met fin pour nous, non pas seulement à de pénibles recherches – ce qui ne pourrait que nous remplir de joie – mais aussi à l'existence d'un certain être, celui que notre imagination avait dénaturé, que notre crainte anxieuse de ne jamais pouvoir être connus de lui avait grandi. Au moment où notre nom résonne dans la bouche du présentateur, surtout si celui-ci l'entoure comme fit Elstir de commentaires élogieux, ce moment sacramentel, analogue à celui où, dans une féerie, le génie ordonne à une personne d'en être soudain une autre, celle que nous avons désiré d'approcher s'évanouit ; d'abord comment resterait-elle pareille à elle-même puisque – de par l'attention que l'inconnue est obligée de prêter à notre nom et de marquer à notre personne – dans les yeux situés à l'infini (et que nous croyions que les nôtres, errants, mal réglés, désespérés, divergents, ne parviendraient jamais à rencontrer) le regard conscient, la pensée inconnaissable que nous cherchions, vient d'être miraculeusement et tout simplement remplacée par notre propre image peinte comme au fond d'un miroir qui sourirait.
