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
      "canonical_name": "Robert de Saint-Loup",
      "surface_forms": [
        "Robert de Saint-Loup"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Robert de Saint-Loup",
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.93,
      "evidence": "« Or la sincérité et le désintéressement de Robert de Saint-Loup étaient au contraire absolus... cette grande pureté morale... le rendait vraiment capable... d’amitié. »",
      "explanation": "The narrator explicitly exalts the probity and moral purity of Robert de Saint-Loup, in contrast with Françoise's biased perceptions."
    }
  ],
  "status_effects": [
    {
      "character": "Robert de Saint-Loup",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.93,
      "explanation": "The narrator highlights his sincerity and selflessness, which clearly elevates his moral status in this passage."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p2-noms-de-pays-le-pays#p-271-p-275"
}

### Candidate characters

[
  "Bloch",
  "Bloch père",
  "Françoise",
  "Odette",
  "baron de Charlus",
  "le narrateur"
]

### Prior local context (optional)

– Robert de Saint-Loup au casque d'airain, dit Bloch, reprenez un peu de ce canard aux cuisses lourdes de graisse sur lesquelles l'illustre sacrificateur des volailles a répandu de nombreuses libations de vin rouge.

### Passage

D'habitude, après avoir sorti de derrière les fagots pour un camarade de marque les histoires sur sir Rufus Israël et autres, Bloch sentant qu'il avait touché son fils jusqu'à l'attendrissement, se retirait pour ne pas se « galvauder » aux yeux du « potache ». Cependant s'il y avait une raison tout à fait capitale, comme quand son fils par exemple fut reçu à l'agrégation, Bloch ajouta à la série habituelle des anecdotes cette réflexion ironique qu'il réservait plutôt pour ses amis personnels et que Bloch jeune fut extrêmement fier de voir débiter pour ses amis à lui : « Le gouvernement a été impardonnable. Il n'a pas consulté M. Coquelin ! M. Coquelin a fait savoir qu'il était mécontent. » (Bloch se piquait d'être réactionnaire et méprisant pour les gens de théâtre).

Mais les demoiselles Bloch et leur frère rougirent jusqu'aux oreilles tant ils furent impressionnés quand Bloch père, pour se montrer royal jusqu'au bout envers les deux « labadens » de son fils, donna l'ordre d'apporter du champagne et annonça négligemment que pour nous « régaler », il avait fait prendre trois fauteuils pour la représentation qu'une troupe d'Opéra Comique donnait le soir même au Casino. Il regrettait de n'avoir pu avoir de loge. Elles étaient toutes prises. D'ailleurs il les avait souvent expérimentées, on était mieux à l'orchestre. Seulement, si le défaut de son fils, c'est-à-dire ce que son fils croyait invisible aux autres, était la grossièreté, celui du père était l'avarice. Aussi, c'est dans une carafe qu'il fit servir sous le nom de champagne un petit vin mousseux et sous celui de fauteuils d'orchestre il avait fait prendre des parterres qui coûtaient moitié moins, miraculeusement persuadé par l'intervention divine de son défaut que ni à table, ni au théâtre (où toutes les loges étaient vides) on ne s'apercevrait de la différence. Quand Bloch nous eut laissé tremper nos lèvres dans les coupes plates que son fils décorait du nom de « cratères aux flancs profondément creusés », il nous fit admirer un tableau qu'il aimait tant qu'il l'apportait avec lui à Balbec. Il nous dit que c'était un Rubens. Saint-Loup lui demanda naïvement s'il était signé. Bloch répondit en rougissant qu'il avait fait couper la signature à cause du cadre, ce qui n'avait pas d'importance, puisqu'il ne voulait pas le vendre. Puis il nous congédia rapidement pour se plonger dans le Journal Officiel dont les numéros encombraient la maison et dont la lecture lui était rendue nécessaire, nous dit-il, « par sa situation parlementaire » sur la nature exacte de laquelle il ne nous fournit pas de lumières. « Je prends un foulard, nous dit Bloch, car Zéphyros et Boréas se disputent à qui mieux mieux la mer poissonneuse, et pour peu que nous nous attardions après le spectacle, nous ne rentrerons qu'aux premières lueurs d'Eôs aux doigts de pourpre. À propos, demanda-t-il à Saint-Loup, quand nous fûmes dehors (et je tremblai car je compris bien vite que c'était de Charlus que Bloch parlait sur ce ton ironique), quel était cet excellent fantoche en costume sombre que je vous ai vu promener avant-hier matin sur la plage ? – C'est mon oncle », répondit Saint-Loup piqué. Malheureusement, une « gaffe » était bien loin de paraître à Bloch chose à éviter. Il se tordit de rire : « Tous mes compliments, j'aurais dû le deviner, il a un excellent chic, et une impayable bobine de gaga de la plus haute lignée. – Vous vous trompez du tout au tout, il est très intelligent, riposta Saint-Loup furieux. – Je le regrette car alors il est moins complet. J'aimerais du reste beaucoup le connaître car je suis sûr que j'écrirais des machines adéquates sur des bonshommes comme ça. Celui-là, à voir passer, est crevant. Mais je négligerais le côté caricatural, au fond assez méprisable pour un artiste épris de la beauté plastique des phrases, de la binette qui, excusez-moi, m'a fait gondoler un bon moment, et je mettrais en relief le côté aristocratique de votre oncle, qui en somme fait un effet boeuf, et la première rigolade passée, frappe par un très grand style. Mais, dit-il, en s'adressant cette fois à moi, il y a une chose, dans un tout autre ordre d'idées, sur laquelle je veux t'interroger et chaque fois que nous sommes ensemble, quelque dieu, bienheureux habitant de l'Olympe, me fait oublier totalement de te demander ce renseignement qui eût pu m'être déjà et me sera sûrement fort utile. Quelle est donc cette belle personne avec laquelle je t'ai rencontré au Jardin d'Acclimatation et qui était accompagnée d'un monsieur que je crois connaître de vue et d'une jeune fille à la longue chevelure ? » J'avais bien vu que Odette ne se rappelait pas le nom de Bloch, puisqu'elle m'en avait dit un autre et avait qualifié mon camarade d'attaché à un ministère où je n'avais jamais pensé depuis à m'informer s'il était entré. Mais comment Bloch qui, à ce qu'elle m'avait dit alors, s'était fait présenter à elle pouvait-il ignorer son nom. J'étais si étonné que je restai un moment sans répondre. « En tous cas, tous mes compliments, me dit-il, tu n'as pas dû t'embêter avec elle. Je l'avais rencontrée quelques jours auparavant dans le train de Ceinture. Elle voulut bien dénouer la sienne en faveur de ton serviteur, je n'ai jamais passé de si bons moments et nous allions prendre toutes dispositions pour nous revoir quand une personne qu'elle connaissait eut le mauvais goût de monter à l'avant-dernière station. » Le silence que je gardai ne parut pas plaire à Bloch. « J'espérais, me dit-il, connaître grâce à toi son adresse et aller goûter chez elle, plusieurs fois par semaine, les plaisirs d'Éros, chers aux Dieux, mais je n'insiste pas puisque tu poses pour la discrétion à l'égard d'une professionnelle qui s'est donnée à moi trois fois de suite et de la manière la plus raffinée entre Paris et le Point-du-Jour. Je la retrouverai bien un soir ou l'autre. »

J'allai voir Bloch à la suite de ce dîner, il me rendit ma visite, mais j'étais sorti et il fut aperçu, me demandant, par Françoise, laquelle par hasard bien qu'il fût venu à Combray ne l'avait jamais vu jusque-là. De sorte qu'elle savait seulement qu'un « des Monsieurs » que je connaissais était passé pour me voir, elle ignorait « à quel effet », vêtu d'une manière quelconque et qui ne lui avait pas fait grande impression. Or j'avais beau savoir que certaines idées sociales de Françoise me resteraient toujours impénétrables, qui reposaient peut-être en partie sur des confusions entre des mots, des noms qu'elle avait pris une fois, et à jamais, les uns pour les autres, je ne pus m'empêcher, moi qui avais depuis longtemps renoncé à me poser des questions dans ces cas-là, de chercher, vainement d'ailleurs, ce que le nom de Bloch pouvait représenter d'immense pour Françoise. Car à peine lui eus-je dit que ce jeune homme qu'elle avait aperçu était Bloch, elle recula de quelques pas, tant furent grandes sa stupeur et sa déception. « Comment, c'est cela, Bloch ! » s'écria-t-elle d'un air atterré comme si un personnage aussi prestigieux eût dû posséder une apparence qui « fît connaître » immédiatement qu'on se trouvait en présence d'un grand de la terre, et à la façon de quelqu'un qui trouve qu'un personnage historique n'est pas à la hauteur de sa réputation, elle répétait d'un ton impressionné, et où on sentait pour l'avenir les germes d'un scepticisme universel : « Comment, c'est ça Bloch ! Ah ! vraiment on ne dirait pas à le voir. » Elle avait l'air de m'en garder rancune comme si je lui eusse jamais « surfait » Bloch. Et pourtant elle eut la bonté d'ajouter : « Hé bien, tout Bloch qu'il est, Monsieur peut dire qu'il est aussi bien que lui. »

Elle eut bientôt à l'égard de Saint-Loup qu'elle adorait une désillusion d'un autre genre, et d'une moindre dureté : elle apprit qu'il était républicain. Or bien qu'en parlant par exemple de la Reine de Portugal, elle dît avec cet irrespect qui dans le peuple est le respect suprême « Amélie, la soeur à Philippe », Françoise était royaliste. Mais surtout un marquis, un marquis qui l'avait éblouie, et qui était pour la République, ne lui paraissait plus vrai. Elle en marquait la même mauvaise humeur que si je lui eusse donné une boîte qu'elle eût cru d'or, de laquelle elle m'eût remercié avec effusion et qu'ensuite un bijoutier lui eût révélé être en plaqué. Elle retira aussitôt son estime à Saint-Loup, mais bientôt après la lui rendit, ayant réfléchi qu'il ne pouvait pas, étant le marquis de Saint-Loup, être républicain, qu'il faisait seulement semblant, par intérêt, car avec le gouvernement qu'on avait, cela pouvait lui rapporter gros. De ce jour sa froideur envers lui, son dépit contre moi cessèrent. Et quand elle parlait de Saint-Loup, elle disait : « C'est un hypocrite », avec un large et bon sourire qui faisait bien comprendre qu'elle le « considérait » de nouveau autant qu'au premier jour et qu'elle lui avait pardonné.

Or la sincérité et le désintéressement de Saint-Loup étaient au contraire absolus et c'était cette grande pureté morale qui, ne pouvant se satisfaire entièrement dans un sentiment égoïste comme l'amour, ne rencontrant pas d'autre part en lui l'impossibilité qui existait par exemple en moi de trouver sa nourriture spirituelle autre part qu'en soi-même, le rendait vraiment capable, autant que moi incapable, d'amitié.
