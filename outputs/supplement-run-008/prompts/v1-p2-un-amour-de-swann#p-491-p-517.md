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
  "duchesse de Guermantes": {
    "aliases": [
      "princesse des Laumes",
      "Mme des Laumes",
      "Mme de Guermantes",
      "princesse"
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
  }
}

### Accepted annotation (fixed context)

{
  "characters_present": [
    {
      "canonical_name": "Swann",
      "surface_forms": [
        "Swann",
        "mon petit Swann"
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
      "type": "narrated_elevation",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.82,
      "evidence": "« l'esprit de Swann était extrêmement apprécié dans sa coterie »; la princesse « se mit à rire aux éclats » à son compliment métaphorique, tandis que la marquise de Saint-Euverte le prenait au pied de la lettre.",
      "explanation": "Swann shines by a poetic compliment that the princess savors as a sign of wit recognized in the Guermantes coterie, which locally places him above the less fine interlocutors."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "rhetorical_position",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.82,
      "explanation": "His wit is publicly savored by the princess and set as the coterie’s norm, which strengthens his position in the exchange."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-491-p-517"
}

### Candidate characters

[
  "Legrandin",
  "Odette",
  "baron de Charlus",
  "duc de Guermantes",
  "duchesse de Guermantes",
  "général de Froberville",
  "le narrateur",
  "marquise de Saint-Euverte",
  "princesse de Parme"
]

### Prior local context (optional)

– Ah ! Mais qu'ils aient des choses intéressantes au point de vue de l'histoire, je ne vous dis pas. Mais ça ne peut pas être beau... puisque c'est horrible ! Moi j'ai aussi des choses comme ça que duc de Guermantes a héritées des Montesquiou. Seulement elles sont dans les greniers de Guermantes où personne ne les voit. Enfin, du reste, ce n'est pas la question, je me précipiterais chez eux avec duc de Guermantes, j'irais les voir même au milieu de leurs sphinx et de leur cuivre si je les connaissais, mais... je ne les connais pas ! Moi, on m'a toujours dit quand j'étais petite que ce n'était pas poli d'aller chez les gens qu'on ne connaissait pas, dit-elle en prenant un ton puéril. Alors, je fais ce qu'on m'a appris. Voyez-vous ces braves gens s'ils voyaient entrer une personne qu'ils ne connaissent pas ? Ils me recevraient peut-être très mal ! dit la princesse des Laumes.

### Passage

Et par coquetterie elle embellit le sourire que cette supposition lui arrachait, en donnant à son regard bleu fixé sur le général une expression rêveuse et douce.

– Ah ! princesse, vous savez bien qu'ils ne se tiendraient pas de joie...

– Mais non, pourquoi ? lui demanda-t-elle avec une extrême vivacité, soit pour ne pas avoir l'air de savoir que c'est parce qu'elle était une des plus grandes dames de France, soit pour avoir le plaisir de l'entendre dire au général. Pourquoi ? Qu'en savez-vous ? Cela leur serait peut-être tout ce qu'il y a de plus désagréable. Moi je ne sais pas, mais si j'en juge par moi, cela m'ennuie déjà tant de voir les personnes que je connais, je crois que s'il fallait voir des gens que je ne connais pas, « même héroïques », je deviendrais folle. D'ailleurs, voyons, sauf lorsqu'il s'agit de vieux amis comme vous qu'on connaît sans cela, je ne sais pas si l'héroïsme serait d'un format très portatif dans le monde. Ça m'ennuie déjà souvent de donner des dîners, mais s'il fallait offrir le bras à Spartacus pour aller à table... Non vraiment, ce ne serait jamais à Vercingétorix que je ferais signe comme quatorzième. Je sens que je le réserverais pour les grandes soirées. Et comme je n'en donne pas...

– Ah ! princesse, vous n'êtes pas Guermantes pour des prunes. Le possédez-vous assez, l'esprit des Guermantes !

– Mais on dit toujours l'esprit des Guermantes, je n'ai jamais pu comprendre pourquoi. Vous en connaissez donc d'autres qui en aient, ajouta-t-elle dans un éclat de rire écumant et joyeux, les traits de son visage concentrés, accouplés dans le réseau de son animation, les yeux étincelants, enflammés d'un ensoleillement radieux de gaieté que seuls avaient le pouvoir de faire rayonner ainsi les propos, fussent-ils tenus par la princesse elle-même, qui étaient une louange de son esprit ou de sa beauté. Tenez, voilà Swann qui a l'air de saluer votre Cambremer ; là... il est à côté de la mère Saint-Euverte, vous ne voyez pas ! Demandez-lui de vous présenter. Mais dépêchez-vous, il cherche à s'en aller !

– Avez-vous remarqué quelle affreuse mine il a ? dit le général.

– Mon petit Swann ! Ah ! enfin il vient, je commençais à supposer qu'il ne voulait pas me voir !

Swann aimait beaucoup la princesse des Laumes, puis sa vue lui rappelait Guermantes, terre voisine de Combray, tout ce pays qu'il aimait tant et où il ne retournait plus pour ne pas s'éloigner d'Odette. Usant des formes mi-artistes, mi-galantes, par lesquelles il savait plaire à la princesse et qu'il retrouvait tout naturellement quand il se retrempait un instant dans son ancien milieu – et voulant d'autre part pour lui-même exprimer la nostalgie qu'il avait de la campagne :

– Ah ! dit-il à la cantonade, pour être entendu à la fois de Mme de Saint-Euverte à qui il parlait et de Mme des Laumes pour qui il parlait, voici la charmante princesse ! Voyez, elle est venue tout exprès de Guermantes pour entendre le Saint François d'Assise de Liszt et elle n'a eu le temps, comme une jolie mésange, que d'aller piquer pour les mettre sur sa tête quelques petits fruits de prunier des oiseaux et d'aubépine ; il y a même encore de petites gouttes de rosée, un peu de la gelée blanche qui doit faire gémir la duchesse. C'est très joli, ma chère princesse.

– Comment, la princesse est venue exprès de Guermantes ? Mais c'est trop ! Je ne savais pas, je suis confuse, s'écria naïvement Mme de Saint-Euverte qui était peu habituée au tour d'esprit de Swann. Et examinant la coiffure de la princesse : Mais c'est vrai, cela imite... comment dirais-je, pas les châtaignes, non oh ! c'est une idée ravissante ! Mais comment la princesse pouvait-elle connaître mon programme ! Les musiciens ne me l'ont même pas communiqué à moi.

Swann, habitué quand il était auprès d'une femme avec qui il avait gardé des habitudes galantes de langage, de dire des choses délicates que beaucoup de gens du monde ne comprenaient pas, ne daigna pas expliquer à Mme de Saint-Euverte qu'il n'avait parlé que par métaphore. Quant à la princesse, elle se mit à rire aux éclats, parce que l'esprit de Swann était extrêmement apprécié dans sa coterie, et aussi parce qu'elle ne pouvait entendre un compliment s'adressant à elle sans lui trouver les grâces les plus fines et une irrésistible drôlerie.

– Hé bien ! je suis ravie, Swann, si mes petits fruits d'aubépine vous plaisent. Pourquoi est-ce que vous saluez cette Cambremer, est-ce que vous êtes aussi son voisin de campagne ?

Mme de Saint-Euverte voyant que la princesse avait l'air content de causer avec Swann s'était éloignée.

– Mais vous l'êtes vous-même, princesse.

– Moi, mais ils ont donc des campagnes partout, ces gens ! Mais comme j'aimerais être à leur place !

– Ce ne sont pas les Cambremer, c'étaient ses parents à elle ; elle est une demoiselle Legrandin qui venait à Combray. Je ne sais pas si vous savez que vous êtes comtesse de Combray et que le chapitre vous doit une redevance.

– Je ne sais pas ce que me doit le chapitre mais je sais que je suis tapée de cent francs tous les ans par le curé, ce dont je me passerais. Enfin ces Cambremer ont un nom bien étonnant. Il finit juste à temps, mais il finit mal ! dit-elle en riant.

– Il ne commence pas mieux, répondit Swann.

– En effet cette double abréviation !...

– C'est quelqu'un de très en colère et de très convenable, qui n'a pas osé aller jusqu'au bout du premier mot.

– Mais puisqu'il ne devait pas pouvoir s'empêcher de commencer le second, il aurait mieux fait d'achever le premier pour en finir une bonne fois. Nous sommes en train de faire des plaisanteries d'un goût charmant, mon petit Swann, mais comme c'est ennuyeux de ne plus vous voir, ajouta-t-elle d'un ton câlin, j'aime tant causer avec vous. Pensez que je n'aurais même pas pu faire comprendre à cet idiot de Froberville que le nom de Cambremer était étonnant. Avouez que la vie est une chose affreuse. Il n'y a que quand je vous vois que je cesse de m'ennuyer.

Et sans doute cela n'était pas vrai. Mais Swann et la princesse avaient une même manière de juger les petites choses qui avait pour effet – à moins que ce ne fût pour cause – une grande analogie dans la façon de s'exprimer et jusque dans la prononciation. Cette ressemblance ne frappait pas parce que rien n'était plus différent que leurs deux voix. Mais si on parvenait par la pensée à ôter aux propos de Swann la sonorité qui les enveloppait, les moustaches d'entre lesquelles ils sortaient, on se rendait compte que c'étaient les mêmes phrases, les mêmes inflexions, le tour de la coterie Guermantes. Pour les choses importantes, Swann et la princesse n'avaient les mêmes idées sur rien. Mais depuis que Swann était si triste, ressentant toujours cette espèce de frisson qui précède le moment où l'on va pleurer, il avait le même besoin de parler du chagrin qu'un assassin a de parler de son crime. En entendant la princesse lui dire que la vie était une chose affreuse, il éprouva la même douceur que si elle lui avait parlé d'Odette.

– Oh ! oui, la vie est une chose affreuse. Il faut que nous nous voyions, ma chère amie. Ce qu'il y a de gentil avec vous, c'est que vous n'êtes pas gaie. On pourrait passer une soirée ensemble.

– Mais je crois bien, pourquoi ne viendriez-vous pas à Guermantes, ma belle-mère serait folle de joie. Cela passe pour très laid, mais je vous dirai que ce pays ne me déplaît pas, j'ai horreur des pays « pittoresques ».

– Je crois bien, c'est admirable, répondit Swann, c'est presque trop beau, trop vivant pour moi, en ce moment ; c'est un pays pour être heureux. C'est peut-être parce que j'y ai vécu, mais les choses m'y parlent tellement. Dès qu'il se lève un souffle d'air, que les blés commencent à remuer, il me semble qu'il y a quelqu'un qui va arriver, que je vais recevoir une nouvelle ; et ces petites maisons au bord de l'eau... je serais bien malheureux !

– Oh ! mon petit Swann, prenez garde, voilà l'affreuse Rampillon qui m'a vue, cachez-moi, rappelez-moi donc ce qui lui est arrivé, je confonds, elle a marié sa fille ou son amant, je ne sais plus ; peut-être les deux... et ensemble !... Ah ! non, je me rappelle, elle a été répudiée par son prince... ayez l'air de me parler, pour que cette Bérénice ne vienne pas m'inviter à dîner. Du reste, je me sauve. Écoutez, mon petit Swann, pour une fois que je vous vois, vous ne voulez pas vous laisser enlever et que je vous emmène chez la princesse de Parme qui serait tellement contente, et duc de Guermantes aussi qui doit m'y rejoindre. Si on n'avait pas de vos nouvelles par Mémé... Pensez que je ne vous vois plus jamais !

Swann refusa ; ayant prévenu Charlus qu'en quittant de chez Mme de Saint-Euverte il rentrerait directement chez lui, il ne se souciait pas en allant chez la princesse de Parme de risquer de manquer un mot qu'il avait tout le temps espéré se voir remettre par un domestique pendant la soirée, et que peut-être il allait trouver chez son concierge. « Ce pauvre Swann, dit ce soir-là Mme des Laumes à son mari, il est toujours gentil, mais il a l'air bien malheureux. Vous le verrez, car il a promis de venir dîner un de ces jours. Je trouve ridicule au fond qu'un homme de son intelligence souffre pour une personne de ce genre et qui n'est même pas intéressante, car on la dit idiote », ajouta-t-elle avec la sagesse des gens non amoureux, qui trouvent qu'un homme d'esprit ne devrait être malheureux que pour une personne qui en valût la peine ; c'est à peu près comme s'étonner qu'on daigne souffrir du choléra par le fait d'un être aussi petit que le bacille virgule.
