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
    },
    {
      "canonical_name": "Norpois",
      "surface_forms": [
        "Norpois",
        "l'Ambassadeur",
        "Ambassadeur"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "Norpois",
      "target": "Françoise",
      "type": "admiration",
      "polarity": "positive",
      "narrative_stance": "endorsed",
      "confidence": 0.9,
      "evidence": "traitée par l'Ambassadeur de « chef de premier ordre » ... « L'Ambassadeur, lui dit la mère du narrateur, assure que nulle part on ne mange de boeuf froid et de soufflés comme les vôtres. »",
      "explanation": "Norpois explicitly lauds Françoise’s culinary skill; the mother ceremoniously relays the compliment, and the narrator underscores Françoise’s legitimate sense of her art."
    }
  ],
  "status_effects": [
    {
      "character": "Françoise",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Within the household, Françoise’s standing rises as a recognized master of her craft, validated by Norpois’s praise and the narrator’s supportive framing."
    }
  ],
  "ambiguities": [],
  "unit_id": "v2-p1-autour-de-mme-swann#p-96-p-100"
}

### Candidate characters

[
  "Gilberte",
  "Odette",
  "Swann",
  "la Berma",
  "la grand-mère",
  "la mère du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Mais oui, comme tu dis là. J'avais remarqué, c'est très fin. On voit qu'il a une profonde expérience de la vie.

### Passage

– C'est extraordinaire qu'il ait dîné chez les Swann et qu'il y ait trouvé en somme des gens réguliers, des fonctionnaires... Où est-ce que Odette a pu aller pêcher tout ce monde-là ?

– As-tu remarqué avec quelle malice il a fait cette réflexion : « C'est une maison où il va surtout des hommes ! »

Et tous deux cherchaient à reproduire la manière dont Norpois avait dit cette phrase, comme ils auraient fait pour quelque intonation de Bressant ou de Thiron dans l'Aventurière ou dans le Gendre de M. Poirier. Mais de tous ses mots, le plus goûté le fut par Françoise qui, encore plusieurs années après, ne pouvait pas « tenir son sérieux » si on lui rappelait qu'elle avait été traitée par l'Ambassadeur de « chef de premier ordre », ce que ma mère était allée lui transmettre comme un ministre de la guerre les félicitations d'un souverain de passage après « la Revue ». Je l'avais d'ailleurs précédée à la cuisine. Car j'avais fait promettre à Françoise, pacifiste mais cruelle, qu'elle ne ferait pas trop souffrir le lapin qu'elle avait à tuer et je n'avais pas eu de nouvelles de cette mort ; Françoise m'assura qu'elle s'était passée le mieux du monde et très rapidement : « J'ai jamais vu une bête comme ça ; elle est morte sans dire seulement une parole, vous auriez dit qu'elle était muette. » Peu au courant du langage des bêtes, j'alléguai que le lapin ne criait peut-être pas comme le poulet. « Attendez un peu voir, me dit Françoise indignée de mon ignorance, si les lapins ne crient pas autant comme les poulets. Ils ont même la voix bien plus forte. » Françoise accepta les compliments de Norpois avec la fière simplicité, le regard joyeux et – fût-ce momentanément – intelligent, d'un artiste à qui on parle de son art. Ma mère l'avait envoyée autrefois dans certains grands restaurants voir comment on y faisait la cuisine. J'eus ce soir-là à l'entendre traiter les plus célèbres de gargotes le même plaisir qu'autrefois à apprendre, pour les artistes dramatiques, que la hiérarchie de leurs mérites n'était pas la même que celle de leurs réputations. « L'Ambassadeur, lui dit ma mère, assure que nulle part on ne mange de boeuf froid et de soufflés comme les vôtres. » Françoise, avec un air de modestie et de rendre hommage à la vérité, l'accorda, sans être, d'ailleurs, impressionnée par le titre d'ambassadeur ; elle disait de Norpois, avec l'amabilité due à quelqu'un qui l'avait prise pour un « chef » : « C'est un bon vieux comme moi. » Elle avait bien cherché à l'apercevoir quand il était arrivé, mais sachant que maman détestait qu'on fût derrière les portes ou aux fenêtres et pensant qu'elle saurait par les autres domestiques ou par les concierges qu'elle avait fait le guet (car Françoise ne voyait partout que « jalousies » et « racontages » qui jouaient dans son imagination le même rôle permanent et funeste que, pour telles autres personnes, les intrigues des jésuites ou des juifs), elle s'était contentée de regarder par la croisée de la cuisine, « pour ne pas avoir des raisons avec Madame », et sous l'aspect sommaire de Norpois elle avait « cru voir Monsieur Legrand », à cause de son agileté, et bien qu'il n'y eût pas un trait commun entre eux. « Mais enfin, lui demanda ma mère, comment expliquez-vous que personne ne fasse la gelée aussi bien que vous (quand vous le voulez) ? – Je ne sais pas d'où ce que ça devient », répondit Françoise (qui n'établissait pas une démarcation bien nette entre le verbe venir, au moins pris dans certaines acceptions, et le verbe devenir). Elle disait vrai du reste, en partie, et n'était pas beaucoup plus capable – ou désireuse – de dévoiler le mystère qui faisait la supériorité de ses gelées ou de ses crèmes, qu'une grande élégante pour ses toilettes, ou une grande cantatrice pour son chant. Leurs explications ne nous disent pas grand'chose ; il en était de même des recettes de notre cuisinière. « Ils font cuire trop à la va-vite, répondit-elle en parlant des grands restaurateurs, et puis pas tout ensemble. Il faut que le boeuf, il devienne comme une éponge, alors il boit tout le jus jusqu'au fond. Pourtant il y avait un de ces Cafés où il me semble qu'on savait bien un peu faire la cuisine. Je ne dis pas que c'était tout à fait ma gelée, mais c'était fait bien doucement et les soufflés ils avaient bien de la crème. – Est-ce Henry ? demanda mon père qui nous avait rejoints et appréciait beaucoup le restaurant de la place Gaillon où il avait à dates fixes des repas de corps. – Oh non ! dit Françoise avec une douceur qui cachait un profond dédain, je parlais d'un petit restaurant. Chez cet Henry c'est très bon bien sûr, mais c'est pas un restaurant, c'est plutôt... un bouillon ! – Weber ? – Ah ! non, Monsieur, je voulais dire un bon restaurant. Weber c'est dans la rue Royale, ce n'est pas un restaurant, c'est une brasserie. Je ne sais pas si ce qu'ils vous donnent est servi. Je crois qu'ils n'ont même pas de nappe, ils posent cela comme cela sur la table, va comme je te pousse. – Cirro ? » Françoise sourit : « Oh ! là je crois qu'en fait de cuisine il y a surtout des dames du monde. (Monde signifiait pour Françoise demi-monde.) Dame, il faut ça pour la jeunesse. » Nous nous apercevions qu'avec son air de simplicité Françoise était pour les cuisiniers célèbres une plus terrible « camarade » que ne peut l'être l'actrice la plus envieuse et la plus infatuée. Nous sentîmes pourtant qu'elle avait un sentiment juste de son art et le respect des traditions, car elle ajouta : « Non, je veux dire un restaurant où c'est qu'il y avait l'air d'avoir une bien bonne petite cuisine bourgeoise. C'est une maison encore assez conséquente. Ça travaillait beaucoup. Ah ! on en ramassait des sous là dedans (Françoise, économe, comptait par sous, non par louis comme les décavés). Madame connaît bien, là-bas, à droite, sur les grands boulevards, un peu en arrière... » Le restaurant dont elle parlait avec cette équité mêlée d'orgueil et de bonhomie, c'était... le Café Anglais.

Quand vint le 1er janvier, je fis d'abord des visites de famille avec maman, qui, pour ne pas me fatiguer, les avait d'avance (à l'aide d'un itinéraire tracé par mon père) classées par quartier plutôt que selon le degré exact de la parenté. Mais à peine entrés dans le salon d'une cousine assez éloignée qui avait comme raison de passer d'abord que sa demeure ne le fût pas de la nôtre, ma mère était épouvantée en voyant, ses marrons glacés ou déguisés à la main, le meilleur ami du plus susceptible de mes oncles auquel il allait rapporter que nous n'avions pas commencé notre tournée par lui. Cet oncle serait sûrement blessé ; il n'eût trouvé que naturel que nous allassions de la Madeleine au Jardin des Plantes où il habitait avant de nous arrêter à Saint-Augustin, pour repartir rue de l'École-de-Médecine.

Les visites finies (ma grand'mère dispensait que nous en fissions chez elle, comme nous y dînions ce jour-là), je courus jusqu'aux Champs-Élysées porter à notre marchande, pour qu'elle la remît à la personne qui venait plusieurs fois par semaine de chez les Swann y chercher du pain d'épices, la lettre que dès le jour où mon amie m'avait fait tant de peine j'avais décidé de lui envoyer au nouvel an, et dans laquelle je lui disais que notre amitié ancienne disparaissait avec l'année finie, que j'oubliais mes griefs et mes déceptions et qu'à partir du 1er janvier, c'était une amitié neuve que nous allions bâtir, si solide que rien ne la détruirait, si merveilleuse que j'espérais que Gilberte mettrait quelque coquetterie à lui garder toute sa beauté et à m'avertir à temps, comme je promettais de le faire moi-même, aussitôt que surviendrait le moindre péril qui pourrait l'endommager. En rentrant, Françoise me fit arrêter, au coin de la rue Royale, devant un étalage en plein vent où elle choisit, pour ses propres étrennes, des photographies de Pie IX et de Raspail et où, pour ma part, j'en achetai une de la Berma. Les innombrables admirations qu'excitait l'artiste donnaient quelque chose d'un peu pauvre à ce visage unique qu'elle avait pour y répondre, immuable et précaire comme ce vêtement des personnes qui n'en ont pas de rechange, et où elle ne pouvait exhiber toujours que le petit pli au-dessus de la lèvre supérieure, le relèvement des sourcils, quelques autres particularités physiques toujours les mêmes qui, en somme, étaient à la merci d'une brûlure ou d'un choc. Ce visage, d'ailleurs, ne m'eût pas à lui seul semblé beau, mais il me donnait l'idée et, par conséquent, l'envie de l'embrasser à cause de tous les baisers qu'il avait dû supporter, et que, du fond de la « carte-album », il semblait appeler encore par ce regard coquettement tendre et ce sourire artificieusement ingénu. Car la Berma devait ressentir effectivement pour bien des jeunes hommes ces désirs qu'elle avouait sous le couvert du personnage de Phèdre, et dont tout, même le prestige de son nom qui ajoutait à sa beauté et prorogeait sa jeunesse, devait lui rendre l'assouvissement si facile. Le soir tombait, je m'arrêtai devant une colonne de théâtre où était affichée la représentation que la Berma donnait pour le 1er janvier. Il soufflait un vent humide et doux. C'était un temps que je connaissais ; j'eus la sensation et le pressentiment que le jour de l'an n'était pas un jour différent des autres, qu'il n'était pas le premier d'un monde nouveau où j'aurais pu, avec une chance encore intacte, refaire la connaissance de Gilberte comme au temps de la Création, comme s'il n'existait pas encore de passé, comme si eussent été anéanties, avec les indices qu'on aurait pu en tirer pour l'avenir, les déceptions qu'elle m'avait parfois causées : un nouveau monde où rien ne subsistât de l'ancien... rien qu'une chose : mon désir que Gilberte m'aimât. Je compris que si mon coeur souhaitait ce renouvellement autour de lui d'un univers qui ne l'avait pas satisfait, c'est que lui, mon coeur, n'avait pas changé, et je me dis qu'il n'y avait pas de raison pour que celui de Gilberte eût changé davantage ; je sentis que cette nouvelle amitié c'était la même, comme ne sont pas séparées des autres par un fossé les années nouvelles que notre désir, sans pouvoir les atteindre et les modifier, recouvre à leur insu d'un nom différent. J'avais beau dédier celle-ci à Gilberte, et comme on superpose une religion aux lois aveugles de la nature essayer d'imprimer au jour de l'an l'idée particulière que je m'étais faite de lui, c'était en vain ; je sentais qu'il ne savait pas qu'on l'appelât le jour de l'an, qu'il finissait dans le crépuscule d'une façon qui ne m'était pas nouvelle : dans le vent doux qui soufflait autour de la colonne d'affiches, j'avais reconnu, j'avais senti reparaître la matière éternelle et commune, l'humidité familière, l'ignorante fluidité des anciens jours.
