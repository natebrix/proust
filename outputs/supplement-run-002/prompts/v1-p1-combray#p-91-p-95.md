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
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [],
  "status_effects": [],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-91-p-95"
}

### Candidate characters

[
  "Swann",
  "duchesse de Guermantes",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

– Mais à l'église, ils doivent y être déjà ; vous ferez bien de ne pas perdre de temps. Allez surveiller votre déjeuner.

### Passage

Pendant que ma tante devisait ainsi avec Françoise, j'accompagnais mes parents à la messe. Que je l'aimais, que je la revois bien, notre église ! Son vieux porche par lequel nous entrions, noir, grêlé comme une écumoire, était dévié et profondément creusé aux angles (de même que le bénitier où il nous conduisait) comme si le doux effleurement des mantes des paysannes entrant à l'église et de leurs doigts timides prenant de l'eau bénite, pouvait, répété pendant des siècles, acquérir une force destructive, infléchir la pierre et l'entailler de sillons comme en trace la roue des carrioles dans la borne contre laquelle elle bute tous les jours. Ses pierres tombales, sous lesquelles la noble poussière des abbés de Combray, enterrés là, faisait au choeur comme un pavage spirituel, n'étaient plus elles-mêmes de la matière inerte et dure, car le temps les avait rendues douces et fait couler comme du miel hors des limites de leur propre équarrissure qu'ici elles avaient dépassées d'un flot blond, entraînant à la dérive une majuscule gothique en fleurs, noyant les violettes blanches du marbre ; et en deçà desquelles, ailleurs, elles s'étaient résorbées, contractant encore l'elliptique inscription latine, introduisant un caprice de plus dans la disposition de ces caractères abrégés, rapprochant deux lettres d'un mot dont les autres avaient été démesurément distendues. Ses vitraux ne chatoyaient jamais tant que les jours où le soleil se montrait peu, de sorte que, fît-il gris dehors, on était sûr qu'il ferait beau dans l'église ; l'un était rempli dans toute sa grandeur par un seul personnage pareil à un Roi de jeu de cartes, qui vivait là-haut, sous un dais architectural, entre ciel et terre ; (et dans le reflet oblique et bleu duquel, parfois les jours de semaine, à midi, quand il n'y a pas d'office – à l'un de ces rares moments où l'église aérée, vacante, plus humaine, luxueuse, avec du soleil sur son riche mobilier, avait l'air presque habitable comme le hall de pierre sculptée et de verre peint, d'un hôtel de style moyen âge – on voyait s'agenouiller un instant Mme Sazerat, posant sur le prie-Dieu voisin un paquet tout ficelé de petits fours qu'elle venait de prendre chez le pâtissier d'en face et qu'elle allait rapporter pour le déjeuner) ; dans un autre une montagne de neige rose, au pied de laquelle se livrait un combat, semblait avoir givré à même la verrière qu'elle boursouflait de son trouble grésil comme une vitre à laquelle il serait resté des flocons éclairés par quelque aurore (par la même sans doute qui empourprait le retable de l'autel de tons si frais qu'ils semblaient plutôt posés là momentanément par une lueur du dehors prête à s'évanouir que par des couleurs attachées à jamais à la pierre) ; et tous étaient si anciens qu'on voyait çà et là leur vieillesse argentée étinceler de la poussière des siècles et montrer brillante et usée jusqu'à la corde la trame de leur douce tapisserie de verre. Il y en avait un qui était un haut compartiment divisé en une centaine de petits vitraux rectangulaires où dominait le bleu, comme un grand jeu de cartes pareil à ceux qui devaient distraire le roi Swann VI ; mais soit qu'un rayon eût brillé, soit que mon regard en bougeant eût promené à travers la verrière tour à tour éteinte et rallumée un mouvant et précieux incendie, l'instant d'après elle avait pris l'éclat changeant d'une traîne de paon, puis elle tremblait et ondulait en une pluie flamboyante et fantastique qui dégouttait du haut de la voûte sombre et rocheuse, le long des parois humides, comme si c'était dans la nef de quelque grotte irisée de sinueux stalactites que je suivais mes parents, qui portaient leur paroissien ; un instant après les petits vitraux en losange avaient pris la transparence profonde, l'infrangible dureté de saphirs qui eussent été juxtaposés sur quelque immense pectoral, mais derrière lesquels on sentait, plus aimé que toutes ces richesses, un sourire momentané de soleil ; il était aussi reconnaissable dans le flot bleu et doux dont il baignait les pierreries que sur le pavé de la place ou la paille du marché ; et, même à nos premiers dimanches quand nous étions arrivés avant Pâques, il me consolait que la terre fût encore nue et noire, en faisant épanouir, comme en un printemps historique et qui datait des successeurs de saint Louis, ce tapis éblouissant et doré de myosotis en verre.

Deux tapisseries de haute lice représentaient le couronnement d'Esther (la tradition voulait qu'on eût donné à Assuérus les traits d'un roi de France et à Esther ceux d'une dame de Guermantes dont il était amoureux) auxquelles leurs couleurs, en fondant, avaient ajouté une expression, un relief, un éclairage : un peu de rose flottait aux lèvres d'Esther au delà du dessin de leur contour ; le jaune de sa robe s'étalait si onctueusement, si grassement, qu'elle en prenait une sorte de consistance et s'enlevait vivement sur l'atmosphère refoulée ; et la verdure des arbres restée vive dans les parties basses du panneau de soie et de laine, mais ayant « passé » dans le haut, faisait se détacher en plus pâle, au-dessus des troncs foncés, les hautes branches jaunissantes, dorées et comme à demi effacées par la brusque et oblique illumination d'un soleil invisible. Tout cela, et plus encore les objets précieux venus à l'église de personnages qui étaient pour moi presque des personnages de légende (la croix d'or travaillée, disait-on, par saint Éloi et donnée par Dagobert, le tombeau des fils de Louis le Germanique, en porphyre et en cuivre émaillé), à cause de quoi je m'avançais dans l'église, quand nous gagnions nos chaises, comme dans une vallée visitée des fées, où le paysan s'émerveille de voir dans un rocher, dans un arbre, dans une mare, la trace palpable de leur passage surnaturel ; tout cela faisait d'elle pour moi quelque chose d'entièrement différent du reste de la ville : un édifice occupant, si l'on peut dire, un espace à quatre dimensions – la quatrième étant celle du Temps – déployant à travers les siècles son vaisseau qui, de travée en travée, de chapelle en chapelle, semblait vaincre et franchir, non pas seulement quelques mètres, mais des époques successives d'où il sortait victorieux ; dérobant le rude et farouche XIe siècle dans l'épaisseur de ses murs, d'où il n'apparaissait avec ses lourds cintres bouchés et aveuglés de grossiers moellons que par la profonde entaille que creusait près du porche l'escalier du clocher, et, même là, dissimulé par les gracieuses arcades gothiques qui se pressaient coquettement devant lui comme de plus grandes soeurs, pour le cacher aux étrangers, se placent en souriant devant un jeune frère rustre, grognon et mal vêtu ; élevant dans le ciel au-dessus de la Place, sa tour qui avait contemplé saint Louis et semblait le voir encore ; et s'enfonçant avec sa crypte dans une nuit mérovingienne où, nous guidant à tâtons sous la voûte obscure et puissamment nervurée comme la membrane d'une immense chauve-souris de pierre, Théodore et sa soeur nous éclairaient d'une bougie le tombeau de la petite fille de Sigebert, sur lequel une profonde valve – comme la trace d'un fossile – avait été creusée, disait-on, « par une lampe de cristal qui, le soir du meurtre de la princesse franque, s'était détachée d'elle-même des chaînes d'or où elle était suspendue à la place de l'actuelle abside, et, sans que le cristal se brisât, sans que la flamme s'éteignît, s'était enfoncée dans la pierre et l'avait fait mollement céder sous elle ».

L'abside de l'église de Combray, peut-on vraiment en parler ? Elle était si grossière, si dénuée de beauté artistique et même d'élan religieux. Du dehors, comme le croisement des rues sur lequel elle donnait était en contre-bas, sa grossière muraille s'exhaussait d'un soubassement en moellons nullement polis, hérissés de cailloux, et qui n'avait rien de particulièrement ecclésiastique, les verrières semblaient percées à une hauteur excessive, et le tout avait plus l'air d'un mur de prison que d'église. Et certes, plus tard, quand je me rappelais toutes les glorieuses absides que j'ai vues, il ne me serait jamais venu à la pensée de rapprocher d'elles l'abside de Combray. Seulement, un jour, au détour d'une petite rue provinciale, j'aperçus, en face du croisement de trois ruelles, une muraille fruste et surélevée, avec des verrières percées en haut et offrant le même aspect asymétrique que l'abside de Combray. Alors je ne me suis pas demandé comme à Chartres ou à Reims avec quelle puissance y était exprimé le sentiment religieux, mais je me suis involontairement écrié : « L'Église ! »

L'église ! Familière ; mitoyenne, rue Saint-Hilaire, où était sa porte nord, de ses deux voisines, la pharmacie de M. Rapin et la maison de Mme Loiseau, qu'elle touchait sans aucune séparation ; simple citoyenne de Combray qui aurait pu avoir son numéro dans la rue si les rues de Combray avaient eu des numéros, et où il semble que le facteur aurait dû s'arrêter le matin quand il faisait sa distribution, avant d'entrer chez Mme Loiseau et en sortant de chez M. Rapin, il y avait pourtant entre elle et tout ce qui n'était pas elle une démarcation que mon esprit n'a jamais pu arriver à franchir. Mme Loiseau avait beau avoir à sa fenêtre des fuchsias, qui prenaient la mauvaise habitude de laisser leurs branches courir toujours partout tête baissée, et dont les fleurs n'avaient rien de plus pressé, quand elles étaient assez grandes, que d'aller rafraîchir leurs joues violettes et congestionnées contre la sombre façade de l'église, les fuchsias ne devenaient pas sacrés pour cela pour moi ; entre les fleurs et la pierre noircie sur laquelle elles s'appuyaient, si mes yeux ne percevaient pas d'intervalle, mon esprit réservait un abîme.

On reconnaissait le clocher de Saint-Hilaire de bien loin, inscrivant sa figure inoubliable à l'horizon où Combray n'apparaissait pas encore ; quand du train qui, la semaine de Pâques, nous amenait de Paris, mon père l'apercevait qui filait tour à tour sur tous les sillons du ciel, faisant courir en tous sens son petit coq de fer, il nous disait : « Allons, prenez les couvertures, on est arrivé. » Et dans une des plus grandes promenades que nous faisions de Combray, il y avait un endroit où la route resserrée débouchait tout à coup sur un immense plateau fermé à l'horizon par des forêts déchiquetées que dépassait seul la fine pointe du clocher de Saint-Hilaire, mais si mince, si rose, qu'elle semblait seulement rayée sur le ciel par un ongle qui aurait voulu donner à ce paysage, à ce tableau rien que de nature, cette petite marque d'art, cette unique indication humaine. Quand on se rapprochait et qu'on pouvait apercevoir le reste de la tour carrée et à demi détruite qui, moins haute, subsistait à côté de lui, on était frappé surtout du ton rougeâtre et sombre des pierres ; et, par un matin brumeux d'automne, on aurait dit, s'élevant au-dessus du violet orageux des vignobles, une ruine de pourpre presque de la couleur de la vigne vierge.
