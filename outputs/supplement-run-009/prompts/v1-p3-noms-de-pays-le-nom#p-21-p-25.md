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
        "Gilberte Swann",
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
      "confidence": 0.86,
      "evidence": "« Tenez, elle est à vous, je vous la donne, gardez-la comme souvenir. »; « Le lendemain elle m'apporta ... la brochure qu'elle avait fait chercher »; « Vous savez, vous pouvez m'appeler Gilberte ... elle la termina par mon petit nom. »",
      "explanation": "The narrator highlights Gilberte's affectionate tokens—gift of the marble, procuring the brochure, and granting first-name intimacy—as small but real advances, which locally cast her in a favorable, generous light."
    },
    {
      "event_id": "E2",
      "source": "narrator",
      "target": "Gilberte",
      "type": "snub",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.8,
      "evidence": "« elle me faisait aussi de la peine en ayant l'air de ne pas avoir de plaisir à me voir »; « Hélas ! aux Champs-Élysées je ne trouvais pas Gilberte, elle n'était pas encore arrivée. »",
      "explanation": "The narrator reports recurring coolness and a key absence on a day of high expectation, which function locally as snubs or setbacks in responsiveness."
    }
  ],
  "status_effects": [
    {
      "character": "Gilberte",
      "dimension": "general_appraisal",
      "delta": 0,
      "based_on_events": [
        "E1",
        "E2"
      ],
      "confidence": 0.78,
      "explanation": "Gilberte is simultaneously elevated by her generosity and intimacy tokens and diminished by episodes of coolness/absence; the net local appraisal is mixed."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p3-noms-de-pays-le-nom#p-21-p-25"
}

### Candidate characters

[
  "Bergotte",
  "Françoise",
  "Swann",
  "la Berma",
  "la mère du narrateur",
  "le narrateur"
]

### Prior local context (optional)

Tout le temps que j'étais loin de Gilberte, j'avais besoin de la voir, parce que cherchant sans cesse à me représenter son image, je finissais par ne plus y réussir, et par ne plus savoir exactement à quoi correspondait mon amour. Puis, elle ne m'avait encore jamais dit qu'elle m'aimait. Bien au contraire, elle avait souvent prétendu qu'elle avait des amis qu'elle me préférait, que j'étais un bon camarade avec qui elle jouait volontiers quoique trop distrait, pas assez au jeu ; enfin elle m'avait donné souvent des marques apparentes de froideur qui auraient pu ébranler ma croyance que j'étais pour elle un être différent des autres, si cette croyance avait pris sa source dans un amour que Gilberte aurait eu pour moi, et non pas, comme cela était, dans l'amour que j'avais pour elle, ce qui la rendait autrement résistante, puisque cela la faisait dépendre de la manière même dont j'étais obligé, par une nécessité intérieure, de penser à Gilberte. Mais les sentiments que je ressentais pour elle, moi-même je ne les lui avais pas encore déclarés. Certes, à toutes les pages de mes cahiers, j'écrivais indéfiniment son nom et son adresse, mais à la vue de ces vagues lignes que je traçais sans qu'elle pensât pour cela à moi, qui lui faisaient prendre autour de moi tant de place apparente sans qu'elle fût mêlée davantage à ma vie, je me sentais découragé parce qu'elles ne me parlaient pas de Gilberte qui ne les verrait même pas, mais de mon propre désir qu'elles semblaient me montrer comme quelque chose de purement personnel, d'irréel, de fastidieux et d'impuissant. Le plus pressé était que nous nous vissions, Gilberte et moi, et que nous puissions nous faire l'aveu réciproque de notre amour, qui jusque-là n'aurait pour ainsi dire pas commencé. Sans doute les diverses raisons qui me rendaient si impatient de la voir auraient été moins impérieuses pour un homme mûr. Plus tard, il arrive que devenus habiles dans la culture de nos plaisirs, nous nous contentions de celui que nous avons à penser à une femme comme je pensais à Gilberte, sans être inquiets de savoir si cette image correspond à la réalité, et aussi de celui de l'aimer sans avoir besoin d'être certain qu'elle nous aime ; ou encore que nous renoncions au plaisir de lui avouer notre inclination pour elle, afin d'entretenir plus vivace l'inclination qu'elle a pour nous, imitant ces jardiniers japonais qui, pour obtenir une plus belle fleur, en sacrifient plusieurs autres. Mais à l'époque où j'aimais Gilberte, je croyais encore que l'Amour existait réellement en dehors de nous ; que, en permettant tout au plus que nous écartions les obstacles, il offrait ses bonheurs dans un ordre auquel on n'était pas libre de rien changer ; il me semblait que si j'avais, de mon chef, substitué à la douceur de l'aveu la simulation de l'indifférence, je ne me serais pas seulement privé d'une des joies dont j'avais le plus rêvé, mais que je me serais fabriqué à ma guise un amour factice et sans valeur, sans communication avec le vrai, dont j'aurais renoncé à suivre les chemins mystérieux et préexistants.

### Passage

Mais quand j'arrivais aux Champs-Élysées – et que d'abord j'allais pouvoir confronter mon amour, pour lui faire subir les rectifications nécessaires, à sa cause vivante, indépendante de moi – dès que j'étais en présence de cette Gilberte Swann sur la vue de laquelle j'avais compté pour rafraîchir les images que ma mémoire fatiguée ne retrouvait plus, de cette Gilberte Swann avec qui j'avais joué hier, et que venait de me faire saluer et reconnaître un instinct aveugle comme celui qui dans la marche nous met un pied devant l'autre avant que nous ayons eu le temps de penser, aussitôt tout se passait comme si elle et la fillette qui était l'objet de mes rêves avaient été deux êtres différents. Par exemple si depuis la veille je portais dans ma mémoire deux yeux de feu dans des joues pleines et brillantes, la figure de Gilberte m'offrait maintenant avec insistance quelque chose que précisément je ne m'étais pas rappelé, un certain effilement aigu du nez qui, s'associant instantanément à d'autres traits, prenait l'importance de ces caractères qui en histoire naturelle définissent une espèce, et la transmuait en une fillette du genre de celles à museau pointu. Tandis que je m'apprêtais à profiter de cet instant désiré pour me livrer, sur l'image de Gilberte que j'avais préparée avant de venir et que je ne retrouvais plus dans ma tête, à la mise au point qui me permettrait dans les longues heures où j'étais seul d'être sûr que c'était bien elle que je me rappelais, que c'était bien mon amour pour elle que j'accroissais peu à peu comme un ouvrage qu'on compose, elle me passait une balle ; et comme le philosophe idéaliste dont le corps tient compte du monde extérieur à la réalité duquel son intelligence ne croit pas, le même moi qui m'avait fait la saluer avant que je l'eusse identifiée, s'empressait de me faire saisir la balle qu'elle me tendait (comme si elle était une camarade avec qui j'étais venu jouer, et non une âme soeur que j'étais venu rejoindre), me faisait lui tenir par bienséance jusqu'à l'heure où elle s'en allait, mille propos, aimables et insignifiants et m'empêchait ainsi, ou de garder le silence pendant lequel j'aurais pu enfin remettre la main sur l'image urgente et égarée, ou de lui dire les paroles qui pouvaient faire faire à notre amour les progrès décisifs sur lesquels j'étais chaque fois obligé de ne plus compter que pour l'après-midi suivante. Il en faisait pourtant quelques-uns. Un jour que nous étions allés avec Gilberte jusqu'à la baraque de notre marchande qui était particulièrement aimable pour nous – car c'était chez elle que Swann faisait acheter son pain d'épices, et par hygiène, il en consommait beaucoup, souffrant d'un eczéma ethnique et de la constipation des Prophètes – Gilberte me montrait en riant deux petits garçons qui étaient comme le petit coloriste et le petit naturaliste des livres d'enfants. Car l'un ne voulait pas d'un sucre d'orge rouge parce qu'il préférait le violet et l'autre, les larmes aux yeux, refusait une prune que voulait lui acheter sa bonne, parce que, finit-il par dire d'une voix passionnée : « J'aime mieux l'autre prune, parce qu'elle a un ver ! » J'achetai deux billes d'un sou. Je regardais avec admiration, lumineuses et captives dans une sébile isolée, les billes d'agate qui me semblaient précieuses parce qu'elles étaient souriantes et blondes comme des jeunes filles et parce qu'elles coûtaient cinquante centimes pièce. Gilberte, à qui on donnait beaucoup plus d'argent qu'à moi, me demanda laquelle je trouvais la plus belle. Elles avaient la transparence et le fondu de la vie. Je n'aurais voulu lui en faire sacrifier aucune. J'aurais aimé qu'elle pût les acheter, les délivrer toutes. Pourtant je lui en désignai une qui avait la couleur de ses yeux. Gilberte la prit, chercha son rayon doré, la caressa, paya sa rançon, mais aussitôt me remit sa captive en me disant : « Tenez, elle est à vous, je vous la donne, gardez-la comme souvenir. »

Une autre fois, toujours préoccupé du désir d'entendre la Berma dans une pièce classique, je lui avais demandé si elle ne possédait pas une brochure où Bergotte parlait de Racine, et qui ne se trouvait plus dans le commerce. Elle m'avait prié de lui en rappeler le titre exact, et le soir je lui avais adressé un petit télégramme en écrivant sur l'enveloppe ce nom de Gilberte Swann que j'avais tant de fois tracé sur mes cahiers. Le lendemain elle m'apporta, dans un paquet noué de faveurs mauves et scellé de cire blanche, la brochure qu'elle avait fait chercher. « Vous voyez que c'est bien ce que vous m'avez demandé, me dit-elle, tirant de son manchon le télégramme que je lui avais envoyé. » Mais dans l'adresse de ce pneumatique – qui, hier encore n'était rien, n'était qu'un petit bleu que j'avais écrit, et qui depuis qu'un télégraphiste l'avait remis au concierge de Gilberte et qu'un domestique l'avait porté jusqu'à sa chambre, était devenu cette chose sans prix, un des petits bleus qu'elle avait reçus ce jour-là – j'eus peine à reconnaître les lignes vaines et solitaires de mon écriture sous les cercles imprimés qu'y avait apposés la poste, sous les inscriptions qu'y avait ajoutées au crayon un des facteurs, signes de réalisation effective, cachets du monde extérieur, violettes ceintures symboliques de la vie, qui pour la première fois venaient épouser, maintenir, relever, réjouir mon rêve.

Et il y eut un jour aussi où elle me dit : « Vous savez, vous pouvez m'appeler Gilberte, en tous cas moi, je vous appellerai par votre nom de baptême. C'est trop gênant. » Pourtant elle continua encore un moment à se contenter de me dire « vous » et comme je le lui faisais remarquer, elle sourit, et composant, construisant une phrase comme celles qui dans les grammaires étrangères n'ont d'autre but que de nous faire employer un mot nouveau, elle la termina par mon petit nom. Et me souvenant plus tard de ce que j'avais senti alors, j'y ai démêlé l'impression d'avoir été tenu un instant dans sa bouche, moi-même, nu, sans plus aucune des modalités sociales qui appartenaient aussi, soit à ses autres camarades, soit, quand elle disait mon nom de famille, à mes parents, et dont ses lèvres – en l'effort qu'elle faisait, un peu comme son père, pour articuler les mots qu'elle voulait mettre en valeur – eurent l'air de me dépouiller, de me dévêtir, comme de sa peau un fruit dont on ne peut avaler que la pulpe, tandis que son regard, se mettant au même degré nouveau d'intimité que prenait sa parole, m'atteignait aussi plus directement, non sans témoigner la conscience, le plaisir et jusque la gratitude qu'il en avait, en se faisant accompagner d'un sourire.

Mais au moment même, je ne pouvais apprécier la valeur de ces plaisirs nouveaux. Ils n'étaient pas donnés par la fillette que j'aimais, au moi qui l'aimait, mais par l'autre, par celle avec qui je jouais, à cet autre moi qui ne possédait ni le souvenir de la vraie Gilberte, ni le coeur indisponible qui seul aurait pu savoir le prix d'un bonheur, parce que seul il l'avait désiré. Même après être rentré à la maison je ne les goûtais pas, car chaque jour, la nécessité qui me faisait espérer que le lendemain j'aurais la contemplation exacte, calme, heureuse de Gilberte, qu'elle m'avouerait enfin son amour, en m'expliquant pour quelles raisons elle avait dû me le cacher jusqu'ici, cette même nécessité me forçait à tenir le passé pour rien, à ne jamais regarder que devant moi, à considérer les petits avantages qu'elle m'avait donnés non pas en eux-mêmes et comme s'ils se suffisaient, mais comme des échelons nouveaux où poser le pied, qui allaient me permettre de faire un pas de plus en avant et d'atteindre enfin le bonheur que je n'avais pas encore rencontré.

Si elle me donnait parfois de ces marques d'amitié, elle me faisait aussi de la peine en ayant l'air de ne pas avoir de plaisir à me voir, et cela arrivait souvent les jours mêmes sur lesquels j'avais le plus compté pour réaliser mes espérances. J'étais sûr que Gilberte viendrait aux Champs-Élysées et j'éprouvais une allégresse qui me paraissait seulement la vague anticipation d'un grand bonheur quand – entrant dès le matin au salon pour embrasser maman déjà toute prête, la tour de ses cheveux noirs entièrement construite, et ses belles mains blanches et potelées sentant encore le savon – j'avais appris, en voyant une colonne de poussière se tenir debout toute seule au-dessus du piano, et en entendant un orgue de Barbarie jouer sous la fenêtre : « En revenant de la revue », que l'hiver recevait jusqu'au soir la visite inopinée et radieuse d'une journée de printemps. Pendant que nous déjeunions, en ouvrant sa croisée, la dame d'en face avait fait décamper en un clin d'oeil, d'à côté de ma chaise – rayant d'un seul bond toute la largeur de notre salle à manger – un rayon qui y avait commencé sa sieste et était déjà revenu la continuer l'instant d'après. Au collège, à la classe d'une heure, le soleil me faisait languir d'impatience et d'ennui en laissant traîner une lueur dorée jusque sur mon pupitre, comme une invitation à la fête où je ne pourrais arriver avant trois heures, jusqu'au moment où Françoise venait me chercher à la sortie, et où nous nous acheminions vers les Champs-Élysées par les rues décorées de lumière, encombrées par la foule, et où les balcons, descellés par le soleil et vaporeux, flottaient devant les maisons comme des nuages d'or. Hélas ! aux Champs-Élysées je ne trouvais pas Gilberte, elle n'était pas encore arrivée. Immobile sur la pelouse nourrie par le soleil invisible qui çà et là faisait flamboyer la pointe d'un brin d'herbe, et sur laquelle les pigeons qui s'y étaient posés avaient l'air de sculptures antiques que la pioche du jardinier a ramenées à la surface d'un sol auguste, je restais les yeux fixés sur l'horizon, je m'attendais à tout moment à voir apparaître l'image de Gilberte suivant son institutrice, derrière la statue qui semblait tendre l'enfant qu'elle portait et qui ruisselait de rayons à la bénédiction du soleil. La vieille lectrice des Débats était assise sur son fauteuil, toujours à la même place, elle interpellait un gardien à qui elle faisait un geste amical de la main en lui criant : « Quel joli temps ! » Et la préposée s'étant approchée d'elle pour percevoir le prix du fauteuil, elle faisait mille minauderies en mettant dans l'ouverture de son gant le ticket de dix centimes comme si ç'avait été un bouquet, pour qui elle cherchait, par amabilité pour le donateur, la place la plus flatteuse possible. Quand elle l'avait trouvée, elle faisait exécuter une évolution circulaire à son cou, redressait son boa, et plantait sur la chaisière, en lui montrant le bout de papier jaune qui dépassait sur son poignet, le beau sourire dont une femme, en indiquant son corsage à un jeune homme, lui dit : « Vous reconnaissez vos roses ! »
