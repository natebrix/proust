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
      "presence_confidence": 0.98
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "collective_social_voice",
      "target": "Swann",
      "type": "discredit_association",
      "polarity": "negative",
      "narrative_stance": "neutral_report",
      "confidence": 0.88,
      "evidence": "« plus rarement depuis qu'il avait fait ce mauvais mariage, parce que mes parents ne voulaient pas recevoir sa femme »",
      "explanation": "The family treats Swann's marriage as 'bad' and refuses to receive his wife, which socially stigmatizes Swann by association and reduces his inclusion at the house."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "social_status",
      "delta": -1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.85,
      "explanation": "His standing with the family is lowered and his visits become less frequent because his marriage is discredited."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p1-combray#p-11-p-15"
}

### Candidate characters

[
  "Françoise",
  "Geneviève",
  "la grand-mère",
  "la mère du narrateur",
  "le grand-père du narrateur",
  "le narrateur",
  "le père du narrateur"
]

### Prior local context (optional)

À Combray, tous les jours dès la fin de l'après-midi, longtemps avant le moment où il faudrait me mettre au lit et rester, sans dormir, loin de la mère du narrateur et de la grand-mère, ma chambre à coucher redevenait le point fixe et douloureux de mes préoccupations. On avait bien inventé, pour me distraire les soirs où on me trouvait l'air trop malheureux, de me donner une lanterne magique, dont, en attendant l'heure du dîner, on coiffait ma lampe ; et, à l'instar des premiers architectes et maîtres verriers de l'âge gothique, elle substituait à l'opacité des murs d'impalpables irisations, de surnaturelles apparitions multicolores, où des légendes étaient dépeintes comme dans un vitrail vacillant et momentané. Mais ma tristesse n'en était qu'accrue, parce que rien que le changement d'éclairage détruisait l'habitude que j'avais de ma chambre et grâce à quoi, sauf le supplice du coucher, elle m'était devenue supportable. Maintenant je ne la reconnaissais plus et j'y étais inquiet, comme dans une chambre d'hôtel ou de « chalet », où je fusse arrivé pour la première fois en descendant de chemin de fer.

### Passage

Au pas saccadé de son cheval, Golo, plein d'un affreux dessein, sortait de la petite forêt triangulaire qui veloutait d'un vert sombre la pente d'une colline, et s'avançait en tressautant vers le château de la pauvre Geneviève. Ce château était coupé selon une ligne courbe qui n'était guère que la limite d'un des ovales de verre ménagés dans le châssis qu'on glissait entre les coulisses de la lanterne. Ce n'était qu'un pan de château, et il avait devant lui une lande où rêvait Geneviève qui portait une ceinture bleue. Le château et la lande étaient jaunes, et je n'avais pas attendu de les voir pour connaître leur couleur, car, avant les verres du châssis, la sonorité mordorée du nom de Brabant me l'avait montrée avec évidence. Golo s'arrêtait un instant pour écouter avec tristesse le boniment lu à haute voix par ma grand'tante et qu'il avait l'air de comprendre parfaitement, conformant son attitude, avec une docilité qui n'excluait pas une certaine majesté, aux indications du texte ; puis il s'éloignait du même pas saccadé. Et rien ne pouvait arrêter sa lente chevauchée. Si on bougeait la lanterne, je distinguais le cheval de Golo qui continuait à s'avancer sur les rideaux de la fenêtre, se bombant de leurs plis, descendant dans leurs fentes. Le corps de Golo lui-même, d'une essence aussi surnaturelle que celui de sa monture, s'arrangeait de tout obstacle matériel, de tout objet gênant qu'il rencontrait en le prenant comme ossature et en se le rendant intérieur, fût-ce le bouton de la porte sur lequel s'adaptait aussitôt et surnageait invinciblement sa robe rouge ou sa figure pâle toujours aussi noble et aussi mélancolique, mais qui ne laissait paraître aucun trouble de cette transvertébration.

Certes je leur trouvais du charme à ces brillantes projections qui semblaient émaner d'un passé mérovingien et promenaient autour de moi des reflets d'histoire si anciens. Mais je ne peux dire quel malaise me causait pourtant cette intrusion du mystère et de la beauté dans une chambre que j'avais fini par remplir de mon moi au point de ne pas faire plus attention à elle qu'à lui-même. L'influence anesthésiante de l'habitude ayant cessé, je me mettais à penser, à sentir, choses si tristes. Ce bouton de la porte de ma chambre, qui différait pour moi de tous les autres boutons de porte du monde en ceci qu'il semblait ouvrir tout seul, sans que j'eusse besoin de le tourner, tant le maniement m'en était devenu inconscient, le voilà qui servait maintenant de corps astral à Golo. Et dès qu'on sonnait le dîner, j'avais hâte de courir à la salle à manger, où la grosse lampe de la suspension, ignorante de Golo et de Barbe-Bleue, et qui connaissait mes parents et le boeuf à la casserole, donnait sa lumière de tous les soirs, et de tomber dans les bras de maman que les malheurs de Geneviève me rendaient plus chère, tandis que les crimes de Golo me faisaient examiner ma propre conscience avec plus de scrupules.

Après le dîner, hélas, j'étais bientôt obligé de quitter maman qui restait à causer avec les autres, au jardin s'il faisait beau, dans le petit salon où tout le monde se retirait s'il faisait mauvais. Tout le monde, sauf ma grand'mère qui trouvait que « c'est une pitié de rester enfermé à la campagne » et qui avait d'incessantes discussions avec mon père, les jours de trop grande pluie, parce qu'il m'envoyait lire dans ma chambre au lieu de rester dehors. « Ce n'est pas comme cela que vous le rendrez robuste et énergique, disait-elle tristement, surtout ce petit qui a tant besoin de prendre des forces et de la volonté. » Mon père haussait les épaules et il examinait le baromètre, car il aimait la météorologie, pendant que ma mère, évitant de faire du bruit pour ne pas le troubler, le regardait avec un respect attendri, mais pas trop fixement pour ne pas chercher à percer le mystère de ses supériorités. Mais ma grand'mère, elle, par tous les temps, même quand la pluie faisait rage et que Françoise avait précipitamment rentré les précieux fauteuils d'osier de peur qu'ils ne fussent mouillés, on la voyait dans le jardin vide et fouetté par l'averse, relevant ses mèches désordonnées et grises pour que son front s'imbibât mieux de la salubrité du vent et de la pluie. Elle disait : « Enfin, on respire ! » et parcourait les allées détrempées – trop symétriquement alignées à son gré par le nouveau jardinier dépourvu du sentiment de la nature et auquel mon père avait demandé depuis le matin si le temps s'arrangerait – de son petit pas enthousiaste et saccadé, réglé sur les mouvements divers qu'excitaient dans son âme l'ivresse de l'orage, la puissance de l'hygiène, la stupidité de mon éducation et la symétrie des jardins, plutôt que sur le désir inconnu d'elle d'éviter à sa jupe prune les taches de boue sous lesquelles elle disparaissait jusqu'à une hauteur qui était toujours pour sa femme de chambre un désespoir et un problème.

Quand ces tours de jardin de ma grand'mère avaient lieu après dîner, une chose avait le pouvoir de la faire rentrer : c'était, à un des moments où la révolution de sa promenade la ramenait périodiquement, comme un insecte, en face des lumières du petit salon où les liqueurs étaient servies sur la table à jeu – si ma grand'tante lui criait : « Bathilde ! viens donc empêcher ton mari de boire du cognac ! » Pour la taquiner, en effet (elle avait apporté dans la famille de mon père un esprit si différent que tout le monde la plaisantait et la tourmentait), comme les liqueurs étaient défendues à mon grand-père, ma grand'tante lui en faisait boire quelques gouttes. Ma pauvre grand'mère entrait, priait ardemment son mari de ne pas goûter au cognac ; il se fâchait, buvait tout de même sa gorgée, et ma grand'mère repartait, triste, découragée, souriante pourtant, car elle était si humble de coeur et si douce que sa tendresse pour les autres et le peu de cas qu'elle faisait de sa propre personne et de ses souffrances, se conciliaient dans son regard en un sourire où, contrairement à ce qu'on voit dans le visage de beaucoup d'humains, il n'y avait d'ironie que pour elle-même, et pour nous tous comme un baiser de ses yeux qui ne pouvaient voir ceux qu'elle chérissait sans les caresser passionnément du regard. Ce supplice que lui infligeait ma grand'tante, le spectacle des vaines prières de ma grand'mère et de sa faiblesse, vaincue d'avance, essayant inutilement d'ôter à mon grand-père le verre à liqueur, c'était de ces choses à la vue desquelles on s'habitue plus tard jusqu'à les considérer en riant et à prendre le parti du persécuteur assez résolument et gaiement pour se persuader à soi-même qu'il ne s'agit pas de persécution ; elles me causaient alors une telle horreur, que j'aurais aimé battre ma grand'tante. Mais dès que j'entendais : « Bathilde, viens donc empêcher ton mari de boire du cognac ! » déjà homme par la lâcheté, je faisais ce que nous faisons tous, une fois que nous sommes grands, quand il y a devant nous des souffrances et des injustices : je ne voulais pas les voir ; je montais sangloter tout en haut de la maison à côté de la salle d'études, sous les toits, dans une petite pièce sentant l'iris, et que parfumait aussi un cassis sauvage poussé au dehors entre les pierres de la muraille et qui passait une branche de fleurs par la fenêtre entr'ouverte. Destinée à un usage plus spécial et plus vulgaire, cette pièce, d'où l'on voyait pendant le jour jusqu'au donjon de Roussainville-le-Pin, servit longtemps de refuge pour moi, sans doute parce qu'elle était la seule qu'il me fût permis de fermer à clef, à toutes celles de mes occupations qui réclamaient une inviolable solitude : la lecture, la rêverie, les larmes et la volupté. Hélas ! je ne savais pas que, bien plus tristement que les petits écarts de régime de son mari, mon manque de volonté, ma santé délicate, l'incertitude qu'ils projetaient sur mon avenir, préoccupaient ma grand'mère, au cours de ces déambulations incessantes, de l'après-midi et du soir, où on voyait passer et repasser, obliquement levé vers le ciel, son beau visage aux joues brunes et sillonnées, devenues au retour de l'âge presque mauves comme les labours à l'automne, barrées, si elle sortait, par une voilette à demi relevée, et sur lesquelles, amené là par le froid ou quelque triste pensée, était toujours en train de sécher un pleur involontaire.

Ma seule consolation, quand je montais me coucher, était que maman viendrait m'embrasser quand je serais dans mon lit. Mais ce bonsoir durait si peu de temps, elle redescendait si vite, que le moment où je l'entendais monter, puis où passait dans le couloir à double porte le bruit léger de sa robe de jardin en mousseline bleue, à laquelle pendaient de petits cordons de paille tressée, était pour moi un moment douloureux. Il annonçait celui qui allait le suivre, où elle m'aurait quitté, où elle serait redescendue. De sorte que ce bonsoir que j'aimais tant, j'en arrivais à souhaiter qu'il vînt le plus tard possible, à ce que se prolongeât le temps de répit où maman n'était pas encore venue. Quelquefois quand, après m'avoir embrassé, elle ouvrait la porte pour partir, je voulais la rappeler, lui dire « embrasse-moi une fois encore », mais je savais qu'aussitôt elle aurait son visage fâché, car la concession qu'elle faisait à ma tristesse et à mon agitation en montant m'embrasser, en m'apportant ce baiser de paix, agaçait mon père qui trouvait ces rites absurdes, et elle eût voulu tâcher de m'en faire perdre le besoin, l'habitude, bien loin de me laisser prendre celle de lui demander, quand elle était déjà sur le pas de la porte, un baiser de plus. Or la voir fâchée détruisait tout le calme qu'elle m'avait apporté un instant avant, quand elle avait penché vers mon lit sa figure aimante, et me l'avait tendue comme une hostie pour une communion de paix où mes lèvres puiseraient sa présence réelle et le pouvoir de m'endormir. Mais ces soirs-là, où maman en somme restait si peu de temps dans ma chambre, étaient doux encore en comparaison de ceux où il y avait du monde à dîner et où, à cause de cela, elle ne montait pas me dire bonsoir. Le monde se bornait habituellement à Swann, qui, en dehors de quelques étrangers de passage, était à peu près la seule personne qui vînt chez nous à Combray, quelquefois pour dîner en voisin (plus rarement depuis qu'il avait fait ce mauvais mariage, parce que mes parents ne voulaient pas recevoir sa femme), quelquefois après le dîner, à l'improviste. Les soirs où, assis devant la maison sous le grand marronnier, autour de la table de fer, nous entendions au bout du jardin, non pas le grelot profus et criard qui arrosait, qui étourdissait au passage de son bruit ferrugineux, intarissable et glacé, toute personne de la maison qui le déclenchait en entrant « sans sonner », mais le double tintement timide, ovale et doré de la clochette pour les étrangers, tout le monde aussitôt se demandait : « Une visite, qui cela peut-il être ? » mais on savait bien que cela ne pouvait être que Swann ; ma grand'tante parlant à haute voix, pour prêcher d'exemple, sur un ton qu'elle s'efforçait de rendre naturel, disait de ne pas chuchoter ainsi ; que rien n'est plus désobligeant pour une personne qui arrive et à qui cela fait croire qu'on est en train de dire des choses qu'elle ne doit pas entendre ; et on envoyait en éclaireur ma grand'mère, toujours heureuse d'avoir un prétexte pour faire un tour de jardin de plus, et qui en profitait pour arracher subrepticement au passage quelques tuteurs de rosiers afin de rendre aux roses un peu de naturel, comme une mère qui, pour les faire bouffer, passe la main dans les cheveux de son fils que le coiffeur a trop aplatis.
