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
      "canonical_name": "Odette",
      "surface_forms": [
        "Odette",
        "Mme de comte de Forcheville"
      ],
      "presence_type": "explicit",
      "presence_confidence": 0.99
    }
  ],
  "appraisal_events": [
    {
      "event_id": "E1",
      "source": "narrator",
      "target": "Odette",
      "type": "narrated_elevation",
      "polarity": "mixed",
      "narrative_stance": "ironized",
      "confidence": 0.78,
      "evidence": "« son aspect … semblait un défi plus miraculeux aux lois de la chronologie »; mais aussi: « Seule peut-être Mme de comte de Forcheville … avait l’air d’une cocotte d’autrefois à jamais ‘naturalisée’ » et « Quel était le fait du fard, de la teinture ? »",
      "explanation": "The narrator singularizes Odette as miraculously defying time, while insinuating artifice (makeup, dye) and a fixed 'naturalized chick,' which elevates and tarnishes her image at the same time."
    }
  ],
  "status_effects": [
    {
      "character": "Odette",
      "dimension": "general_appraisal",
      "delta": 1,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.72,
      "explanation": "Locally, Odette stands out as a preserved exception against the surrounding aging, despite an ironic coloring on the artificial nature of this preservation."
    }
  ],
  "ambiguities": [],
  "unit_id": "v7-p4-le-bal-de-tetes#p-31-p-35"
}

### Candidate characters

[
  "Gilberte",
  "comte de Forcheville",
  "la mère du narrateur",
  "le narrateur",
  "princesse de Guermantes"
]

### Prior local context (optional)

Sans doute certaines femmes étaient encore très reconnaissables, le visage était resté presque le même, et elles avaient seulement, comme par une harmonie convenable avec la saison, revêtu les cheveux gris, qui étaient leur parure d'automne. Mais pour d'autres, et pour des hommes aussi, la transformation était si complète, l'identité si impossible à établir – par exemple entre un noir viveur qu'on se rappelait et le vieux moine qu'on avait sous les yeux – que plus même qu'à l'art de l'acteur, c'était à celui de certains prodigieux mimes, dont Fregoli reste le type, que faisaient penser ces fabuleuses transformations. La vieille femme avait envie de pleurer en comprenant que l'indéfinissable et mélancolique sourire qui avait fait son charme ne pouvait plus arriver à irradier jusqu'à la surface de ce masque de plâtre que lui avait appliqué la vieillesse. Puis tout à coup découragée de plaire, trouvant plus spirituel de se résigner, elle s'en servait comme d'un masque de théâtre pour faire rire ! Mais presque toutes les femmes n'avaient pas de trêve dans leur effort pour lutter contre l'âge et tendaient vers la beauté qui s'éloignait comme un soleil couchant et dont elles voulaient passionnément conserver les derniers rayons, le miroir de leur visage. Pour y réussir certaines cherchaient à l'aplanir, à élargir la blanche superficie, renonçant au piquant des fossettes menacées, aux mutineries d'un sourire condamné et déjà à demi désarmé ; tandis que d'autres, voyant la beauté définitivement disparue et obligées de se réfugier dans l'expression, comme on compense par l'art de la diction la perte de la voix, se raccrochaient à une moue, à une patte d'oie, à un regard vague, parfois à un sourire qui, à cause de l'incoordination de muscles qui n'obéissaient plus, leur donnait l'air de pleurer.

### Passage

Une grosse dame me dit un bonjour pendant la courte durée duquel les pensées les plus différentes se pressèrent dans mon esprit. J'hésitai un instant à lui répondre, craignant que, ne reconnaissant pas les gens mieux que moi, elle eût cru que j'étais quelqu'un d'autre, puis son assurance me fit au contraire, de peur que ce fût quelqu'un avec qui j'avais été lié, exagérer l'amabilité de mon sourire, pendant que mes regards continuaient à chercher dans ses traits le nom que je ne trouvais pas. Tel un candidat au baccalauréat, incertain de ce qu'il doit répondre, attache ses regards sur la figure de l'examinateur et espère vainement y trouver la réponse qu'il ferait mieux de chercher dans sa propre mémoire, tel, tout en lui souriant, j'attachais mes regards sur les traits de la grosse dame. Ils me semblèrent être ceux de Mme de Forcheville, aussi mon sourire se nuança-t-il de respect, pendant que mon indécision commençait à cesser. Alors j'entendis la grosse dame me dire, une seconde plus tard : « Vous me preniez pour maman, en effet je commence à lui ressembler beaucoup. » Et je reconnus Gilberte.

D'ailleurs, même chez les hommes qui n'avaient subi qu'un léger changement, dont seule la moustache était devenue blanche, on sentait que ce changement n'était pas positivement matériel. C'était comme si on les avait vus à travers une vapeur colorante, ou mieux un verre peint qui changeait l'aspect de leur figure mais surtout par ce qu'il y ajoutait de trouble, montrait que ce qu'il nous permettait de voir « grandeur nature » était en réalité très loin de nous, dans un éloignement différent, il est vrai, de celui de l'espace, mais du fond duquel, comme d'un autre rivage, nous sentions qu'ils avaient autant de peine à nous reconnaître que nous eux. Seule peut-être Mme de Forcheville, que j'aperçus alors comme injectée d'un liquide, d'une espèce de paraffine qui gonfle la peau mais l'empêche de se modifier, avait l'air d'une cocotte d'autrefois à jamais « naturalisée ». « Vous me prenez pour ma mère », m'avait dit Gilberte. C'était vrai. C'eût été, d'ailleurs, aimable pour la fille. D'ailleurs, il n'y avait pas que chez cette dernière qu'avaient apparu des traits familiaux qui jusque-là étaient restés aussi invisibles dans sa figure que ces parties d'une graine repliées à l'intérieur et dont on ne peut deviner la saillie qu'elles feront un jour en dehors. Ainsi un énorme busquage maternel venait, chez l'une ou chez l'autre, transformer vers la cinquantaine un nez jusque-là droit et pur. Chez une autre fille de banquier, le teint, d'une fraîcheur de jardinière, se roussissait, se cuivrait, et prenait comme le reflet de l'or qu'avait tant manié le père. Certains même avaient fini par ressembler à leur quartier, portaient sur eux comme le reflet de la rue de l'Arcade, de l'avenue du Bois, de la rue de l'Élysée. Mais surtout ils reproduisaient les traits de leurs parents.

On part de l'idée que les gens sont restés les mêmes et on les trouve vieux. Mais une fois que l'idée dont on part est qu'ils sont vieux, on les retrouve, on ne les trouve pas si mal. Pour Odette, ce n'était pas seulement cela ; son aspect, une fois qu'on savait son âge et qu'on s'attendait à une vieille femme, semblait un défi plus miraculeux aux lois de la chronologie que la conservation du radium à celles de la nature. Elle, si je ne la reconnus pas d'abord, ce fut non parce qu'elle avait, mais parce qu'elle n'avait pas changé. Me rendant compte depuis une heure de ce que le temps ajoutait de nouveau aux êtres et de ce qu'il fallait soustraire pour les retrouver tels que je les avais connus, je faisais maintenant rapidement ce calcul et, ajoutant à l'ancienne Odette le chiffre d'années qui avait passé sur elle, le résultat que je trouvai fut une personne qui me semblait ne pas pouvoir être celle que j'avais sous les yeux, précisément parce que celle-là était pareille à celle d'autrefois.

Quel était le fait du fard, de la teinture ? Elle avait l'air, sous ses cheveux dorés tout plats – un peu un chignon ébouriffé de grosse poupée mécanique sur une figure étonnée et immuable également de poupée – auxquels se superposait un chapeau de paille plat aussi, de l'Exposition de 1878 (dont elle eût certes été alors, et surtout si elle eût eu alors l'âge d'aujourd'hui, la plus fantastique merveille) venant débiter son compliment dans une revue de fin d'année, mais de l'Exposition de 1878 représentée par une femme encore jeune.

À côté de nous, un ministre d'avant l'époque boulangiste, et qui l'était de nouveau, passait, lui aussi, en envoyant aux dames un sourire tremblotant et lointain, mais comme emprisonné dans les mille liens du passé, comme un petit fantôme qu'une main invisible promenait, diminué de taille, changé dans sa substance et ayant l'air d'une réduction en pierre ponce de soi-même. Cet ancien président du Conseil, si bien reçu dans le Faubourg Saint-Germain, avait jadis été l'objet de poursuites criminelles, exécré du monde et du peuple. Mais grâce au renouvellement des individus qui composent l'un et l'autre, et, dans les individus subsistant, des passions et même des souvenirs, personne ne le savait plus et il était honoré. Aussi n'y a-t-il pas d'humiliation si grande dont on ne devrait prendre aisément son parti, sachant qu'au bout de quelques années, nos fautes ensevelies ne seront plus qu'une invisible poussière sur laquelle sourira la paix souriante et fleurie de la nature. L'individu momentanément taré se trouvera, par le jeu d'équilibre du temps, pris entre deux couches sociales nouvelles qui n'auront pour lui que déférence et admiration, et au-dessus desquelles il se prélassera aisément. Seulement c'est au temps qu'est confié ce travail ; et, au moment de ses ennuis, rien ne peut le consoler que la jeune laitière d'en face l'ait entendu appeler « chéquard » par la foule qui montrait le poing tandis qu'il entrait dans le « panier à salade », la jeune laitière qui ne voit pas les choses dans le plan du temps, qui ignore que les hommes qu'encense le journal du matin furent déconsidérés jadis, et que l'homme qui frise la prison en ce moment, et peut-être en pensant à cette jeune laitière, n'aura pas les paroles humbles qui lui concilieraient la sympathie, sera un jour célébré par la presse et recherché par les duchesses. Le temps éloigne pareillement les querelles de famille. Et chez la princesse de Guermantes on voyait un couple où le mari et la femme avaient pour oncles, morts aujourd'hui, deux hommes qui ne s'étaient pas contentés de se souffleter mais dont l'un pour humilier l'autre lui avait envoyé comme témoins son concierge et son maître d'hôtel, jugeant que des gens du monde eussent été trop bien pour lui. Mais ces histoires dormaient dans les journaux d'il y a trente ans et personne ne les savait plus. Et ainsi le salon de la princesse de Guermantes était illuminé, oublieux et fleuri, comme un paisible cimetière. Le temps n'y avait pas seulement défait d'anciennes créatures, il y avait rendu possibles, il y avait créé des associations nouvelles.
