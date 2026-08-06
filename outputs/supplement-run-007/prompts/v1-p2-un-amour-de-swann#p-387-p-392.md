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
      "confidence": 0.88,
      "evidence": "Odette lui dit: «Vous ne voudriez pas m'attendre cinq minutes... nous reviendrions ensemble, vous me ramèneriez chez moi.» Puis, devant Forcheville: «Cela dépend de ce monsieur-là... il aime causer tranquillement avec moi.» Ensuite, «toutes les idées terribles... s'évanouissaient» et «après ces tranquilles soirées, les soupçons de Swann étaient calmés; il bénissait Odette».",
      "explanation": "The narrator shows Odette including and favoring Swann in public and in private, which dispels his jealousy and gives him back emotional peace. This public favor and the acknowledged intimacy locally raise Swann’s position."
    }
  ],
  "status_effects": [
    {
      "character": "Swann",
      "dimension": "emotional_position",
      "delta": 2,
      "based_on_events": [
        "E1"
      ],
      "confidence": 0.88,
      "explanation": "Thanks to Odette’s gestures and words, his jealousy calms and he feels marked relief, going so far as to «bénir» Odette and to send her jewels."
    }
  ],
  "ambiguities": [],
  "unit_id": "v1-p2-un-amour-de-swann#p-387-p-392"
}

### Candidate characters

[
  "Odette",
  "comte de Forcheville",
  "le narrateur",
  "le peintre"
]

### Prior local context (optional)

C'est qu'elle n'avait même pas pensé à lui. Et de tels moments, où elle oubliait jusqu'à l'existence de Swann étaient plus utiles à Odette, servaient mieux à lui attacher Swann, que toute sa coquetterie. Car ainsi Swann vivait dans cette agitation douloureuse qui avait déjà été assez puissante pour faire éclore son amour, le soir où il n'avait pas trouvé Odette chez les M. Verdurin et l'avait cherchée toute la soirée. Et il n'avait pas, comme j'eus à Combray dans mon enfance, des journées heureuses pendant lesquelles s'oublient les souffrances qui renaîtront le soir. Les journées, Swann les passait sans Odette ; et par moments il se disait que laisser une aussi jolie femme sortir ainsi seule dans Paris était aussi imprudent que de poser un écrin plein de bijoux au milieu de la rue. Alors il s'indignait contre tous les passants comme contre autant de voleurs. Mais leur visage collectif et informe échappant à son imagination ne nourrissait pas sa jalousie. Il fatiguait la pensée de Swann, lequel, se passant la main sur les yeux, s'écriait : « À la grâce de Dieu », comme ceux qui après s'être acharnés à étreindre le problème de la réalité du monde extérieur ou de l'immortalité de l'âme accordent la détente d'un acte de foi à leur cerveau lassé. Mais toujours la pensée de l'absente était indissolublement mêlée aux actes les plus simples de la vie de Swann – déjeuner, recevoir son courrier, sortir, se coucher – par la tristesse même qu'il avait à les accomplir sans elle, comme ces initiales de Philibert le Beau que dans l'église de Brou, à cause du regret qu'elle avait de lui, Marguerite d'Autriche entrelaça partout aux siennes. Certains jours, au lieu de rester chez lui, il allait prendre son déjeuner dans un restaurant assez voisin dont il avait apprécié autrefois la bonne cuisine et où maintenant il n'allait plus que pour une de ces raisons à la fois mystiques et saugrenues, qu'on appelle romanesques ; c'est que ce restaurant (lequel existe encore) portait le même nom que la rue habitée par Odette : Lapérouse. Quelquefois, quand elle avait fait un court déplacement, ce n'est qu'après plusieurs jours qu'elle songeait à lui faire savoir qu'elle était revenue à Paris. Et elle lui disait tout simplement, sans plus prendre comme autrefois la précaution de se couvrir à tout hasard d'un petit morceau emprunté à la vérité, qu'elle venait d'y rentrer à l'instant même par le train du matin. Ces paroles étaient mensongères ; du moins pour Odette elles étaient mensongères, inconsistantes, n'ayant pas, comme si elles avaient été vraies, un point d'appui dans le souvenir de son arrivée à la gare ; même elle était empêchée de se les représenter au moment où elle les prononçait, par l'image contradictoire de ce qu'elle avait fait de tout différent au moment où elle prétendait être descendue du train. Mais dans l'esprit de Swann au contraire, ces paroles qui ne rencontraient aucun obstacle venaient s'incruster et prendre l'inamovibilité d'une vérité si indubitable que, si un ami lui disait être venu par ce train et ne pas avoir vu Odette, il était persuadé que c'était l'ami qui se trompait de jour ou d'heure, puisque son dire ne se conciliait pas avec les paroles d'Odette. Celles-ci ne lui eussent paru mensongères que s'il s'était d'abord défié qu'elles le fussent. Pour qu'il crût qu'elle mentait, un soupçon préalable était une condition nécessaire. C'était d'ailleurs aussi une condition suffisante. Alors tout ce que disait Odette lui paraissait suspect. L'entendait-il citer un nom, c'était certainement celui d'un de ses amants ; une fois cette supposition forgée, il passait des semaines à se désoler ; il s'aboucha même une fois avec une agence de renseignements pour savoir l'adresse, l'emploi du temps de l'inconnu qui ne le laisserait respirer que quand il serait parti en voyage, et dont il finit par apprendre que c'était un oncle d'Odette mort depuis vingt ans.

### Passage

Bien qu'elle ne lui permît pas en général de la rejoindre dans des lieux publics, disant que cela ferait jaser, il arrivait que dans une soirée où il était invité comme elle – chez Forcheville, chez le peintre, ou à un bal de charité dans un ministère – il se trouvât en même temps qu'elle. Il la voyait mais n'osait pas rester de peur de l'irriter en ayant l'air d'épier les plaisirs qu'elle prenait avec d'autres et qui – tandis qu'il rentrait solitaire, qu'il allait se coucher anxieux comme je devais l'être moi-même quelques années plus tard les soirs où il viendrait dîner à la maison, à Combray – lui semblaient illimités parce qu'il n'en avait pas vu la fin. Et une fois ou deux il connut par de tels soirs de ces joies qu'on serait tenté, si elles ne subissaient avec tant de violence le choc en retour de l'inquiétude brusquement arrêtée, d'appeler des joies calmes, parce qu'elles consistent en un apaisement : il était allé passer un instant à un raout chez le peintre et s'apprêtait à le quitter ; il y laissait Odette muée en une brillante étrangère au milieu d'hommes à qui ses regards et sa gaieté, qui n'étaient pas pour lui, semblaient parler de quelque volupté, qui serait goûtée là ou ailleurs (peut-être au « Bal des Incohérents » où il tremblait qu'elle n'allât ensuite) et qui causait à Swann plus de jalousie que l'union charnelle même parce qu'il l'imaginait plus difficilement ; il était déjà prêt à passer la porte de l'atelier quand il s'entendait rappeler par ces mots (qui en retranchant de la fête cette fin qui l'épouvantait, la lui rendaient rétrospectivement innocente, faisaient du retour d'Odette une chose non plus inconcevable et terrible, mais douce et connue et qui tiendrait à côté de lui, pareille à un peu de sa vie de tous les jours, dans sa voiture, et dépouillait Odette elle-même de son apparence trop brillante et gaie, montraient que ce n'était qu'un déguisement qu'elle avait revêtu un moment, pour lui-même, non en vue de mystérieux plaisirs, et duquel elle était déjà lasse), par ces mots qu'Odette lui jetait, comme il était déjà sur le seuil : « Vous ne voudriez pas m'attendre cinq minutes, je vais partir, nous reviendrions ensemble, vous me ramèneriez chez moi.

Il est vrai qu'un jour Forcheville avait demandé à être ramené en même temps, mais comme, arrivé devant la porte d'Odette, il avait sollicité la permission d'entrer aussi, Odette lui avait répondu en montrant Swann : « Ah ! cela dépend de ce monsieur-là, demandez-lui. Enfin, entrez un moment si vous voulez, mais pas longtemps, parce que je vous préviens qu'il aime causer tranquillement avec moi, et qu'il n'aime pas beaucoup qu'il y ait des visites quand il vient. Ah ! si vous connaissiez cet être-là autant que je le connais ; n'est-ce pas, my love, il n'y a que moi qui vous connaisse bien ? »

Et Swann était peut-être encore plus touché de la voir ainsi lui adresser en présence de Forcheville, non seulement ces paroles de tendresse, de prédilection, mais encore certaines critiques comme : « Je suis sûre que vous n'avez pas encore répondu à vos amis pour votre dîner de dimanche. N'y allez pas si vous ne voulez pas, mais soyez au moins poli », ou : « Avez-vous laissé seulement ici votre essai sur Ver Meer pour pouvoir l'avancer un peu demain ? Quel paresseux ! Je vous ferai travailler, moi ! », qui prouvaient qu'Odette se tenait au courant de ses invitations dans le monde et de ses études d'art, qu'ils avaient bien une vie à eux deux. Et en disant cela, elle lui adressait un sourire au fond duquel il la sentait toute à lui.

Alors à ces moments-là, pendant qu'elle leur faisait de l'orangeade, tout d'un coup, comme quand un réflecteur mal réglé d'abord promène autour d'un objet, sur la muraille, de grandes ombres fantastiques, qui viennent ensuite se replier et s'anéantir en lui, toutes les idées terribles et mouvantes qu'il se faisait d'Odette s'évanouissaient, rejoignaient le corps charmant que Swann avait devant lui. Il avait le brusque soupçon que cette heure passée chez Odette, sous la lampe, n'était peut-être pas une heure factice, à son usage à lui (destinée à masquer cette chose effrayante et délicieuse à laquelle il pensait sans cesse sans pouvoir bien se la représenter, une heure de la vraie vie d'Odette, de la vie d'Odette quand lui n'était pas là), avec des accessoires de théâtre et des fruits de carton, mais était peut-être une heure pour de bon de la vie d'Odette ; que s'il n'avait pas été là, elle eût avancé à Forcheville le même fauteuil et lui eût versé non un breuvage inconnu, mais précisément cette orangeade ; que le monde habité par Odette n'était pas cet autre monde effroyable et surnaturel où il passait son temps à la situer et qui n'existait peut-être que dans son imagination, mais l'univers réel, ne dégageant aucune tristesse spéciale, comprenant cette table où il allait pouvoir écrire et cette boisson à laquelle il lui serait permis de goûter ; tous ces objets qu'il contemplait avec autant de curiosité et d'admiration que de gratitude, car si en absorbant ses rêves ils l'en avaient délivré, eux en revanche, s'en étaient enrichis, ils lui en montraient la réalisation palpable, et ils intéressaient son esprit, ils prenaient du relief devant ses regards, en même temps qu'ils tranquillisaient son coeur. Ah ! si le destin avait permis qu'il pût n'avoir qu'une seule demeure avec Odette et que chez elle il fût chez lui, si en demandant au domestique ce qu'il y avait à déjeuner, c'eût été le menu d'Odette qu'il avait appris en réponse, si quand Odette voulait aller le matin se promener avenue du Bois-de-Boulogne, son devoir de bon mari l'avait obligé, n'eût-il pas envie de sortir, à l'accompagner, portant son manteau quand elle avait trop chaud, et le soir après le dîner si elle avait envie de rester chez elle en déshabillé, s'il avait été forcé de rester là près d'elle, à faire ce qu'elle voudrait ; alors combien tous les riens de la vie de Swann qui lui semblaient si tristes, au contraire parce qu'ils auraient en même temps fait partie de la vie d'Odette auraient pris, même les plus familiers – et comme cette lampe, cette orangeade, ce fauteuil qui contenaient tant de rêve, qui matérialisaient tant de désir – une sorte de douceur surabondante et de densité mystérieuse.

Pourtant il se doutait bien que ce qu'il regrettait ainsi, c'était un calme, une paix qui n'auraient pas été pour son amour une atmosphère favorable. Quand Odette cesserait d'être pour lui une créature toujours absente, regrettée, imaginaire ; quand le sentiment qu'il aurait pour elle ne serait plus ce même trouble mystérieux que lui causait la phrase de la sonate, mais de l'affection, de la reconnaissance ; quand s'établiraient entre eux des rapports normaux qui mettraient fin à sa folie et à sa tristesse, alors sans doute les actes de la vie d'Odette lui paraîtraient peu intéressants en eux-mêmes – comme il avait déjà eu plusieurs fois le soupçon qu'ils étaient, par exemple le jour où il avait lu à travers l'enveloppe la lettre adressée à Forcheville. Considérant son mal avec autant de sagacité que s'il se l'était inoculé pour en faire l'étude, il se disait que, quand il serait guéri, ce que pourrait faire Odette lui serait indifférent. Mais du sein de son état morbide, à vrai dire, il redoutait à l'égal de la mort une telle guérison, qui eût été en effet la mort de tout ce qu'il était actuellement.

Après ces tranquilles soirées, les soupçons de Swann étaient calmés ; il bénissait Odette et le lendemain, dès le matin, il faisait envoyer chez elle les plus beaux bijoux, parce que ces bontés de la veille avaient excité ou sa gratitude, ou le désir de les voir se renouveler, ou un paroxysme d'amour qui avait besoin de se dépenser.
