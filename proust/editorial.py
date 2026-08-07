CHARACTER_PORTRAIT_SLUGS = {
    "Albertine": "albertine",
    "Odette": "odette",
    "Robert de Saint-Loup": "saint-loup",
    "Swann": "swann",
    "baron de Charlus": "charlus",
    "le narrateur": "le-narrateur",
}


CHARACTER_PAGE_PILOT_EDITORIAL = {
    "le narrateur": {
        "subheading": "The perpetually admitted observer: first in advantage and inclusion among the novel's central figures, with average scores that stay stubbornly near zero.",
        "summary": "The narrator is the novel's \"I\": nearly every scene passes through him, so he meets more of the cast, more often, than any other figure. Scene by scene he tends to come out ahead of whoever shares the room — admitted, favored, received — while his own overall readings stay anxious and nearly neutral. It is as close as the scores come to restating the novel's split between lived unease and narrated mastery.",
        "why_interesting": [
            "He leads the advantage and inclusion standings despite near-zero average scores: he prevails relative to whoever shares his scenes, not by dominating them.",
            "Because the whole novel passes through him, his fortunes brush against more of the cast than anyone else's — his outcomes are the connective tissue of the book's social world.",
            "The gap between how consistently he comes out ahead and how anxiously he reports it is the cleanest measurable version of the book's central irony.",
        ],
        "primary_pattern": "relational_positive_understated",
        "reading_path": [
            {
                "chapter_id": "v2-p2-noms-de-pays-le-pays",
                "label": "Balbec thresholds: the machinery of being received",
            },
            {
                "chapter_id": "v3-p2",
                "label": "Guermantes admission: the observer absorbed",
            },
            {
                "chapter_id": "v7-p4-le-bal-de-tetes",
                "label": "The bal de têtes: survivor among the masks",
            },
        ],
    },
    "Odette": {
        "subheading": "Prestige-positive but inclusion-negative, with her sharpest gains and reversals concentrated in a few high-pressure chapters.",
        "summary": "Odette is one of the clearest cross-lens split figures in the corpus: she rises strongly in prestige while remaining far more unstable in belonging and immediate advantage.",
        "why_interesting": [
            "Her prestige and inclusion readings diverge much more sharply than her raw frequency alone would predict.",
            "Her profile is driven by a few concentrated chapter zones rather than a flat corpus-wide pattern.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Prestige ascent around Mme Swann"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Negative counterweight in Swann's love"},
            {"chapter_id": "v3-p1", "label": "Later reversals in Guermantes-adjacent society"},
        ],
    },
    "Robert de Saint-Loup": {
        "subheading": "A highly frequent aristocratic figure whose prestige often holds even where belonging and immediate advantage give way.",
        "summary": "Robert de Saint-Loup combines very high annotation frequency with one of the largest lens spreads in the corpus, especially where aristocratic polish and emotional belonging pull apart.",
        "why_interesting": [
            "He is central enough to matter structurally, not just as a curiosity of one chapter.",
            "His strongest divergence is chapter-shaped rather than corpus-flat, especially in the Guermantes material.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Main prestige / inclusion divergence"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Earlier positive concentration"},
            {"chapter_id": "v7-p1-a-tansonville", "label": "Late negative pressure"},
        ],
    },
    "Swann": {
        "subheading": "The corpus's most annotated figure, overwhelmingly shaped by repeated immediate, social, and emotional losses.",
        "summary": "Swann dominates the corpus by sheer annotation footprint, and his aggregate profile remains broadly and repeatedly negative across all three lenses.",
        "why_interesting": [
            "He is both the most annotated character and one of the clearest broad negative cases.",
            "His profile is not a narrow anomaly but a corpus-defining social pattern, especially in Un amour de Swann.",
        ],
        "primary_pattern": "broad_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Primary negative concentration"},
            {"chapter_id": "v1-p1-combray", "label": "Early counterweight and setup"},
            {"chapter_id": "v4-p2", "label": "Later negative reinforcement"},
        ],
    },
    "Albertine": {
        "subheading": "A heavily annotated recurring figure whose strongest shaping comes from imprisonment, suspicion, disappearance, and loss.",
        "summary": "Albertine is one of the largest and most persistently negative figures in the corpus, with her strongest shaping concentrated in the prison and disappearance chapters.",
        "why_interesting": [
            "Her profile is both high-volume and highly concentrated in a few major late narrative blocks.",
            "She helps distinguish broad negative centrality from the more split prestige/inclusion cases.",
        ],
        "primary_pattern": "broad_negative",
        "reading_path": [
            {"chapter_id": "v5", "label": "Main negative concentration in La Prisonnière"},
            {"chapter_id": "v6-p1", "label": "Afterlife of loss in Albertine disparue"},
            {"chapter_id": "v6-p2", "label": "Continuing exclusion pressure"},
        ],
    },
    "baron de Charlus": {
        "subheading": "A highly volatile major figure whose aggregate negatives are spread across salon, sexual, and wartime terrains.",
        "summary": "baron de Charlus is a highly annotated and highly volatile figure whose negative aggregate treatment is spread across salon, sexual, and wartime configurations rather than one single narrative block.",
        "why_interesting": [
            "He is too frequent and too spread out to read as a one-zone anomaly.",
            "His profile shows how a major character can be broadly negative without collapsing into a single repeated scene type.",
        ],
        "primary_pattern": "volatile_broad_negative",
        "reading_path": [
            {"chapter_id": "v4-p2", "label": "Salon-world negative pressure"},
            {"chapter_id": "v5", "label": "Late negative cluster with Morel"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Wartime degradation"},
        ],
    },
    "duchesse de Guermantes": {
        "subheading": "The corpus's clearest uniformly positive great-world figure, with command and symbolic force holding across all three lenses.",
        "summary": "duchesse de Guermantes is the strongest uniformly positive figure in the current corpus surface, with her social command and symbolic force holding across every lens rather than depending on a narrow chapter exception.",
        "why_interesting": [
            "She provides a counterexample to the many elite figures whose rank does not convert cleanly into broad advantage.",
            "Her positivity is repeated enough to matter structurally, not merely as a prestige cameo.",
        ],
        "primary_pattern": "uniform_positive",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "High Guermantes concentration"},
            {"chapter_id": "v3-p2", "label": "Continued positive confirmation"},
            {"chapter_id": "v4-p2", "label": "Late reinforcing appearances"},
        ],
    },
    "Mme de Villeparisis": {
        "subheading": "A moderate but revealing split figure, relatively strong in prestige while advantage and inclusion drift downward.",
        "summary": "Mme de Villeparisis is one of the clearest moderate split figures in the corpus: she remains comparatively strong in prestige while advantage and inclusion drift downward or oscillate by chapter.",
        "why_interesting": [
            "She shows the lens split in a quieter register than Odette or Saint-Loup.",
            "Her chapter distribution helps distinguish sustained social authority from weaker interpersonal footing.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Main split concentration"},
            {"chapter_id": "v3-p2", "label": "Prestige support zone"},
            {"chapter_id": "v6-p3", "label": "Late negative counterweight"},
        ],
    },
    "Françoise": {
        "subheading": "A frequent recurring figure whose overall downward pull includes a few brief local reversals.",
        "summary": "Françoise accumulates as a broadly negative figure across the corpus, though her profile is not flat: a small number of chapters briefly reverse the trend before the longer downward pull returns.",
        "why_interesting": [
            "She is frequent enough to matter, but not in the same aristocratic pattern as Swann or Charlus.",
            "Her profile is useful for distinguishing domestic/local authority from broader belonging and valuation.",
        ],
        "primary_pattern": "broad_negative_with_reversals",
        "reading_path": [
            {"chapter_id": "v1-p1-combray", "label": "Early domestic concentration"},
            {"chapter_id": "v5", "label": "Late negative reinforcement"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Brief local reversal"},
        ],
    },
    "Mme Verdurin": {
        "subheading": "A salon figure whose aggregate reading is broadly negative across all three lenses.",
        "summary": "Mme Verdurin is one of the clearest broadly negative salon figures in the corpus, with losses in advantage, prestige, and inclusion all reinforcing rather than offsetting one another.",
        "why_interesting": [
            "She is central enough to shape multiple social zones without becoming a prestige split case.",
            "Her profile helps define the project's recurrent salon-world negative pattern.",
        ],
        "primary_pattern": "broad_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Primary Verdurin-world concentration"},
            {"chapter_id": "v5", "label": "Late negative counterpoint"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Wartime reversal zone"},
        ],
    },
    "Gilberte": {
        "subheading": "A compact but telling figure whose strength in prestige and immediate advantage does not translate into equal belonging.",
        "summary": "Gilberte is a compact but revealing cross-lens figure: she scores very well in prestige and immediate advantage, yet her inclusion profile remains markedly less secure.",
        "why_interesting": [
            "She is a smaller but especially clear example of lens divergence.",
            "Her chapter locations make her useful for app-facing spotlighting without requiring a massive corpus footprint.",
        ],
        "primary_pattern": "advantage_positive_prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v1-p3-noms-de-pays-le-nom", "label": "Early positive concentration"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Mme Swann-world extension"},
            {"chapter_id": "v6-p2", "label": "Late instability in belonging"},
        ],
    },
    "Norpois": {
        "subheading": "A strongly positive figure whose main force comes from durable rhetorical and social authority rather than intimacy.",
        "summary": "Norpois is a strongly positive figure across all three lenses, driven less by intimacy than by durable rhetorical authority and socially legible judgment.",
        "why_interesting": [
            "He helps separate prestige-positive authority from the more emotionally charged positive figures.",
            "His positivity is anchored in repeated interpretive command rather than one dramatic narrative reversal.",
        ],
        "primary_pattern": "authority_positive",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Main authority concentration"},
            {"chapter_id": "v3-p1", "label": "Secondary Guermantes reinforcement"},
            {"chapter_id": "v6-p3", "label": "Late echo of rhetorical force"},
        ],
    },
    "la grand-mère": {
        "subheading": "A recurring family figure whose harshest pressure falls on belonging and broader valuation.",
        "summary": "la grand-mère accumulates as one of the corpus's more strongly negative recurring figures, with the harshest pressure falling on inclusion and broad valuation rather than on a narrow prestige story alone.",
        "why_interesting": [
            "She is a strongly negative recurring figure outside the main salon/aristocratic pattern.",
            "Her profile is useful for showing how intimate and familial figures can be damaged without being prestige-centered.",
        ],
        "primary_pattern": "inclusion_negative",
        "reading_path": [
            {"chapter_id": "v1-p1-combray", "label": "Early family-world pressure"},
            {"chapter_id": "v3-p1", "label": "Guermantes-world counterweight"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Main split concentration"},
        ],
    },
    "Bloch": {
        "subheading": "A strongly negative recurring figure whose losses reinforce each other across all three lenses.",
        "summary": "Bloch is one of the clearest aggregate negative cases in the corpus, with repeated losses in advantage, prestige, and inclusion reinforcing each other rather than splitting apart.",
        "why_interesting": [
            "He is one of the cleanest examples of consistent multi-lens damage.",
            "His profile helps clarify the difference between broad negative treatment and more prestige-divergent cases.",
        ],
        "primary_pattern": "broad_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Primary Guermantes-world humiliation zone"},
            {"chapter_id": "v1-p1-combray", "label": "Early negative setup"},
            {"chapter_id": "v3-p2", "label": "Continued social diminishment"},
        ],
    },
    "duc de Guermantes": {
        "subheading": "A revealing reversal-of-expectation figure: high rank without correspondingly positive aggregate treatment.",
        "summary": "duc de Guermantes is one of the project's most revealing reversals of expectation: despite formal rank, his annotation surface is broadly negative across all three lenses.",
        "why_interesting": [
            "He demonstrates that aristocratic title alone does not guarantee positive aggregate treatment in the corpus.",
            "His profile sharpens the distinction between formal status and actual advantage or belonging.",
        ],
        "primary_pattern": "prestige_expectation_reversed",
        "reading_path": [
            {"chapter_id": "v3-p2", "label": "Primary Guermantes counterexample"},
            {"chapter_id": "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle", "label": "Late decline reinforcement"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "Final negative return"},
        ],
    },
    "docteur Cottard": {
        "subheading": "A mid-tier negative figure shaped by a strong Swann-world concentration and then smaller, uneven later reversals.",
        "summary": "docteur Cottard is a mid-tier negative figure whose profile is shaped by one strong Swann-world concentration, then complicated by smaller later recoveries and uneven prestige moments.",
        "why_interesting": [
            "He is a useful moderate case rather than an extreme corpus outlier.",
            "His chapter pattern helps show how one major concentration can dominate an otherwise mixed profile.",
        ],
        "primary_pattern": "swann_world_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Primary negative concentration"},
            {"chapter_id": "v1-p3-noms-de-pays-le-nom", "label": "Brief positive counterweight"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Later positive echo"},
        ],
    },
    "la mère du narrateur": {
        "subheading": "A quiet but high-performing recurring figure, especially strong in advantage and inclusion.",
        "summary": "la mère du narrateur is a quietly high-performing figure across all three lenses, with especially strong advantage and inclusion values driven by stable interpretive and familial force.",
        "why_interesting": [
            "She offers a positive familial counterpoint to the more socially competitive figures.",
            "Her profile shows how recurring emotional authority can register positively without requiring prestige spectacle.",
        ],
        "primary_pattern": "familial_positive",
        "reading_path": [
            {"chapter_id": "v6-p3", "label": "Main positive family-world concentration"},
            {"chapter_id": "v1-p3-noms-de-pays-le-nom", "label": "Earlier positive presence"},
            {"chapter_id": "v1-p1-combray", "label": "Foundational domestic context"},
        ],
    },
    "Bergotte": {
        "subheading": "A symbolic positive figure whose literary authority carries strongly across advantage and prestige.",
        "summary": "Bergotte is one of the corpus's clearest positive symbolic figures, with his literary authority translating into very high advantage and prestige across several distinct narrative zones.",
        "why_interesting": [
            "He is strongly positive without being primarily a belonging-driven figure.",
            "His profile is sharply chapter-shaped, which makes him useful for showing how aggregate positivity can arise from uneven terrain.",
        ],
        "primary_pattern": "rehabilitated_positive",
        "reading_path": [
            {"chapter_id": "v5", "label": "Main late positive recovery"},
            {"chapter_id": "v1-p1-combray", "label": "Early negative counterweight"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Intermediate positive reinforcement"},
        ],
    },
    "Legrandin": {
        "subheading": "A broadly negative recurring figure marked by repeated self-positioning failures and social discredit.",
        "summary": "Legrandin is a broadly negative figure whose profile is shaped by repeated discredit and awkward self-positioning, even though a few isolated units briefly interrupt the downward pattern.",
        "why_interesting": [
            "He is a clear recurring loser without depending on a single chapter block.",
            "His profile sharpens the category of embarrassment-driven negative treatment.",
        ],
        "primary_pattern": "awkward_negative",
        "reading_path": [
            {"chapter_id": "v1-p1-combray", "label": "Primary negative concentration"},
            {"chapter_id": "v6-p4", "label": "Late positive interruption"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "Final return in diminished society"},
        ],
    },
    "Mme de Cambremer": {
        "subheading": "A compact but stable negative case: limited in footprint, but strongly downward where it appears.",
        "summary": "Mme de Cambremer is a compact but stable negative case: she does not dominate the corpus by volume, but what is there reads overwhelmingly downward in advantage, prestige, and inclusion.",
        "why_interesting": [
            "She is useful as a smaller-scale confirmation that consistent multi-lens negativity is not limited to the biggest characters.",
            "Her appearances are sparse enough to stay legible but numerous enough to matter.",
        ],
        "primary_pattern": "compact_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Primary negative concentration"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Supporting negative evidence"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Prestige-world reinforcement"},
        ],
    },
    "M. Vinteuil": {
        "subheading": "A surprising aggregate positive whose late recoveries outweigh strongly negative early material.",
        "summary": "M. Vinteuil is one of the more surprising positive figures in the corpus: despite some strongly negative early material, his aggregate treatment ends up decisively positive, especially in inclusion.",
        "why_interesting": [
            "He is a genuine reversal case rather than a merely stable positive.",
            "His profile is sharply chapter-shaped, which makes him useful for showing how aggregate positivity can arise from uneven terrain.",
        ],
        "primary_pattern": "rehabilitated_positive",
        "reading_path": [
            {"chapter_id": "v5", "label": "Main late positive recovery"},
            {"chapter_id": "v1-p1-combray", "label": "Early negative counterweight"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Intermediate positive reinforcement"},
        ],
    },
}


CHAPTER_SUMMARY_EDITORIAL = {
    "v1-p1-combray": (
        "Combray is organized around the household and its visitors, with Françoise, Swann, Legrandin, and the Vinteuil circle repeatedly coming under scrutiny. "
        "What stands out is not one decisive reversal but a steady accumulation of slights, embarrassments, and exclusions, especially around Swann and Legrandin. "
        "The chapter's social world feels watchful and punitive, turning ordinary encounters into tests of place and acceptance."
    ),
    "v1-p2-un-amour-de-swann": (
        "This chapter is dominated by Swann's pursuit of Odette and by the salon world around the Verdurins, Cottard, and their circle. "
        "Attachment, jealousy, dependence, and access to Odette matter more here than formal rank, and Swann absorbs the heaviest losses by far. "
        "Even when other figures are humiliated or briefly advanced, the chapter keeps returning to Swann's growing emotional subjection."
    ),
    "v1-p3-noms-de-pays-le-nom": (
        "This section is centered on the young narrator's idealizing imagination, especially around Gilberte, Swann, and Odette. "
        "Unlike the darker social weather of the earlier chapters, it is largely buoyed by fascination, projection, and moments of elevation, though Gilberte's hold over the scene can flip from promise to hurt. "
        "The chapter feels lighter and more aspirational, with desire and fantasy doing more work than humiliation."
    ),
    "v2-p1-autour-de-mme-swann": (
        "The chapter revolves around Odette's renewed brilliance, Swann's diminished position beside her, and the narrator's fascination with the Swann household and its orbit. "
        "Its most revealing pattern is the contrast between Odette's steady rise and Swann's repeated weakening, which gives the chapter the shape of a social reordering inside an apparently elegant world. "
        "Rather than one sharp crisis, it builds its meaning through repeated scenes in which glamour and access gather around Odette."
    ),
    "v2-p2-noms-de-pays-le-pays": (
        "Balbec is organized around new encounters and shifting attractions, especially Elstir, Albertine, Charlus, Saint-Loup, and the narrator's movements among them. "
        "The chapter's most striking pattern is its mixed social weather: artistic and social arrival on one side, awkwardness and exclusion on the other, so that discovery and discomfort keep alternating. "
        "Elstir emerges as the clearest source of uplift, while the narrator's own footing remains noticeably less secure."
    ),
    "v3-p1": (
        "This chapter is centered on entry into the Guermantes world, with the duchesse, Saint-Loup, Bloch, Charlus, and Odette all helping define its social landscape. "
        "What makes it interesting is the split between glitter at the top and repeated embarrassment lower down: the duchesse is repeatedly confirmed, while Bloch is persistently cut down, and Saint-Loup looks admirable in public yet less secure in matters of closeness and belonging. "
        "High society dazzles here, but it also exposes how unevenly its rewards are distributed."
    ),
    "v3-p2": (
        "This chapter is dominated by the Guermantes world, with the duchesse repeatedly gathering brilliance and authority around herself while the duc, Swann, and the narrator's family move through a more exposed atmosphere. "
        "What stands out is the contrast between her sustained command of the room and the steady diminishment of figures around her, especially the duc and Swann. "
        "The chapter feels less like a single reversal than a long society performance in which one woman keeps controlling the terms."
    ),
    "v4-p1": (
        "This short chapter is centered almost entirely on the charged encounter between Charlus and Jupien. "
        "Its interest lies in the abrupt swing from Charlus's initial excitement to his loss of footing, while Jupien emerges unexpectedly strengthened. "
        "The scene reads like a compressed seduction and reversal, with power passing more quickly than the surface first suggests."
    ),
    "v4-p2": (
        "This chapter moves through salons, humiliations, and uneasy encounters, with Swann, Albertine, Vaugoubert, and Charlus all caught in a social world that keeps turning against them. "
        "The strongest pressure is not just embarrassment but repeated public diminishment, especially in the scenes around Swann and the smaller cruelties inflicted on figures like Saniette and Saint-Euverte. "
        "Even when someone briefly recovers, the larger impression is of a chapter that strips people of ease, dignity, and welcome."
    ),
    "v5": (
        "This chapter is overwhelmingly organized around Albertine, with Charlus and Morel forming a secondary line of strain around her. "
        "The most striking feature is how relentlessly Albertine is placed under suspicion, confinement, and emotional pressure, so that nearly every temporary recovery gives way to another loss of ground. "
        "The chapter is shaped less by one scandal than by sustained possession, surveillance, and attrition."
    ),
    "v6-p1": (
        "This chapter centers on Albertine in absence, with Saint-Loup and a few others entering a field shaped by memory, inquiry, and grief after her disappearance. "
        "The pressure is quieter and more uneven than in the previous chapter: Albertine still absorbs the deepest losses, especially around intimacy and belonging, but the chapter keeps wavering between recollection, idealization, and renewed hurt. "
        "It feels like mourning that cannot settle into a single story."
    ),
    "v6-p2": (
        "This chapter turns around Swann, Albertine, and Gilberte, with Gilberte especially moving in and out of favor as the social mood shifts around her. "
        "Its most revealing pattern is the contrast between brief moments of renewed brightness and the stronger undertow of estrangement, particularly for Swann and Albertine. "
        "The chapter feels like one of unstable remembrance, where recognition and loss keep interrupting one another."
    ),
    "v6-p3": (
        "This chapter is organized around Albertine's absence, the narrator's family world, and a smaller set of recurring figures such as Norpois and Mme de Villeparisis. "
        "What stands out is the contrast between private consolation and lingering damage: the narrator's mother can still steady the scene, while Albertine and Villeparisis keep pulling it back toward loss and diminishment. "
        "The effect is quieter than the surrounding chapters, but the grief remains active beneath the surface."
    ),
    "v6-p4": (
        "This short section centers on Legrandin, Robert de Saint-Loup, and Gilberte. "
        "Its most interesting feature is the unstable balance between brief social recovery and renewed distance, especially around Saint-Loup, who can look momentarily restored before the chapter turns again. "
        "The result is a compact chapter of partial repair that never fully settles into reassurance."
    ),
    "v7-p1-a-tansonville": (
        "This chapter is organized around Robert de Saint-Loup, Mme Bontemps, Swann, and the lingering afterlife of earlier attachments. "
        "What dominates is not brilliance but loss: even when Swann briefly brightens the scene, Saint-Loup and those around him are pulled back toward diminishment and damage. "
        "The chapter feels like a return under shadow, with memory and decline arriving together."
    ),
    "v7-p2-m-de-charlus-pendant-la-guerre": (
        "This chapter is dominated by Charlus in wartime, with Gilberte and Mme Verdurin forming the most important counterweights around him. "
        "Charlus absorbs the heaviest damage, but the chapter is not flatly downward: figures like Gilberte and Mme Verdurin can still rise sharply inside the same harsh field. "
        "The result is a wartime chapter of exposure and reversal, where ruin and sudden social advantage coexist."
    ),
    "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle": (
        "This chapter returns to the high society world in a late, exhausted form, with Charlus, the duc de Guermantes, Swann, and other major figures appearing under the sign of decline. "
        "What matters most is the cumulative sense of diminishment: prestige remains visible, but it is no longer renewing anyone, and even the most famous figures now appear worn down. "
        "The chapter reads like an inventory of faded greatness rather than a fresh social ascent."
    ),
    "v7-p4-le-bal-de-tetes": (
        "The final chapter is centered on the spectacle of aging and reappearance, with the duc de Guermantes, la Berma, Gilberte, and other once-commanding figures returning in altered form. "
        "The dominant pattern is broad diminishment: rank, poise, and belonging all come under pressure at once, so that even famous names seem exposed to time rather than protected by status. "
        "Swann stands out as an especially uneven figure here, but the larger effect is a society seen through its losses."
    ),
}
