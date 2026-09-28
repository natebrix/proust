CHARACTER_PORTRAIT_SLUGS = {
    "Albertine": "albertine",
    "Odette": "odette",
    "Robert de Saint-Loup": "saint-loup",
    "Swann": "swann",
    "baron de Charlus": "charlus",
    "le narrateur": "le-narrateur",
    "Morel": "morel",
    "Rachel": "rachel",
}


CHARACTER_PAGE_PILOT_EDITORIAL = {
    "Morel": {
        "subheading": "The violinist ranks 2nd of 14 in standing, behind only the duchesse de Guermantes, and loses most of his scenes, where he is 25th of 31.",
        "summary": "Morel ranks 2nd of 14 in standing, behind only the duchesse de Guermantes. His talent, and the protection Charlus buys for it, set him above dukes and barons in the salons. The scenes themselves go against him. He is 25th of 31 in the scenes, and for every passage that leaves him better off, four or more leave him worse. His belonging is staged too rarely to rank. His fortune reaches its low point in La Prisonnière, where he breaks with Charlus, and recovers by the Bal de têtes, where the wartime deserter has become a decorated and respected man.",
        "why_interesting": [
            "His standing rests on protectors, Charlus above all, which makes it real and precarious at once.",
            "His scenes are sharply negative: 22 passages leave him worse off and 5 leave him better.",
            "His fortune bottoms out in La Prisonnière and climbs back by the Bal de têtes, where the deserter ends the novel honored.",
        ],
        "primary_pattern": "prestige_high_scene_negative",
        "reading_path": [
            {"chapter_id": "v4-p2", "label": "The Verdurin salon, under Charlus's patronage"},
            {"chapter_id": "v5", "label": "The break with Charlus"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Wartime: the protégé outlives the protector"},
        ],
    },
    "Rachel": {
        "subheading": "Saint-Loup's mistress becomes the celebrated actress of the last chapter, and she ranks 8th of 31 in the scenes.",
        "summary": "Rachel appears across the whole novel, from the young woman Saint-Loup keeps in Le Côté de Guermantes to the actress whose recital empties la Berma's salon at the Bal de têtes. She ranks 8th of 31 in the scenes, with 27 wins, 20 losses and 9 draws. Her standing looks high, but the novel shows it in only 10 passages, too few to rank, and her belonging is thinner still. Her fortune rises from Le Côté de Guermantes II to the end of the book, though not steeply enough to call a clear arc.",
        "why_interesting": [
            "Her triumph at the Bal de têtes is one of the sharpest reversals in the novel. The woman once sold for twenty francs is celebrated at the princesse's matinée while la Berma waits for guests who never come.",
            "She wins more of her scenes than she loses, 27 to 20, which is striking for a character the narrator first meets as a woman for sale.",
            "Her standing is the open question. It looks high, but the novel shows it too seldom to rank.",
        ],
        "primary_pattern": "scenes_strong_standing_unranked",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "\"Rachel quand du Seigneur\" and Saint-Loup's love"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "Her recital, and la Berma's empty salon"},
        ],
    },
    "le narrateur": {
        "subheading": "He loses more scenes than he wins, yet he ranks 1st of 8 in belonging, 4th of 14 in standing and 7th of 31 in the scenes.",
        "summary": "Nearly every passage in the novel passes through the narrator, and the scenes often go badly for him: 168 wins against 200 losses, with 81 passages leaving him worse off and 46 better. Across the book his welcome holds all the same. He ranks 1st of 8 in belonging and 4th of 14 in standing, and his place in the scenes, 7th of 31, owes as much to how often he is seen as to what he wins, since no one's rating is more certain. His fortune rises through Le Côté de Guermantes, peaks in the Guermantes salons, and falls in La Prisonnière. The book's last turn, the discovery of his vocation in L'Adoration perpétuelle, depends on no one's regard but his own, so it leaves no trace in his ratings.",
        "why_interesting": [
            "The rooms keep receiving him while the scenes keep wounding him. That split is the book's central irony about its narrator.",
            "Because the whole novel passes through him, his rating in the scenes has the narrowest uncertainty of anyone's.",
            "His one lasting victory, the vocation he finds in L'Adoration perpétuelle, happens alone, where no other character can grant or refuse it.",
        ],
        "primary_pattern": "relational_positive_understated",
        "reading_path": [
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec: new people, new rooms"},
            {"chapter_id": "v3-p2", "label": "Received by the Guermantes"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "The Bal de têtes: the survivor among the masks"},
        ],
    },
    "Odette": {
        "subheading": "Odette ends the novel with more standing than most of the people who once refused to receive her, while her scenes stay even and her welcome stays thin.",
        "summary": "Odette ranks 3rd of 14 in standing, behind only the duchesse de Guermantes and Morel. In her scenes she sits at the middle, 15th of 31, with 112 wins to 107 losses, and in belonging she is 5th of 8. This is the pattern Proust gives her: the Faubourg learns to defer to Mme Swann, and later to Mme de Forcheville, long before it lets her in. Across the novel her standing edges up from Combray to Sodome et Gomorrhe II and holds to the end, while the passages as a whole drift slowly against her, from 1482 in Combray to 1346 at the Bal de têtes. Neither movement is large enough to call a clear arc.",
        "why_interesting": [
            "Standing is her strongest measure, and it is the one the salons fought hardest. The demi-mondaine of Un amour de Swann ends above most of the Faubourg.",
            "Her three rankings pull apart: high standing, even scenes, modest belonging. She is received as a name before she is received as a person.",
            "Her scene record is nearly even, which suits a woman who advances by marriage and patience more than by winning rooms.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Mme Swann's salon"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Swann's love, and the Verdurin circle"},
            {"chapter_id": "v3-p1", "label": "Glimpsed from the Guermantes world"},
        ],
    },
    "Robert de Saint-Loup": {
        "subheading": "Saint-Loup is in more scenes than almost anyone, and ranks in the middle of them, 16th of 31, and 9th of 14 in standing.",
        "summary": "Saint-Loup is one of the most present figures in the novel, and his record is close to even: 16th of 31 in the scenes, with 105 wins and 108 losses. In standing, the measure his name should command, he is 9th of 14, behind his uncle Charlus, his great-aunt Mme de Villeparisis, Odette, and Gilberte, the woman he marries. His belonging is staged too rarely to rank. His fortune dips in Le Côté de Guermantes I, where Rachel and the barracks at Doncières fill his scenes, then rises to the end of the book, where the soldier killed in the war is remembered well.",
        "why_interesting": [
            "His standing sits below his family's: 9th of 14, behind his uncle and his great-aunt, and behind Odette.",
            "His scene record is almost exactly even, 105 wins to 108 losses. Presence, more than triumph, holds his place.",
            "His fortune climbs through the last volumes, from the low of Doncières and Rachel to the war that kills him.",
        ],
        "primary_pattern": "broad_presence_middling",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Doncières, Rachel, and the family salons"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "The friendship begins at Balbec"},
            {"chapter_id": "v7-p1-a-tansonville", "label": "Tansonville: the unhappy marriage"},
        ],
    },
    "Swann": {
        "subheading": "Swann is one of the most present men in the novel and one of its steadiest losers, 19th of 31 in the scenes, 10th of 14 in standing and 7th of 8 in belonging.",
        "summary": "Swann appears in 177 passages, more than anyone but the narrator and the duchesse, and the scenes go against him: 144 wins, 197 losses, and 83 passages that leave him worse off against 46 that leave him better. He ranks 19th of 31 in the scenes and 10th of 14 in standing, a modest place for the man Combray never knew dined with princes, because the novel shows his standing mostly after his marriage to Odette has cost him. In belonging he is 7th of 8. His fortune falls from Combray to Albertine disparue II, where the Guermantes will not speak his name, the second largest fall among the novel's main figures after Charlus's.",
        "why_interesting": [
            "He appears in so many passages that his decline is no accident of a few scenes.",
            "His standing is shown mostly on its way down, after the marriage, so the Swann the novel lets us watch in society is already the diminished one.",
            "He is 7th of 8 in belonging. The man who once dined with princes ends as a name the Guermantes avoid.",
        ],
        "primary_pattern": "broad_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Un amour de Swann"},
            {"chapter_id": "v1-p1-combray", "label": "Combray: the neighbor who dines with princes"},
            {"chapter_id": "v4-p2", "label": "The last evenings, ill and unwelcome"},
        ],
    },
    "Albertine": {
        "subheading": "Albertine falls from the girl on the beach at Balbec to the captive of La Prisonnière, one of the clearest declines in the novel, while her scenes split almost evenly.",
        "summary": "Albertine's scenes are among the most conflicted in the book, with 80 wins, 84 losses and more mixed passages than most characters. She ranks 12th of 31 in the scenes. Her standing and her belonging are both staged too seldom to rank, and much of her confinement is the narrator's doing, one man shutting her in more than society shutting her out. Her fortune falls clearly, from its high point at Balbec in Noms de pays : le pays to its low in La Prisonnière. The decline shows in her scenes as well, which keep turning against her into Albertine disparue.",
        "why_interesting": [
            "Hers is one of the clearest declines in the novel. The girl the narrator first sees on the Balbec beach becomes the prisoner of his apartment.",
            "Her scenes split nearly evenly, 80 wins to 84 losses, with an unusual share of mixed passages, the sign of a character pulled both ways at once.",
            "Her exclusion is private. It is the narrator who shuts her in, and the world's verdict on her is staged too rarely to rank.",
        ],
        "primary_pattern": "volatile_scenes_standing_holds",
        "reading_path": [
            {"chapter_id": "v5", "label": "La Prisonnière"},
            {"chapter_id": "v6-p1", "label": "After her flight and death"},
            {"chapter_id": "v6-p2", "label": "Forgetting Albertine"},
        ],
    },
    "baron de Charlus": {
        "subheading": "The largest fall in the novel, from his first appearances at Balbec to the ruined old man of L'Adoration perpétuelle.",
        "summary": "Charlus ranks in all three measures: 13th of 31 in the scenes, 7th of 14 in standing and 3rd of 8 in belonging. Those middling places average a long height and a steep fall. His fortune is the largest decline in the book, from 1627 at Balbec to 1149 in L'Adoration perpétuelle, and the fall is clear in his scenes, in his standing, and in his fortune as a whole. It runs through the Verdurins' expulsion of him in La Prisonnière and the wartime Paris of M. de Charlus pendant la guerre, and it ends with the old baron bowing to Mme de Saint-Euverte, a woman he once refused to acknowledge.",
        "why_interesting": [
            "His is the largest fall in the novel, 479 points, and the only one that is clear in the scenes, in standing and overall at once.",
            "His middling ranks hide the shape of his story, a long summit and a collapse averaged into a place near the middle.",
            "He is 3rd of 8 in belonging. The novel installs the clubbable baron everywhere before it evicts him.",
        ],
        "primary_pattern": "ranked_everywhere_late_fall",
        "reading_path": [
            {"chapter_id": "v4-p2", "label": "The Verdurin salon at la Raspelière"},
            {"chapter_id": "v5", "label": "The expulsion from the Verdurins"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Wartime Paris and Jupien's hotel"},
        ],
    },
    "duchesse de Guermantes": {
        "subheading": "First in the scenes, first in standing, and second only to the narrator in belonging.",
        "summary": "The duchesse ranks 1st of 31 in the scenes, 1st of 14 in standing and 2nd of 8 in belonging, behind only the narrator. No one else places in the top three of every measure. Her scenes bear it out, with 225 wins against 92 losses, the wit carrying the evening far more often than it fails her. Her fortune rises into Le Côté de Guermantes, where the narrator enters her world, and declines after it. By the Bal de têtes her standing has clearly fallen, as the aging duchesse takes up with actresses and Rachel outshines la Berma.",
        "why_interesting": [
            "She tops two of the three measures and is second in the third, the most complete high placement in the book.",
            "Her scene record, 225 wins and 92 losses, is the most one-sided winning record of any often-seen character.",
            "Her standing is also one of the few clear declines at the end of the book. The queen of the Faubourg fades at the Bal de têtes.",
        ],
        "primary_pattern": "uniform_positive",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "The Guermantes seen from outside"},
            {"chapter_id": "v3-p2", "label": "Dinner at the Guermantes"},
            {"chapter_id": "v4-p2", "label": "The princesse's soirée"},
        ],
    },
    "Mme de Villeparisis": {
        "subheading": "The hostess of the Balbec hotel and the Paris matinée ranks 4th of 31 in the scenes and 8th of 14 in standing.",
        "summary": "Mme de Villeparisis ranks 4th of 31 in the scenes, on 72 wins and 38 losses, and 8th of 14 in standing. Her matinée in Le Côté de Guermantes is some of the most closely staged social machinery in the novel, and she runs it, taking as much credit as the guests who shine there. Her belonging is harder to place. Everyone receives her and no one knows quite where to seat her, and the novel stages it too rarely to rank. Her fortune declines gently from Balbec to Albertine disparue III, where she is last seen in Venice with Norpois.",
        "why_interesting": [
            "Among often-seen characters, only the duchesse wins a larger share of her scenes.",
            "Her standing is real but unsettled. She is grand enough to bring the princesse de Luxembourg to the Balbec hotel and still doubtful to the Guermantes.",
            "She fades quietly, last seen in Venice with Norpois, both of them old.",
        ],
        "primary_pattern": "advantage_strong_prestige_ranked",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Her matinée"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec, and the princesse de Luxembourg"},
            {"chapter_id": "v6-p3", "label": "Venice with Norpois"},
        ],
    },
    "Françoise": {
        "subheading": "The family's cook ranks 5th of 31 in the scenes, ahead of most of the aristocrats in the novel.",
        "summary": "Françoise ranks 5th of 31 in the scenes, on a winning record of 51 wins, 38 losses and 11 draws. The kitchen, the sickroom and the servants' table are her ground, and the novel lets her win there more often than most of the aristocrats win in theirs. The deference she commands from footmen, doctors and households is standing too, but the novel shows it too seldom to rank. Her fortune peaks in Autour de Mme Swann and settles back near where it began.",
        "why_interesting": [
            "She wins more of her scenes than all but four characters, on genuinely winning footing.",
            "The deference she commands counts as standing on the same scale as a duchesse's, though the novel shows it too rarely to rank.",
            "Her record is durable more than dazzling: 51 wins, 38 losses and 11 draws.",
        ],
        "primary_pattern": "advantage_high_durable",
        "reading_path": [
            {"chapter_id": "v1-p1-combray", "label": "The Combray kitchen"},
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec"},
            {"chapter_id": "v4-p2", "label": "Paris, and the second Balbec"},
        ],
    },
    "Mme Verdurin": {
        "subheading": "Mme Verdurin ranks 6th of 14 in standing and 10th of 31 in the scenes, and she is last of 8 in belonging.",
        "summary": "Mme Verdurin ranks 6th of 14 in standing, 10th of 31 in the scenes, and 8th of 8 in belonging. Her standing rises across the book, from the bourgeois patronne of Un amour de Swann to the princesse de Guermantes of the last chapters. Her scenes are harsh, with 33 passages leaving her worse off against 4 that leave her better, and her belonging is the lowest of anyone ranked there. The woman who built the most exclusive little clan in Paris is rarely shown securely inside anything. Her fortune peaks in wartime Paris, in M. de Charlus pendant la guerre, and sinks again at the Bal de têtes, where the new princesse is a figure of fun.",
        "why_interesting": [
            "Her three ranks tell three different stories: upper half in standing, upper third in the scenes, and last in belonging.",
            "She loses belonging at her own parties. At the soirée Charlus arranges in her salon, his aristocratic guests walk past her to thank him.",
            "Her title arrives as a joke. By the Bal de têtes the new princesse de Guermantes is mocked, and her fortune falls from its wartime peak.",
        ],
        "primary_pattern": "prestige_positive_inclusion_negative",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "The little clan"},
            {"chapter_id": "v5", "label": "The soirée Charlus arranges, and his expulsion"},
            {"chapter_id": "v7-p2-m-de-charlus-pendant-la-guerre", "label": "Wartime: the salon at its height"},
        ],
    },
    "Gilberte": {
        "subheading": "Swann's daughter changes her name twice and moves further inside society each time, ending 5th of 14 in standing and 4th of 8 in belonging.",
        "summary": "Gilberte ranks 5th of 14 in standing, 4th of 8 in belonging and 17th of 31 in the scenes, where she wins and loses almost equally, 64 to 61. Her standing and belonging follow the novel's study of changed names: Mlle Swann, then Mlle de Forcheville, then the marquise de Saint-Loup, each name opening a door the last one could not. The salon that would not receive Mlle Swann receives Mlle de Forcheville. Her scenes run the other way. Her fortune in them falls clearly, from the Champs-Élysées of Noms de pays : le nom to the Bal de têtes.",
        "why_interesting": [
            "The Guermantes salon that would not receive Mlle Swann receives Mlle de Forcheville, one of the plainest boundary crossings in the book.",
            "Her strengths run opposite to her father's. His standing and belonging collapse as hers grow.",
            "Her scenes, even overall, decline clearly across the novel, from the girl of the Champs-Élysées to the marquise of the Bal de têtes.",
        ],
        "primary_pattern": "inclusion_positive_prestige_positive_advantage_negative",
        "reading_path": [
            {"chapter_id": "v1-p3-noms-de-pays-le-nom", "label": "The Champs-Élysées"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "The Swann household"},
            {"chapter_id": "v6-p2", "label": "Mlle de Forcheville"},
        ],
    },
    "Norpois": {
        "subheading": "The ambassador is deferred to everywhere and wins about half his scenes, ranking 11th of 14 in standing and 20th of 31 in the scenes.",
        "summary": "Norpois ranks 11th of 14 in standing and 20th of 31 in the scenes, where his record is almost a perfect draw: 45 wins, 44 losses and 12 draws. The deference paid to an ambassador is shown constantly, even in passages where his conversation wins nothing, which is very close to the joke the novel tells about him. His fortune slides from Le Côté de Guermantes I to the wartime chapters, where his newspaper articles have become a target of the narrator's irony.",
        "why_interesting": [
            "The novel stages the ceremony around him relentlessly, and mostly with irony.",
            "His scene record is nearly a perfect draw, 45-44-12. The master of official language neither wins nor loses rooms.",
            "His standing outlasts his scenes: 11th of 14 in standing, 20th of 31 in the scenes.",
        ],
        "primary_pattern": "reputation_ranked_scenes_even",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Dinner with the narrator's parents"},
            {"chapter_id": "v3-p1", "label": "Mme de Villeparisis's matinée"},
            {"chapter_id": "v6-p3", "label": "Venice"},
        ],
    },
    "la grand-mère": {
        "subheading": "The narrator's grandmother wins her scenes, 9th of 31, and her fortune rises through the book, most of all after her death.",
        "summary": "The narrator's grandmother ranks 9th of 31 in the scenes, on 37 wins and 31 losses, with more passages leaving her better off than worse. Her belonging and standing are staged too rarely to rank, though both lean warmly upward, including the princesse de Luxembourg's greeting at Balbec. Her fortune rises from Combray to Sodome et Gomorrhe II, where the narrator, a year after her death, finally grieves for her. It is one of the few arcs in the book that climb after the character has died.",
        "why_interesting": [
            "Her scenes genuinely go her way, with 17 passages leaving her better off against 14 worse, which is rarer in this novel than it sounds.",
            "The princesse de Luxembourg's greeting at Balbec, treating a bourgeois grandmother as her equal, is one of the plainest displays of standing in the book.",
            "Her fortune peaks in the passages of grief in Sodome et Gomorrhe II, when the narrator understands for the first time that she is gone.",
        ],
        "primary_pattern": "advantage_strongly_positive",
        "reading_path": [
            {"chapter_id": "v2-p2-noms-de-pays-le-pays", "label": "Balbec"},
            {"chapter_id": "v1-p1-combray", "label": "Combray"},
            {"chapter_id": "v3-p1", "label": "Paris, and the telephone call from Doncières"},
        ],
    },
    "Bloch": {
        "subheading": "Bloch is near the bottom wherever others can see him, 30th of 31 in the scenes, 13th of 14 in standing and 6th of 8 in belonging.",
        "summary": "Bloch ranks 30th of 31 in the scenes, where he loses 100 times and wins 31, and 43 passages leave him worse off against 8 that leave him better. He is 13th of 14 in standing and 6th of 8 in belonging. The novel gives him the gaffes, the wrong clothes, the family manners, and later the new name, Jacques du Rozier. His success as a playwright is real but happens mostly offstage, and the rooms the novel shows are the ones that cost him. His fortune recovers a little from Le Côté de Guermantes II to the Bal de têtes, where he is an established man of letters.",
        "why_interesting": [
            "His scene record, 31 wins and 100 losses, is the most one-sided losing record of any often-seen character.",
            "All three measures agree about him, which is rare. Together they make him the book's most relentless study of the outsider the salons will not absorb.",
            "His late rise is small but real. By the Bal de têtes, the young defer to him.",
        ],
        "primary_pattern": "broad_negative",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "Mme de Villeparisis's matinée"},
            {"chapter_id": "v1-p1-combray", "label": "Combray: the school friend the family distrusts"},
            {"chapter_id": "v3-p2", "label": "Among the Guermantes"},
        ],
    },
    "duc de Guermantes": {
        "subheading": "The head of the Guermantes family ranks last in standing, 14th of 14, and 27th of 31 in the scenes.",
        "summary": "The duc de Guermantes ranks 14th of 14 in standing and 27th of 31 in the scenes, where he loses 126 times and wins 75. Passages that cut him outnumber those that lift him 53 to 5: the Jockey Club election he loses, the dying cousin he will not mourn for fear of missing a costume ball, his wife's wit at his expense. His belonging is staged too rarely to rank. His fortune falls from his first appearance, as the prince des Laumes of Un amour de Swann, to La Prisonnière, and stays low to the end, where the old duc is Odette's lover.",
        "why_interesting": [
            "He is last of 14 in standing and 27th of 31 in the scenes, which suits the book's running joke about him.",
            "Among the great aristocrats, his scenes are the most one-sided against him, with 5 passages lifting him and 53 cutting him.",
            "Against his wife the contrast is complete. She leads two measures; he reaches the top half of none.",
        ],
        "primary_pattern": "title_and_scenes_low",
        "reading_path": [
            {"chapter_id": "v3-p2", "label": "Dinner at the Guermantes"},
            {"chapter_id": "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle", "label": "The matinée"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "The Bal de têtes"},
        ],
    },
    "docteur Cottard": {
        "subheading": "The Verdurins' doctor wins more scenes than he loses, 14th of 31 in the scenes, while the narration laughs at him.",
        "summary": "Cottard ranks 14th of 31 in the scenes, on 50 wins and 43 losses. His wins come mostly from the Verdurin salon's crowded evenings, where his puns land with the faithful and his diagnoses impress them. He is a figure of fun in the telling, with 19 passages mocking him against 9 that flatter him, yet the outcomes go his way, which is precisely Proust's joke about medicine. His standing as an eminent specialist is shown too rarely to rank. His fortune rises from Un amour de Swann to Le Côté de Guermantes II, the doctor making his way up.",
        "why_interesting": [
            "The narration mocks him, 19 passages against 9, while the scenes keep handing him the win.",
            "He is both the idiot of the salon and the great clinician, and his record carries both.",
            "His standing as a specialist is asserted in the later volumes but shown too rarely to rank.",
        ],
        "primary_pattern": "advantage_positive_texture_mocking",
        "reading_path": [
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "The little clan"},
            {"chapter_id": "v4-p2", "label": "La Raspelière"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "The Swann circle"},
        ],
    },
    "la mère du narrateur": {
        "subheading": "The narrator's mother has the cleanest winning record in the book, 6th of 31 in the scenes, with nine passages lifting her for every one that cuts.",
        "summary": "The narrator's mother ranks 6th of 31 in the scenes, the highest of anyone in the family, on 34 wins and 16 losses. Nine passages leave her better off for every one that leaves her worse, the most consistently favorable record of any ranked character. Her authority is domestic and effective: the goodnight kiss, the verdicts the household accepts, the quiet management of the father. Her standing and belonging are staged too rarely to rank, and her belonging leans slightly negative, the cost of being the one who decides who reaches the child rather than the one admitted anywhere herself.",
        "why_interesting": [
            "Nine passages lift her for every one that cuts, the most favorable ratio of any ranked character.",
            "She ranks 6th of 31 in the scenes, above everyone else in the family and above most of the salons.",
            "Her belonging leans against her, a quiet irony. The guardian of the family's inside is rarely shown crossing into anyone else's.",
        ],
        "primary_pattern": "familial_positive",
        "reading_path": [
            {"chapter_id": "v1-p1-combray", "label": "Combray and the goodnight kiss"},
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Paris"},
            {"chapter_id": "v3-p2", "label": "After the grandmother's death"},
        ],
    },
    "Bergotte": {
        "subheading": "The admired writer wins a little more often than he loses, 21st of 31 in the scenes, while his fame happens mostly offstage.",
        "summary": "Bergotte ranks 21st of 31 in the scenes, with 25 wins and 21 losses. The aura of the name is real, and it belongs to his standing, which leans high but is staged too rarely to rank. The novel shows him less than his fame makes it feel: the author of the narrator's youth, met at the Swanns' table, and later the sick old man who dies before Vermeer's View of Delft. His fortune rises to La Prisonnière, where he dies before the little patch of yellow wall, and slips in the last volume.",
        "why_interesting": [
            "His scenes are winning but modest, 25 wins to 21 losses, closer to the dying man before the Vermeer than to the legend at the Swanns' table.",
            "His standing leans among the highest of anyone the novel shows too rarely to rank. The fame is real, and the novel mostly keeps it offstage.",
            "Like Norpois, he is more a reputation than a presence in the rooms the novel shows.",
        ],
        "primary_pattern": "advantage_positive_reputation_offstage",
        "reading_path": [
            {"chapter_id": "v2-p1-autour-de-mme-swann", "label": "Lunch at the Swanns'"},
            {"chapter_id": "v3-p1", "label": "The Guermantes world"},
            {"chapter_id": "v1-p1-combray", "label": "Combray: the books before the man"},
        ],
    },
    "Legrandin": {
        "subheading": "The snob loses nearly every scene he is in, 8 wins against 28 losses, while he performs a standing he does not have.",
        "summary": "Legrandin loses 28 of his scenes and wins 8, and 15 passages leave him worse off against 2 that leave him better, one of the lowest ratings in the book. He appears in too few passages for his scenes to be ranked. What the novel records of him is his performance of standing: the bows calibrated for aristocratic eyes, the exquisite phrases, the snobbery he denounces in others. His standing, too thinly staged to rank, leans upward, because the pose is what the narration keeps witnessing. His fortune improves late, once he has become the comte de Méséglise. His profile is snobbery complete, the floor of the scenes and the ceiling of the pose.",
        "why_interesting": [
            "He loses more than three scenes for every one he wins, though in too few passages to be ranked.",
            "His standing leans among the highest of those shown too rarely to rank, because what the narration witnesses is the performance of standing.",
            "The pairing, the floor of the scenes and the ceiling of the pose, is the anatomy of snobbery.",
        ],
        "primary_pattern": "advantage_negative_prestige_performed",
        "reading_path": [
            {"chapter_id": "v3-p1", "label": "The Paris salons"},
            {"chapter_id": "v1-p1-combray", "label": "Combray: the snob on the church steps"},
            {"chapter_id": "v7-p4-le-bal-de-tetes", "label": "The Bal de têtes"},
        ],
    },
    "Mme de Cambremer": {
        "subheading": "Legrandin's sister, the young marquise, is last of 31 in the scenes, and no passage about her leaves her better off.",
        "summary": "Mme de Cambremer ranks 31st of 31 in the scenes, last of every ranked character, with 15 wins and 41 losses and no passage that leaves her better off. Her standing leans hard downward but is staged too rarely to rank. She is the provincial snob, Legrandin's sister, whose pretensions every Parisian room declines to honor, and Charlus's treatment of her at la Raspelière is one of the book's plainest snubs. She marks the floor of the scenes as the duchesse marks their ceiling.",
        "why_interesting": [
            "She is last in the scenes, and all 17 passages that move her leave her worse off.",
            "The novel shows her rarely and defeats her reliably.",
            "Her fortune reaches its low in Le Côté de Guermantes I and recovers somewhat by the last volume.",
        ],
        "primary_pattern": "compact_negative",
        "reading_path": [
            {"chapter_id": "v4-p2", "label": "La Raspelière"},
            {"chapter_id": "v3-p1", "label": "The Guermantes world"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "Mme de Saint-Euverte's soirée"},
        ],
    },
    "M. Vinteuil": {
        "subheading": "The humiliated piano teacher of Combray becomes, after his death, the composer whose septet transfigures La Prisonnière, the largest rise in the novel.",
        "summary": "M. Vinteuil appears in only 9 passages, too few to rank in any measure, and their order tells a complete story. In Combray he is the shy music teacher shamed by his daughter's reputation. In Un amour de Swann his sonata, not yet known to be his, becomes the anthem of Swann's love. In La Prisonnière his septet, deciphered from his notes by his daughter's friend, reveals him as a great composer. His fortune rises from 1449 in Combray to 1917 in La Prisonnière, a clear rise and the largest in the book, earned entirely after his death.",
        "why_interesting": [
            "His is the largest rise in the book, and it comes entirely after his death, through his music.",
            "The arc follows the chapters exactly: shame in Combray, the sonata in Un amour de Swann, the septet in La Prisonnière.",
            "His daughter's friend, who helped shame him, is the one who restores his work, one of the novel's strangest reparations.",
        ],
        "primary_pattern": "rehabilitated_positive",
        "reading_path": [
            {"chapter_id": "v5", "label": "The septet"},
            {"chapter_id": "v1-p1-combray", "label": "Combray"},
            {"chapter_id": "v1-p2-un-amour-de-swann", "label": "The sonata"},
        ],
    },
}


CHAPTER_SUMMARY_EDITORIAL = {
    "v1-p1-combray": (
        "Combray is the household and its visitors: the great-aunts, the grandmother, Swann on the garden path, Legrandin on the church square, and the Vinteuils at Montjouvain. "
        "Almost everyone loses a little here. Bloch, sent away after one visit, fares worst, followed by Legrandin, whose snubs the family learns to read, and M. Vinteuil, shamed by the gossip about his daughter. "
        "The one clear winner is the duchesse de Guermantes, glimpsed once in the church at Combray and still more a name than a woman."
    ),
    "v1-p2-un-amour-de-swann": (
        "This chapter belongs to Swann, and it goes badly for him. He is in almost every passage, and the passages that cost him outnumber the rest by far. "
        "The Verdurin salon that takes him in and then throws him out loses ground too: Mme Verdurin, Cottard and M. Verdurin all come off worse than they began, and so does Odette, whom Swann's jealousy follows everywhere. "
        "By the end Swann has spent years of his life on a woman who, as he says, was not his type."
    ),
    "v1-p3-noms-de-pays-le-nom": (
        "The short close of the first volume is the lightest chapter in the novel. "
        "Odette, seen walking in the Bois, and Gilberte, playing on the Champs-Élysées, both rise in the narrator's eyes. He himself loses a little, because Gilberte holds the power to make his afternoons happy or miserable. "
        "Desire and fantasy do most of the work here, and humiliation very little."
    ),
    "v2-p1-autour-de-mme-swann": (
        "The narrator gets his wish and is received in the Swann household. The chapter is less kind to its hosts than his admiration suggests. "
        "Odette, now Mme Swann, is at the center of more passages than anyone, and on balance they cost her. Swann loses ground beside her, and Gilberte's friendship cools. Bergotte, met at their table, comes off a little worse than his reputation. "
        "The winners are few: Françoise, and la Berma, whose Phèdre the narrator finally sees."
    ),
    "v2-p2-noms-de-pays-le-pays": (
        "Balbec brings the narrator new people: Saint-Loup, Charlus, Elstir, and the band of girls with Albertine among them. "
        "Charlus, Elstir, Albertine and the grandmother all come out ahead. The losers are Bloch, whose manners and family embarrass him at every turn, and the narrator himself, shy and uncertain in the Grand-Hôtel. "
        "It is a chapter of discovery, and its discomforts belong mostly to the one doing the discovering."
    ),
    "v3-p1": (
        "The narrator's family moves into a wing of the Hôtel de Guermantes, and the chapter follows his approach to the duchesse through Saint-Loup, Mme de Villeparisis's salon and Doncières. "
        "The duchesse rises steadily. Saint-Loup is here more than anyone and loses more than anyone, through his quarrels with Rachel and his awkwardness in his family's salons. Bloch, at the Villeparisis matinée, is cut down again and again. "
        "The world of the Guermantes shines at the top and is hard on nearly everyone below it."
    ),
    "v3-p2": (
        "The grandmother dies, and the narrator is finally invited to dine with the duc and duchesse de Guermantes. "
        "The duchesse dominates, and her wit wins her nearly every passage. Her husband pays for it. The duc loses more ground here than anyone, as the man she corrects and outshines, and at the end the husband who tells the dying Swann that he will outlive the doctors. "
        "The narrator, a guest for the first time, comes out slightly ahead."
    ),
    "v4-p1": (
        "This short chapter is the courtyard meeting between Charlus and Jupien, watched by the narrator from the stairs. "
        "Both men come out ahead. Charlus's pursuit succeeds, and Jupien, the waistcoat-maker, gains even more than the baron. "
        "It is one of only two chapters in the novel in which the passages, taken together, leave the people in them better off."
    ),
    "v4-p2": (
        "Sodome et Gomorrhe moves from the princesse de Guermantes's soirée to the second stay at Balbec and the little train to La Raspelière. "
        "It is one of the harshest chapters in the book. Swann, visibly ill, loses ground at the soirée. So do Mme de Saint-Euverte, Mme de Cambremer and Saniette, the Verdurins' favorite victim. Albertine, now under suspicion, and the duc de Guermantes lose too. "
        "Even the people who recover for a moment end the chapter with less ease and less welcome than they had."
    ),
    "v5": (
        "La Prisonnière belongs to Albertine, kept in the narrator's apartment under his watch. "
        "She is in more of its passages than anyone, and they wear her down one after another. Charlus loses next most, ending with his public disgrace at the Verdurins' musical evening. Morel, who turns on him that night, loses too, and so does the narrator. "
        "Almost nobody in this chapter comes out ahead."
    ),
    "v6-p1": (
        "Albertine has left, and then she dies. "
        "The chapter is almost entirely the narrator and his memories of her, and both lose ground. His inquiries into her past keep turning up new hurts, and his grief will not settle. "
        "Saint-Loup's mission to bring her back fails, and he loses a little as well."
    ),
    "v6-p2": (
        "The chapter turns on Gilberte, now Mlle de Forcheville, and on what she does with her father's memory. "
        "Swann, though dead, loses more here than anyone. His daughter drops his name, and the society he loved stops mentioning him. Gilberte's own standing holds steady while she loses ground in her friendships and attachments. "
        "The narrator's grief for Albertine fades in the background."
    ),
    "v6-p3": (
        "The narrator and his mother go to Venice. "
        "The few people the chapter lingers on lose ground. Norpois and Mme de Villeparisis, seen together in the hotel dining room, are an old couple whose influence is fading. Albertine, the object of a telegram and a last flicker of feeling, loses too, and the mother comes out almost even. "
        "It is a quiet chapter, and its losses are small."
    ),
    "v6-p4": (
        "This short section is a series of marriages: Gilberte to Saint-Loup, and Jupien's niece, made Mlle d'Oloron by Charlus, to the young Cambremer. "
        "Mlle d'Oloron rises, though she dies within weeks. Saint-Loup loses in every passage he appears in, as the narrator learns of his affair with Morel. "
        "Legrandin, the groom's uncle, gains a little."
    ),
    "v7-p1-a-tansonville": (
        "The narrator stays with Gilberte at Tansonville and reads the Goncourt journal. "
        "Every figure in the chapter loses a little. Saint-Loup loses most, as a husband who deceives Gilberte with men. Gilberte, the Verdurins, Elstir and Charlus all appear diminished in memory or in the Goncourt pages. "
        "The chapter is short and uniformly sad."
    ),
    "v7-p2-m-de-charlus-pendant-la-guerre": (
        "Paris in wartime, and Charlus in his worst days. "
        "He loses more than anyone here, from his walk with the narrator through the blackout to the night at Jupien's hotel. Others rise around him. Saint-Loup, a hero and then dead at the front, gains ground, and so do Gilberte and Mme Bontemps. Mme Verdurin, whose salon has become the center of wartime Paris, gains a little. "
        "The war ruins some people and makes others."
    ),
    "v7-p3-matinee-chez-la-princesse-de-guermantes-ladoration-perpetuelle": (
        "The narrator returns to society after years away, and before he reaches the matinée he meets Charlus, old and half-paralyzed, bowing to Mme de Saint-Euverte. "
        "Charlus loses heavily in both of his passages. The few others the chapter touches, Bergotte and the duc de Guermantes among them, lose a little. "
        "The rest of the chapter turns from people to the narrator's discovery of his vocation."
    ),
    "v7-p4-le-bal-de-tetes": (
        "The last chapter is the matinée itself, where the narrator finds everyone he has known made old. "
        "The duchesse de Guermantes loses the most, her salon in decline and her loyalty now given to Rachel. La Berma, whose own reception that afternoon goes unattended while Rachel recites at the matinée, loses nearly as much. Odette, Gilberte, Bloch and the duc lose ground too. "
        "Time spares almost nobody, and the new princesse de Guermantes is Mme Verdurin."
    ),
}
