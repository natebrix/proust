2. Recommended alias map structure

I would make the alias map a little richer than a simple flat dictionary. Not because the prompt strictly needs it, but because it will help you later with reference hygiene.

I recommend a structure like this:

Writing

{
"Swann": {
"aliases": [
"Swann",
"M. Swann",
"Monsieur Swann",
"Charles Swann"
],
"notes": "Use only for the character Charles Swann, not for household or family references unless clearly personal."
},
"la grand-mère": {
"aliases": [
"ma grand-mère",
"grand-mère",
"la grand-mère"
],
"notes": "Narrator's grandmother."
},
"la mère du narrateur": {
"aliases": [
"maman",
"ma mère",
"mère",
"la mère"
],
"notes": "Use only when context clearly indicates the narrator's mother."
},
"Mme de Villeparisis": {
"aliases": [
"Mme de Villeparisis",
"Madame de Villeparisis"
],
"notes": ""
},
"le narrateur": {
"aliases": [
"je",
"moi",
"le narrateur"
],
"notes": "Only resolve first-person references to this canonical name if the passage is clearly in narrator-centered voice and the annotation task requires treating the narrator as a named character."
}
}

I prefer this over a plain list for three reasons.

First, it gives you a place to store warnings like “do not resolve la mère unless local context is clear.”

Second, it gives you room later for metadata like:

family/group membership
stable IDs
volume/source provenance
whether this character is eligible for scoring in a given experiment

Third, it makes later debugging much easier.

My recommendation for the canonical key

Use a stable, human-readable canonical name in French. For example:

Swann
la grand-mère
la mère du narrateur
Mme de Guermantes

That will keep your annotations readable.

Optional richer version

If you want to future-proof the map, I would use:

Writing

{
"Swann": {
"character_id": "char_swann",
"display_name": "Swann",
"aliases": ["Swann", "M. Swann", "Monsieur Swann", "Charles Swann"],
"notes": "Charles Swann"
},
"la grand-mère": {
"character_id": "char_grandmere",
"display_name": "la grand-mère",
"aliases": ["ma grand-mère", "grand-mère", "la grand-mère"],
"notes": "Narrator's grandmother"
}
}

Then your annotations can still output display_name or character_id, depending on what you prefer.

My instinct: start with the simpler human-readable version, add IDs later.

