# proust
Text analytics on Proust's In Search of Lost Time. You can read my posts on Proust here: https://nathanbrixius.wordpress.com/category/proust/

For now, this repository contains code to:
- Download the full public domain French text of «À la recherche du temps perdu» by Marcel Proust
- spaCy code to preprocess the text
- counts occurrences of proper names
- a bit of visualization

I am not an expert in NLP / spacy, so buyer beware.                           
                                                                               
Packages you'll need:                                                     
-   bs4, pandas, spacy, matplotlib                                              
- also, you will need the largest french-language spacy model: python -m spacy download fr_core_news_lg     
