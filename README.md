# Named-Entity-Recognition
Named Entity Recognition (NER) in Natural Language Processing using spaCy and NLTK

```python
import spacy 
from spacy import displacy

```

Lets add the sample data
```python


sample_text="""The Top 100 Companies of the World: U.S. vs Everyone. When it comes to breaking down the top 100 companies of the world, the United States still commands the largest slice of the pie.
Throughout the 20th century and before globalization reached its current peaks, American companies made the country an economic powerhouse and the source of a majority of global market value.
But even as countries like China have made headway with multi-billion dollar companies of their own, and the market’s most important sectors have shifted, the U.S. has managed to stay on top.
"""
```

let's load only the NER model of spacy
```python
NER = spacy.load("en_core_web_sm")
```

let us fit the model on the sample text.
```python
word = NER(sample_text)
```

Printing the NEs named entity found by the model in our sample text.
```python
for w in word.ents:
    print(w.text,w.label_)
```

In case you are confused about the named entity code, you can easily check it wuth the explain() method
```python
spacy.explain("GPE")
spacy.explain("DATE")
spacy.explain("EVENT")
spacy.explain("PRODUCT")
```

lets visualize the name entities with the data using the displacy package of spacy.
```python
displacy.render(word,style="ent",jupyter=True)
```

Happy coding!
