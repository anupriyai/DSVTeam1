import mwparserfromhell
import os




def parsetobm25(file_name= "venv\enwiki-20241120-pages-articles-multistream1.xml-p1p41242\enwiki-20241120-pages-articles-multistream1.xml-p1p41242"):

    shittycorpus = []
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            # Detect and parse articles
            if "<text" in line:
                wikicode = mwparserfromhell.parse(line)
                shittycorpus.append(wikicode.strip_code())

    return shittycorpus

