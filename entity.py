import sys
import codecs
import math
import nltk
from nltk import bigrams

def GetTokens(file):
 tokensTot=[]
 frasi = GetFrasi(file)
 for frase in frasi:
  tokensTot = tokensTot + nltk.word_tokenize(frase)
 return tokensTot
 
def GetFrasi(file):
 sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
 fileInput = codecs.open(file, "r", "utf-8")
 raw = fileInput.read()
 frasi = sent_tokenizer.tokenize(raw)
 return frasi

def AnalisiLinguistica(frasi):
 listanomi = []
 listluoghi = []
 for frase in frasi:
  tokens = nltk.word_tokenize(frase)
  tokensPos = nltk.pos_tag(tokens)
  analisi = nltk.ne_chunk(tokensPos)
  #print analisi
  for nodo in analisi:
   if hasattr(nodo, 'label'):
   #and nodo.label() in ["PERSON", "GPE", "ORGANIZATION", "GSP"]:
    if nodo.label() == "PERSON":
     #print nodo
     for leave in nodo.leaves():
	  if leave[1] == "NNP" or leave[1] == "NNPS":
	   listanomi.append(leave[0])
    elif nodo.label() == "GPE" or nodo.label() == "GSP":
     for leave in nodo.leaves():
	  listluoghi.append(leave[0])
 print listanomi
 print listluoghi
 nomiordered = nltk.FreqDist(listanomi)
 luoghiordered = nltk.FreqDist(listluoghi)
 print nomiordered.most_common(20)
 print luoghiordered.most_common(20)

def main(file):
 tokens = []
 bigr = []
 frasi = GetFrasi(file)
 AnalisiLinguistica(frasi)

main(sys.argv[1])