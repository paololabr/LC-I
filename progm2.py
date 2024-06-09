
import sys
import codecs
import math
import nltk
from nltk import bigrams
from nltk import trigrams

def GetTokens(file):
# fz che dato in input un file ne ritorna i token e le frasi tramite la fz GetFrasi
 tokensTot=[]
 frasi = GetFrasi(file)
 for frase in frasi:
  tokensTot = tokensTot + nltk.word_tokenize(frase)
 return tokensTot, frasi
 
def GetFrasi(file):
# fz che dato in input un file ne ritorna le frasi
 sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
 fileInput = codecs.open(file, "r", "utf-8")
 raw = fileInput.read()
 frasi = sent_tokenizer.tokenize(raw)
 return frasi
 
# fz che data una frase la lunghezza del suo corpus e la distribuzione
# di frequenza calcola la probabilita della frase con Markov 0
def getProbFraseMarkov0(frase, luncorpus, distfreq):
 tokens = nltk.word_tokenize(frase)
 probfrase = 1.0
 for i in range(len(tokens)): 
  probtoken = distfreq[tokens[i]] * 1.0 / luncorpus * 1.0
  probfrase = probfrase * probtoken
 return probfrase

# fz che data una frase la lunghezza del suo corpus, la distribuzione
# di frequenza e i bigrammi calcola la probabilita della frase con Markov 1 
def getProbFraseMarkov1(frase, luncorpus, distfreq, bigrams):
 tokens = nltk.word_tokenize(frase)
 probfrase = 1.0
 for i in range(len(tokens)):
  if (i == 0):
   probfrase = distfreq[tokens[i]] * 1.0 / luncorpus * 1.0
  else:
   probfrase = probfrase * (bigrams.count((tokens[i - 1], tokens[i])) * 1.0 / distfreq[tokens[i - 1]] * 1.0)
 return probfrase

# fz che stampa i 10 pos tag piu frequenti data una lista di token in input
def mostFrequentPos(tokenCorpus):
 pos_tags = nltk.pos_tag(tokenCorpus)
 listaPos = []
 for bigr in pos_tags:
  listaPos.append(bigr[1])
  
 PosPerFreq = nltk.FreqDist(listaPos)
 print "Le 10 PoS (Part-of-Speech) piu frequenti:"
 for pos in PosPerFreq.most_common(10):
  print "'" + pos[0] + "'"
 
# fz che scrive i venti token piu frequenti punteggiatura esclusa
def mostFrequentTokenNoPunct(tokenCorpus):
 pos_tags = nltk.pos_tag(tokenCorpus)
 # filtro punteggiatura
 listaf = [elem for elem in pos_tags if elem[1] != ',' and elem[1] != '.' and elem[1] != ':']
 restmp = []
 for bigr in listaf:
  restmp.append(bigr[0])
 res = nltk.FreqDist(restmp)
 print "I 20 token piu frequenti escludendo la punteggiatura:"
 for tok in res.most_common(20):
  print "'" + tok[0] + "'"

# fz che scrive i venti aggettivi piu frequenti
def mostFrequentAdjective(tokenCorpus):
 pos_tags = nltk.pos_tag(tokenCorpus)
 # filtro punteggiatura
 listaf = [elem for elem in pos_tags if elem[1].startswith('J')]
 restmp = []
 for bigr in listaf:
  restmp.append(bigr[0])
 res = nltk.FreqDist(restmp)
 print "I 20 aggettivi piu frequenti:"
 for tok in res.most_common(20):
  print "'" + tok[0] + "'"
  
# fz che scrive i venti verbi piu frequenti
def mostFrequentVerb(tokenCorpus):
 pos_tags = nltk.pos_tag(tokenCorpus)
 # filtro punteggiatura
 listaf = [elem for elem in pos_tags if elem[1].startswith('V')]
 restmp = []
 for bigr in listaf:
  restmp.append(bigr[0])
 res = nltk.FreqDist(restmp)
 print "I 20 verbi piu frequenti:"
 for tok in res.most_common(20):
  print "'" + tok[0] + "'"

# stampo i 20 trigrammi di pos piu frequenti
def mostFrequentTrigram(tokenCorpus):
 
 pos_tags = nltk.pos_tag(tokenCorpus)
 trigrammiPos = list(trigrams(pos_tags))
 
 listaOnlyTrigram = []
 for trigr in trigrammiPos:
  listaOnlyTrigram.append((trigr[0][1], trigr[1][1], trigr[2][1]))
 
 orderedTrigram = nltk.FreqDist(listaOnlyTrigram)
 print "I 20 trigrammi di pos piu frequenti"
 for trigr in orderedTrigram.most_common(20):
  print "('" + trigr[0][0] + "','" + trigr[0][1] + "','" + trigr[0][2] + "')"

# i 10 bigrammi di Pos con probabilita congiunta massima, indicando anche la relativa probabilita
def mostTwentyBigramsMaxConj(tokens):
 pos_tags = nltk.pos_tag(tokens)
 
 onlyps = [elem[1] for elem in pos_tags]
 
 bigrammiPos = list(bigrams(pos_tags))
 bigramsRes = [(elem[0][1],elem[1][1]) for elem in bigrammiPos]
 
 # P(Art, N) = P(Art) * P(N|Art) (regola del prodotto
 # P(N|Art) = P(Art, N) / P(Art)
 listbigr = []
 for bigr in set(bigramsRes):
  subprob = bigramsRes.count(bigr) * 1.0 / onlyps.count(bigr[0]) * 1.0
  tokenprob = onlyps.count(bigr[0]) * 1.0 / len(onlyps) * 1.0
  conjprob = subprob * tokenprob
  if (len(listbigr) < 10):
   listbigr.append((conjprob, bigr))
   listbigr = sorted(listbigr)
  else:
   minprob = listbigr[0][0]
   if conjprob > minprob:
    listbigr[0] = (conjprob, bigr)
    listbigr = sorted(listbigr)
 print "I 10 bigrammi di Pos con probabilita congiunta massima, con relativa probabilita:"
 for tbigr in listbigr:
  print "{:<30} {:>20} ".format("('" + tbigr[1][0] + "','" + tbigr[1][1] + "')", tbigr[0])

# I 10 bigrammi con probabilita condizionata massima, indicando anche la relativa probabilita 
def mostTenPosCond(tokens):

 pos_tags = nltk.pos_tag(tokens)
 onlyps = [elem[1] for elem in pos_tags]
 bigrammiPos = list(bigrams(pos_tags))
 bigramsRes = [(elem[0][1],elem[1][1]) for elem in bigrammiPos]
 
 listbigr = []
 for bigr in set(bigramsRes):
  subprob = bigramsRes.count(bigr) * 1.0 / onlyps.count(bigr[0]) * 1.0
  if (len(listbigr) < 10):
   listbigr.append((subprob, bigr))
   listbigr = sorted(listbigr)
  else:
   minprob = listbigr[0][0]
   if subprob > minprob:
    listbigr[0] = (subprob, bigr)
    listbigr = sorted(listbigr)
 print "I 10 bigrammi di pos con probabilita condizionata massima, con relativa probabilita:"
 for tbigr in listbigr:
  print "{:<30} {:>20} ".format("('" + tbigr[1][0] + "','" + tbigr[1][1] + "')", tbigr[0])

# fz che stampa i 10 pos tag piu frequenti data una lista di token in input
def createuniqueList(tokenCorpus):
 listaPos = []
 pos_tags = nltk.pos_tag(tokenCorpus)
 
 for i in range(len(pos_tags)):
  if pos_tags[i][1].startswith('N'):
   listaPos.append(pos_tags[i])
   
 PosPerFreq = nltk.FreqDist(listaPos)
 
 listaPos = PosPerFreq.most_common(10)
 
 resNouns = []
 for item in listaPos:
  resNouns.append(item[0][0])

 return resNouns

def orderPrecedingAdj(tokenCorpus, NounList):
 pos_tags = nltk.pos_tag(tokenCorpus)
 
 listaAdjNoun = []
 
 for i in range(len(pos_tags)):
  if ((i > 0) and pos_tags[i][1].startswith('N') and (pos_tags[i][0] in NounList)):
   if pos_tags[i - 1][1].startswith('J'):
    listaAdjNoun.append((pos_tags[i - 1][0], pos_tags[i][0]))
 
 listbigr = []
 for bigr in set(listaAdjNoun):
  if ((bigr[0] in tokenCorpus) and (bigr[1] in tokenCorpus)):
   subprob = listaAdjNoun.count(bigr) * 1.0 / tokenCorpus.count(bigr[0]) * 1.0
   tokenprob1 = tokenCorpus.count(bigr[0]) * 1.0 / len(tokenCorpus) * 1.0
   tokenprob2 = tokenCorpus.count(bigr[1]) * 1.0 / len(tokenCorpus) * 1.0
   mi = math.log((subprob * tokenprob1 * 1.0) / (tokenprob1 * tokenprob2 * 1.0), 2)
  else:
   mi = 0
  
  listbigr.append((mi, bigr))
  listbigr = sorted(listbigr)
   
 print "Aggettivi che li precedono rispetto alla forza associativa (calcolata in termini di Local Mutual Information):"
 for tbigr in listbigr:
  print "{:<30} {:>20} ".format(tbigr[1][0], tbigr[0])
# fz che estrae i 20 nomi propri di persona e i 20 nomi propri di luogo usando le entita nominate
# ordinandoli per frequenza
def ExtractNameAndPlace(frasi):
 listluoghi = []
 for frase in frasi:
  tokens = nltk.word_tokenize(frase)
  tokensPos = nltk.pos_tag(tokens)
  # prendo le entita nominate
  analisi = nltk.ne_chunk(tokensPos)
  for nodo in analisi:
   if hasattr(nodo, 'label'):
    if nodo.label() == "GPE":
     for leave in nodo.leaves():
	  listluoghi.append(leave[0])
 luoghiordered = nltk.FreqDist(listluoghi)
 print "I 20 nomi propri di luogo piu frequenti (tipi), ordinati per frequenza:"
 for ent in luoghiordered.most_common(20):
  print "{:<30} {:>20} ".format(ent[0],ent[1])

def DoJob(file, tokensCorpus, frasi):
 # setto l'encoding di default del sistema ad utf8
 reload(sys)  
 sys.setdefaultencoding('utf8')
 
 #tokensCorpus, frasi = GetTokens(file)
 
 print "*******************************"
 print "File:", file
 print "Lunghezza in token:", len(tokensCorpus)
 print "Numero frasi:", len(frasi)
 print "*******************************"
 print
 
 mostFrequentTokenNoPunct(tokensCorpus)
 print
 mostFrequentAdjective(tokensCorpus)
 print
 mostFrequentVerb(tokensCorpus)
 print
 mostFrequentPos(tokensCorpus)
 print
 mostFrequentTrigram(tokensCorpus)
 print
 mostTwentyBigramsMaxConj(tokensCorpus)
 print
 mostTenPosCond(tokensCorpus)

# il programma ha un numero di argomenti (file) variabile 
def main(args):
 if (len(args) < 3):
  print "program need at least 2 corpora file as argument"
  print
  quit()
 
 tokensCorpus, frasi = GetTokens(args[1])
 tokensCorpus2, frasi2 = GetTokens(args[2])
 
 DoJob(args[1], tokensCorpus,frasi)
 uniqueList = createuniqueList(tokensCorpus) + createuniqueList(tokensCorpus2)
 orderPrecedingAdj(tokensCorpus, uniqueList)
 print
 orderPrecedingAdj(tokensCorpus2, uniqueList)
 print
 #ExtractNameAndPlace(frasi)
 print
 #ExtractNameAndPlace(frasi2)

main(sys.argv)