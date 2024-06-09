import sys
import codecs
import nltk
import math
from nltk import bigrams
from nltk import trigrams
#!/usr/bin/env python
# -*- coding: utf-8 -*-
def EstraiTestoTokenizzato(frasi):
    tokensTOT=[]
    for frase in frasi:
        #Divido la frase in token                                                                                                                                                                           
        tokens=nltk.word_tokenize(frase)
        #Creo la lista che contiene tutti itoken del testo                                                                                                                                                  
        tokensTOT=tokensTOT+tokens
    #restituisco i risultati                                                                                                                                                                                
    return tokensTOT

def AnnotazioneLinguistica (frasi):
    tokensPOStot=[]
    for frase in frasi:
        tokens=nltk.word_tokenize(frase)
        #Analizzo attraverso la Part-Of-Speech  token                                                                                                                                                       
        tokensPOS=nltk.pos_tag(tokens)
        #Creo la lista che contiene tutti i token analizzati                                                                                                                                                
        tokensPOStot=tokensPOStot+tokensPOS
    #Restituisco i risultati                                                                                                                                                                                
    return tokensPOStot

def DieciPOS (TestoAnalizzatoPOS):
    ListaPOS=[]
    for bigramma in TestoAnalizzatoPOS:
        #Inserisco il bigramma nella lista "ListaPOS"                                                                                                                                                       
        ListaPOS.append(bigramma[1])
        #calcolo la distribuzione della lista                                                                                                                                                               
        DistrPOS=nltk.FreqDist(ListaPOS)
        #Prendo esclusivamente i primi 10                                                                                                                                                                   
        dieciPOS=DistrPOS.most_common(10)
    #restituisco i risultati                                                                                                                                                                                
    return dieciPOS
def Ventitoken (tokensTOT):
    listaToken=[]
    for tok in tokensTOT:
        if tok not in [".", ",", "?", ":", ";", "!"]:
           #Inserisco il token nella lista "ListaToken"                                                                                                                                                     
           listaToken.append(tok)
           #calcolo la distribuzione della lista                                                                                                                                                            
           DistrToken=nltk.FreqDist(listaToken)
           #Prendo esclusivamente i primi 20                                                                                                                                                                
           VentiToken=DistrToken.most_common(20)
    #Restituisco  risultati                                                                                                                                                                                 
    return VentiToken
def EstraiBigrammiPOS(TestoAnalizzatoPOS):
    BigrammiEstratti=[]
    #Creo i bigrammi                                                                                                                                                                                        
    bigrammaTokPOS=bigrams(TestoAnalizzatoPOS)
    for bigramma in bigrammaTokPOS:
        #controllo che i bigrammi non siano formati da punteggiatura, congiunzioni e articoli                                                                                                               
        if ((bigramma [0][1] not in [ "." , ",","?",":","!",";", "CC","DT"]) and (bigramma [1][1] not in [".",",","!","?",":",";","CC","DT"])):
           #Inserisco il bigramma nella lista "BigrammiEstratti"                                                                                                                                            
           BigrammiEstratti.append(bigramma)
    #Restituisco i risultati                                                                                                                                                                                
    return BigrammiEstratti
def EstraiTrigrammiPOS(TestoAnalizzatoPOS):
    TrigrammiEstratti=[]
    #Creo i trigrammi                                                                                                                                                                                       
    trigrammaTokPOS=trigrams(TestoAnalizzatoPOS)
    for trigramma in trigrammaTokPOS:
        #controllo che i trigrammi non siano formati da punteggiatura, congiunzioni e articoli                                                                                                              
        if ((trigramma [0][1] not in [ "." , ",","?","!",":",";", "CC","DT"]) and (trigramma [1][1] not in [".",",","?","!",":",";", "CC","DT"]) and (trigramma [2][1] not in [".",",", "CC","DT"])):
           #inserisco il trigramma nella lista "TrigrammiEstratti"                                                                                                                                          
           TrigrammiEstratti.append(trigramma)
    #restituisco i risultati                                                                                                                                                                                
    return TrigrammiEstratti

def VentiBigrammi (bigrammi):
    listaBigrammi=[]
    for bigramma in bigrammi:
        #Inserisco il trigramma nella lista "listaBigrammi"                                                                                                                                                 
        listaBigrammi.append(bigramma)
        #Calcolo la distribuzione della lista "listaBigrammi"                                                                                                                                               
        DistrBigrammi=nltk.FreqDist(listaBigrammi)
        #Ottengo i  20 primi risultati                                                                                                                                                                      
        ventiBig=DistrBigrammi.most_common(20)
    #Restituisco i risultati                                                                                                                                                                                
    return ventiBig

def VentiTrigrammi (trigrammi):
    listaTrigrammi=[]
    for trigramma in trigrammi:
        #Inserisco il trigramma nella lista "listaTrigrammi"                                                                                                                                                
        listaTrigrammi.append(trigramma)
        #Calcolo la distribuzione della lista "listaTrigrammi"                                                                                                                                              
        DistrTrigrammi=nltk.FreqDist(listaTrigrammi)
        #Ottengo i 20 primi risultati                                                                                                                                                                       
        ventiTrigr=DistrTrigrammi.most_common(20)
    #Restituisco i risultati                                                                                                                                                                                
    return ventiTrigr

def TokenFreqDue (Testo):
    TestoOK=[]
    for tok in Testo:
        #Calcolo la frequenza di ogni singolo token                                                                                                                                                         
        frequenza=Testo.count(tok)
        #controllo chela funzione sia maggiore di due                                                                                                                                                       
        if frequenza > 2:
           #inserisco il token all'interno della lista "TestoOK"                                                                                                                                            
           TestoOK.append(tok)
    #Restituisco i risultati                                                                                                                                                                                
    return TestoOK

def TokenFreqDueFrase (frasi,NToken):
    FraseOK=[]
    i=0
    for frase in frasi:
        #Divido la frase in token                                                                                                                                                                           
        tokens=nltk.word_tokenize(frase)
        for tok in tokens:
            #Calcolo la frequenzadi ogni token                                                                                                                                                              
            frequenza=NToken.count(tok)
            #Controllo che la frequenza sia maggiore di due                                                                                                                                                 
            if frequenza > 2:
               #Se ok incremento il coefficente "i" di uno                                                                                                                                                  
               i=i+1
            #Se il coefficente "i" raggiunge il valore di 9                                                                                                                                                 
        if i >10:
           #Inserisco la frase nella lista "FraseOK"                                                                                                                                                        
           FraseOK.append(frase)
    #Restituisco i risultati                                                                                                                                                                                
    return FraseOK

def FrasiDieciToken (frasi):
    FrasiOK=[]
    for frase in frasi:
        #Divido la frase in token                                                                                                                                                                           
        tokens=nltk.word_tokenize(frase)
        #Controllo che la lunghezza della frase sia di dieci token                                                                                                                                          
        if len(tokens)>10:
           #Aggiungo la frase presa in analisi alla lista "FrasiOK"                                                                                                                                         
           FrasiOK.append(frase)
    #Restituisco i risultati                                                                                                                                                                                
    return FrasiOK

def EstraiBigrammiPOSDue(TestoAnalizzatoPOS):
    BigrammiEstratti=[]
    #Trasformo il testoPOS in bigrammi                                                                                                                                                                      
    bigrammaTokPOS=bigrams(TestoAnalizzatoPOS)
    for bigramma in bigrammaTokPOS:
        #Controllo che il bigramma sia formato da un aggettivo e da un sostantivo                                                                                                                           
        if ( ((bigramma [0][1] in [ "JJ", "JJR","JJS"]) and (bigramma [1][1] in ["NNP","NN","NNS","NNPS"]))):
           #Aggiungo il bigramma alla lista "BigrammiEstratti"                                                                                                                                              
           BigrammiEstratti.append(bigramma)
    #Restiuisco i risultati                                                                                                                                                                                 
    return BigrammiEstratti                                    
def ProbabilitaCongiunta(Bigrammi2,TestoTokenizzato):
    Dizionario={}
    #Calcolo la lunghezza del testo tokenizzato                                                                                                                                                             
    NToken=len(TestoTokenizzato)
    for bigramma in Bigrammi2:
        #Calcolo la probabilita congiunta                                                                                                                                                                   
        probabilitaCongiunta=((TestoTokenizzato.count(bigramma[0][0])*1.0)/(NToken*1.0))*((TestoTokenizzato.count(bigramma[1][0])*1.0)/(NToken*1.0))
        #Inserisco nel dizionario la probabilita calcolata assegnata al relativo bigramma                                                                                                                   
        Dizionario[bigramma]=probabilitaCongiunta
    #Ordino il dizionario                                                                                                                                                                                   
    DizionarioOrd= sorted(Dizionario.items(), key=lambda x: x[1], reverse=True)
    #Estraggo gli elementi del dizionario dal zero al ventesimo                                                                                                                                             
    VentiProb=DizionarioOrd[0:20]
    #Restituisco i risultati                                                                                                                                                                                
    return VentiProb
def ProbabilitaCondizionata(Bigrammi2,TestoTokenizzato):
    Dizionario={}
    #Conto il numero di token all'interno del testo tokenizzato                                                                                                                                             
    NTok=len(TestoTokenizzato)
    for bigramma in Bigrammi2:
        #Calcolo la probabilita condizionata                                                                                                                                                                
        probabilitaCondizionata=(Bigrammi2.count(bigramma)*1.0)/(TestoTokenizzato.count(bigramma[0][0]))
        #Inserisco nel dizionario la probabilita calcolata assegnata al relativo bigramma                                                                                                                   
        Dizionario[bigramma]=probabilitaCondizionata
    #Ordino il dizionario                                                                                                                                                                                   
    DizionarioOrd= sorted(Dizionario.items(), key=lambda x: x[1], reverse=True)
    #Estraggo gli elementi del dizionario dal zero al ventesimo                                                                                                                                             
    VentiProb=DizionarioOrd[0:20]
    #Restituisco i risultati                                                                                                                                                                                
    return VentiProb
def ForzaAssociativa(Bigrammi2,TestoTokenizzato):
     Dizionario={}
     #Calcolo il numero dei bigrammi                                                                                                                                                                        
     NBigrammi=len(Bigrammi2)
     #Calcololalunghezza del testo                                                                                                                                                                          
     NToken=len(TestoTokenizzato)
     for bigramma in Bigrammi2:
         #Calcolo la frequenza del bigramma                                                                                                                                                                 
         FrequenzaBig=Bigrammi2.count(bigramma)
         #Calcolo la frequenza del primo token del bigramma                                                                                                                                                 
         F1=TestoTokenizzato.count(bigramma[0][0])
         #Calcolo la frequenza del secondo token del bigramma                                                                                                                                               
         F2=TestoTokenizzato.count(bigramma[1][0])
          #calcolo la frequenza in MI                                                                                                                                                                       
         Frequenza=(FrequenzaBig*1.0/NBigrammi*1.0)/((F1*1.0/NToken*1.0)*(F2*1.0/NToken*1.0))
          #Calcolo la frequenza in LMI                                                                                                                                                                      
         localMutual=FrequenzaBig*math.log(Frequenza, 2)
          #Assegno al dizionario il valore della localMutual in corrispondenza del bigramma                                                                                                                 
         Dizionario[bigramma]=localMutual
      #Ordino il dizionario                                                                                                                                                                                 
     DizionarioOrd= sorted(Dizionario.items(), key=lambda x: x[1], reverse=True)
      #Estraggo gli elementi del dizionario dal zero al ventesimo                                                                                                                                           
     VentiFA=DizionarioOrd[0:20]
     #Restituisco i risultati                                                                                                                                                                               
     return  VentiFA


def CalcolaProbabilitaFraseMarkovZero(LunghezzaCorpus, DistribuzioneDiFrequenzaToken, frasi):
    probabilita=1.0
    Dizionario={}
    for frase in frasi:
        #Divido il testo in token                                                                                                                                                                           
        tokens=nltk.word_tokenize(frase)
        for tok in tokens:
            #calcolo la probabilia del token preso in analisi                                                                                                                                               
            probabilitaToken=(DistribuzioneDiFrequenzaToken[tok]*1.0/LunghezzaCorpus*1.0)
            #Moltiplico la probabilia calcolata con quella precedentemente ottenuta                                                                                                                         
            probabilita=probabilita*probabilitaToken
        #Assegno al dizionario il valore di probabilita calcolato in relazione alla frase                                                                                                                   
        Dizionario[frase]=probabilita
    #Ordino il dizionario                                                                                                                                                                                   
    DizionarioOrd= sorted(Dizionario.items(), key=lambda x: x[1], reverse=True)
    #Estraggo la prima frase del dizionario                                                                                                                                                                 
    Frase=DizionarioOrd[0]
    #Restituisco i risultati                                                                                                                                                                                
    return Frase

def CalcolaProbabilitaFraseMarkovUno(tokensTOT, DistribuzioneDiFrequenzaToken,DistribuzioneDiFrequenzaBigrammi,frasi):
    Dizionario={}
    for frase in frasi:
        #Divido il testo in token                                                                                                                                                                           
        tokens=nltk.word_tokenize(frase)
        #divido il testo in bigrammi                                                                                                                                                                        
        bigrammi=bigrams(tokens)
        #Assegno il primo token alla variabile "Tok"                                                                                                                                                        
        Tok=tokens[0]
        #Calcolo la probabilita della prima parola                                                                                                                                                          
        PM=(DistribuzioneDiFrequenzaToken[Tok]*1.0)/((tokensTOT)*1.0)
        for big in bigrammi:
            #calcolo la probabilita del bigramma                                                                                                                                                            
            ProbBig=(1.0*DistribuzioneDiFrequenzaBigrammi[big])/(1.0*DistribuzioneDiFrequenzaToken[big[0]]/tokensTOT)
            #Moltiplico la probabilita della prima parola per quella del bigramma                                                                                                                           
            PM=PM*ProbBig
        #Assegno al dizionario il valore di probabilita calcolato in relazione alla frase                                                                                                                   
        Dizionario[frase]=PM
    #Ordino il dizionario                                                                                                                                                                                   
    DizionarioOrd= sorted(Dizionario.items(), key=lambda x: x[1], reverse=True)
    #Estraggo la prima frase del dizionario                                                                                                                                                                 
    Frase=DizionarioOrd[1]

    #Restituisco i risultati                                                                                                                                                                                
    return Frase


def NETNomiPropri(frasi):
    NamedEntityListPers=[]
    NamedEntityListLuog=[]
    for frase in frasi:
        #Divido la frase in token                                                                                                                                                                           
        tokens=nltk.word_tokenize(frase)
        #Analizzo i token tramite Part-Of-Speech                                                                                                                                                            
        tokensPOS=nltk.pos_tag(tokens)
        #Eseguo classificazione delle entita nominate                                                                                                                                                       
        analisi=nltk.ne_chunk(tokensPOS)
        for nodo in analisi:
            NE=''
            #Controllo che "nodo" sia un nodo intermedio o foglia                                                                                                                                           
            if hasattr(nodo, 'label'):
                #Estraggo l'etichetta del nodo e controllo se si stratta di un nome di persona                                                                                                              
                if nodo.label() in ["PERSON"]:
                    #Ciclo le foglie del nodo selezionato                                                                                                                                                   
                    for partNE in nodo.leaves():
                        NE=NE+' '+partNE[0]
                    #Inserisco la NE all'internodella lista "NamedEntityListPers"                                                                                                                           
                    NamedEntityListPers.append(NE)
            #Controllo che "nodo" sia un nodo intermedio o foglia                                                                                                                                           
            if hasattr(nodo, 'label'):
                #Estraggo l'etichetta del nodo e controllo se si stratta di un nome di persona                                                                                                              
                if nodo.label() in ["GPE"]:
                    for partNE in nodo.leaves():
                        NE=NE+' '+partNE[0]
                    #Inserisco la NE all'internodella lista "NamedEntityListPers"                                                                                                                           
                    NamedEntityListLuog.append(NE)
    #Restituisco i risultati                                                                                                                                                                                
    return NamedEntityListPers, NamedEntityListLuog

def StampaVentiNET (NET):
   listaNET=[]
   for singoloNET in NET:
        #Aggiungo il singoloNET preso in analisi alla lista "listaNET"                                                                                                                                      
        listaNET.append(singoloNET)
   #Calcolo la distribuzione degli elementi della lista                                                                                                                                                     
   DistrNET=nltk.FreqDist(listaNET)
   #Estraggo venti elementi per i quali era stata calcolata la frequenza precedentemente                                                                                                                    
   VentiNET=DistrNET.most_common(20)
   #Restituisco i risultati                                                                                                                                                                                 
   return VentiNET






def main (file1,file2):
    #Carico i due file                                                                                                                                                                                      
    fileInput1= codecs.open(file1, "r", "utf-8")
    fileInput2= codecs.open(file2, "r", "utf-8")
    #Leggo i due file                                                                                                                                                                                       
    raw1= fileInput1.read()
    raw2= fileInput2.read()
    sent_tokenizer= nltk.data.load('tokenizers/punkt/english.pickle')
    #Divido i due file in frasi                                                                                                                                                                             
    frasi1= sent_tokenizer.tokenize(raw1)
    frasi2= sent_tokenizer.tokenize(raw2)
    #Divido i due file in token attraverso la fuzione "EstraiTestoTokenizzato"                                                                                                                              
    TestoTokenizzato1=EstraiTestoTokenizzato(frasi1)
    TestoTokenizzato2=EstraiTestoTokenizzato(frasi2)
    #Annoto i due file tokenizzati attraverso la funzione "AnnotazioneLinguistica"                                                                                                                          
    AnnotaTesto1=AnnotazioneLinguistica(frasi1)
    AnnotaTesto2=AnnotazioneLinguistica(frasi2)
                                                                                                                               
#Calcolo i dieci POS piu frequenti all'interno dei due file attraverso la funzione "DieciPOS"                                                                                                           
    DieciPartOfSpeech1=DieciPOS(AnnotaTesto1)
    DieciPartOfSpeech2=DieciPOS(AnnotaTesto2)
 #Stampo i venti token piu frequenti dei due file attraverso la funzione "Ventitoken"                                                                                                                    
    StampaVentiToken1=Ventitoken(TestoTokenizzato1)
    StampaVentiToken2=Ventitoken(TestoTokenizzato2)
#Divido i due file in bigrammi attraverso la funzione "EstraiBigrammiPOS"                                                                                                                               
    bigrammi1=EstraiBigrammiPOS(AnnotaTesto1)
    bigrammi2=EstraiBigrammiPOS(AnnotaTesto2)
#Divido i due file in tirgrammi attraverso la funzione "EstraiTrigrammiPOS"                                                                                                                             
    trigrammi1=EstraiTrigrammiPOS(AnnotaTesto1)
    trigrammi2=EstraiTrigrammiPOS(AnnotaTesto2)
    #Stampo i primi venti bigrammi attraverso la funzione "VentiBigrammi"                                                                                                                                   
    StampaVentiBigrammi1=VentiBigrammi(bigrammi1)
    StampaVentiBigrammi2=VentiBigrammi(bigrammi2)
    #Stampo i primi trenta trigrammi attraverso la funzione "VentiTrigrammi"                                                                                                                                
    StampaVentiTrigrammi1=VentiTrigrammi(trigrammi1)
    StampaVentiTrigrammi2=VentiTrigrammi(trigrammi2) 
 #Calcolo i dieci POS piu frequenti all'interno dei due file attraverso la funzione "DieciPOS"                                                                                                           
    DieciPartOfSpeech1=DieciPOS(AnnotaTesto1)
    DieciPartOfSpeech2=DieciPOS(AnnotaTesto2)
    #Stampo i venti token piu frequenti dei due file attraverso la funzione "Ventitoken"                                                                                                                    
    StampaVentiToken1=Ventitoken(TestoTokenizzato1)
    StampaVentiToken2=Ventitoken(TestoTokenizzato2)

    #Divido il testo in bigrammi dei due file attraverso la fuzione "EstraiBigrammiPOSDue"                                                                                                                  
    tokenFreq=TokenFreqDue(TestoTokenizzato1)
    ATesto=AnnotazioneLinguistica(tokenFreq)
    #Estraggo i bigrammi dei due file che soddisfano le condizioni all'interno della funzione "BigrammiPOSDue"                                                                                              
    Bigrm1=EstraiBigrammiPOSDue(ATesto)
    Bigrm2=EstraiBigrammiPOSDue(ATesto)

     #Calcolo la probablita congiunta dei duel file attraverso la funzione "ProbabilitaCongiunta"                                                                                                            
    ProbabilitaCong1=ProbabilitaCongiunta(Bigrm1,tokenFreq)
    ProbabilitaCong2=ProbabilitaCongiunta(Bigrm2,tokenFreq)

   #Calcolo la probablita condizionata dei duel file attraverso la funzione "ProbabilitaCondizionata"                                                                                                      
    ProbabilitaCond1=ProbabilitaCondizionata(Bigrm1,tokenFreq)
    ProbabilitaCond2=ProbabilitaCondizionata(Bigrm2,tokenFreq)
    #Calcolo la forza associativa dei due file attraverso la funzione "forzaAssociativa"                                                                                                                    
    FAssociativa1=ForzaAssociativa(Bigrm1,tokenFreq)
    FAssociativa2=ForzaAssociativa(Bigrm2,tokenFreq)

    #Estraggo le frasi dai due file che soddosfano le condizioni della funzione "FrasiDieciToken"                                                                                                           
    FrasiDieci1=FrasiDieciToken(frasi1)
    FrasiDieci2=FrasiDieciToken(frasi2)
    #Controllo che i token delle frasi prese in analisi soddisfino le condizioni della funzione "TokenFreqDuefrase"                                                                                         
    FrasiTokenDue1=TokenFreqDueFrase(FrasiDieci1,TestoTokenizzato1)
    FrasiTokenDue2=TokenFreqDueFrase(FrasiDieci2, TestoTokenizzato2)
    #Divido in token i due file precedentemente analizzati attraverso la funzione "EstraiTestoTokenizzato"                                                                                                  
    TestoTokenizzatoDue1=EstraiTestoTokenizzato(FrasiTokenDue1)
    TestoTokenizzatoDue2=EstraiTestoTokenizzato(FrasiTokenDue2)
    #Calcolo la distribuzione di frequenza dei token dei due file                                                                                                                                           
    DisDiFreqToken1=nltk.FreqDist(TestoTokenizzatoDue1)
    DisDiFreqToken2=nltk.FreqDist(TestoTokenizzatoDue2)
    #Divido il testo in bigrammi                                                                                                                                                                            
    bigrammiDue1=bigrams(TestoTokenizzatoDue1)
    bigrammiDue2=bigrams(TestoTokenizzatoDue2)
    #Calcolo la lunghezza dei due file                                                                                                                                                                      
    NTok1=len(TestoTokenizzatoDue1)
    NTok2=len(TestoTokenizzatoDue2)
    #Calcolo la frequenza dei bigrammi dei due file                                                                                                                                                        
    DisDiFreqBigrammi1=nltk.FreqDist(bigrammiDue1)
    DisDiFreqBigrammi2=nltk.FreqDist(bigrammiDue2)

    #Estraggo la frase dei due file con probabilita piu alta, utilizzando markov zero, attraverso la funzione "CalcolaProbabilitaFraseMarkovZero"                                                           
    MarkovZero1=CalcolaProbabilitaFraseMarkovZero(NTok1,DisDiFreqToken1,FrasiTokenDue1)
    MarkovZero2=CalcolaProbabilitaFraseMarkovZero(NTok2,DisDiFreqToken2,FrasiTokenDue2)

    #Estraggo la frase dei due file con probabilita piu alta, utilizzando markov uno, attraverso la funzione "CalcolaProbabilitaFraseMarkovUno"                                                             
    MarkovUno1=CalcolaProbabilitaFraseMarkovUno(NTok1,DisDiFreqToken1,DisDiFreqBigrammi1,FrasiTokenDue1)
    MarkovUno2=CalcolaProbabilitaFraseMarkovUno(NTok2,DisDiFreqToken2,DisDiFreqBigrammi2,FrasiTokenDue2)



    #Estraggo i nomi propri di persona e di luoghi classificati attraverso NE tagging                                                                                                                       
    NETPers1,NETLuog1=NETNomiPropri(frasi1)
    NETPers2,NETLuog2=NETNomiPropri(frasi2)
    
    print "-Secondo progetto dell'esame di linguistica computazionale A.S 2016/2017-"
    print "Le dieci Part-Of-Speech del file", file1, "maggiormente frequenti sono:\n",
    for i in  DieciPartOfSpeech1:
        print "Part-Of-Speech:", i[0],"   ","Frequenza:", i[1]

    print "I venti token, escludendo la punteggiatura, del file",file1, "maggiormente frequenti sono:\n",
    for i in  StampaVentiToken1:
        print "Token:", i[0],"   ","Frequenza:", i[1]
    print "I venti token, escludendo la punteggiatura, del file",file2, "maggiormente frequenti sono:\n",
    for i in  StampaVentiToken2:
        print "Token:", i[0],"   ","Frequenza:", i[1]
    for i in  StampaVentiBigrammi1:
         print "Bigramma:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"    ","Frequenza:", i[1]
    print "I venti bigrammi, che non contengono congiunzioni, articoli e punteggiatura, del file", file2,"maggiormente frequenti sono:\n",
    for i in  StampaVentiBigrammi2:
        print "Bigramma:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"    ","Frequenza:", i[1]
    print "I venti trigrammi, che non contengono congiunzioni, articoli e punteggiatura, del file", file1,"maggiormente frequenti sono:\n",
    for i in StampaVentiTrigrammi1:
        print "Trigramma:","Primo token del trigramma:", i[0][0][0],"   ","Secondo token del trigramma:",i[0][1][0],"   ","Terzo token del trigramma:",i[0][2][0],"    ","Frequenza:", i[1]
    print "I venti trigrammi, che non contengono congiunzioni, articoli e punteggiatura, del file", file2,"maggiormente frequenti sono:\n",
    for i in StampaVentiTrigrammi2:
        print "Trigramma:","Primo token del trigramma:", i[0][0][0],"   ","Secondo token del trigramma:",i[0][1][0],"   ","Terzo token del trigramma:",i[0][2][0],"    ","Frequenza:", i[1]
    print"I venti bigrammi, composti da aggettivo e sostantivo, del file", file1,"con probabilita congiunta massima sono:\n",
    for i in ProbabilitaCong1:
        print "Bigramma:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"   ","Probabilita congiunta:", i[1]
    print"I venti bigrammi, composti da aggettivo e sostantivo, del file", file2,"con probabilita congiunta massima sono:\n",
    for i in ProbabilitaCong2:
        print "Bigramma:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"   ","Probabilita congiunta:", i[1]
    print"I venti bigrammi, composti da aggettivo e sostantivo, del file", file1,"con probabilita condizionata massima sono:\n",
    for i in ProbabilitaCond1:
        print "Bigramma:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"   ","Probabilita condizionata:", i[1]
    print"I venti bigrammi, composti da aggettivo e sostantivo, del file", file2,"con probabilita condizionata massima sono:\n",
    for i in ProbabilitaCond2:
        print "Bigramma:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"   ","Probabilita condizionata:", i[1]
    print"I venti bigrammi, composti da aggettivo e sostantivo, delfile", file1,"con forza associativa massima sono:\n",
    for i in FAssociativa1:
        print "Bigrammi:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"   ","Forza associativa:", i[1]
    print"I venti bigrammi, composti da aggettivo e sostantivo, delfile", file2,"con forza associativa massima sono:\n",
    for i in FAssociativa2:
        print "Bigrammi:","Primo token del bigramma:", i[0][0][0],"   ","Secondo token del bigramma:",i[0][1][0],"   ","Forza associativa:", i[1]
    print" La frase con probabilita piu alta del file", file1, " analizzata con una catena di Markov di ordine 0 :\n", "Frase:", MarkovZero1[0], "   ",  "Valore:",MarkovZero1[1]
    print" La frase con probabilita piu alta del file", file2, " analizzata con una catena di Markov di ordine 0 :\n", "Frase:", MarkovZero2[0], "   ",  "Valore:",MarkovZero2[1]
    print" La frase con probabilita piu alta del file",file1, " analizzata con una catena di Markov di ordine 1 :\n", "Frase:", MarkovUno1[0], "   ",  "Valore:",MarkovUno1[1]
    print" La frase con probabilita pii alta del file",file2, " analizzata con una catena di Markov di ordine 1 :\n", "Frase:", MarkovUno2[0], "   ",  "Valore:",MarkovUno2[1]
    for i in VentiPers1:
        print "Nome:", i[0], "    ","Frequenza:", i[1]
    print" I venti nomi propri di persona piu frequenti del file", file2," sono:\n",
    for i in VentiPers2:
        print "Nome:", i[0], "    ","Frequenza:", i[1]
    print" I venti nomi propri di luoghi piu frequenti del file", file1," sono:\n",
    for i in  VentiLuoghi1:
        print "Nome:", i[0], "    ","Frequenza:", i[1]
    print" I venti nomi propri di luoghi piu frequenti del file", file2," sono:\n",
    for i in  VentiLuoghi2:
        print "Nome:", i[0], "    ","Frequenza:", i[1]
main(sys.argv[1],sys.argv[2])
