import re


def split_into_sentences(text):
    #Regular Expressions
    #For splitting into sentences
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    #Split into sentences
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def family_check(sentence):
    test = [
        '[S,s]on',
        '[G,g]randmother',
        '[G,g]randfather',
        '[t,T]win\s[a|A|b|B|1|2]',
        '[M,m]other',
        '[C,c]ousin',
        '[F,f]ather',
        '[U,u]ncle',
        '[A,a]unt',
        '[S,s]ibling',
        '[B,b]rother',
        '[S,s]ister',
        '[F,f]amily',
        '[F,f]amilial',
        '[K,k]indred']

    familyFlag = 0
    i = []
    words = sentence.split()
    for word in words:
        for exp in test:
            id = re.match(exp, word, re.MULTILINE | re.IGNORECASE)
            if id != None:
                matches = id.group(0)
                familyFlag = 1
                i.append(matches)
    return familyFlag, i

def unusual_check(sentence):
    test = [
    '[U,u]nique',
    '[W,w]ithout',
    '[A,a]typical',
    '[N,n]ovo',
    '[N,n]ovel',
    '[N,n]ew',
    '[U,u]nreported']
    unusualFlag = 0
    i = []
    words = sentence.split()
    for word in words:
        for exp in test:
            id = re.match(exp, word, re.MULTILINE | re.IGNORECASE)
            if id != None:
                matches = id.group(0)
                unusualFlag = 1
                i.append(matches)
    return unusualFlag, i

def number_of_patients(sentence):
    test = [
    '[0-9][0-9]*',
    '[H,h]undred',
    '[N,n]inety',
    '[E,e]ighty',
    '[S,s]eventy',
    '[S,s]ixty',
    '[F,f]ifty',
    '[F,f]orty',
    '[T,t]hirty',
    '[T,t]wenty',
    '[N,n]ineteen',
    '[E,e]ighteen',
    '[S,s]eventeen',
    '[S,s]ixteen',
    '[F,f]ifteen',
    '[F,f]ourteen',
    '[T,t]hirteen',
    '[T,t]welve',
    '[E,e]leven',
    '[T,t]en',
    '[N,n]ine',
    '[E,e]ight',
    '[S,s]even',
    '[S,s]ix',
    '[F,f]ive',
    '[F,f]our',
    '[T,t]hree',
    '[T,t]wo',
    '[O,o]ne']
    #Making the assumption for now that the biggest number mentioned is the total number of patients mentioned in the article
    number = 0
    flag = 0
    words = sentence.split()
    for word in words:
        if flag != 1:
            for exp in test:
                if flag != 1:        #Stop when we find oen
                    id = re.match(exp, word, re.MULTILINE | re.IGNORECASE)
                    if id != None:
                        matches = id.group(0)
                        flag = 1
                        number = matches
    return number
