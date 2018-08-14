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
        '[t,T]win',
        '[t,T]wins',
        '[M,m]other',
        '[C,c]ousin',
        '[F,f]ather',
        '[U,u]ncle',
        '[A,a]unt',
        '[S,s]ibling',
        '[B,b]rother',
        '[S,s]ister',
        'family',
        '[F,f]amilies',
        '[F,f]amilial',
        '[K,k]indred',
        'families']

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
    #Only return the unique elements of the set - ie dont have any of the words in there twice
    used = set()
    unique = [x for x in i if x not in used and (used.add(x) or True)]
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
    used = set()
    unique = [x for x in i if x not in used and (used.add(x) or True)]
    return unusualFlag, unique

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

def indentifier_analyis(sentence):
    test = [
    '[P,p]atient\s(\d+[a-zA-Z])',
    '[T,t]he\s(patient|proband|individual|propositus)',
    '[P,p]atient\s(one|two|three|four|five|six|seven|eight|nine)',
    '[p,P]atient\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))[\-,/,\.,:]((\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3})))[\-,/,\.,:](\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))))',
    '[p,P]atient\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))[\-,/,\.,:](\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3})))', #layer 5 of patients class,
    '[p,P]atient\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))(\*?)', #layer 4 of patients class
    '[p,P]atient\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))', #layer 3 of patients clas,
    '[P,p]atient\s((\d+[a-zA-Z]))*',
    '[P,p]atient\s(\d+[a-zA-Z])',
    '[i,I]ndex\spatient',
    '[P,p]atient\s(#\d+)',
    '[P,p]atient\sno.\s\d+',
    '[p,P]atient\s([P,p]\d+|\d+|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)|[a-zA-Z]|\#\d+)', #layer 3 of the individual class


    '[p,P]roband\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))[\-,/,\.,](\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3})))', #layer 5 of proband class
    '[p,P]roband\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))(\*?)', #layer 4 of proband class
    '[p,P]roband\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3}))|[a-zA-Z]|(no.\s\d+))', #layer 3 of proband clas,
    '[T,t]he\s(patient|proband|individual|propositus)',
    '[P,p]atient',
    '[P,p]roband',
    '[P,p]ropositus',
    '[P,p]ropositis',

    '[P,p]articipant\s\d+',
    '[P,p]articipant',

    '[I,i]ndividual\s((\d+[a-zA-Z]\d+)|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)[-,.,:](\d+|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)))', #layer 5 of the indivudual class
    '[I,i]ndividual\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3})))[-,.,:](\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3})))',
    '[I,i]ndividual\s([P,p]\d+|\d+|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)|[a-zA-Z]|\#\d+)', #layer 3 of the individual class
    '[I,i]ndividual\s(\d+|((XC|XL|L?X{0,3})(X|IV|V?I{0,3})))'
    '[I,i]ndividual\s(\d+[a-zA-Z])', #layer 4 of the individual class
    '[I,i]ndividual',


    '(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|I{1,3})[:,.,-](\d+|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|I{1,3}))[:,.,-](\d+|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|I{1,3}))', #layer 5 of the roman numeral class
    '(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|I{1,3})[:,.,-](\d+|(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|I{1,3}))', #layer 3 of the roman numeral class
    '(XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|I{1,3})\d+', #layer 2 of the roman numeral class

    '([a-zA-Z]|[a-zA-Z]\d+|\d+)[-,:](XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)\d+', #layer 4 of alphanumeric string class
    '([a-zA-Z]|[a-zA-Z]\d+|\d+)[-,:]((XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)|\d+)[-,:]((XC|XL|L?X{0,3})(IX|IV|VI{1,3}|V|III|II|I)|\d+)', #layer 5 of alphaqnumeric string class
    '[C,c]ase\s(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
    '[T,t]win\s(1|2|one|two)',
    '([S,s]ubject\s\d+)',
    '([P,p]\d+)',
    '[B,b]oy',
    '[G,g]irl',
    '[W,w]oman',
    '[M,m]an',
    '[F,f]emale',
    '[M,m]ale',
    '[S,s]on',
    '[t,T]win\s[a|A|b|B|1|2]',
    '[M,m]other',
    '[C,c]ousin',
    '[F,f]ather',
    '[U,u]ncle',
    '[A,a]unt',
    '[S,s]ibling',
    '[B,b]rother',
    '[S,s]ister']
    indentifiers =  []
    words = sentence.split()
    for word in words:
        for exp in test:
            id = re.match(exp, word, re.MULTILINE | re.IGNORECASE)
            if id != None:
                matches = id.group(0)
                indentifiers.append(matches)
    #Only return the uniquely found identifiers
    used = set()
    unique = [x for x in indentifiers if x not in used and (used.add(x) or True)]
    return unique

def surgery(sentence):
    test = [
    '[S,s]urgery',
    '[R,r]econstruction',
    '[R,r]econstructive'
    ]
    surgery_flag = 0
    i = []
    words = sentence.split()
    for word in words:
        for exp in test:
            id = re.match(exp, word, re.MULTILINE | re.IGNORECASE)
            if id != None:
                matches = id.group(0)
                i.append(matches)
                surgery_flag = 1
    used = set()
    unique = [x for x in i if x not in used and (used.add(x) or True)]
    return surgery_flag, unique

def unaffectedCheck(sentence):
    test = [
    '[U,u]naffected'
    ]
    flag = 0
    i = []
    words = sentence.split()
    for word in words:
        for exp in test:
            id = re.match(exp, word, re.MULTILINE | re.IGNORECASE)
            if id != None:
                matches = id.group(0)
                i.append(matches)
                flag = 1
    used = set()
    unique = [x for x in i if x not in used and (used.add(x) or True)]
    return flag, unique
