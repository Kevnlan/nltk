import nltk
grammar1 = nltk.CFG.fromstring("""
 S -> NP VP
 V -> "saw" | "called" | "ate" | "walked" | "is" | "wanted" | "died" | "put"
 VP -> V NP | V NP PV | V Adv | V Adj | V CP
 PN -> "John" | "Mary" | "Bob" | "Oscar" | "Paris" | "Sally"
 NP -> Det N | Det N PV | PN | PN CP
 PV -> PP | PP NP
 CP -> C NP | C VP
 Det -> "a" | "an" | "the" | "my" | "The"
 N -> "man" | "dog" | "cat" | "telescope" | "park" | "sandwich" | "president" | "waiter" | "chairs" | "tables"
 PP -> "in" | "on" | "by" | "with"
 Adv -> "suddenly" | "quickly" | "slowly" |"very" | "very" | "very"
 Adj -> "lazy" | "very"
 C -> "and"
 """)
sent = "Oscar called the waiter".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
    print(tree)