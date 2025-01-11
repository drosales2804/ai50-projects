import nltk
import sys
import re

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
NP -> N | Det N | Det AP N | P NP | NP P NP
VP -> V | Adv VP | V Adv | VP NP | V NP Adv
AP -> Adj | AP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # Get the sentence (file or user input)
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    # Preprocess the sentence into words
    s = preprocess(s)

    # Try parsing the sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print parsed trees and noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` into a list of words.
    Make everything lowercase and skip words w/o letters.
    """
    # Match words w/ at least one letter
    test = re.compile('[a-zA-Z]')

    # Tokenize sentence into words
    tokens = nltk.word_tokenize(sentence)

    # Return words in lowercase, ignore non-alphabetic ones
    return [word.lower() for word in tokens if test.match(word)]


def np_chunk(tree):
    """
    Get all noun phrase chunks in a sentence tree.
    A noun phrase chunk is an "NP" subtree w/o other "NP" subtrees inside.
    """
    chunks = []

    # Check every subtree in the tree
    for subtree in tree.subtrees():
        # If it's labeled "NP", check if it's simple
        if subtree.label() == "NP":
            # Add it if no child subtree is also "NP"
            if not any(child.label() == "NP" for child in subtree.subtrees() if child != subtree):
                chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
