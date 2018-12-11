#!/usr/bin/env python3

import random
import sys

def readSentences(filename):
    sentences = list()
    f = open(filename, 'r')
    for line in f.readlines():
        sentences.append(line[:-1])
    return sentences

def extractVocabulary(sentences):
    words = set()
    for s in sentences:
        if s[-1] == '.':
            s = s[:-1]
        for word in s.split(" "):
            word = word.lower().strip()
            if len(word) > 0:
                words.add(word)
    return list(words)

def genGibberish(vocabulary, expectedLenght):
    word = random.choice(vocabulary)
    words = [ word[0].upper() + word[1:] ]
    length = len(word)
    word = random.choice(vocabulary)
    while length + len(word) < expectedLenght:
        words.append(word)
        length += len(word)
        word = random.choice(vocabulary)
    return " ".join(words)

def randomSequence(expectedLenght):
    s =  "".join((random.choice("    abcdefghijklmnopqrstuvwxyz") for i in range(expectedLenght)))
    return s[0].upper() + s[1:]

if __name__ == '__main__':
    filename = sys.argv[1] # database of sentences
    sentences = readSentences(filename)
    n = len(sentences)
    minLength = min(map(len, sentences))
    maxLength = max(map(len, sentences))
    averageLength = sum(map(len, sentences)) // n
    vocabulary = extractVocabulary(sentences)
    for s in sentences:
        print('"{}",1,1'.format(s))
        print('"{}",0,1'.format(genGibberish(vocabulary, averageLength)))
        print('"{}",0,0'.format(randomSequence(random.randint(minLength, maxLength))))






