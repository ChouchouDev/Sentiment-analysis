class SentiWordNet():
    def __init__(self, netpath):
        self.fileGenerated = open("./SentiWordNet_simple_format.txt", "w") # we generate a lexicon of simple format
        self.netpath = netpath
        self.dictionary = {}
        self.infoextract()

    def infoextract(self):
        tempdict = {}
        try:
            f = open(self.netpath, "r")
        except IOError:
            print("failed to open file!")
            exit()

        print('loading lexicon...')
        # Example line:
        # POS     ID     PosS  NegS SynsetTerm#sensenumber Desc
        # a   00009618  0.5    0.25  spartan#4 austere#3 ascetical#2  ……

        lines = f.readlines()
        for sor in lines:
            if sor.strip().startswith("#"):
                pass
            else:
                data = sor.split("\t")
                if len(data) != 6:
                    print('invalid data')
                    continue
                wordTypeMarker = data[0]
                synsetScore = float(data[2]) - float(data[3])  # // Calculate synset score as score = PosS - NegS

                synTermsSplit = data[4].split(" ")  # word#sentimentscore
                for w in synTermsSplit:
                    synTermAndRank = w.split("#")  #
                    synTerm = synTermAndRank[0]
                    synTermRank = int(synTermAndRank[1])
                    if synTerm in tempdict.keys():
                        t = [synTermRank, synsetScore]
                        tempdict.get(synTerm).append(t)
                    else:
                        temp = {synTerm: []}
                        t = [synTermRank, synsetScore]
                        temp.get(synTerm).append(t)
                        tempdict.update(temp)

        for key in tempdict.keys():
            score = 0.0
            ssum = 0.0
            for wordlist in tempdict.get(key):
                score += wordlist[1] / wordlist[0]   #score = ∑ synserScore/rank
                ssum += 1.0 / wordlist[0]           #ssum = ∑ 1/rank
            score /= ssum   # score = score/ssum
            self.fileGenerated.write(key + ":" + str(score) + "\n")
            # self.dictionary.update({key: score})
        self.fileGenerated.close()

    def getDic(self):
        return self.dictionary

# pathNet = "SentiWordNet_3.0.0_20130122.txt"
# SentiWordNet(pathNet)