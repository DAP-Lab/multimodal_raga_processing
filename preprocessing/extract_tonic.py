import essentia.standard
from sys import argv

src=argv[1]
loader = essentia.standard.MonoLoader(filename=src)
audio = loader()
tonic = essentia.standard.TonicIndianArtMusic(maxTonicFrequency=530)(audio)

print (tonic)