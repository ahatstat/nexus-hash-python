# skein 1024 implementation for use in Nexus hash
# Copyright Aperture Mining LLC
# Andrew Hatstat
# June 16, 2018

import numpy as np
# ignore numpy overflow warnings
np.seterr(over='ignore')

regSize = np.uint64(64) #bits
maxUInt = np.uint64(0xFFFFFFFFFFFFFFFF)  #max value of uint64
# special key constant in threefish
C240 = np.uint64(0x1BD11BDAA9FC1A22)

#word permutation constants
permuteIndices = [0, 9, 2, 13, 6, 11, 4, 15, 10, 7, 12, 3, 14, 5, 8, 1]
# The mix function rotation constants.  These changed between version 1.1 and 1.2 of Skein
R = np.array([[24, 13, 8, 47, 8, 17, 22, 37],[38, 19, 10, 55, 49, 18, 23, 52],
             [33, 4, 51, 13, 34, 41, 59, 17],[5, 20, 48, 41, 47, 28, 16, 25],
             [41, 9, 37, 31, 12, 47, 44, 30],[16, 34, 56, 51, 4, 53, 42, 41],
             [31, 44, 47, 46, 19, 42, 44, 25],[9, 48, 35, 52, 23, 31, 37, 20]], dtype=np.uint64)

# The Skein-1024 Configuration String for Pure Hashing
ConfigString = b'SHA3' + int.to_bytes(1,2,'little') + int.to_bytes(0,2,'little') + int.to_bytes(1024,8,'little')\
               + int.to_bytes(0,16,'little')

# print (ConfigString.hex())
# print (len(ConfigString))
# A few of the message types that we use with hashing
Tcfg = 4  # Configuration block
Tmsg = 48 # Message Block
Tout = 63 # Output Block

# byte-word converters from skein paper
def WordsToBytes(someWords):
    # Convert a word or array of words (uint64) to a byte string
    # someWords must be a numpy array or a numpy scalar
    # check for scalar
    if (np.isscalar(someWords)):
        someWords = np.array([someWords])
    return someWords.tobytes()

def BytesToWords(someBytes):
    # convert a byte string to an array of words (uint64)
    return np.frombuffer(someBytes, dtype=np.uint64)

def ceildiv(a, b):
    # ceiling division (divides and then rounds up)
    return -(-a // b)

# Rotate left: 0b1001 --> 0b0011
def rol(val, r_bits):
    # all inputs must be the same numpy uint type (ie uint64)
    return (val << r_bits) | (val >> (regSize - r_bits))

def mix(x0,x1,d,j):
    #core mix function of threefish
    #x0 and x1 are the two 64 bit inputs
    #y0 and y1 are the two 64 bit outputs
    #d is the round or row (0 to Nr)
    #j is the column (0 to Nw/2 - 1 = 7 for 1024 bit threefish)
    #the first output is the addition of the inputs
    y0 = x0 + x1  # overflow is ok here
    #the second output is the xor of x0 with a rotated version of x1
    # the rotation constants repeat every 8 rows hence the d%8
    #bitwise rotation by the rotation constant and bitwise xor with y0
    y1 = rol(x1, R[d%8,j]) ^ y0

    return y0,y1

def permute(someWords, p):
    # rearrange the words based on the algorithm's permute values
    # p is an array of indices that defines the permute
    return someWords[p]

def threefish1024(K, T, P):
    #P is plaintext, the 1024 bit message we want to encode (16 words)
    #K is the 1024 bit key (16 words)
    #T is the 128 bit tweak (2 words)
    # P K and T are all byte strings

    # convert byte strings to 64 bit words
    p = BytesToWords(P)
    k = BytesToWords(K)
    t = BytesToWords(T)

    #operations in threefish are broken down into 64 bit words
    Nr = 80  #number of rounds
    Nw = 16  #number of 64 bit words (1024 bits / 64)

    # Every four rounds we inject a subkey.  In 1024 bit threefish there are Nr/4 + 1 = 21 subkeys
    # Subkeys are stored as 21 rows by 16 words 2D array of uint64
    # Generate subkeys
    #bitwise xor C240 and all the key words together to generate a special key
    kNw = C240 ^ np.bitwise_xor.reduce(k)
    # append this to the end of the key words
    k = np.append(k, kNw)
    # print('kNw: {0:#X}'.format(kNw))
    # generate t2 by xoring t0 and t1
    t2 = t[0] ^ t[1]
    t = np.append(t,t2)
    subkeyCount = Nr//4 + 1
    # initialize the subkey array
    subkey = np.zeros((subkeyCount,Nw),dtype=np.uint64)
    # iterate through the subkeys and generate the subkey values
    for s in range(subkeyCount):
        for i in range(Nw):
            subkey[s,i] = k[(s+i) % (Nw + 1)]
            if (i == Nw - 3):
                subkey[s,i] += t[s % 3]  # overflow is ok here
            if (i == Nw - 2):
                subkey[s,i] += t[(s+1) % 3] # overflow is ok here
            if (i == Nw - 1):
                subkey[s,i] += np.uint64(s)
    # print(subkey)
    # v is the current state, same size as the key
    # initialize to the plaintext
    v = np.copy(p)
    # initialize f, the output of the mix operation
    f = np.empty_like(v)
    # iterate through each threefish round
    for d in range(Nr):
        if ((d % 4) == 0):
            # add a subkey
            v += subkey[d//4,:]
        # 8 mixes per round
        for j in range(Nw//2):
            f[2*j], f[2*j+1] = mix(v[2*j],v[2*j+1],d,j)
        # 1 permute per round
        v = permute(f, permuteIndices)

    # add the final subkey
    v += subkey[subkeyCount - 1]
    # convert to a byte string
    c = WordsToBytes(v)
    return c


def UBI(G,M,Ts):
    # Skein UBI function
    # G is the starting value.  This is a byte string.  The length must match the key length. For 1024 bits Nb = 16 bytes long
    # M is the message.  This is an arbitrary length byte string.
    # Ts is the starting value of the tweak.  This is a 128 bit integer. (Not a byte string)
    # Only messages with complete bytes are supported.  No tree stuff is supported.  This is for straight hashing.
    Nb = 1024//8 # bytes in the key
    #number of bytes in the message
    Nm = len(M)
    # number of 1024 bit blocks in the message
    Nk = ceildiv(Nm,Nb)
    if (Nm == 0):
        padByteCount = Nb # The message is empty.  Pad with one key length's worth of zeros
        Nk = 1  #There is now one block
    else:
        padByteCount = -Nm % Nb
    #print(padByteCount)
    # pad the message with zeros so that it is an even number of words
    M = M + b'\x00'*padByteCount
    # print (M)
    # print (len(M))

    #initialize
    H = G
    # set first only for the first block
    first = 1
    last = 0
    # Iterate through the message words.  Each UBI round works on one word.
    for i in range(Nk):
        # Update the tweak
        # ToBytes(Ts + min(NM,(i + 1)Nb) + ai2^126 + bi(B2^119 + 2^127), 16)
        # check if this is the last word
        if i == Nk - 1:
            # This is the last block in the message
            last = 1  # set the last flag
        T = Ts + min(Nm, (i+1)*Nb) + first*2**126 + last*2**127  #Updated tweak
        #break the message into 1024 bit blocks
        messageBlock = M[i*128:(i+1)*128]
        # Run Threefish
        tf = threefish1024(H, int.to_bytes(T,16,'little'), messageBlock)
        # convert to numpy byte arrays to do bitwise xor
        tfnp = np.frombuffer(tf,dtype=np.uint8)
        messageBlockNP = np.frombuffer(messageBlock,dtype=np.uint8)
        # bitwise xor and convert back to byte string
        H = (tfnp ^ messageBlockNP).tobytes()
        #clear the first flag
        first = 0

    return H

def OutputBlock(G):
    # For our application we fix the output length to match key size.
    return UBI(G,int.to_bytes(0,8,'little'),Tout*2**120)

def skein1024(m):
    # top level skein1024 function.
    # m is the plaintext message to be hashed as a byte string
    # The key for the config block is zero
    K = int.to_bytes(0,128,'little')
    # Config block
    G0 = UBI(K,ConfigString,Tcfg*2**120)
    G1 = UBI(G0, m, Tmsg*2**120)
    H = OutputBlock(G1)
    return H


# The initial key for pure hasing.  These are valid only for skein-1024 with 1024 bit output for version 1.3.
# Copied from the spec.
KeyInit = np.array([0xD593DA0741E72355, 0x15B5E511AC73E00C, 0x5180E5AEBAF2C4F0, 0x03BD41D3FCBCAFAF,
0x1CAEC6FD1983A898, 0x6E510B8BCDD0589F, 0x77E2BDFDC6394ADA, 0xC11E1DB524DCB0A3,
0xD6D14AF9C6329AB5, 0x6A9B0BFC6EB67E0D, 0x9243C60DCCFF1332, 0x1A1F1DDE743F02D4,
0x0996753C10ED0BB8, 0x6572DD22F2B4969A, 0x61FD3062D00A579A, 0x1DE0536E8682E539], dtype=np.uint64)
KeyInitBytes = WordsToBytes(KeyInit)

# Test vectors for Skein version 1.3
# Threefish test vectors from https://sites.google.com/site/bartoszmalkowski/threefish
# three fish test vector 0
key = WordsToBytes(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.uint64))
tweak = WordsToBytes(np.array([0,0],dtype=np.uint64))
plainText = WordsToBytes(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.uint64))
cipherTextStr = "f05c3d0a3d05b304f785ddc7d1e036015c8aa76e2f217b06c6e1544c0bc1a90d\
f0accb9473c24e0fd54fea68057f43329cb454761d6df5cf7b2e9b3614fbd5a2\
0b2e4760b40603540d82eabc5482c171c832afbe68406bc39500367a592943fa\
9a5b4a43286ca3c4cf46104b443143d560a4b230488311df4feef7e1dfe8391e"

tf = threefish1024(key, tweak, plainText)
if (tf == bytes.fromhex(cipherTextStr)):
    print("Threefish test vector 0 matches.")
else:
    print("Threefish test vector 0 does not match.")
    print("Expected: ", cipherTextStr)
    print("Got: ",tf.hex())

# test vector 1
keyStr = "101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f\
303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f\
505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f\
707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f"
tweakStr = "000102030405060708090a0b0c0d0e0f"
plainTextStr = "fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0efeeedecebeae9e8e7e6e5e4e3e2e1e0\
dfdedddcdbdad9d8d7d6d5d4d3d2d1d0cfcecdcccbcac9c8c7c6c5c4c3c2c1c0\
bfbebdbcbbbab9b8b7b6b5b4b3b2b1b0afaeadacabaaa9a8a7a6a5a4a3a2a1a0\
9f9e9d9c9b9a999897969594939291908f8e8d8c8b8a89888786858483828180"
cipherTextStr = "a6654ddbd73cc3b05dd777105aa849bce49372eaaffc5568d254771bab85531c\
94f780e7ffaae430d5d8af8c70eebbe1760f3b42b737a89cb363490d670314bd\
8aa41ee63c2e1f45fbd477922f8360b388d6125ea6c7af0ad7056d01796e90c8\
3313f4150a5716b30ed5f569288ae974ce2b4347926fce57de44512177dd7cde"

key = bytes.fromhex(keyStr)
tweak = bytes.fromhex(tweakStr)
plainText = bytes.fromhex(plainTextStr)
# print(tweak)

tf = threefish1024(key, tweak, plainText)
if (tf == bytes.fromhex(cipherTextStr)):
    print("Threefish test vector 1 matches.")
else:
    print("Threefish test vector 1 does not match.")
    print("Expected: ", cipherTextStr)
    print("Got: ",tf.hex())

# UBI test
K = int.to_bytes(0,128,'little')
# Generate the config block hash for straight hashing. This should match KeyInit from the spec.
G0 = UBI(K,ConfigString,Tcfg*2**120)
if (G0 == KeyInitBytes):
    print("UBI config block test vector matches.")
else:
    print("UBI config block test vector does not match.")
    print("Expected: ", KeyInitBytes.hex())
    print("Got: ",G0.hex())

# skein1024 test
m = b'\xFF'
expectedResult = "E62C05802EA0152407CDD8787FDA9E35703DE862A4FBC119CFF8590AFE79250B\
CCC8B3FAF1BD2422AB5C0D263FB2F8AFB3F796F048000381531B6F00D85161BC0FFF4BEF2486B1EBCD3773FABF50AD4A\
D5639AF9040E3F29C6C931301BF79832E9DA09857E831E82EF8B4691C235656515D437D2BDA33BCEC001C67FFDE15BA8"
sk1 = skein1024(m)
if (sk1 == bytes.fromhex(expectedResult)):
    print("Skein-1024 test vector 0 matches.")
else:
    print("Skein-1024 test vector 0 does not match.")
    print("Expected: ", expectedResult)
    print("Got: ",sk1.hex())

mStr = "FFFEFDFCFBFAF9F8F7F6F5F4F3F2F1F0EFEEEDECEBEAE9E8E7E6E5E4E3E2E1E0\
DFDEDDDCDBDAD9D8D7D6D5D4D3D2D1D0CFCECDCCCBCAC9C8C7C6C5C4C3C2C1C0BFBEBDBCBBBAB9B8B7B6B5B4B3B2B1B0\
AFAEADACABAAA9A8A7A6A5A4A3A2A1A09F9E9D9C9B9A999897969594939291908F8E8D8C8B8A89888786858483828180"
expectedResult = "1F3E02C46FB80A3FCD2DFBBC7C173800B40C60C2354AF551189EBF433C3D85F9\
FF1803E6D920493179ED7AE7FCE69C3581A5A2F82D3E0C7A295574D0CD7D217C484D2F6313D59A7718EAD07D0729C248\
51D7E7D2491B902D489194E6B7D369DB0AB7AA106F0EE0A39A42EFC54F18D93776080985F907574F995EC6A37153A578"
m = bytes.fromhex(mStr)
sk1 = skein1024(m)
if (sk1 == bytes.fromhex(expectedResult)):
    print("Skein-1024 test vector 1 matches.")
else:
    print("Skein-1024 test vector 1 does not match.")
    print("Expected: ", expectedResult)
    print("Got: ",sk1.hex())

mStr = "FFFEFDFCFBFAF9F8F7F6F5F4F3F2F1F0EFEEEDECEBEAE9E8E7E6E5E4E3E2E1E0\
DFDEDDDCDBDAD9D8D7D6D5D4D3D2D1D0CFCECDCCCBCAC9C8C7C6C5C4C3C2C1C0BFBEBDBCBBBAB9B8B7B6B5B4B3B2B1B0\
AFAEADACABAAA9A8A7A6A5A4A3A2A1A09F9E9D9C9B9A999897969594939291908F8E8D8C8B8A89888786858483828180\
7F7E7D7C7B7A797877767574737271706F6E6D6C6B6A696867666564636261605F5E5D5C5B5A59585756555453525150\
4F4E4D4C4B4A494847464544434241403F3E3D3C3B3A393837363534333231302F2E2D2C2B2A29282726252423222120\
1F1E1D1C1B1A191817161514131211100F0E0D0C0B0A09080706050403020100"

expectedResult = "842A53C99C12B0CF80CF69491BE5E2F7515DE8733B6EA9422DFD676665B5FA42\
FFB3A9C48C217777950848CECDB48F640F81FB92BEF6F88F7A85C1F7CD1446C9161C0AFE8F25AE444F40D3680081C35A\
A43F640FD5FA3C3C030BCC06ABAC01D098BCC984EBD8322712921E00B1BA07D6D01F26907050255EF2C8E24F716C52A5"

m = bytes.fromhex(mStr)
sk1 = skein1024(m)
if (sk1 == bytes.fromhex(expectedResult)):
    print("Skein-1024 test vector 2 matches.")
else:
    print("Skein-1024 test vector 2 does not match.")
    print("Expected: ", expectedResult)
    print("Got: ",sk1.hex())

# Nexus uses an old version of skein - version 1.1. Modify the constants to revert from 1.3 to 1.1
C240_old = np.uint64(0x5555555555555555)  #key constant used prior to skein 1.3
# Deprecated rotation contants used prior to skein 1.2
R_old = np.array([[55, 43, 37, 40, 16, 22, 38, 12], [25, 25, 46, 13, 14, 13, 52, 57],
                  [33, 8, 18, 57, 21, 12, 32, 54], [34, 43, 25, 60, 44, 9, 59, 34],
                  [28, 7, 47, 48, 51, 9, 35, 41], [17, 6, 18, 25, 43, 42, 40, 15],
                  [58, 7, 32, 45, 19, 18, 2, 56], [47, 49, 27, 58, 37, 48, 53, 56]], dtype=np.uint64)
ConfigString_old = ConfigString+b'\x00'*96 # used in skein 1.0 only.  Not used in Nexus

R = R_old  #Rotation constants updated in version 1.2
C240 = C240_old #Key Constant updated in version 1.3
# ConfigString=ConfigString_old - Changed in version 1.1.  1.0 there was a bug in the reference implementation
# that resulted in the config string being the wrong length for skein-512 and skein-1024

# use a sample Nexus block as a test vector
blockHeight = 2023276
version = 4
channel = 2  # Hash channel is 2
bits = 0x7b032ed8
nonce = 21155560019
merkleStr = "31f5a458fc4207cd30fd1c4f43c26a3140193ed088f75004aa5b07beebf6be905fd49a412294c73850b422437c414429a6160df514b8ec919ea8a2346d3a3025"
prevHashStr = "00000902546301d2a29b00cad593cf05c798469b0e3f39fe623e6762111d6f9eed3a6a18e0e5453e81da8d0db5e89808e68e96c8df13005b714b1e63d7fa44a5025d1370f6f255af2d5121c4f65624489f1b401f651b5bd505002d3a5efc098aa6fa762d270433a51697d7d8d3252d56bbbfbe62f487f258c757690d31e493a7"
expectedHashResult = "00000000000072d507b3b1cf8829e6e8201cd5288494b53b379e9f33fcaeeec82d1415330bbb4746354db60b3dbb86ed5008d27877ada92194e5d54d3bfb247ede1b0305db3f208e7e4a51a237dcb0ccc342d345ad7221f0bbe30561e517d0adb2190870bd24ab6b17e9dd895383f183eab21d5d045e438ad9c3d004983eed6b"

endian = 'little'
# convert header data to byte strings
blockHeightB = blockHeight.to_bytes(4,endian)
versionB = version.to_bytes(4,endian)
channelB = channel.to_bytes(4,endian)
bitsB = bits.to_bytes(4,endian)
nonceB = nonce.to_bytes(8,endian)
merkleB = bytes.fromhex(merkleStr)[::-1]  # hex strings are big endian in nexus world but stored as little endian so we need to reverse this
prevHashB = bytes.fromhex(prevHashStr)[::-1]
# Assemble the nexus header as a byte string
header = versionB + prevHashB + merkleB + channelB + blockHeightB + bitsB + nonceB
# The header length should be 216 bytes
# Set the message to the header byte string
m = header

# m = b'\xAB\xFF' + b'\x00'*126
# m = b'\x00'*128
expectedResult = "bc6f384f3f42402d271b91db917552db58e3a533ba19bd69ea259ca3edc0df6f6110b1f95db3208d1f8ecfc8c1dca11a25b02689dee7e8e73df4a891eb13086d3856dd0e0fac1594d0909199ab3ad9ec45f79ec4397b1d78b238fb816b563be6808f81c6205b113a7ff769de6764ced2dd20addd59ced6e0f55c821a9c659a1c"
sk = skein1024(m)
#reverse hex string endianness
sk1 = sk[::-1]
if (sk1 == bytes.fromhex(expectedResult)):
    print("Nexus Skein-1024 test vector 0 matches.")
else:
    print("Nexus Skein-1024 test vector 0 does not match.")
    print("Expected: ", expectedResult)
    print("Got: ",sk1.hex())

#Keccak test vector using canned library
expectedResult = "00000000000072d507b3b1cf8829e6e8201cd5288494b53b379e9f33fcaeeec82d1415330bbb4746354db60b3dbb86ed5008d27877ada92194e5d54d3bfb247ede1b0305db3f208e7e4a51a237dcb0ccc342d345ad7221f0bbe30561e517d0adb2190870bd24ab6b17e9dd895383f183eab21d5d045e438ad9c3d004983eed6b"
# write skein results to a string
skeinStr = sk.hex()
print ("Keccak Input:",skeinStr)

#import the Keccak library
import KeccakInPython.Keccak as keccak

#Nexus uses 0x05 suffix
delimitedSuffix = 0x05
rate = 576
capacity = 1024
outputBitLength = 1024
Len = 1024
Msg = skeinStr
myKeccak = keccak.Keccak()
kec = myKeccak.Keccak((Len,Msg), rate, capacity, delimitedSuffix, outputBitLength, True)
print ("Keccak Output:",kec)
#convert to bytes, reverse the byte order
kecB = bytes.fromhex(kec)[::-1]
# print(kecB.hex())
if (kecB == bytes.fromhex(expectedHashResult)):
    print("Keccak-1024 test vector 0 matches.")
else:
    print("Keccak-1024 test vector 0 does not match.")
    print("Expected: ", expectedHashResult)
    print("Got: ",kecB.hex())

# Try another Nexus test vector
blockHeight = 206400
version = 3
channel = 2  # Hash channel is 2
bits = 0x7c01fa49
nonce = 836706567
merkleStr = "5b20bb6dde3afe3f02ec0897b04afe406d19f818672791f986a09823ea1cbfa3dacde3ec7911359153b903631cfb8085ce182216a7af2a58b844f59c8e83ad1d"
prevHashStr = "00000000015aac018c92c3cd6475661768e1e6f803ad31631e7b1c9322d039710e4b9be5b09cb11c1ae9a38b19c1e269b7818abf362cdf2f289661f8d7d7e37337e4f7af3674c6704bffbe48a323242dee084d3c26e57a992e5134ea632ed7e247eae71b4a5c54f9923bb2c9705f43cfb93730cf9712a5335d914d1ae74b016b"
expectedHashResult = "0000000000c951d32c80f7b6ca2153d1446a56b63926b2c7948f81ff630c81244dcad059ba1e8ba6dd867df7f47d60a8462a713d4e50c68c5dd21c6fc1ef3e7dce14a2f35455b366a9b8d48101243360695b765618db00c9b21a244d8f375b522735d6b6fd2bbbca8011bbd346364728b298d60e4e976253aea7464e37b4ff24"

endian = 'little'
# convert header data to byte strings
blockHeightB = blockHeight.to_bytes(4,endian)
versionB = version.to_bytes(4,endian)
channelB = channel.to_bytes(4,endian)
bitsB = bits.to_bytes(4,endian)
nonceB = nonce.to_bytes(8,endian)
merkleB = bytes.fromhex(merkleStr)[::-1]  # hex strings are big endian in nexus world but stored as little endian so we need to reverse this
prevHashB = bytes.fromhex(prevHashStr)[::-1]
# Assemble the nexus header as a byte string
header = versionB + prevHashB + merkleB + channelB + blockHeightB + bitsB + nonceB
# The header length should be 216 bytes
# Set the message to the header byte string
m = header

expectedResult = "93cf1838ebdb1b09c466ad0023b444746e342e33eb003700b627e451163775dc3d15c5718ed12f9ebf410665071f1ec49a36d5e6b45f07dbdfa4f6f09a9f50292ec92cfa3fccd00c1b2f73293cb6544def8ad5bd53d1fec08f0e82d4aa4a4c4f1c6583043997e0a6e805450119d969da2ecf8337cf779cc0e420abb8c491e524"
sk = skein1024(m)
#reverse hex string endianness
sk1 = sk[::-1]
if (sk1 == bytes.fromhex(expectedResult)):
    print("Nexus Skein-1024 test vector 1 matches.")
else:
    print("Nexus Skein-1024 test vector 1 does not match.")
    print("Expected: ", expectedResult)
    print("Got: ",sk1.hex())

#Keccak test vector using canned library
expectedResult = expectedHashResult
# write skein results to a string
skeinStr = sk.hex()
print ("Keccak Input:",skeinStr)

#Nexus uses 0x05 suffix
delimitedSuffix = 0x05
rate = 576
capacity = 1024
outputBitLength = 1024
Len = 1024
Msg = skeinStr
myKeccak = keccak.Keccak()
kec = myKeccak.Keccak((Len,Msg), rate, capacity, delimitedSuffix, outputBitLength, False)
#convert to bytes, reverse the byte order
kecB = bytes.fromhex(kec)[::-1]
# print(kecB.hex())
if (kecB == bytes.fromhex(expectedHashResult)):
    print("Keccak-1024 test vector 1 matches.")
else:
    print("Keccak-1024 test vector 1 does not match.")
    print("Expected: ", expectedHashResult)
    print("Got: ",kecB.hex())

print ("Keccak Output:",kec)
