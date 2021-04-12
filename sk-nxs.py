# skein-keccak 1024 implementation stripped down for NXS

# skein optimization notes
# The config stage has fixed inputs and should be precomputed.
# There are two calls to the UBI function.  One for the message, one for the output.  We will call these UBI1 and UBI2
# There are three 128 bit tweaks resulting in nine 64 bit subtweaks.  Two tweaks for UBI1 and one for UBI2.  The tweaks are fixed and can be precomputed.
# The key schedule for UBI1 is fixed but the subkeys for UBI2 must be calculated using the output of UBI1.
# UBI1 calls threefish twice.  UBI2 calls threefish once for three total calls to threefish. Each threefish call is 80 rounds for a total of 240 rounds.
# So for Nexus there are three threefish calls.  We will call them threefish1, threefish2, and threefish3.  Each has some unique properties that can be precomputed.
# Each threefish takes a 1024 bit key and a 1024 bit message as input.  These are stored as arrays of 16 64 bit words.
# The key and message for threefish2 are key2 and message2.
# Threefish1 processes the first 1024 bits of the header.  For a given block, this is fixed.  The key for threefish1 is also fixed anc can be precomputed.
# Threefish1 is therefore fixed for a given block and can be computed once per block in software.
# Threefish2 takes the output of threefish1 to make it's key.  Key2 is therefore also fixed per block and can be computed once for each new block in software.
# The message for threefish 2 (aka message2) is the last half of the nexus header padded with zeros.  The nonce is included in this message.
# Most of message2 is fixed per block.  The only part that changes is the 64 bit nonce section.
# Message3 is fixed at all zeros.  Key3 is a function of threefish2 and message2.
# We will send the hardware accelerator these values per block as inputs:
# Message2 as 11 unsigned 64 bit numbers (includes the starting nonce).
# Key2 as 17 unsigned 64 bit numbers


import numpy as np
# ignore numpy overflow warnings
np.seterr(over='ignore')

regSize = np.uint64(64) #bits
maxUInt = np.uint64(0xFFFFFFFFFFFFFFFF)  #max value of uint64
# special key constant in threefish.  This is the original constant.  It was updated in version 1.3
C240 = np.uint64(0x5555555555555555)

# operations in threefish are broken down into 64 bit words
Nr = 80  # number of rounds
Nw = 16  # number of 64 bit words (1024 bits / 64)
subkeyCount = Nr // 4 + 1  #21 subkeys

# initialize the subkey array
subkey = np.zeros((subkeyCount, Nw), dtype=np.uint64)

#word permutation constants
permuteIndices = [0, 9, 2, 13, 6, 11, 4, 15, 10, 7, 12, 3, 14, 5, 8, 1]  # real

# The original mix function rotation constants.  These changed between version 1.1 and 1.2 of Skein
R = np.array([[55, 43, 37, 40, 16, 22, 38, 12], [25, 25, 46, 13, 14, 13, 52, 57],
                  [33, 8, 18, 57, 21, 12, 32, 54], [34, 43, 25, 60, 44, 9, 59, 34],
                  [28, 7, 47, 48, 51, 9, 35, 41], [17, 6, 18, 25, 43, 42, 40, 15],
                  [58, 7, 32, 45, 19, 18, 2, 56], [47, 49, 27, 58, 37, 48, 53, 56]], dtype=np.uint64)


# This is the precomputed config key for the first threefish call.
hashInitStr = "56210962be52435aca01f0721a8b6e5f26cea2a19cfecbffca8b036796c3236c6ceb34cefc8b3a583e6aa4d411fbdb3f980930a8fcac0433d20f7fa15f67f6b26babf70e7399259de4a9fe3d0da21409d3db94a4af9c1acc8c38a6a00d032898dce3deaa5d9d330d86a0e2c435de46fcd1a6192ef5e4d653dd1d5d712f956356"

# precomputed tweaks
tweak1Str = "80000000000000000000000000000070"
tweak2Str = "d80000000000000000000000000000b0"
tweak3Str = "080000000000000000000000000000ff"
tweak1 = bytes.fromhex(tweak1Str)
tweak2 = bytes.fromhex(tweak2Str)
tweak3 = bytes.fromhex(tweak3Str)
# tweaks stored as uint64 words
t1 = np.array([0X00000000000080, 0X7000000000000000, 0X7000000000000080],dtype=np.uint64)
t2 = np.array([0X000000000000D8, 0XB000000000000000, 0XB0000000000000D8],dtype=np.uint64)
t3 = np.array([0X00000000000008, 0XFF00000000000000, 0XFF00000000000008],dtype=np.uint64)

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

def printWords(someWords):
    print(", ".join(["{0:#018X}".format(someWords[j]) for j in range(len(someWords))]))

def printWordsVHDL(someWords):
    print(", ".join(["x\"{0:016X}\"".format(someWords[j]) for j in range(len(someWords))]))

# Rotate left: 0b1001 --> 0b0011
def rol(val, r_bits):
    # all inputs must be the same numpy uint type (ie uint64)
    return (val << r_bits) | (val >> (regSize - r_bits))

def mix(x0,x1,d,j):
    #core mix function of threefish
    #x0 and x1 are the two 64 bit inputs
    #y0 and y1 are the two 64 bit outputs
    #d is the threefish round (0 to Nr=80)
    #j is the column (0 to 7)
    #the first output is the addition of the inputs
    y0 = x0 + x1  # overflow is ok here
    #the second output is the xor of y0 with a rotated version of x1
    # the rotation constants repeat every 8 rows hence the d%8
    #bitwise rotation by the rotation constant and bitwise xor with y0
    y1 = rol(x1, R[d%8,j]) ^ y0

    return y0,y1

def permute(someWords, p):
    # rearrange the words based on the algorithm's permute values
    # p is an array of indices that defines the permute
    return someWords[p]

def threefish1024(P):
    #P is plaintext, the 1024 bit message we want to encode (16 words)

    # print("Message: ", P.hex())

    # convert byte strings to 64 bit words
    p = BytesToWords(P)
    # print("Message Words:")
    # printWords(p)

    # v is the current state, stored as 16 64 bit words
    # initialize to the plaintext
    v = np.copy(p)
    # initialize f, the output of the mix operation
    f = np.empty_like(v)
    # iterate through each threefish round
    for d in range(Nr):
        if ((d % 4) == 0):
            # add a subkey every fourth round
            v += subkey[d//4,:]
        # 8 mixes per round
        for j in range(Nw//2):
            f[2*j], f[2*j+1] = mix(v[2*j],v[2*j+1],d,j)
        # 1 permute per round
        v = permute(f, permuteIndices)
        # if ((d % 4) == 3):
        #     print("Threefish round {0} output: ".format(d // 4))
        #     printWords(v)

    # add the final subkey
    v += subkey[subkeyCount - 1]
    # convert to a byte string
    c = WordsToBytes(v)
    # print("Threefish Output:",c.hex())
    return c

def makeSubkeys(key, t):
    # given a key and tweak, generate the 21 x 16 array of 64 bit subkeys used in threefish
    # key is the 1024 bit key (16 words)
    # t is the tweak array of 3 words

    # print("Key: ", key.hex())

    k = BytesToWords(key)


    # bitwise xor C240 and all the key words together to generate a special key
    kNw = C240 ^ np.bitwise_xor.reduce(k)
    # append this to the end of the key words
    k = np.append(k, kNw)
    # print("Key words: " + ", ".join(["{0:#018X}".format(k[j]) for j in range(len(k))]))
    # print("tweak words: {0:#018X}, {1:#018X}, {2:#018X}".format(t[0], t[1], t[2]))

    # iterate through the subkeys and generate the subkey values
    for s in range(subkeyCount):
        for i in range(Nw):
            subkey[s, i] = k[(s + i) % (Nw + 1)]
            if (i == Nw - 3):
                subkey[s, i] += t[s % 3]  # overflow is ok here
            if (i == Nw - 2):
                subkey[s, i] += t[(s + 1) % 3]  # overflow is ok here
            if (i == Nw - 1):
                subkey[s, i] += np.uint64(s)
            # print("Subkey {0},{1}: {2:#X}".format(s,i,subkey[s,i]))

    return subkey


def skein1024(m):
    # top level skein1024 function.  This makes three calls to threefish.

    # Setup the first call to threefish
    # Initialize the state using the precomputed initialization key
    H = bytes.fromhex(hashInitStr)
    # Get the first 1024 bits from the message
    messageBlock = m[0:128]
    # generate the 21 by 16 subkey words for this round of threefish
    makeSubkeys(H, t1)
    # Run Threefish
    # print("First Threefish Call.")
    tf = threefish1024(messageBlock)
    # convert to numpy byte arrays and do bitwise xor
    tfnp = np.frombuffer(tf, dtype=np.uint8)
    messageBlockNP = np.frombuffer(messageBlock, dtype=np.uint8)
    # bitwise xor and convert back to byte string
    H = (tfnp ^ messageBlockNP).tobytes()

    key2 = BytesToWords(H)
    kNw = C240 ^ np.bitwise_xor.reduce(key2)
    # append this to the end of the key words
    key2 = np.append(key2, kNw)
    print("Key2:")
    printWordsVHDL(key2)

    # Setup the second call to threefish
    # Pad the last part of the message with zeros to make it 1024 bits
    messageBlock = m[128:216] + b'\x00' * 40
    message2 = BytesToWords(messageBlock)
    print("message2:")
    printWordsVHDL(message2)

    # Second threefish call
    makeSubkeys(H, t2)
    # print ("Second Threefish Call.")
    tf = threefish1024(messageBlock)
    # convert to numpy byte arrays to do bitwise xor
    tfnp = np.frombuffer(tf, dtype=np.uint8)
    messageBlockNP = np.frombuffer(messageBlock, dtype=np.uint8)
    # bitwise xor and convert back to byte string
    H = (tfnp ^ messageBlockNP).tobytes()

    # Setup the third call to threefish

    # The message is a string of 0s
    messageBlock = b'\x00' * 128
    makeSubkeys(H, t3)
    # Run Threefish
    # print("Third Threefish Call.")
    tf = threefish1024(messageBlock)
    # no final xor is required because the message is all zeros
    print ("Result:")
    printWordsVHDL(BytesToWords(tf))
    return tf


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
# print ("Header:",m.hex())

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

# Expected result after applying Keccak - This is the final hash output from sk1024.
expectedResult = "00000000000072d507b3b1cf8829e6e8201cd5288494b53b379e9f33fcaeeec82d1415330bbb4746354db60b3dbb86ed5008d27877ada92194e5d54d3bfb247ede1b0305db3f208e7e4a51a237dcb0ccc342d345ad7221f0bbe30561e517d0adb2190870bd24ab6b17e9dd895383f183eab21d5d045e438ad9c3d004983eed6b"
# write skein results to a string
skeinStr = sk.hex()

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
kec = myKeccak.Keccak((Len,Msg), rate, capacity, delimitedSuffix, outputBitLength, False)
#convert to bytes, reverse the byte order
kecB = bytes.fromhex(kec)[::-1]
# print(kecB.hex())
if (kecB == bytes.fromhex(expectedHashResult)):
    print("Keccak-1024 test vector 0 matches.")
else:
    print("Keccak-1024 test vector 0 does not match.")
    print("Expected: ", expectedHashResult)
    print("Got: ",kecB.hex())

# simulate mining - increment nonce and check the hash
# nonce = nonce - 10


for i in range(3):
    nonce+=1
    nonceB = nonce.to_bytes(8,endian)
    header = versionB + prevHashB + merkleB + channelB + blockHeightB + bitsB + nonceB

    # Set the message to the header byte string
    m = header
    sk = skein1024(m)
    # print ("Nonce:", nonce, "Nexus Hash:",sk.hex())
    Msg = sk.hex();
    myKeccak = keccak.Keccak()
    kec = myKeccak.Keccak((Len, Msg), rate, capacity, delimitedSuffix, outputBitLength, False)
    # convert to bytes, reverse the byte order
    kecB = bytes.fromhex(kec)[::-1]
    # print ("Nonce:", nonce, "SK Hash:",kecB.hex())

