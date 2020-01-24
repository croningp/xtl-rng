import os
import sys
import random
import inspect
import time

from MTcracker.randcrack import RandCrack

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PROJECT_DIR =os.path.dirname(os.path.dirname(HERE_PATH))

def from_twister(number_of_keys=1):
    total_ints = int(number_of_keys/4)
    randints = [random.randint(0,4294967294) for _ in range(total_ints)]
    randbits = [to_bitarray(i) for i in randints]
    randkeys = []
    for bits in randbits:
        for i in range(4):
            randkeys.append(bits[i*8:(i+1)*8])
    return randkeys

def from_file(number_of_keys=1, filepath=''):
    string = open(filepath, 'r').read().strip()
    randkeys = []
    for i in range(number_of_keys):
        chunk = string[i*8:(i+1)*8]
        key = [int(i) for i in chunk]
        randkeys.append(key)
    return randkeys

def encrypt(message, key):
    encrypted_message = ''

    for letter in message:
        letter_bits = [int(i) for i in bin(ord(letter))[2:].zfill(8)]
        encrypted_letter = [l ^ k for l, k in zip(letter_bits, key)]
        encrypted_letter = chr(to_int(encrypted_letter))
        encrypted_message += encrypted_letter
    return encrypted_message

def predict(total_keys, cracker):
    total_ints = int(total_keys/4) - 624
    predints = [cracker.predict_getrandbits(32) for i in range(total_ints)]
    predbits = [to_bitarray(i) for i in predints]

    randkeys = []
    for bits in predbits:
        for i in range(4):
            randkeys.append(bits[i*8:(i+1)*8])
    return randkeys

def decrypt(message, key=None):
    return encrypt(message, key)


def brute_force(message):
    for key in range(2**8):
        key = [int(bit) for bit in bin(key)[2:].zfill(8)]
        decrypted_message = decrypt(message, key)
        if decrypted_message == original_message:
            return key
    print('Did not find key')
    sys.exit()

def to_bitarray(num):
    k = [int(x) for x in bin(num)[2:]]
    return [0] * (32-len(k)) + k

def to_int(bits):
    return int("".join(str(i) for i in bits), 2)

def run(total_keys, compound='', attempt_crack=False):

    if compound:
        filepath = os.path.join(PROJECT_DIR, 'results', 'numbers', '{}.txt'.format(compound))
        print('Running ', compound)
        keys = from_file(total_keys, filepath)
    else:
        keys = from_twister(total_keys)
        print('Running Mersenne Twister')

    encrypted_messages = [encrypt(original_message, key) for key in keys]

    cracked = False
    known_keys = []
    new_32_bit_key = []
    pred_keys = []
    kdx = 0
    cracker = RandCrack()
    durations = []
    for idx, encrypted in enumerate(encrypted_messages):
        start_time = time.time()
        if not cracked:
            key = brute_force(encrypted)
            new_32_bit_key.extend(key)
            if attempt_crack:
                if idx % 4 == 3:
                    cracker.submit(to_int(new_32_bit_key))
                    new_32_bit_key = []
                    if cracker.state:
                        cracked = True
                        predkeys = predict(total_keys, cracker)
        else:
            if original_message == decrypt(encrypted, predkeys[kdx]):
                key = predkeys[kdx]
            else:
                key = brute_force(encrypted)
            kdx += 1
        end_time = time.time()
        durations.append(end_time-start_time)
    return durations

original_message = 'crystal'
total_keys = 12800

compound = 'W19'

CuSO4_durations = run(total_keys=total_keys, compound='CuSO4')
W19_durations = run(total_keys=total_keys, compound='W19')
Co4_durations = run(total_keys=total_keys, compound='Co4')
MT_durations = run(total_keys=total_keys, compound='', attempt_crack=True)

MT1 = MT_durations[:624*4]
MT2 = MT_durations[624*4:]
