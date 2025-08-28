#!/usr/bin/env python3
import random
import sys
import json

ALPHABET = "QWERTYUIOP"

def generate_sequence(size, alphabet=ALPHABET):
	return ''.join(random.choices(alphabet, k=size))

def mutated_sequences(seq, k=10, target=None, alphabet=ALPHABET):
	if target is None:
		target = len(seq)
	for _ in range(k):
		mutation = ''
		for c in seq:
			chance = random.random()
			if chance < 0.8: # none
				mutation += c
			elif chance < 0.9: # mutation
				mutation += random.choice(alphabet)
			elif chance < 0.95: # insertion
				mutation += '-'
				mutation += c
			else: # deletion
				pass
		while len(mutation) > target:
			deletion = random.randint(0, len(mutation) - 1)
			mutation = mutation[:deletion] + mutation[deletion+1:]
		while len(mutation) < target:
			insertion = random.randint(0, len(mutation) - 1)
			mutation = mutation[:insertion] + random.choice(alphabet) + mutation[insertion:]
		yield mutation

def main(size, amount):
	size = int(size)
	amount = int(amount)
	actual = size + random.randint(0, max(4, size // 10 - 1)) - size // 10
	
	seq = generate_sequence(actual)
	seq_size = size + random.randint(0, max(4, size // 5 - 1)) - size // 5
	generated_seq = list(mutated_sequences(seq, k=1, target=seq_size))[0].replace("-", "")

	print(f'profile_HMM_sequence_alignment(\"{generated_seq}\", 0.25, 0.04, \"{ALPHABET}\", {json.dumps(list(mutated_sequences(seq, k=amount, target=size)))})')

if __name__ == "__main__":
	main(*sys.argv[1:])
