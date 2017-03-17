# Author info: Mahyar Khayatkhoei @ m.khayatkhoei@gmail.com

import sys
import re
from collections import defaultdict
import argparse
import spell_check
import nltk.stem.wordnet as wn

lem = wn.WordNetLemmatizer()

parse_re = re.compile(r'([a-zA-Z0-9_:]+)\(([-a-zA-Z0-9/]+), ([-a-zA-Z0-9/]+)\)')
parse_sent = re.compile(r'(\w(?:[-,a-zA-Z0-9_]|\s)+$)')
parse_nn = re.compile(r'NN.*')
parse_jj = re.compile(r'JJ.*')
parse_nmod = re.compile(r'nmod:\w+')
hyphoned = re.compile(r'[a-z]+-[a-z]+')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
        "-f", "--filename", dest="fname", required=True,
        help="name of the current file")
def main():
	args = arg_parser.parse_args()
	fname = args.fname
	gram_type = dict()
	stream = sys.stdin
	head = False
	start = False
	for l in stream:
		sl = l.strip(' !?.\n')
		if not start:
			### Start a parsedb instance
			sm = parse_sent.match(sl)
			if sm:
				#print(sm.group(0))
				parsedb = ParseDB(sm.group(0))
				start = True
		else:
			if sl == '':
				if not head:
					head = True
				else:
					head = False
					start = False
					#print(parsedb)
					parsedb.extract()
					parsedb.save_to_file(fname)
				continue

			### Check for the parse triple
			m = parse_re.match(sl)
			if m:
				#print([m.group(1), break_down(m.group(2)), break_down(m.group(3))])
				#print(sl)
				assert len(m.groups()) == 3
				parsedb.store(m.groups())
				continue

class ParseDB:
	def __init__(self, sentence):
		self.sent = sentence
		self.grams = defaultdict(list)
		self.words = defaultdict(lambda: defaultdict(list))
		self.rels = list()

	def store(self, parse_triple):
		self.grams[parse_triple[0]].append(parse_triple[1:])
		self.words[parse_triple[1]][parse_triple[0]].append(parse_triple[2])
		self.words[parse_triple[2]][parse_triple[0]].append(parse_triple[1])

	def __str__(self):
		print('\n===========ParseDB IS PRINTING')
		#print(self.sent)
		#print(self.grams)
		#print(self.words)
		return self.sent

	def preproc_rels(self, rel):
		rel_list = list()
		for w in rel:
			hm = hyphoned.match(w)
			if hm:
				wl, wr = hm.group(0).split('-')
				wr = lem.lemmatize(wr,'v')
				rel_list += [wr, wl]
			else:
				rel_list.append(w)

		for w_i, w in enumerate(rel_list):
			rel_list[w_i] = spell_check.correct(w)

		return tuple(rel_list)

	def save_to_file(self, filename):
		rel_form2 = '(is (and %s %s))'
		rel_form3 = '(is (and %s %s %s))'
		### Generate parses
		with open(filename+'.sps2', 'a+') as spf, open(filename+'.sent', 'a+') as sentf:
			for r in self.rels:
				r = self.preproc_rels(r)
				rs = ' '.join(r)
				s = '(is (and '+rs+'))'
				#if len(r) == 2:
				#	s = rel_form2 % r
				#elif len(r) == 3:
				#	s = rel_form3 % r
				#elif len(r) == 4:
				#	s = rel_form3 % r
				#else:
				#	s = '(is _thing)'
				print >>spf, s
				print >>sentf, self.sent

	def solve_compound(self, w):
		if 'compound' in self.words[w]:
			cn = self.words[w]['compound'][0] # should be only one noun
			return cn
		else:
			return None

	def solve_cop(self, w):
		if 'cop' in self.words[w]:
			if 'nsubj' in self.words[w]:
				n = self.words[w]['nsubj'][0] # should be only one noun
				return n
		return None

	def solve_amod(self, w):
		if 'amod' in self.words[w]:
			for nj in self.words[w]['amod']:
				_, tnj, _ = break_down(nj)
				if parse_nn.match(tnj):
					return nj
		return None

	def solve_nmod(self, w):
		if 'nmod:on' in self.words[w]:
			return self.words[w]['nmod:on'][0]
		if 'nmod:under' in self.words[w]:
			return self.words[w]['nmod:under'][0]
		if 'nmod:over' in self.words[w]:
			return self.words[w]['nmod:over'][0]
		if 'nmod:on_top_of' in self.words[w]:
			return self.words[w]['nmod:on_top_of'][0]
		return None

	### Not used, solves general form verbs for their object
	def solve_nsubj(self, w):
		if 'nsubj' in self.words[w]:
			if 'dobj' in self.words[w]:
				n = self.words[w]['dobj'][0]
				return n
		return None

	def solve_amod_chain(self, w, pw):
		_, tw, _ = break_down(w)
		if parse_nn.match(tw):
			return w
		n = self.solve_cop(w)
		if n:
			return n
		n = self.solve_amod(w)
		if n:
			return n
		if 'amod' in self.words[w]:			
			# works because it's a tree and there is no loop
			for j in self.words[w]['amod']:
				if j==pw:
					continue
				n = self.solve_amod_chain(j,w)
				if n:
					return n 
		return None

	def solve_and(self, w):
		wordlist = list()
		_, tw, _ = break_down(w)
		pf = parse_nn if parse_nn.match(tw) else parse_jj
		if 'conj:and' in self.words[w]:
			for nj in self.words[w]['conj:and']:
				_, tnj, _ = break_down(nj)
				if pf.match(tnj) and not 'amod' in self.words[nj]:
					wordlist.append(nj)
		return wordlist

	def solve_rels(self, w, j):
		_, tw, _ = break_down(w)
		_, tj, _ = break_down(j)
		nlist = list()
		jlist = list()
		if parse_nn.match(tw):
			nlist = self.solve_and(w)
		if parse_jj.match(tj):
			jlist = self.solve_and(j)
		nlist.append(w)
		jlist.append(j)
		for n in nlist:
			cn = self.solve_compound(n)
			nm = self.solve_nmod(n)
			n = 'bird' if n=='this' or n=='its' else n
			cn = 'bird' if cn=='this' or cn=='its' else cn
			nm = 'bird' if nm=='this' or nm=='its' else nm
			for j in jlist:
				if cn:
					rel = (break_down(n)[0], break_down(cn)[0], break_down(j)[0])
				elif nm:
					rel = (break_down(n)[0], break_down(nm)[0], break_down(j)[0])
				else:
					rel = (break_down(n)[0], break_down(j)[0])
				self.rels.append(rel)

	def extract(self):
		rel_form2 = '(is (and %s %s))'
		rel_form3 = '(is (and %s %s %s))'
		
		### Solve copula
		if 'cop' in self.grams:
			for j,v in self.grams['cop']:
				n = self.solve_cop(j)
				if n:
					self.solve_rels(n, j)
		
		### Solve amod chain
		if 'amod' in self.grams:
			for nj,j in self.grams['amod']:
				n = self.solve_amod_chain(nj,j)
				if n:
					self.solve_rels(n, j)
				else:
					self.solve_rels(nj, j)

		### Generate parses
		#for r in self.rels:
		#	if len(r) == 2:
		#		print(rel_form2 % r)
		#	elif len(r) == 3:
		#		print(rel_form3 % r)
		#	else:
		#		print('(is _thing)')

def break_down(text):
	if text == 'ROOT-0':
		return ('ROOT', 'ROOT', 0)
	#print(text)
	w, tn = text.split('/')
	return [w]+tn.split('-')

if __name__ == '__main__':
	main()
	