#!/usr/bin/env bash
#
# Runs the English PCFG parser on one or more files, printing trees only

if [ ! $# -ge 1 ]; then
  echo Usage: `basename $0` 'file(s)'
  echo
  exit
fi

scriptdir=`dirname $0`

#java -mx150m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
# -outputFormat "penn,typedDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $*

java -mx600m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
 -outputFormat "words,typedDependencies" -outputFormatOptions "stem,collapsedDependencies,includeTags" \
 -sentences newline \
 edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $*
