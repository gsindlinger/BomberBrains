#! /bin/bash
# Works on Linux Fedora37 and NOT on OSX !? (grep)
if (($# != 1))
   then echo "Usage: $0 logfile"
   exit 1
fi

echo "Analyse Logfile:" $1

WORKFILE=$1".work"
echo "Analyse workfile:" $WORKFILE
STATISTICS_FILE=$1".statistics"
echo "Analyse statistics file:" $STATISTICS_FILE

# Correction Log (end_of_round)
#      DEBUG: end_of_round Encountered event(s) 'INVALID_ACTION'
# --> #DEBUG: Encountered game event(s) 'INVALID_ACTION'
cat $1 | sed "s/DEBUG: end_of_round Encountered event/DEBUG: Encountered game event/g" > $WORKFILE

# -----------------------------------
#  Analyse: Encountered game event(s)
# -----------------------------------
grep -Po '(?<='Encountered\ game\ event...')[^in]*' $WORKFILE | sed 's/,/\n/g' | sed "s/'//g" | awk -v file=$STATISTICS_FILE ' 
    {if (NR == 1){
       print "\nEVENT\t COUNT" > file;
    }
    count[$1]++} END {for (word in count) print count[word], "\t", word  >> file;
    }'


# ----------------------------------
#  Analyse: Action via Model
# ----------------------------------
grep -Eo '.*Action via Model:.*' $WORKFILE | uniq -c | awk -v file=$STATISTICS_FILE ' 
    {if (NR == 1){
       print "\nAKTION\t COUNT" word >> file;
    }
    count[$9]++} END {for (word in count) print word, "\t", count[word] >> file;
    }'