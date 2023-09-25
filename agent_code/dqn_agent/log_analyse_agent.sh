#! /bin/bash
if (($# != 1))
    then echo "Usage: $0 logfile"
    exit 1
fi

echo "Analyse Logfile:" $1

# -----------------------------------
#  Analyse: Encountered game event(s)
# -----------------------------------
#2023-09-23 16:57:24,497 [best_ruehl_code] DEBUG: Encountered game event(s) 'INVALID_ACTION' in step 5

grep -Eo '(?<='Encountered\ game\ event...')[^in]*' $1 | sed 's/,/\n/g' | sed "s/'//g" | awk ' 
     {if (NR == 1){
        print "\nEVENT\t COUNT" # > "fileresult.txt";
     }
     count[$1]++} END {for (word in count) print count[word], "\t", word
     }'


# ----------------------------------
#  Analyse: Action via Model
# ----------------------------------
#2023-09-23 16:57:24,497 [best_ruehl_code] DEBUG: Encountered game event(s) 'INVALID_ACTION' in step 5
grep -Eo '.*Action via Model:.*' $1 | uniq -c | awk ' 
     {if (NR == 1){
        print "\nAKTION\t COUNT"  # > "fileresult.txt";
     }
     count[$9]++} END {for (word in count) print word, "\t", count[word]
     }'