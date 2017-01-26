
minp=0.0001
maxp=10.0
ofac=.1
npeaks=1


for fname in "$@";
do
	vartools -i $fname -inputlcformat t:1,mag:2,err:3 skipnum 1 -LS $minp $maxp $ofac $npeaks 1 `dirname $fname` noGLS
done
