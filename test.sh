
set -x

OVER=5
HIFAC=3
NPEAKS=5
#THRESH=1E-2
PEAK_THRESH=1
NTHREADS=10
MEM=512
NBOOTSTRAPS=0
FLAGS='--save-maxp --floating-mean' #-pow2' # --floating-mean'

VT_MINP=0.01
VT_MAXP=1000.0
VT_OFAC=0.1
VT_NPEAKS=1

args="--over=${OVER} --hifac=${HIFAC} --npeaks=${NPEAKS} --nbootstraps=${NBOOTSTRAPS} --nthreads=${NTHREADS} --memory-per-thread=${MEM} ${FLAGS}"

time ./cunfftlsf --list-in=list.dat --list-out=out.dat ${args}

set +x

echo "Running nfftls"
cd ../nfftls-1.1
for fname in `cat ../cunfftls/list.dat | awk 'NR > 1 { print $1 }'`;
do 
	FNAME=../cunfftls/${fname}
	cat $FNAME | awk 'NR > 1 { print $1, $2 }' > ${FNAME}.2col
	./nfftls --data-file ${FNAME}.2col --oversampling ${OVER} --hifreq-factor ${HIFAC} -o ${FNAME}.nfftls

	vartools -i $FNAME -inputlcformat t:1,mag:2,err:3 skipnum 1 -LS $VT_MINP $VT_MAXP $VT_OFAC $VT_NPEAKS 1 `dirname $FNAME`
	python ../cunfftls/plot_lsp.py ${FNAME}.lsp ${FNAME}.nfftls ${FNAME}.ls
done


