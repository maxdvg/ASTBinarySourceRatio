export PATH=$PATH:$HOME/csmith/bin

for i in $(seq 1 100);
do
	csmith --stop-by-stmt 35 > random$i.c
	gcc random$i.c -I$HOME/csmith/include -o random$i
	./random$i
done

