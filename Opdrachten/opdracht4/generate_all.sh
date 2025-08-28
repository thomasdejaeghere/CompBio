

for SIZE in 100 300 400; do
for AMOUNT in 10 100; do
python3 ./generator.py ${SIZE} ${AMOUNT} > data_size${SIZE}_amount${AMOUNT}.py
done
done
