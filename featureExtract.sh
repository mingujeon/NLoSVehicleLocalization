FILES=$(find cls_features/SA/Dynamic3/Dynamic3_Site1_L_10_Trial9_1 -name "*.wav")
for i in $FILES ; do
  echo $i
  python featureExtract.py --input=$i
done