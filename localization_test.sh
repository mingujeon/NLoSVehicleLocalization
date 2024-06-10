# Site1 left case
FILES=$(find cls_features/SA/Site1_L/Site1_L_10_Trial9 -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site1_Left.py --input=$i 
done

# Site1 right case
FILES=$(find cls_features/SA/Site1_R/Site1_R_10_Trial6 -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site1_Right.py --input=$i
done

# Site2 left case 
FILES=$(find cls_features/SB/Site2_L/Site2_L_10_Trial8 -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site2.py --input=$i
done

# Site2 right case
FILES=$(find cls_features/SB/Site2_R/Site2_R_10_Trial6 -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site2.py --input=$i
done


