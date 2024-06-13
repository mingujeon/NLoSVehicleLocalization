# Site1 left case
FILES=$(find ARILDataset/SA/Site1_L/ -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site1_Left.py --input=$i 
done

# Site1 right case
FILES=$(find ARILDataset/SA/Site1_R/ -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site1_Right.py --input=$i
done

# Site2 left case 
FILES=$(find ARILDataset/SB/Site2_L/ -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site2.py --input=$i
done

# Site2 right case
FILES=$(find ARILDataset/SB/Site2_R/ -name "*.csv")
for i in $FILES ; do
  echo $i
  python tracking_test_Site2.py --input=$i
done


