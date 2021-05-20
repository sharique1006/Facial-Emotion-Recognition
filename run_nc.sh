question=$1
train_data=$2
test_data=$3
output_file=$4

if [ ${question} == "1" ] ; then
	python3 Q1/Q1a.py $train_data $test_data $output_file
elif [ ${question} == "2" ] ; then
	python3 Q1/Q1b.py $train_data $test_data $output_file
elif [ ${question} == "3" ] ; then
	python3 Q1/Q1c.py $train_data $test_data $output_file
fi