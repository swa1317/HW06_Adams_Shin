
import sys
import pandas as pd


def csv_to_array_trained(csv_filename):
	csv_pandas = pd.read_csv(csv_filename)
	attributes_with_class = csv_pandas[['Age', 'Ht', 'TailLn', 'HairLn', 'BangLn', 'Reach', 'EarLobes']].copy()

	data = attributes_with_class.to_numpy()
	return data


def predict(data):
	classifications_filename = "HW05_Adams_MyClassifications.csv"
	file_object = open(classifications_filename, "wt")  # create file

	for row in data:
		if( row[2] < 9.8814697265625):
			class_val="-1"
		else:
			if( row[3] < 9.816938920454549):
				class_val="-1"
			else:
				class_val="+1"
		print(class_val)
		file_object.write(class_val + '\n')
	file_object.close()
  
if __name__ == '__main__':
	parameter = sys.argv[1:]
	if len(parameter) == 0:
		print("the parameter is empty")
	else:
		parameter = parameter[0]
		data = csv_to_array_trained(parameter)
		predict(data)

