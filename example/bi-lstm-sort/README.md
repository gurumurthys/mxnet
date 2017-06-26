This is an example of using bidirection lstm to sort an array.

Firstly, generate data by:

    cd data
    python ../gen_data.py

Then, train the model by:

    python lstm_sort.py

At last, test model by:

    python infer_sort.py 234 189 785 763 225

and will output sorted seq

    189
    225
    234
    763
    785
