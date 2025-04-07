import os
import json

is_export=lambda l : "#export"==l.lower().replace(" ","").replace("\n","")

def nb_to_lib(nb,lib):
    nb=json.load(open(nb))

    code_cells=tuple(c['source'] for c  in nb['cells'] if c['cell_type']=='code')

    cells_to_export=tuple("".join(c) for c in code_cells if c and is_export(c[0]))

    f=open(lib,'w')
    f.writelines(cells_to_export)
    f.write("\n")
    f.close()


def nb_to_test(nb, test):
    nb=json.load(open(nb))

    code_cells=tuple(c['source'] for c  in nb['cells'] if c['cell_type']=='code')

    cells_to_export=tuple("".join(map(lambda x:'\t'+x,c)) for c in code_cells if c and is_export(c[0]))

    f=open(test,'w')
    tf=open('.testTemplate').readlines()
    begin=tf[:-3]
    end=tf[-3:]
    f.writelines(begin)
    f.writelines(cells_to_export)
    f.write("\n")
    f.writelines(end)
    f.close()

nblib="./matrix_operations.ipynb"
lib="./humble.py"
nbtest="./tests.ipynb"
test="./test/tests.py"


print("--- converting lib notebook into lib ---")
nb_to_lib(nblib, lib)
print("--- done ---")
print("--- converting test notebook into test file ---")
nb_to_test(nbtest, test)
print("--- done ---")

## Start the tests
import nose2
nose2.discover()