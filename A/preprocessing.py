import pandas as pd
import os
import re
import numpy as np
#config
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class preprocessor:
    #constructor
    def __init__(self,filename:str):
        self._filename:str = filename;
        self.dataset_raw:str = "";

    #internal methods
    def normalize_string(self,regulars,normal_val,input_val,regulars_2,normal_val_2):
        # print("input val: " + str(input_val))
        # print("regulars: " + str(regulars))
        if(input_val in regulars):
            # print("yes")
            output = re.sub(regulars, normal_val, input_val)
            #print(output)
            return output
        elif(input_val in regulars_2):
            #print("yes_2")
            output = re.sub(regulars_2,normal_val_2,input_val)
            #print(output)
            return output


    def import_file(self):
        print("\nImporting the input file...")

        cwd = os.getcwd() + '/dataset/';
        self.dataset_raw = pd.read_csv(cwd + self._filename);
        self.dataset_raw.columns = [c.replace(' ', '_') for c in self.dataset_raw.columns]


    def normalize(self):
        #find and replace skin color 
        arr = []
        for col in self.dataset_raw.columns:
            arr.append(col)
        print("\nData labels are: " + str(arr))
        print("\nNormalizing data...")

        self.dataset_raw['Gender'] = self.dataset_raw['Gender'].map(lambda x:self.normalize_string('male?|Male?|MALE?',"1",x,"female?|Female?|FEMALE?","0"))
        self.dataset_raw['Skin_Color_'] = self.dataset_raw['Skin_Color_'].map(lambda x:self.normalize_string('brown?|Brown?|BROWN?',"1",x,"white?|White?|WHITE?","0"))
        self.dataset_raw['Eye_Color'] = self.dataset_raw['Eye_Color'].map(lambda x:self.normalize_string('brown?|Brown?|BROWN?',"1",x,"black?|Black?|BLACK?","0"))
        self.dataset_raw['Hair_Color'] = self.dataset_raw['Hair_Color'].map(lambda x:self.normalize_string('brown?|Brown?|BROWN?',"1",x,"black?|Black?|BLACK?","0"))
        self.dataset_raw = self.dataset_raw.dropna()

        #print(self.dataset_raw.head(100));


    def split_dataframe(self):
        msk = np.random.rand(len(self.dataset_raw)) < 0.8
        self.train_data = self.dataset_raw[msk]
        self.test_data = self.dataset_raw[~msk]
        # print(len(self.train_data))
        # print(len(self.test_data))
        return self.train_data,self.test_data


    def get_train_test_set(self):
        self.import_file();
        self.normalize();
        test_data,train_data = self.split_dataframe();
        return test_data, train_data

if __name__ == "__main__":
    myobj = preprocessor("raw_data.csv");
    myobj.import_file();
    myobj.normalize();
    print("Splitting the data for testing and training...")
    myobj.split_dataframe();
