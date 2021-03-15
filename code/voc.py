import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def label_voc(voc_path, label_path):

    xl = pd.ExcelFile(voc_path)
    egg_data = pd.read_csv(label_path)
    sheet_names = xl.sheet_names

    for sheet in sheet_names:
        print(sheet)

        day = xl.parse(sheet)
        day.insert(1, 'label',-1)

        for name in day['Data range']:
            id_split = name.split('_')

            if len(id_split) > 2:
                if (id_split[1][:2] == 'ID') and (len(id_split[1]) == 6):
                    index = day.loc[day['Data range'] == name].index[0]
                    id = int(id_split[1][2:])
                    egg = egg_data.loc[egg_data['ID'] == id]

                    if len(egg) == 0:
                        day['label'][index] = -1
                    elif egg['label'].values[0] == 'M':
                        day['label'][index] = 0
                    elif egg['label'].values[0] == 'F':
                        day['label'][index] = 1
                    else:
                        day['label'][index] = -1

        save_path = '../data/voc/labeled/round5/' + sheet + '.csv'
        day.to_csv(save_path)
def main():
    voc_path = '../data/voc/r5-uniform.xlsx'
    label_path = '../data/measurements/round5_complete.csv'
    label_voc(voc_path, label_path)

if __name__ == '__main__':
    main()