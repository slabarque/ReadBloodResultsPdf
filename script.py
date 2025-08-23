#############################################################################################################################
# Auteur: Stif Labarque                                                                                                     #
# Datum: 22/08/2025                                                                                                         #
# Getest met Python 3.13.1                                                                                                  #
#                                                                                                                           #
# Dit python script leest pdf bestanden (in het formaat van Jan Palfijn ziekenhuis) met resultaten van bloedonderzoeken in. #
# De output is een data.csv bestand en een grafiek die enkele parameters plot ifv de tijd.                                  #
# Plaats de bestanden in inputDir                                                                                           #
# Plaats de gewenste parameters in parameters array                                                                         #
# In de events array kunnen gebeurtenissen worden toegevoegd die in de grafiek worden weergegeven                           #
# Meer info over kleur en lijntypes: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html                  #
# Het csv bestand komt in outputDir terecht                                                                                 #
# De grafiek met parameters word full screen geopend                                                                        #
#                                                                                                                           #
# De data die uit de pdf bestanden komt is niet altijd even consequent. Als daar problemen mee zijn 
# (bvb: "Unknown row pattern") dan kan een aanpassing in clean_row nodig zijn.                                                                                                                         #
#############################################################################################################################

# Deze waarden kunnen worden aangepast
inputDir = "input"
outputDir = "output"
parameters = ["Creatinine", "CRP", "Vancomycine cont. infuus", "Gamma GT", "AST (OT)", "ALT (PT)"]
#             datum     ,kleur  ,lijntype  ,label
events = [('2025-05-16' , 'g'   , '--'     ,'Ongeval'),
          ('2025-06-23' , 'b'   , '--'     ,'Debridement en spiertransfer'),
          ('2025-08-07' , 'r'   , '--'     ,'Opstart Vanco'),
          ]


# Script. Here be dragons. Do not touch, unless you know what you're doing ;-)
import camelot
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import os
from datetime import datetime
import re
import numpy as np
import functools as f

def get_date(fileName):
    fileDateRegex = r"\d{8}"
    dateStr = re.findall(fileDateRegex, fileName)[0]
    return dateStr

def build_file_path(inputDir):
    def build(fileName):
        return (os.path.join(inputDir, fileName),get_date(fileName))
    return build

def get_table_rows(pdfFile):
    tables = camelot.read_pdf(pdfFile, flavor='stream', table_areas=['29,44,563,591'])
    return [row for row in tables[0].data[1:]]#skip the title column

def is_relevant_row(row):
    return len(list(filter(lambda val:len(val)>0, row)))>=4 or len(row[len(row)-1]) > 0

def get_relevant_rows(pdfFile):
    return [x for x in filter(is_relevant_row, get_table_rows(pdfFile))]

def get_rows_with_date(fileInfo):
    (pdfFile, date) = fileInfo
    def prepend(x):
        def p(l):
            return [x] + l
        return p
    return list(map(prepend(date), get_relevant_rows(pdfFile)))

def clean_row(row, shouldLog=False):
    def log(msg):
        if(shouldLog):
            print(msg)
    log(row)
    match row:
        case [date,name,outsideminmax,value,minmax,unit] if (outsideminmax == "" or outsideminmax =="*"):
            result = [date,name,value,minmax,unit]
            log("*****[date,name,outsideminmax,value,minmax,unit]*****")
            return result
        case [date,name,outsideminmax,value,empty,minmax,unit] if (outsideminmax == "" or outsideminmax =="*") and empty == "":
            result = [date,name,value,minmax,unit]
            log("+++++[date,name,outsideminmax,value,empty,minmax,unit]+++++")
        case [date,name,value,empty, empty2,unit] if empty == "" and empty2 == "":
            result = [date,name,value,empty,unit]
            log(":::::[date,name,value,empty, empty2,unit]:::::")
        case [date,name,value,minmax,unit]:
            result = [date,name,value,minmax,unit]
            log("<<<<<[date,name,value,minmax,unit]>>>>>")
        case _:
            raise Exception("Unknown row pattern")
    log(result)
    return result

def clean_data(row):
    row[0] = datetime.strftime(datetime.strptime(row[0], "%Y%m%d"),"%Y-%m-%d")
    row[2] = row[2].strip('>').strip('<')
    min= 0
    max= 0
    if(row[3].startswith('<')):
        min = 0
        max = row[3].strip('<')
    else:
        min,sep,max = row[3].partition('-')
    row.insert(3, min)
    row[4] = max
    return row

def pad_row(row):
    return list(map(lambda t: f"{t[1]:<15}" if t[0]!=1 else f"{t[1]:<30}", enumerate(row)))

def get_data(directory, shouldClean=True, shouldLog=False):
    mapToFileInfos = f.partial(map, build_file_path(directory))
    mapToTables = f.partial(map, get_rows_with_date)
    clean_row_logged = f.partial(clean_row, shouldLog=shouldLog)
    cleanRows = f.partial(map, clean_row_logged)
    cleanData = f.partial(map, clean_data)
    flattenRows = lambda tables : [row for rows in tables for row in rows]
    allRows = flattenRows(mapToTables(mapToFileInfos(os.listdir(directory))))
    if(shouldClean):
        allRows = cleanData(cleanRows(allRows))
    return allRows

def write_to_csv(data, directory, filename, shouldPad):
    padRows = f.partial(map, pad_row)
    rows = padRows(data) if shouldPad else data
    outputFile = open(os.path.join(directory,filename), "w", encoding="utf-16")
    outputFile.writelines(list(map(lambda row : f"{row},\n".replace("[","").replace("]",""), rows)))
    outputFile.close()

# def write_dataframe_to_csv(data:pd.DataFrame, directory, filename):
#     outputFile = open(os.path.join(directory,filename), "w", encoding="utf-16")
#     data.to_csv(outputFile, encoding="utf-16")
#     outputFile.close()

def write_dataframe_to_csv(data:pd.DataFrame, directory, filename):
    df.to_csv(os.path.join(directory, filename), encoding="utf-16")

def get_dataframe(data):
    df = pd.DataFrame(data, columns=["date","name","value","min","max","unit"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    df["min"] = pd.to_numeric(df["min"])
    df["max"] = pd.to_numeric(df["max"])
    df.set_index("date")
    names = df["name"].unique()
    return df

def plot(events, *labels):
    colors = pl.cm.jet(np.linspace(0,1,len(labels)))
    fig, ax1 = plt.subplots(figsize=(13,8))
    handles = list()
    rightAxes = list[plt.Axes]()
    for key, color in zip(labels, range(len(labels))):
        subset = df.loc[df['name']==key]
        data_x = subset['date']
        data_y = subset['value']
        data_min = subset['min']
        data_max = subset['max']
        unit = subset['unit'].values[0]
        handles.append(ax1.plot(data_x, data_y, color=colors[color], label=key))
        ax1.plot(data_x, data_min, ':', color=colors[color])
        ax1.plot(data_x, data_max, '-.', color=colors[color])
        ax1.set_ylabel(unit)
        if(color > 1):
            ax1.spines['right'].set_position(('outward', (color-1) * 40))
            ax1.spines['right'].set_color(colors[color])
            rightAxes.append(ax1)
        elif (color == 1):
            ax1.spines['right'].set_color(colors[color])
            rightAxes.append(ax1)
        elif (color == 0):
            ax1.spines['left'].set_color(colors[color])
        if(color<len(labels)-1):
            ax1 = ax1.twinx()

    for ax in rightAxes:
        y = ax.get_yaxis()
        y.tick_right()

    handles.append(list(map(lambda e: plt.axvline(x=pd.to_datetime(e[0]), color=e[1], linestyle=e[2], label=e[3]), events)))
    ax1.legend(handles=f.reduce(lambda x1, x2:x1 + x2, handles), loc="upper center")

    plt.subplots_adjust(left=0.04, bottom=0.25, right=0.65, top=0.98)
    figManager = plt.get_current_fig_manager()
    figManager.set_window_title('Bloedwaarden')
    figManager.full_screen_toggle()
    plt.show()


# rawdata = get_data(inputDir, False)
# write_to_csv(data, outputDir, "rawData.csv", True)
data = get_data(inputDir)
df = get_dataframe(data)
write_dataframe_to_csv(df, outputDir, "data.csv")
plot(events, *parameters)
