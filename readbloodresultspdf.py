#############################################################################################################################
# Auteur: Stif Labarque
# Date: 15/09/2025
# Tested with Python 3.13.1
#
# This Python script reads PDF files (in the Jan Palfijn Hospital format) containing blood test results.
# The output is a data.csv file and a graph that plots several parameters over time.
# - Place the files in the input directory. Example file name: *LastName-FirstName-20250822-AZ_Jan_Palfijn_Gent-0_A_68813785_5.pdf*
# - Place the desired parameters in the parameters array.
# - Events can be added to the events array to be displayed in the graph.
# More information about color and line types: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# - The CSV file is placed in the output directory.
# - The graph with parameters opens full screen.

# *The data extracted from the PDF files is not always consistent. If there are any problems with this (e.g. "Unknown row pattern") then an adjustment in clean_row may be necessary. The --verbose flag might help identifying the row that caused problems.*'"""

#############################################################################################################################

# Deze waarden kunnen worden aangepast
# inputDir = "input"
# outputDir = "output"
# parameters = [#"Creatinine", "CRP", "Vancomycine cont. infuus", "Gamma GT", "AST (OT)", "ALT (PT)", 
#     "Lymfocyten", "Leukocyten", "Normoblasten"]
# #             datum     ,kleur  ,lijntype  ,label
# events = [('2025-05-16' , 'g'   , '--'     ,'Ongeval'),
#           ('2025-06-23' , 'b'   , '--'     ,'Debridement en spiertransfer'),
#           ('2025-08-07' , 'r'   , '--'     ,'Opstart Vanco'),
#           ]


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
from math import isnan
import argparse

description = """This Python script reads PDF files (in the Jan Palfijn Hospital format) containing blood test results.
The output is a data.csv file and a graph that plots several parameters over time.
- Place the files in the input directory. Example file name: *LastName-FirstName-20250822-AZ_Jan_Palfijn_Gent-0_A_68813785_5.pdf*
- Place the desired parameters in the parameters array.
- Events can be added to the events array to be displayed in the graph.
More information about color and line types: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
- The CSV file is placed in the output directory.
- The graph with parameters opens full screen.

*The data extracted from the PDF files is not always consistent. If there are any problems with this (e.g. "Unknown row pattern") then an adjustment in clean_row may be necessary. The --verbose flag might help identifying the row that caused problems.*'"""


# Parse commandline arguments
def parse_event(s: str):
    parts = s.split(",", 3)  # max 4 fields
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Event must have 4 fields: date,color,linetype,description. E.g. '2025-05-16' , 'g'   , '--'     ,'Ongeval'")
    return tuple(parts)

parser = argparse.ArgumentParser(
                    prog='readbloodresultspdf',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description=description,
                    epilog='')

parser.add_argument('-i', '--input', default='input', help='name of the directory that contains the PDF files')
parser.add_argument('-o', '--output', default='output', help='name of the directory where the resulting CSV should be saved')
parser.add_argument('-p', '--parameter', dest='parameters', action='append', help='add a parameter that should be plotted', required=True)
parser.add_argument('-e', '--event', dest='events', action='append', type=parse_event, help='add a specific event that should be plotted by a vertical line (format: yyyy-mm-dd, color, linetype, description)')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output flag')

args = parser.parse_args()
inputDir = args.input
outputDir = args.output
parameters = args.parameters
events = args.events or []
shouldLog = args.verbose

def log(msg):
    if(shouldLog):
        print(msg)

print("INPUT:")
print(inputDir)
print("OUTPUT:")
print(outputDir)
print("PARAMETERS:")
print(parameters)
print("EVENTS:")
print(events)

def log(msg):
    if(shouldLog):
        print(msg)

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

def get_table_rows_labo_nuytinck(pdfFile):
    tables = camelot.read_pdf(pdfFile, flavor='stream', table_areas=['10,44,583,632'])
    return [row for row in tables[0].data[1:]]#skip the title column

def is_relevant_row(row):
    return len(list(filter(lambda val:len(val)>0, row)))>=4 or len(row[len(row)-1]) > 0

def get_relevant_rows(pdfFile):
    if "Labo_Nuytinck" in pdfFile:
        return [x for x in filter(is_relevant_row, get_table_rows_labo_nuytinck(pdfFile))]
    return [x for x in filter(is_relevant_row, get_table_rows(pdfFile))]

def get_rows_with_date(fileInfo):
    (pdfFile, date) = fileInfo
    def prepend(x):
        def p(l):
            return [x] + l
        return p
    return list(map(prepend(date), get_relevant_rows(pdfFile)))

def clean_row(row):
    log(row)
    match row:
        case [date,name,outsideminmax,value,minmax,unit] if (outsideminmax == "" or outsideminmax =="*" or outsideminmax =="L" or outsideminmax =="H"):
            result = [date,name,value,minmax,unit]
            log("*****[date,name,outsideminmax,value,minmax,unit]*****")
        case [date,name,outsideminmax,value,empty,minmax,unit] if (outsideminmax == "" or outsideminmax =="*") and empty == "":
            result = [date,name,value,minmax,unit]
            log("+++++[date,name,outsideminmax,value,empty,minmax,unit]+++++")
        case [date,name,value,empty,empty2,unit] if empty == "" and empty2 == "":
            result = [date,name,value,empty,unit]
            log(":::::[date,name,value,empty,empty2,unit]:::::")
        case [date,name,value,empty,minmax,unit] if empty == "":
            result = [date,name,value,minmax,unit]
            log("=====[date,name,value,empty,minmax,unit]=====")
        case [date,name,value,minmax,unit]:
            result = [date,name,value,minmax,unit]
            log("<<<<<[date,name,value,minmax,unit]>>>>>")
        case [date,emptyorcaret,name,outsideminmax,value,minmax,unit] if (emptyorcaret == "" or emptyorcaret == "^") and (outsideminmax == "" or outsideminmax =="*"):
            result = [date,name,value,minmax,unit]
            log("?????[date,emptyorcaret,name,outsideminmax,value]?????")
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

def get_data(directory, shouldClean=True):
    mapToFileInfos = f.partial(map, build_file_path(directory))
    mapToTables = f.partial(map, get_rows_with_date)
    clean_row_logged = f.partial(clean_row)
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

def write_dataframe_to_csv(data:pd.DataFrame, directory, filename):
    data.to_csv(os.path.join(directory, filename), encoding="utf-16")

def get_dataframe(data):
    df = pd.DataFrame(data, columns=["date","name","value","min","max","unit"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    df["min"] = pd.to_numeric(df["min"])
    df["max"] = pd.to_numeric(df["max"])
    df.set_index("date")
    names = df["name"].unique()
    return df

def plot(dataFrame, events, *labels):
    def get_colors(count):
        return pl.cm.jet(np.linspace(0,1,count))
    
    def scaly_y_axis(ax: plt.Axes, scale):
        ymin, ymax = ax.get_ylim()
        center = (ymax + ymin) / 2
        half_range = (ymax - ymin) * scale / 2
        ax.set_ylim(center - half_range, center + half_range)

    colors = get_colors(len(labels))
    fig, ax1 = plt.subplots(figsize=(13,8))
    handles = list[list[plt.Line2D]]()
    rightAxes = list[plt.Axes]()
    for index, key in enumerate(labels):
        subset = dataFrame.loc[dataFrame['name']==key]
        data_x = subset['date']
        data_y = subset['value']
        data_min = subset['min']
        data_max = subset['max']
        min = data_min.values[0]
        max = data_max.values[0]
        label = f"{key}{f" ({min}-{max})" if not isnan(min) and not isnan(max) else ""}"
        unit = subset['unit'].values[0]
        handles.append(ax1.plot(data_x, data_y, color=colors[index], label=label))
        ax1.plot(data_x, data_min, ':', color=colors[index])
        ax1.plot(data_x, data_max, '-.', color=colors[index])
        scaly_y_axis(ax1, 1 + 0.05 * index)#this makes sure the data_max and data_min for the different axes twins do not overlap
        ax1.set_ylabel(unit)
        if(index > 1):
            ax1.spines['right'].set_position(('outward', (index-1) * 40))
            ax1.spines['right'].set_color(colors[index])
            rightAxes.append(ax1)
        elif (index == 1):
            ax1.spines['right'].set_color(colors[index])
            rightAxes.append(ax1)
        elif (index == 0):
            ax1.spines['left'].set_color(colors[index])
        if(index<len(labels)-1):
            ax1 = ax1.twinx()

    for ax in rightAxes:
        y = ax.get_yaxis()
        y.tick_right()

    handles.append(list(map(lambda e: plt.axvline(x=pd.to_datetime(e[0]), color=e[1], linestyle=e[2], label=e[3]), events)))
    ax1.legend(handles=f.reduce(lambda x1, x2:x1 + x2, handles), loc="upper center")

    plt.subplots_adjust(left=0.04, bottom=0.52, right=0.41, top=0.98)
    figManager = plt.get_current_fig_manager()
    figManager.set_window_title('Bloedwaarden')
    figManager.full_screen_toggle()
    plt.show()


# rawdata = get_data(inputDir, False)
# write_to_csv(rawdata, outputDir, "rawData.csv", True)
data = get_data(inputDir)
dfr = get_dataframe(data)
write_dataframe_to_csv(dfr, outputDir, "data.csv")
plot(dfr, events, *parameters)
