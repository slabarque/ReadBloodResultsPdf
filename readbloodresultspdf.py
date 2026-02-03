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
def parse_event(s: str) -> tuple[str, str, str, str]:
    """Parse event string into tuple of (date, color, linetype, description)."""
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

print("INPUT:")
print(inputDir)
print("OUTPUT:")
print(outputDir)
print("PARAMETERS:")
print(parameters)
print("EVENTS:")
print(events)

def log(msg: str) -> None:
    """Print message if verbose logging is enabled."""
    if shouldLog:
        print(msg)

def get_date(fileName):
    fileDateRegex = r"\d{8}"
    dateStr = re.findall(fileDateRegex, fileName)[0]
    return dateStr

def build_file_path(input_dir: str) -> callable:
    """Return function that builds file path and extracts date from filename."""
    def build(file_name: str) -> tuple[str, str]:
        return (os.path.join(input_dir, file_name), get_date(file_name))
    return build

def get_table_rows(pdf_file: str) -> list[list[str]]:
    """Extract table rows from PDF file (Jan Palfijn Hospital format)."""
    tables = camelot.read_pdf(pdf_file, flavor='stream', table_areas=['29,44,563,591'])
    return [row for row in tables[0].data[1:]]  # skip the title column

def get_table_rows_labo_nuytinck(pdf_file: str) -> list[list[str]]:
    """Extract table rows from Labo Nuytinck format PDF."""
    tables = camelot.read_pdf(pdf_file, flavor='stream', table_areas=['10,44,583,632'])
    return [row for row in tables[0].data[1:]]  # skip the title column

def is_relevant_row(row: list[str]) -> bool:
    """Check if row contains enough relevant data."""
    return len(list(filter(lambda val: len(val) > 0, row))) >= 4 or len(row[len(row) - 1]) > 0

def get_relevant_rows(pdf_file: str) -> list[list[str]]:
    """Get relevant rows from PDF, handling both standard and Labo Nuytinck formats."""
    if "Labo_Nuytinck" in pdf_file:
        return [x for x in filter(is_relevant_row, get_table_rows_labo_nuytinck(pdf_file))]
    return [x for x in filter(is_relevant_row, get_table_rows(pdf_file))]

def get_rows_with_date(file_info: tuple[str, str]) -> list[list[str]]:
    """Get rows from PDF with date prepended to each row."""
    pdf_file, date = file_info
    def prepend(x: str) -> callable:
        def p(l: list[str]) -> list[str]:
            return [x] + l
        return p
    return list(map(prepend(date), get_relevant_rows(pdf_file)))

def clean_row(row: list[str]) -> list[str]:
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

def clean_data(row: list[str]) -> list[str]:
    """Clean and parse blood test data row."""
    row[0] = datetime.strftime(datetime.strptime(row[0], "%Y%m%d"), "%Y-%m-%d")
    row[2] = row[2].strip('>').strip('<')
    min_val = 0
    max_val = 0
    if row[3].startswith('<'):
        min_val = 0
        max_val = row[3].strip('<')
    else:
        min_val, sep, max_val = row[3].partition('-')
    row.insert(3, min_val)
    row[4] = max_val
    return row

def pad_row(row: list[str]) -> list[str]:
    """Pad row for formatted output."""
    return list(map(lambda t: f"{t[1]:<15}" if t[0] != 1 else f"{t[1]:<30}", enumerate(row)))

def get_data(directory: str, should_clean: bool = True) -> list[list[str]]:
    """Load and optionally clean blood test data from all PDFs in directory."""
    map_to_file_infos = f.partial(map, build_file_path(directory))
    map_to_tables = f.partial(map, get_rows_with_date)
    clean_row_logged = f.partial(clean_row)
    clean_rows = f.partial(map, clean_row_logged)
    clean_data_map = f.partial(map, clean_data)
    flatten_rows = lambda tables: [row for rows in tables for row in rows]
    all_rows = flatten_rows(map_to_tables(map_to_file_infos(os.listdir(directory))))
    if should_clean:
        all_rows = clean_data_map(clean_rows(all_rows))
    return all_rows

def write_to_csv(data: list[list[str]], directory: str, filename: str, should_pad: bool) -> None:
    """Write data rows to CSV file."""
    pad_rows = f.partial(map, pad_row)
    rows = pad_rows(data) if should_pad else data
    output_file = open(os.path.join(directory, filename), "w", encoding="utf-16")
    output_file.writelines(list(map(lambda row: f"{row},\n".replace("[", "").replace("]", ""), rows)))
    output_file.close()

def write_dataframe_to_csv(data: pd.DataFrame, directory: str, filename: str) -> None:
    """Write pandas DataFrame to CSV file."""
    data.to_csv(os.path.join(directory, filename), encoding="utf-16")

def get_dataframe(data: list[list[str]]) -> pd.DataFrame:
    """Convert cleaned blood test data to pandas DataFrame with proper types."""
    df = pd.DataFrame(data, columns=["date", "name", "value", "min", "max", "unit"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    df["min"] = pd.to_numeric(df["min"])
    df["max"] = pd.to_numeric(df["max"])
    df.set_index("date")
    return df

def plot(data_frame: pd.DataFrame, events: list[tuple[str, str, str, str]], *labels: str) -> None:
    """Plot blood test parameters over time with medical events marked."""
    def get_colors(count: int) -> np.ndarray:
        """Generate color map for multiple parameters."""
        return pl.cm.jet(np.linspace(0, 1, count))
    
    def scaly_y_axis(ax: plt.Axes, scale: float) -> None:
        """Scale y-axis to prevent overlapping of multiple axes."""
        ymin, ymax = ax.get_ylim()
        center = (ymax + ymin) / 2
        half_range = (ymax - ymin) * scale / 2
        ax.set_ylim(center - half_range, center + half_range)

    colors = get_colors(len(labels))
    fig, ax1 = plt.subplots(figsize=(13, 8))
    handles: list[list[plt.Line2D]] = []
    right_axes: list[plt.Axes] = []
    
    for index, key in enumerate(labels):
        subset = data_frame.loc[data_frame['name'] == key]
        data_x = subset['date']
        data_y = subset['value']
        data_min = subset['min']
        data_max = subset['max']
        min_val = data_min.values[0]
        max_val = data_max.values[0]
        label = f"{key}{f' ({min_val}-{max_val})' if not isnan(min_val) and not isnan(max_val) else ''}"
        unit = subset['unit'].values[0]
        handles.append(ax1.plot(data_x, data_y, color=colors[index], label=label))
        ax1.plot(data_x, data_min, ':', color=colors[index])
        ax1.plot(data_x, data_max, '-.', color=colors[index])
        scaly_y_axis(ax1, 1 + 0.05 * index)  # prevent overlapping of different axes
        ax1.set_ylabel(unit)
        if index > 1:
            ax1.spines['right'].set_position(('outward', (index - 1) * 40))
            ax1.spines['right'].set_color(colors[index])
            right_axes.append(ax1)
        elif index == 1:
            ax1.spines['right'].set_color(colors[index])
            right_axes.append(ax1)
        elif index == 0:
            ax1.spines['left'].set_color(colors[index])
        if index < len(labels) - 1:
            ax1 = ax1.twinx()

    for ax in right_axes:
        y = ax.get_yaxis()
        y.tick_right()

    handles.append(list(map(lambda e: plt.axvline(x=pd.to_datetime(e[0]), color=e[1], linestyle=e[2], label=e[3]), events)))
    ax1.legend(handles=f.reduce(lambda x1, x2: x1 + x2, handles), loc="upper center")

    plt.subplots_adjust(left=0.04, bottom=0.52, right=0.41, top=0.98)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title('Bloedwaarden')
    fig_manager.full_screen_toggle()
    plt.show()


def main() -> None:
    """Main entry point."""
    # rawdata = get_data(inputDir, False)
    # write_to_csv(rawdata, outputDir, "rawData.csv", True)
    data = get_data(inputDir)
    dfr = get_dataframe(data)
    write_dataframe_to_csv(dfr, outputDir, "data.csv")
    plot(dfr, events, *parameters)


if __name__ == "__main__":
    main()
