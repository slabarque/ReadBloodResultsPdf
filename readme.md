This Python script reads PDF files (in the Jan Palfijn Hospital format) containing blood test results.
The output is a data.csv file and a graph that plots several parameters over time.
- Place the files in the input directory. Example file name: *LastName-FirstName-20250822-AZ_Jan_Palfijn_Gent-0_A_68813785_5.pdf*
- Place the desired parameters in the parameters array.
- Events can be added to the events array to be displayed in the graph.
More information about color and line types: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
- The CSV file is placed in the output directory.
- The graph with parameters opens full screen.

![Bloedwaarden](img/Bloedwaarden.png)

*The data extracted from the PDF files is not always consistent. If there are any problems with this (e.g. "Unknown row pattern") then an adjustment in clean_row may be necessary. The --verbose flag might help identifying the row that caused problems.*