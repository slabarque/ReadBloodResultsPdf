rem py readbloodresultspdf.py -h
rem -p Leukocyten -p Lymfocyten -p "Lymfocyten abs" -p Normoblasten -p Trombocyten ^

py readbloodresultspdf.py ^
-p Creatinine -p CRP -p "Vancomycine cont. infuus" -p "Gamma GT" -p "AST (OT)" -p "ALT (PT)" ^
-e 2025-05-16,g,--,Ongeval ^
-e "2025-06-23,b,--,Debridement en spiertransfer" ^
-e "2025-08-07,r,--,Opstart Vanco" ^
-e "2025-09-19,y,--,?"
