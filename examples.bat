rem py readbloodresultspdf.py -h
rem -p Leukocyten -p "Neutrofiele granulocyten" -p "Lymfocyten abs" -p Normoblasten -p Trombocyten ^

rem Opvolging Vancomycine en nier- en leverfuncties
@REM py readbloodresultspdf.py ^
@REM -p Creatinine -p CRP -p "Vancomycine cont. infuus" -p "Gamma GT" -p "AST (OT)" -p "ALT (PT)" ^
@REM -e 2025-05-16,g,--,Ongeval ^
@REM -e "2025-06-23,b,--,Debridement en spiertransfer" ^
@REM -e "2025-08-07,r,--,Opstart Vanco" ^
@REM -e "2025-09-19,y,--,?" ^
@REM -e "2026-01-15,g,--,Infectie"


rem Infectie indicatoren
py readbloodresultspdf.py ^
-p CRP -p Leukocyten -p "Neutrofiele granulocyten" ^
-e 2025-05-16,g,--,Ongeval ^
-e "2025-06-23,b,--,Debridement en spiertransfer" ^
-e "2025-08-07,r,--,Opstart Vanco" ^
-e "2025-09-19,y,--,?" ^
-e "2026-01-15,g,--,Infectie"
