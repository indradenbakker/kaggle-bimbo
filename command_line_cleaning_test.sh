grep -vwE "(Semana)" test_extended.csv > test_extended_f.csv 

echo id,Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Demanda_uni_equil,NombreCliente,NombreProducto,shortName,brand,weight,pieces,Town,State,wtpcs,lag_3,lag_2,lag_1,client_town,product_town,lag_sum,sum_demand > test_extended_f_2.csv

cat test_extended_f.csv >> test_extended_f_2.csv 

awk -F'"' -v OFS='' '{ for (i=2; i<=NF; i+=2) gsub(",", "", $i) } 1' test_extended_f_2.csv > test_extended_f_2_c.csv
