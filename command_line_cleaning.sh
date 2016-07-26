# Need to clean the lines from text, because composing on Google Cloud returned the header on some lines
grep -vwE "(Semana)" train_extended.csv > train_extended_f.csv

# Sort the rows on week
sort --field-separator=',' --key=1 train_extended_f.csv > train_extended_f_s.csv

# Add the column names to the first line
#sed -i `1iSemana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Venta_hoy,Dev_uni_proxima,Dev_proxima,Demanda_uni_equil,NombreCliente,NombreProducto,shortName,brand,weight,pieces,Town,State,wtpcs,lag_3,lag_2,lag_1` train_extended_f.csv 
echo 'Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Venta_hoy,Dev_uni_proxima,Dev_proxima,Demanda_uni_equil,NombreCliente,NombreProducto,shortName,brand,weight,pieces,Town,State,wtpcs,lag_3,lag_2,lag_1' > train_extended_f_2.csv 
cat train_extended_f.csv >> train_extended_f_2.csv

# Remove the quoted commas
awk -F'"' -v OFS='' '{ for (i=2; i<=NF; i+=2) gsub(",", "", $i) } 1' train_extended_f_2.csv > train_extended_f_2_c.csv

# Only use subset of columns
cut -d, -f1-11,20-21 train_extended_f_2_c.csv > train_extended_f_2_c_f.csv
