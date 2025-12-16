import pandas as pd
import pyarrow.parquet as pq

# chunking large files (csv)
chunks = pd.read_csv('files/example_csv.csv', chunksize=10000)  
df_list = []
for chunk in chunks:
    chunk_final = (
        chunk
        # long dot-chain processing pipe goes here
    )
    df_list.append(chunk_final)
df = pd.concat(df_list, ignore_index=True)

# chunking large files (parquet)
parquet_file = pq.ParquetFile('files/example.parquet')
df_list = []
for batch in parquet_file.iter_batches(batch_size=10000):
    chunk = batch.to_pandas()
    chunk_final = (
        chunk
        # long dot-chain processing pipe goes here
    )
    df_list.append(chunk_final)
df = pd.concat(df_list, ignore_index=True)  