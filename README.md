# bigann

This mini crate benches the crate [hnsw-rs](https://crates.io/crates/hnsw_rs) on u8 vectors sampled data from the BIGANN benchmark. See [BIGANN](https://big-ann-benchmarks.com/)

Files learn.100M.u8bin, query.public.10K.u8bin and public_query_gt100.bin must first be downloaded (it amounts to 12Gb)
and installed in some directory.

If you run only on the first 10M slices of data you must download the corresponding ground truth 
as documented on the BIGANN web page and extract the file GT_10M/bigann-10M to replace the file public_query_gt100.bin which corresponds to the ground truth on the 
## Results

