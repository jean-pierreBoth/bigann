# bigann

This mini crate benches the crate [hnsw-rs](https://crates.io/crates/hnsw_rs) on u8 vectors sampled data from the BIGANN benchmark. See [BIGANN](https://big-ann-benchmarks.com/)

Files bigann_base.bvecs, bigann_query.bvecs must be dowloaded and installed in some directory (This amount to 133Gb).
Then depending on the size of the data you want to run on (10M, 100M, 1B) you download the corresponding groun truth
as explained in the BIGANN web page.

To run on the first 10M slices of data you must download the corresponding ground truth 
as documented on the BIGANN web page and extract the file GT_10M/bigann-10M to replace the file public_query_gt100.bin which corresponds to the ground truth on the totality.


## Results

