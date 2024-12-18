# bigann

This mini crate benches the crate [hnsw-rs](https://crates.io/crates/hnsw_rs) on sampled u8 vectors from the BIGANN benchmark. See [BIGANN](https://big-ann-benchmarks.com/neurips21.html) and [IRISA](http://corpus-texmex.irisa.fr/)

Files bigann_base.bvecs, bigann_query.bvecs must be dowloaded and installed in some directory (This amounts to 133Gb).
Then depending on the size of the data you want to run on the first (10M, 100M, 1B) of the large file *bigann_base.bvecs* you download the corresponding ground truth as explained in the BIGANN web page.

To run on the first 10M slices of data you download the corresponding ground truth corresponding to this size and extract the file dis_10M.fvecs and idx_10M.ivecs from the gnd directory to put it in the same directory as bigann_base.bvecs and bigann_query.bvecs check in source.

## commandline

- bigann --dir DataDir --nbdata 10 (or 100 or 1000) to specify the number million data you want to run.

Use --dump to dump the hnsw structure for search variations
 - bigann --dir DataDir --dump --nbdata 10 (or 100 or 1000) and
 - bigann --dir DataDir --hnsw "dumpbigann" --nbdata 10 

For more see documentation (cargo doc --no-deps as usual) 
## Results for the first 10 Million data points.

### Results with standard level sampling

Results on Intel E5-2630 v3 @2.4GHz
16 cores 2 thread / core

All parameters are explained in doc of  [hnsw-rs](https://crates.io/crates/hnsw_rs).


| knbn  | max_nb_conn | ef_cons | ef_search | extend | keep pruned | recall | req/s | last ratio |
| :---: | :---------: | :-----: | :-------: | :----: | :---------: | :----: | :---: | :--------: |
|  10   |     64      |   100   |    128    |   no   |     no      | 0.995  | 2610  |   1.0002   |
|  100  |     64      |   100   |    128    |   no   |     no      | 0.983  | 1350  |   1.0006   |
|  10   |     24      |   100   |    128    |   no   |     no      | 0.970  | 4845  |   1.001    |
|  100  |     24      |   100   |    128    |   no   |     no      | 0.923  | 2411  |   1.003    |



| knbn  | max_nb_conn | ef_cons | ef_search | extend | keep pruned | recall | req/s | last ratio |
| :---: | :---------: | :-----: | :-------: | :----: | :---------: | :----: | :---: | :--------: |
|  10   |     24      |   100   |    128    |   no   |     no      | 0.960  | 5900  |   1.001    |
|  100  |     24      |   100   |    128    |   no   |     no      | 0.907  | 2800  |   1.004    |
|  10   |     24      |   400   |    128    |   no   |     no      | 0.972  | 4678  |   1.001    |
|  100  |     24      |   400   |    128    |   no   |     no      | 0.938  | 2338  |   1.003    |
|  10   |     24      |   800   |    128    |   no   |     no      | 0.975  | 4313  |   1.001    |
|  100  |     24      |   800   |    128    |   no   |     no      | 0.9428 | 2151  |   1.0025   |


###  Results with Amd Ryzen 9 7950 16 core and 0.5 scale modification factor

With modified level sampling level (as documented in [hnsw-rs](https://crates.io/crates/hnsw_rs))
we increase recall and have with max_nb_conn 48 better results than with max_nb_conn=24 without scale modification
which decrease memory consumption

| knbn  | max_nb_conn | ef_cons | ef_search | extend | keep pruned | recall | req/s | last ratio |
| :---: | :---------: | :-----: | :-------: | :----: | :---------: | :----: | :---: | :--------: |
|  10   |     48      |   100   |    128    |   no   |     no      | 0.997  | 6283  |   1.0001   |
|  100  |     48      |   100   |    128    |   no   |     no      | 0.980  | 3152  |   1.0007   |
|  10   |     24      |   100   |    128    |   no   |     no      | 0.989  | 9825  |   1.0003   |
|  100  |     24      |   100   |    128    |   no   |     no      | 0.958  | 4897  |   1.0017   |
