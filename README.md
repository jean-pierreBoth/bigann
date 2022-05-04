# bigann

This mini crate benches the crate [hnsw-rs](https://crates.io/crates/hnsw_rs) on sampled u8 vectors from the BIGANN benchmark. See [BIGANN](https://big-ann-benchmarks.com/)

Files bigann_base.bvecs, bigann_query.bvecs must be dowloaded and installed in some directory (This amounts to 133Gb).
Then depending on the size of the data you want to run on the first (10M, 100M, 1B) of the large file *bigann_base.bvecs* you download the corresponding ground truth as explained in the BIGANN web page.

To run on the first 10M slices of data you download the corresponding ground truth 
as documented on the BIGANN web page and extract the file GT_10M/bigann-10M to replace the file *public_query_gt100.bin* which corresponds to the ground truth on the totality.

## commandline

bigann --dir DataDir -hnsw dumpname  or bigann --dir DataDir --nbdata 10 (or 100 or 1000M if you have a access to a big machine).  
For more see documentation (cargo doc --no-dpes as usual) 
## Results for the first 10 Million data points.

Results on Intel E5-2630 v3@2.4GHz
16 cores 2 thread / core

All parameters are explained in doc of  [hnsw-rs](https://crates.io/crates/hnsw_rs).


|  knbn         | max_nb_conn  |  ef_cons   | ef_search   |  extend     | keep pruned  |   recall  |    req/s  |  last ratio |
|  :----------: |  :--------:  | :-------:  |  :-------:  |   :-------: |  :-------:   |   :-----: |  :----:   | :-------:   |
|     10        |   64         |  100       |   128       |    no       |    no        |   0.995   |  2610     |  1.0002     | 
|     100       |   64         |  100       |   128       |    no       |    no        |   0.983   |  1350     |  1.0006     |
|      10       |   24         |  100       |   128       |    no       |    no        |   0.970   |  4845     |  1.001      |     
|     10        |   24         |  100       |   128       |    no       |    no        |   0.923   |  2411     |  1.003      |

Results on Intel  i7-10875H CPU @ 2.30GHz  8 core 2 Thread /core

time for Hnsw structure construction user :1370 s,  cpu time 21493

|  knbn         | max_nb_conn  |  ef_cons   | ef_search   |  extend     | keep pruned  |   recall  |    req/s  |  last ratio |
|      10       |   24         |  100       |   128       |    no       |    no        |   0.960   |  5900     |  1.001      |     
|     10        |   24         |  100       |   128       |    no       |    no        |   0.907   |  2800     |  1.004      |
