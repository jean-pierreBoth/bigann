//! test BigANN benchmark with 100M points u8 in d = 128.
//! command syntax is:
//! bigann --dir filesdir [--hnsw name] [--dump] --nbdata nbdata
//! where :
//!     -  --dir filedir gives directory where data query files are (and possible hnsw dump)
//!     -  --dump to get a dump of hnsw structure in dumpbigann.hnsw.data and dumpbigann.hnsw.graph
//!     -  --hnsw name is optional and gives name of dump of hnsw to reload (useful to change query parameters
//!     -  --nbdata nbdata gives the number of data (in millions) to run on. expect 10 100 or 1000

// learn and query format : for each data : get dim on u32 little endian then for each d * data value
// so 4 + d * size of data in bytes
// Then each point, each dim u8
//
//  ground truth format
// nbquery int32 , KNN as uint32
// Then nbquery * num_id as uint42 and last nbquery * d as f32

use cpu_time::ProcessTime;
use memory_stats::memory_stats;
use std::time::{Duration, SystemTime};

use anyhow::anyhow;

use clap::{Arg, ArgAction, Command};

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use hnsw_rs::hnsw::DataId;
use hnsw_rs::hnswio::*;
use hnsw_rs::prelude::*;

const DIM: usize = 128;

/// read data from .bvecs file
fn read_data_block<const SIZE: usize>(
    data_buf: &mut BufReader<File>,
    nb_data: usize,
) -> Result<Vec<Vec<u8>>, anyhow::Error> {
    // read number of asked number of points of size size * u8 until EOF
    //
    let mut data = [0u8; SIZE];
    let mut datas = Vec::<Vec<u8>>::with_capacity(nb_data);
    for _ in 0..nb_data {
        let dim = data_buf.read_u32::<LittleEndian>().unwrap();
        if dim as usize != SIZE {
            log::error!(
                "data has not the correct dimenson, found : {} expected {}",
                dim,
                SIZE
            );
            return Err(anyhow!(
                "data has not the correct dimenson, found : {} expected {}",
                dim,
                SIZE
            ));
        }
        let read_res = data_buf.read_exact(&mut data);
        if read_res.is_ok() {
            datas.push(data.to_vec());
        } else {
            match read_res.err().unwrap().kind() {
                std::io::ErrorKind::UnexpectedEof => {
                    // got EOF
                    log::info!("read_data_block got EOF after {} vectors", datas.len());
                    return Ok(datas);
                }
                _ => {
                    return Err(anyhow!("unexpected error"));
                }
            } // end match
        }
    } // end of for
    return Ok(datas);
} // end of read_data

// read fvec file
fn read_f32_block(data_buf: &mut BufReader<File>) -> Result<Vec<f32>, anyhow::Error> {
    //
    let dim = data_buf.read_u32::<LittleEndian>().unwrap();
    let v: Vec<f32> = (0..dim)
        .map(|_| data_buf.read_f32::<LittleEndian>().unwrap())
        .collect();
    //
    Ok(v)
}

// read ivec file
fn read_u32_block(data_buf: &mut BufReader<File>) -> Result<Vec<u32>, anyhow::Error> {
    //
    let dim = data_buf.read_u32::<LittleEndian>().unwrap();
    let v: Vec<u32> = (0..dim)
        .map(|_| data_buf.read_u32::<LittleEndian>().unwrap())
        .collect();
    //
    Ok(v)
}

// read ground truth format
fn read_ground_truth(
    i_path: &PathBuf, // ivecs file
    f_path: &PathBuf, // fvec file
    nb_query: usize,
) -> Result<Vec<Vec<(u32, f32)>>, anyhow::Error> {
    //
    log::info!("in read_ground_truth");
    // read nbquery(uint32 little endian)
    let gt_file_i = OpenOptions::new().read(true).open(i_path);
    if gt_file_i.is_err() {
        log::error!("could not open file : {}", i_path.display());
    }
    let gt_file_i = gt_file_i.unwrap();
    let mut gt_buf = BufReader::new(gt_file_i);
    log::info!("read_ground_truth got nb_query : {}", nb_query,);
    let ids: Vec<Vec<u32>> = (0..nb_query)
        .map(|_| read_u32_block(&mut gt_buf).unwrap())
        .collect();
    // get number of neighbours
    let knn: usize = ids[0].len();
    log::info!("read_ground_truth got knn : {}", knn);
    //
    // now we treat fvecs
    //
    let gt_file_f = OpenOptions::new().read(true).open(&f_path);
    if gt_file_f.is_err() {
        log::error!("could not open file : {}", f_path.display());
    }
    let gt_file_f = gt_file_f.unwrap();
    let mut gt_buf = BufReader::new(gt_file_f);
    let distances: Vec<Vec<f32>> = (0..nb_query)
        .map(|_| read_f32_block(&mut gt_buf).unwrap())
        .collect();
    let knn_f: usize = distances[0].len();
    assert_eq!(knn_f, knn);
    // We zip the vectors to have all info on each request in one block
    let mut ground_truth = Vec::<Vec<(u32, f32)>>::with_capacity(nb_query);
    for q in 0..nb_query {
        let mut q_answer = Vec::<(u32, f32)>::with_capacity(knn);
        for k in 0..knn {
            q_answer.push((ids[q][k], distances[q][k]));
        }
        ground_truth.push(q_answer);
    }
    log::info!("exiting read_ground_truth");
    //
    return Ok(ground_truth);
} // end of read_ground_truth

// read query vector
fn read_query(path: PathBuf, nb_data: usize) -> Result<Vec<Vec<u8>>, anyhow::Error> {
    //
    log::info!("in read_query");
    //
    let query_file = OpenOptions::new().read(true).open(&path);
    if query_file.is_err() {
        log::error!("read_query , could not open file : {}", path.display());
    }
    let query_file = query_file.unwrap();
    let mut query_buf = BufReader::new(query_file);
    let res = read_data_block::<DIM>(&mut query_buf, nb_data);
    log::info!("exiting read_query");
    return res;
} // end of read_query

// read data and construct hnsw
fn fill_hnsw<'a, 'b>(
    data_buf: &'a mut BufReader<File>,
    nb_data: usize,
    ef_c: usize,
    max_nb_connection: usize,
    dump: bool,
    test: bool,
) -> Result<Hnsw<'b, u8, DistL2>, anyhow::Error>
where
    'b: 'a,
{
    //
    // we will read data by block doing parallel insertion in Hnsw, read_data_block running async
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    let default_block: usize = 50000;
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let mut hnsw = Hnsw::<u8, DistL2>::new(max_nb_connection, nb_data, nb_layer, ef_c, DistL2 {});
    //
    hnsw.modify_level_scale(0.25);
    //
    let mut nb_data_read = 0;
    loop {
        // we adjust block to read to get exactly nb_data
        let block = if nb_data_read + default_block < nb_data {
            default_block
        } else {
            nb_data - nb_data_read
        };
        let new_datas = read_data_block::<DIM>(data_buf, block);
        if new_datas.is_err() {
            log::error!("read_data_block , nb blocks read : {}", nb_data_read);
            std::panic!(
                "read_data_block failed with error: {:?}",
                new_datas.as_ref().err().unwrap()
            );
        }
        let new_data = new_datas.unwrap();
        if new_data.len() == 0 {
            break;
        }
        // insert in Hnsw
        let data_with_id: Vec<(&Vec<u8>, DataId)> = new_data
            .iter()
            .zip(0..new_data.len())
            .map(|v| (v.0, v.1 + nb_data_read))
            .collect();
        if !test {
            hnsw.parallel_insert(&data_with_id);
        }
        nb_data_read += new_data.len();
        if nb_data_read == nb_data {
            log::info!("exiting loop nb data read : {}", nb_data_read);
            break;
        }
    } // end of loop
    let cpu_time: Duration = cpu_start.elapsed();
    println!(
        " ann construction sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    hnsw.dump_layer_info();
    // dump in a file. Must take care of name as tests runs in //
    if dump {
        let fname = String::from("dumpbigann");
        log::info!("dumping in files : {} ... ", fname);
        let _res = hnsw.file_dump(Path::new("."), &fname);
        log::info!("dump finished");
    }
    //
    log::debug!("fill_hnsw finished");
    //
    Ok(hnsw)
} // end of fill_hnsw

// TODO  clap args to avoid re filling a Hnsw if already existing somewhere
pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let bigann_arg = Command::new("bigann")
        .arg(
            Arg::new("dirname")
                .long("dir")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .required(true)
                .help("expecting dirname containing .bvecs"),
        )
        .arg(
            Arg::new("nbdata")
                .long("nbdata")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .help("expecting numberof data to run on : 10 100 or 1000"),
        )
        .arg(
            Arg::new("dump")
                .long("dump")
                .action(ArgAction::SetTrue)
                .help("flag to get a dump in files .hnsw.graph and .hnsw.data"),
        )
        .arg(
            Arg::new("hnsw")
                .long("hnsw")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(String))
                .help("name to use to access to name.hnsw.data and name.hnsw.graph"),
        )
        .get_matches();
    //
    // parse cmd
    //
    let dirname = bigann_arg.get_one::<String>("dirname").unwrap();
    log::info!("got dir : {}", dirname);
    // no hnsw, we must have nb_data to know what we must read
    let mut nb_data = *bigann_arg.get_one::<usize>("nbdata").unwrap();
    log::info!("got nb_data : {}", nb_data);
    if nb_data >= 10 && nb_data != 10 && nb_data != 100 && nb_data != 1000 {
        log::error!("nb_data must be 1, 10, 100 or 1000");
        std::process::exit(1);
    };
    nb_data *= 1_000_000;
    log::info!("running on first : {}", nb_data);
    // check if we want the hnsw structure to be dumped
    let to_dump = bigann_arg.get_flag("dump");
    log::info!("dump flag : {}", to_dump);
    // get hnsw if any
    let mut hnsw_name: Option<String> = None;
    if let Some(hnsw) = bigann_arg.get_one::<String>("hnsw") {
        log::debug!("hnsw present, will reload from it");
        log::info!("got hnsw name : {}", hnsw);
        hnsw_name = Some(hnsw.clone());
    } else {
        // no hnsw, we must have nb_data to know what we must read
        nb_data = *bigann_arg.get_one::<usize>("nbdata").unwrap();
        log::info!("got nb_data : {}", nb_data);
        if nb_data >= 10 && nb_data != 10 && nb_data != 100 && nb_data != 1000 {
            log::error!("nb_data must be 1, 10, 100 or 1000");
            std::process::exit(1);
        };
        nb_data *= 1_000_000;
    }

    //
    let mut query_fname = PathBuf::from(dirname.clone());
    query_fname.push("bigann_query.bvecs");
    //
    // read ground truth for 100 knn for the first 10M vectors. To modify to run on 100M or whole data
    //================================================================================================
    //
    let mut ground_truth_fname_ivec = PathBuf::from(dirname.clone());
    let gt_fname_i = "idx_10M.ivecs";
    ground_truth_fname_ivec.push(gt_fname_i);
    let gt_fname_f = "dis_10M.fvecs";
    let mut ground_truth_fname_fvec = PathBuf::from(dirname.clone());
    ground_truth_fname_fvec.push(gt_fname_f);
    //
    // read queries
    //
    let nb_query = 10000;
    let query = read_query(query_fname, nb_query);
    if query.is_err() {
        std::panic!(
            "could not read queries, error : {:?}",
            query.as_ref().err().unwrap()
        );
    }
    let query = query.unwrap();
    log::info!("loaded nb_query : {}", query.len());
    //
    // load ground truth to know how many neighbours we must search
    //
    let gtruth = read_ground_truth(&ground_truth_fname_ivec, &ground_truth_fname_fvec, nb_query);
    if gtruth.is_err() {
        log::error!(
            "error opening ground truth file : {}, {} err : {:?}",
            ground_truth_fname_ivec.display(),
            ground_truth_fname_fvec.display(),
            gtruth.as_ref().err().unwrap()
        );
    }
    let gtruth = gtruth.unwrap();
    //
    // if test is true we run without doing the hnsw insertion, this enbales testing
    let mut reloader: HnswIo;
    let test = false;
    let hnsw_res = if hnsw_name.is_none() {
        // parameters to initialize hnsw
        // =============================
        let ef_c: usize = 800;
        let max_nb_connection: usize = 24;

        //=============================
        log::info!(
            "no hnsw to reload from, will read data from file bigann_base.bvecs in dir : {}",
            dirname
        );
        assert!(nb_data > 0);
        let mut data_fname = PathBuf::from(dirname.clone());
        data_fname.push("bigann_base.bvecs");
        let path = PathBuf::from(data_fname);
        let data_file_res = OpenOptions::new().read(true).open(&path);
        if data_file_res.is_err() {
            log::error!("cannot open file : {:?}", path);
        }
        let data_file = data_file_res.unwrap();
        let mut data_buf = BufReader::new(data_file);
        let hnsw_res = fill_hnsw(
            &mut data_buf,
            nb_data,
            ef_c,
            max_nb_connection,
            to_dump,
            test,
        );
        hnsw_res
    } else {
        log::info!(
            " hnsw passed , will reload from hnsw data in dir : {}",
            dirname
        );
        let hnsw_str = hnsw_name.unwrap();
        reloader = HnswIo::new(Path::new(dirname), &hnsw_str);
        let hnsw_loaded = reloader.load_hnsw::<u8, DistL2>().unwrap();
        //
        Ok(hnsw_loaded)
    };
    //
    if hnsw_res.is_err() {
        log::error!("could not get hnsw");
        std::process::exit(1);
    }
    let hnsw = hnsw_res.unwrap();
    //
    if let Some(usage) = memory_stats() {
        println!("memory usage after loading hnsw");
        println!("Current physical memory usage: {}", usage.physical_mem);
        println!("Current virtual memory usage: {}", usage.virtual_mem);
    } else {
        println!("Couldn't get the current memory usage :(");
    }
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    log::info!("starting requests");
    // TODO check knn is constant!?
    let ef_search = 128;
    //
    // first search with knn = 10
    //
    let knbn = 10;
    log::info!("Results with knbn : {}", knbn);
    let knn_answers = hnsw.parallel_search(&query, knbn, ef_search);
    let cpu_time: Duration = cpu_start.elapsed();
    let sys_time = sys_now.elapsed().unwrap().as_micros() as f32;
    println!(
        " ann requests sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    let mut recalls = Vec::<usize>::with_capacity(nb_query);
    let mut nb_returned = Vec::<usize>::with_capacity(nb_query);
    let mut last_distances_ratio = Vec::<f32>::with_capacity(nb_query);
    // now compute recall rate
    for i in 0..nb_query {
        let answer = &gtruth[i];
        if answer.len() <= 0 {
            std::process::exit(1);
        }
        // it seems that ground truth returns || a-b||**2 !!!
        let max_dist = answer[knbn - 1].1.sqrt();
        let mut _knn_neighbours_id: Vec<usize> = knn_answers[i].iter().map(|p| p.d_id).collect();
        let knn_neighbours_dist: Vec<f32> = knn_answers[i].iter().map(|p| p.distance).collect();
        nb_returned.push(answer.len());
        // count how many distances of knn_neighbours_dist are less than
        let recall = knn_neighbours_dist
            .iter()
            .filter(|x| *x <= &max_dist)
            .count();
        recalls.push(recall);
        let mut ratio = 0.;
        if knn_neighbours_dist.len() >= 1 {
            ratio = knn_neighbours_dist[knn_neighbours_dist.len() - 1] / max_dist;
        }
        last_distances_ratio.push(ratio);
        if i <= 10 {
            log::debug!("request num : {}", i);
            log::debug!("distances found : {:?}", knn_neighbours_dist);
            log::debug!("ids found : {:?}", _knn_neighbours_id);
            for i in 0..knbn {
                log::debug!(
                    "ground truth id : {}, distance : {:.3e}",
                    answer[i].0,
                    answer[i].1.sqrt()
                );
            }
        }
    }
    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((knbn * recalls.len()) as f32);
    println!(
        "\n recall rate for  {:?} , nb req /s {:?}",
        mean_recall,
        (nb_query as f32) * 1.0e+6_f32 / sys_time
    );
    println!(
        "\n last distances ratio {:?} ",
        last_distances_ratio.iter().sum::<f32>() / last_distances_ratio.len() as f32
    );

    //
    // second search with knn = 100
    //
    let knbn = 100;
    log::info!("Results with knbn : {}", knbn);
    let knn_answers = hnsw.parallel_search(&query, knbn, ef_search);
    let cpu_time: Duration = cpu_start.elapsed();
    let sys_time = sys_now.elapsed().unwrap().as_micros() as f32;
    println!(
        " ann requests sys time(s) {:?} cpu time {:?}",
        sys_now.elapsed().unwrap().as_secs(),
        cpu_time.as_secs()
    );
    let mut recalls = Vec::<usize>::with_capacity(nb_query);
    let mut nb_returned = Vec::<usize>::with_capacity(nb_query);
    let mut last_distances_ratio = Vec::<f32>::with_capacity(nb_query);
    // now compute recall rate
    for i in 0..nb_query {
        let answer = &gtruth[i];
        if answer.len() <= 0 {
            std::process::exit(1);
        }
        // it seems that ground truth returns || a-b||**2 !!!
        let max_dist = answer[knbn - 1].1.sqrt();
        let mut _knn_neighbours_id: Vec<usize> = knn_answers[i].iter().map(|p| p.d_id).collect();
        let knn_neighbours_dist: Vec<f32> = knn_answers[i].iter().map(|p| p.distance).collect();
        nb_returned.push(answer.len());
        // count how many distances of knn_neighbours_dist are less than
        let recall = knn_neighbours_dist
            .iter()
            .filter(|x| *x <= &max_dist)
            .count();
        recalls.push(recall);
        let mut ratio = 0.;
        if knn_neighbours_dist.len() >= 1 {
            ratio = knn_neighbours_dist[knn_neighbours_dist.len() - 1] / max_dist;
        }
        last_distances_ratio.push(ratio);
        if i <= 5 {
            log::debug!("request num : {}", i);
            log::debug!("distances found : {:?}", knn_neighbours_dist);
            log::debug!("ids found : {:?}", _knn_neighbours_id);
            for i in 0..knbn {
                log::debug!(
                    "ground truth id : {}, distance : {:.3e}",
                    answer[i].0,
                    answer[i].1.sqrt()
                );
            }
        }
    }
    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((knbn * recalls.len()) as f32);
    println!(
        "\n recall rate for  {:?} , nb req /s {:?}",
        mean_recall,
        (nb_query as f32) * 1.0e+6_f32 / sys_time
    );
    println!(
        "\n last distances ratio {:?} ",
        last_distances_ratio.iter().sum::<f32>() / last_distances_ratio.len() as f32
    );
} // end of main
