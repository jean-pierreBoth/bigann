//! test BigANN benchmark with 100M points u8 in d = 128.
//! command syntax is:
//! bigann --dir filesdir [--hnsw name] --nbdata nbdata
//! where :
//!     -  --dir filedir gives directory where data query files are.
//!     -  --hnsw name is optional and gives nale of dump of hnsw to reload (useful to change query parameters
//!     -  --nbdata nbdata gives the number of data (in millions) to run on. expect 10 100 or 1000


// learn and query format : for each data : get dim on u32 little endian then for each d * data value 
// so 4 + d * size of data in bytes
// Then each point, each dim u8
//
//  ground truth format
// nbquery int32 , KNN as uint32 
// Then nbquery * num_id as uint43 and last nbquery * d as f32

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use anyhow::{anyhow};

use clap::{Arg,Command, ArgMatches};

use std::io::prelude::*;
use std::io::{BufReader};
use std::fs::{File, OpenOptions};
use std::path::{PathBuf};
use byteorder::{LittleEndian, ReadBytesExt};

use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::{DataId};

// const BIGANN_DIR : &'static str = "/home/jpboth/Data/ANN/BigANN";
const BIGANN_DIR : &'static str = "/home.2/jpboth/Data/Ann/BigAnn";

const DIM : usize = 128;

/// read data from .bvecs file
fn read_data_block<const SIZE: usize>(data_buf : &mut BufReader<File>, nb_data: usize) -> Result< Vec<Vec<u8>>, anyhow::Error> {
    // read number of asked number of points of size size * u8 until EOF
    //
    let mut data = [0u8; SIZE];
    let mut datas = Vec::<Vec<u8>>::with_capacity(nb_data);
    for _ in 0..nb_data {
        let dim = data_buf.read_u32::<LittleEndian>().unwrap();
        if dim as usize != SIZE {
            log::error!("data has not the correct dimenson, found : {} expected {}", dim, SIZE);
            return Err(anyhow!("data has not the correct dimenson, found : {} expected {}", dim, SIZE));
        }
        let read_res = data_buf.read_exact(&mut data);
        if read_res.is_ok() {
            datas.push(data.to_vec());
        }
        else {
            match read_res.err().unwrap().kind() {
                std::io::ErrorKind::UnexpectedEof => { // got EOF
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



// read ground truth format
fn read_ground_truth(path : PathBuf) -> Result< Vec<Vec<(u32, f32)>>, anyhow::Error> {
    // read nbquery(uint32 little endian)
    let gt_file = OpenOptions::new().read(true).open(&path);
    if gt_file.is_err() {
        log::error!("could not open file : {}", path.display());
    }
    let gt_file = gt_file.unwrap();
    let mut gt_buf = BufReader::new(gt_file);
    let nb_query : usize = gt_buf.read_u32::<LittleEndian>().unwrap() as usize;
    let knn : usize = gt_buf.read_u32::<LittleEndian>().unwrap() as usize; // get number of neighbours
    log::info!("read_ground_truth got nb_query : {}, knn by answer : {}", nb_query, knn);
    //
    let mut ids = Vec::<Vec<u32>>::with_capacity(nb_query);
    // now we have for each query : knn u32 giving its corresponding answers
    for _ in 0..nb_query {
        let mut curent_ids = Vec::<u32>::with_capacity(knn);
        for _ in 0..knn {
            let id = gt_buf.read_u32::<LittleEndian>().unwrap();
            curent_ids.push(id);
        }
        ids.push(curent_ids);
    }
    // then for each data : knn f32 giving its distances
    let mut distances = Vec::<Vec<f32>>::with_capacity(nb_query);
    for _ in 0..nb_query {
        let mut current_dists = Vec::<f32>::with_capacity(knn);
        for _ in 0..knn {
            let dist = gt_buf.read_f32::<LittleEndian>().unwrap();
            current_dists.push(dist);
        }
        distances.push(current_dists);
    }
    // We zip the vectors to have all info on each request in one block
    let mut ground_truth = Vec::<Vec<(u32, f32)>>::with_capacity(nb_query);
    for q in 0..nb_query {
        let mut q_answer = Vec::<(u32,f32)>::with_capacity(knn);
        for k in 0..knn {
            q_answer.push((ids[q][k],distances[q][k]));
        }
        ground_truth.push(q_answer);
    }
    //
    return Ok(ground_truth);
} // end of read_ground_truth


// read query vector
fn read_query(path : PathBuf, nb_data : usize) -> Result< Vec<Vec<u8>>, anyhow::Error> {
    //
    let query_file = OpenOptions::new().read(true).open(&path);
    if query_file.is_err() {
        log::error!("read_query , could not open file : {}", path.display());
    }
    let query_file = query_file.unwrap();
    let mut query_buf = BufReader::new(query_file);
    let res = read_data_block::<DIM>(&mut query_buf, nb_data);
    return res;
} // end of read_query




// TODO  clap args to avoid re filling a Hnsw if already existing somewhere
pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    // 
    let bigann_cmd = Command::new("bigann")
        .arg(Arg::new("dirname")
            .long("dir")
            .takes_value(true)
            .help("expecting dirname containing .bvecs"))
        .arg(Arg::new("nbdata")
            .long("nbdata")
            .takes_value(true)
            .help("expecting numberof data to run on : 10 100 or 1000"))
        .arg(Arg::new("hnsw")
            .long("hnsw")
            .help("name to use to access to name.hnsw.data and name.hnsw.graph"));
    //
    let dirname = String::from(BIGANN_DIR);
    //
    let mut data_fname = PathBuf::from(dirname.clone());
    data_fname.push("bigann_base.bvecs");
    //
    let mut query_fname = PathBuf::from(dirname.clone());
    query_fname.push("bigann_query.bvecs");
    //
    let mut ground_truth_fname = PathBuf::from(dirname.clone());
    // read ground truth for 100 knn for the first 10M vectors
    ground_truth_fname.push("bigann-gt-10M");
    //
    let path = PathBuf::from(data_fname);
    let data_file_res = OpenOptions::new().read(true).open(&path);
    if data_file_res.is_err() {
        log::error!("cannot open file : {:?}", path);
    }
    let data_file = data_file_res.unwrap();
    let mut data_buf = BufReader::new(data_file);
    //
    let nb_data = 10_000_000 as usize;
    // we will read data by block doing parallel insertion in Hnsw, read_data_block running async
    let ef_c = 64;
    let max_nb_connection = 100;
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    let default_block : usize = 10000;
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let hnsw = Hnsw::<u8, DistL2>::new(max_nb_connection, nb_data, nb_layer, ef_c, DistL2{});
    // now we loop reading inserting
    let mut nb_data_read = 0;
    //==============  When reading is ok set test to true
    let test = false;
    //==============
    loop {
        // we adjust block to read to get exactly nb_data
        let block = if nb_data_read + default_block < nb_data { default_block } else { nb_data - nb_data_read};
        let new_datas = read_data_block::<DIM>(&mut data_buf, block);
        if new_datas.is_err() {
            log::error!("read_data_block , nb blocks read : {}", nb_data_read);
            std::panic!("read_data_block failed with error: {:?}", new_datas.as_ref().err().unwrap());
        }
        let new_data = new_datas.unwrap();
        if new_data.len() == 0 {
            break;
        }
        // insert in Hnsw
        let data_with_id : Vec<(&Vec<u8>, DataId)>= new_data.iter().zip(0..new_data.len()).map(|v| (v.0, v.1 + nb_data_read)).collect();
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
    println!(" ann construction sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.as_secs());
    hnsw.dump_layer_info();
    // dump in a file. Must take care of name as tests runs in // !!!
    if !test {
        let fname = String::from("dumpbigann");
        log::info!("dumping in files : {} ... ", fname);
        let _res = hnsw.file_dump(&fname); 
        log::info!("dump finished");
    }   
    //
    // load ground truth to know how many neighbours we must search
    //
    let gtruth = read_ground_truth(ground_truth_fname.clone());
    if gtruth.is_err() {
        log::error!("error opening ground truth file : {}, err : {:?}", ground_truth_fname.display(), gtruth.as_ref().err().unwrap());
    }
    let gtruth = gtruth.unwrap();
    //
    // read queries
    //
    let nb_query = 10000;
    let query = read_query(query_fname, nb_query);
    if query.is_err() {
        std::panic!("could not read queries, error : {:?}", query.as_ref().err().unwrap());
    }
    let query = query.unwrap();
    log::info!("loaded nb_query : {}", query.len());
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now(); 
    log::info!("starting requests"); 
    // TODO check knn is constant!?  
    let knbn = gtruth[0].len();
    let ef_search = 128;
//    let knn_answers = hnsw.parallel_search(&query, knbn, ef_search);
    let cpu_time: Duration = cpu_start.elapsed();
    println!(" ann construction sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.as_secs());
} // end of main