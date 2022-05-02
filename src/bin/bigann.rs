//! test BigANN benchmark with 100M points u8 in d = 128.
//! 



// learn.100M.u8bin and query.public.10K.u8bin format
// nb points , dimension as uint32 little endian
// Then each point, each dime u8
//
//  ground truth format
// nbquery int32 , KNN as uint32 
// Then nbquery * num_id as uint43 and last nbquery * d as f32

use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;

use anyhow::{anyhow};


use std::io::prelude::*;

use std::io::{BufReader};
use std::fs::{File, OpenOptions};
use std::path::{PathBuf};
use byteorder::{LittleEndian, ReadBytesExt};

use hnsw_rs::prelude::*;
use hnsw_rs::hnsw::{DataId};

const BIGANN_DIR : &'static str = "/home/jpboth/Data/ANN/BigANN";

/// read learn.100M.u8bin or query.public.10K.u8bin
fn read_data_block(data_buf : &mut BufReader<File>, nb_data: usize, size : usize) -> Result< Vec<Vec<u8>>, anyhow::Error> {
    // read number of asked number of points of size size * u8 until EOF
    //
    for i in 0..nb_data {

    }
    Err(anyhow!("not yet"))
} // end of read_data


// read ground truth format
fn read_ground_truth(fname : String) -> Result< Vec<(u32, f32)>, anyhow::Error> {
    // read nbquery(uint32 little endian)

    // read knn (uint32 little endian)
    Err(anyhow!("not yet"))
} // end of read_ground_truth



pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let dirname = String::from(BIGANN_DIR);
    //
    let mut data_fname = dirname.clone();
    data_fname.push_str("learn.100M.u8bin");
 //   data_fname.push_str("learn.100M.u8bin");
    //
    let query_fname = dirname.clone().push_str("query.public.10K.u8bin");
    let ground_truth = dirname.clone().push_str("public_query_gt100.bin");
    //
    let path = PathBuf::from(data_fname);
    let data_file = OpenOptions::new().read(true).open(&path).unwrap();
    let mut data_buf = BufReader::new(data_file);
    //
    let nb_data : u32 = data_buf.read_u32::<LittleEndian>().unwrap();
    let nb_data = nb_data as usize;
    let dim = data_buf.read_u32::<LittleEndian>().unwrap() as usize;
    // we will read data by block doing parallel insertion in Hnsw, read_data_block running async
    let ef_c = 50;
    let max_nb_connection = 70;
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    let block = 10000;
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let hnsw = Hnsw::<u8, DistL2>::new(max_nb_connection, nb_data, nb_layer, ef_c, DistL2{});
    // now we loop reading inserting
     

} // end of main