//! test BigANN benchmark with 100M points u8 in d = 128.
//! 



// learn.100M.u8bin and query.public.10K.u8bin format
// nb points , dimension as uint32 little endian
// Then each point, each dim u8
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
fn read_data_block<const SIZE: usize>(data_buf : &mut BufReader<File>, nb_data: usize) -> Result< Vec<Vec<u8>>, anyhow::Error> {
    // read number of asked number of points of size size * u8 until EOF
    //
    let mut data = [0u8; SIZE];
    let mut datas = Vec::<Vec<u8>>::with_capacity(nb_data);
    for _ in 0..nb_data {
        let nb_read = data_buf.read(&mut data)?;
        match nb_read {
            100 => { datas.push(data.to_vec()); }
            0 => { // got EOF
                    log::info!("read_data_block got EOF after {} vectors", datas.len());
                    return Ok(datas);
                 }
            _ => {
                    return Err(anyhow!("could read only {} dimensions", nb_read));
            }
        }
    } // end of for 
    return Ok(datas);
} // end of read_data



// read ground truth format
fn read_ground_truth(fname : String) -> Result< Vec<(u32, f32)>, anyhow::Error> {
    // read nbquery(uint32 little endian)

    // read knn (uint32 little endian)
    Err(anyhow!("not yet"))
} // end of read_ground_truth


const DIM : usize = 100;

// TODO  clap args to avoid re filling a Hnsw if already existing somewhere
pub fn main() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    let dirname = String::from(BIGANN_DIR);
    //
    let mut data_fname = dirname.clone();
    data_fname.push_str("learn.100M.u8bin");
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
    log::info!("number of vectors to read : {}, dimension : {}", nb_data, dim);
    if dim != DIM {
        std::panic!("expected dim {}, got {}", DIM, dim);
    }
    // we will read data by block doing parallel insertion in Hnsw, read_data_block running async
    let ef_c = 50;
    let max_nb_connection = 70;
    let nb_layer = 16.min((nb_data as f32).ln().trunc() as usize);
    let block : usize = 10000;
    //
    let cpu_start = ProcessTime::now();
    let sys_now = SystemTime::now();
    let hnsw = Hnsw::<u8, DistL2>::new(max_nb_connection, nb_data, nb_layer, ef_c, DistL2{});
    // now we loop reading inserting
    let mut nb_data_read = 0;
    loop {
        let new_datas = read_data_block::<DIM>(&mut data_buf, block);
        if new_datas.is_err() {
            std::panic!("read_data_block failed with error {:?}", new_datas.as_ref());
        }
        let new_data = new_datas.unwrap();
        if new_data.len() == 0 {
            break;
        }
        // insert in Hnsw
        let data_with_id : Vec<(&Vec<u8>, DataId)>= new_data.iter().zip(0..new_data.len()).map(|v| (v.0, v.1 + nb_data_read)).collect();
        hnsw.parallel_insert(&data_with_id);
        nb_data_read += new_data.len();
        if nb_data_read == nb_data {
            break;
        }
    } // end of loop
    let cpu_time: Duration = cpu_start.elapsed();
    println!(" ann construction sys time(s) {:?} cpu time {:?}", sys_now.elapsed().unwrap().as_secs(), cpu_time.as_secs());
    hnsw.dump_layer_info();

} // end of main