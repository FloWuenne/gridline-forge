//! TSV I/O for transcript XYZ coordinate files.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use super::TranscriptRecord;
use crate::types::{Error, Result};

/// Read a tab-separated XYZ gene coordinates file (no header).
///
/// Expected columns: `x  y  z  gene` (tab-separated). Extra columns are ignored.
pub fn read_tsv(path: &Path) -> Result<Vec<TranscriptRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // Split on tab; take at most 5 parts (ignores extra columns)
        let fields: Vec<&str> = line.splitn(5, '\t').collect();
        if fields.len() < 4 {
            return Err(Error::TsvParse(format!(
                "line {}: expected at least 4 tab-separated columns, got {}",
                line_num + 1,
                fields.len()
            )));
        }
        let x = fields[0].trim().parse::<f32>().map_err(|e| {
            Error::TsvParse(format!(
                "line {}: invalid x '{}': {}",
                line_num + 1,
                fields[0],
                e
            ))
        })?;
        let y = fields[1].trim().parse::<f32>().map_err(|e| {
            Error::TsvParse(format!(
                "line {}: invalid y '{}': {}",
                line_num + 1,
                fields[1],
                e
            ))
        })?;
        let z = fields[2].trim().parse::<f32>().map_err(|e| {
            Error::TsvParse(format!(
                "line {}: invalid z '{}': {}",
                line_num + 1,
                fields[2],
                e
            ))
        })?;
        let gene = fields[3].trim().to_string();
        records.push(TranscriptRecord { x, y, z, gene });
    }

    Ok(records)
}

/// Write records to a TSV file, replacing the gene name with `"Duplicated"` for
/// any index in `duplicates`.
pub fn write_tsv(
    path: &Path,
    records: &[TranscriptRecord],
    duplicates: &HashSet<usize>,
) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for (i, rec) in records.iter().enumerate() {
        let gene = if duplicates.contains(&i) {
            "Duplicated"
        } else {
            &rec.gene
        };
        writeln!(writer, "{}\t{}\t{}\t{}", rec.x, rec.y, rec.z, gene)?;
    }
    Ok(())
}
