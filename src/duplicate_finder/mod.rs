//! Duplicate finder: identifies gene transcripts duplicated at grid line boundaries.
//!
//! Matches the MindaGap `duplicate_finder.py` algorithm: for each tile-pair boundary,
//! finds potential duplicate transcript pairs, computes the mode XYZ shift, and
//! confirms duplicates via mutual best-partner matching.

use std::collections::{HashMap, HashSet};

pub mod tsv_io;

/// A single transcript record with XYZ coordinates and gene name.
#[derive(Debug, Clone)]
pub struct TranscriptRecord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: String,
}

/// Configuration for the duplicate finder algorithm.
#[derive(Debug, Clone)]
pub struct DuplicateFinderConfig {
    /// Tile size on X axis (distance between vertical gridlines)
    pub x_tilesize: u32,
    /// Tile size on Y axis (distance between horizontal gridlines)
    pub y_tilesize: u32,
    /// Window size around gridlines to search (pixels to each side)
    pub window_size: u32,
    /// Maximum gene count; genes above this are excluded from shift calculation
    pub max_freq: usize,
    /// Minimum occurrences of a ~XYZ shift to consider it valid
    pub min_mode: usize,
    /// Print progress messages
    pub show_progress: bool,
}

impl Default for DuplicateFinderConfig {
    fn default() -> Self {
        Self {
            x_tilesize: 2144,
            y_tilesize: 2144,
            window_size: 30,
            max_freq: 400,
            min_mode: 10,
            show_progress: true,
        }
    }
}

/// Statistics from a duplicate-finding run.
#[derive(Debug, Default)]
pub struct DuplicateStats {
    pub tile_pairs: usize,
    pub tile_overlaps: usize,
    pub total_dups: usize,
}

/// Find duplicate transcripts at grid line boundaries.
///
/// Returns the set of original record indices that are duplicates, plus stats.
pub fn find_duplicates(
    records: &[TranscriptRecord],
    config: &DuplicateFinderConfig,
) -> (HashSet<usize>, DuplicateStats) {
    let mut duplicated: HashSet<usize> = HashSet::new();
    let mut stats = DuplicateStats::default();

    let w = config.window_size as f32;
    let x_tile = config.x_tilesize as f32;
    let y_tile = config.y_tilesize as f32;

    let x_max = records
        .iter()
        .map(|r| r.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let y_max = records
        .iter()
        .map(|r| r.y)
        .fold(f32::NEG_INFINITY, f32::max);

    // First pass: vertical gridlines (x = Xtilesize, 2·Xtilesize, …)
    // Outer loop over tile rows (y bands), inner over vertical grid line positions.
    let mut yjump = 0.0_f32;
    while yjump < y_max {
        let y_min_b = yjump;
        let y_max_b = yjump + y_tile;

        let mut xjump = x_tile;
        while xjump < x_max {
            let x_min_b = xjump - w;
            let x_max_b = xjump + w;

            let mut window: Vec<(usize, &TranscriptRecord)> = records
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    r.x > x_min_b && r.x < x_max_b && r.y > y_min_b && r.y < y_max_b
                })
                .collect();

            if window.is_empty() {
                if config.show_progress {
                    println!(
                        "Found no transcripts within Y{:.0}:{:.0}, X{:.0}:{:.0}",
                        x_min_b, x_max_b, y_min_b, y_max_b
                    );
                }
                xjump += x_tile;
                continue;
            }

            stats.tile_pairs += 1;

            // Sort by y (the axis perpendicular to the vertical gridline)
            window.sort_by(|a, b| {
                a.1.y
                    .partial_cmp(&b.1.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let common_genes = count_common_genes(&window, config.max_freq);
            let mut partners_map: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut pair_xd = Vec::new();
            let mut pair_yd = Vec::new();
            let mut pair_zd = Vec::new();

            for i in 0..window.len() {
                let (orig_i, rec_i) = window[i];
                if rec_i.x >= xjump {
                    continue; // only left transcripts drive the search
                }

                // Backward scan (decreasing y)
                for ni in (i.saturating_sub(30)..i).rev() {
                    let (orig_n, rec_n) = window[ni];
                    if rec_n.y < rec_i.y - 7.0 {
                        break;
                    }
                    check_vertical_partner(
                        orig_i,
                        rec_i,
                        orig_n,
                        rec_n,
                        xjump,
                        w,
                        &common_genes,
                        &mut pair_xd,
                        &mut pair_yd,
                        &mut pair_zd,
                        &mut partners_map,
                    );
                }

                // Forward scan (increasing y)
                for ni in (i + 1)..window.len().min(i + 31) {
                    let (orig_n, rec_n) = window[ni];
                    if rec_n.y > rec_i.y + 7.0 {
                        break;
                    }
                    check_vertical_partner(
                        orig_i,
                        rec_i,
                        orig_n,
                        rec_n,
                        xjump,
                        w,
                        &common_genes,
                        &mut pair_xd,
                        &mut pair_yd,
                        &mut pair_zd,
                        &mut partners_map,
                    );
                }
            }

            let shift = find_mode_shift(&pair_xd, &pair_yd, &pair_zd, config.min_mode);

            if shift.iter().any(|&v| v > 2) {
                let mut dups = 0;
                for i in 0..window.len() {
                    let (orig_i, rec_i) = window[i];
                    if rec_i.x >= xjump {
                        continue;
                    }
                    let Some(partners_i) = partners_map.get(&orig_i) else {
                        continue;
                    };
                    let partners_i = partners_i.clone();

                    let Some((best_p, multdist)) =
                        find_best_partner(rec_i, &partners_i, records, shift)
                    else {
                        continue;
                    };
                    if multdist >= 20 {
                        continue;
                    }

                    let Some(partners_p) = partners_map.get(&best_p) else {
                        continue;
                    };
                    let partners_p = partners_p.clone();

                    if let Some((best_pp, _)) =
                        find_best_partner(&records[best_p], &partners_p, records, shift)
                    {
                        if best_pp == orig_i {
                            duplicated.insert(orig_i);
                            dups += 1;
                        }
                    }
                }

                if dups > 4 {
                    stats.total_dups += dups;
                    stats.tile_overlaps += 1;
                }
            }

            xjump += x_tile;
        }
        yjump += y_tile;
    }

    // Second pass: horizontal gridlines (y = Ytilesize, 2·Ytilesize, …)
    let mut xjump = 0.0_f32;
    while xjump < x_max {
        let x_min_b = xjump;
        let x_max_b = xjump + x_tile;

        let mut yjump = y_tile;
        while yjump < y_max {
            let y_min_b = yjump - w;
            let y_max_b = yjump + w;

            let mut window: Vec<(usize, &TranscriptRecord)> = records
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    r.x > x_min_b && r.x < x_max_b && r.y > y_min_b && r.y < y_max_b
                })
                .collect();

            if window.is_empty() {
                if config.show_progress {
                    println!(
                        "Found no transcripts within Y{:.0}:{:.0}, X{:.0}:{:.0}",
                        x_min_b, x_max_b, y_min_b, y_max_b
                    );
                }
                yjump += y_tile;
                continue;
            }

            stats.tile_pairs += 1;

            // Sort by x (the axis perpendicular to the horizontal gridline)
            window.sort_by(|a, b| {
                a.1.x
                    .partial_cmp(&b.1.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let common_genes = count_common_genes(&window, config.max_freq);
            let mut partners_map: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut pair_xd = Vec::new();
            let mut pair_yd = Vec::new();
            let mut pair_zd = Vec::new();

            for i in 0..window.len() {
                let (orig_i, rec_i) = window[i];
                if rec_i.y >= yjump {
                    continue; // only bottom transcripts drive the search
                }

                // Backward scan (decreasing x)
                for ni in (i.saturating_sub(30)..i).rev() {
                    let (orig_n, rec_n) = window[ni];
                    if rec_n.x < rec_i.x - 7.0 {
                        break;
                    }
                    check_horizontal_partner(
                        orig_i,
                        rec_i,
                        orig_n,
                        rec_n,
                        w,
                        &common_genes,
                        &mut pair_xd,
                        &mut pair_yd,
                        &mut pair_zd,
                        &mut partners_map,
                    );
                }

                // Forward scan (increasing x)
                for ni in (i + 1)..window.len().min(i + 31) {
                    let (orig_n, rec_n) = window[ni];
                    if rec_n.x > rec_i.x + 7.0 {
                        break;
                    }
                    check_horizontal_partner(
                        orig_i,
                        rec_i,
                        orig_n,
                        rec_n,
                        w,
                        &common_genes,
                        &mut pair_xd,
                        &mut pair_yd,
                        &mut pair_zd,
                        &mut partners_map,
                    );
                }
            }

            let shift = find_mode_shift(&pair_xd, &pair_yd, &pair_zd, config.min_mode);

            if shift.iter().any(|&v| v > 2) {
                let mut dups = 0;
                for i in 0..window.len() {
                    let (orig_i, rec_i) = window[i];
                    if rec_i.y >= yjump {
                        continue;
                    }
                    let Some(partners_i) = partners_map.get(&orig_i) else {
                        continue;
                    };
                    let partners_i = partners_i.clone();

                    let Some((best_p, multdist)) =
                        find_best_partner(rec_i, &partners_i, records, shift)
                    else {
                        continue;
                    };
                    if multdist >= 20 {
                        continue;
                    }

                    let Some(partners_p) = partners_map.get(&best_p) else {
                        continue;
                    };
                    let partners_p = partners_p.clone();

                    if let Some((best_pp, _)) =
                        find_best_partner(&records[best_p], &partners_p, records, shift)
                    {
                        if best_pp == orig_i {
                            duplicated.insert(orig_i);
                            dups += 1;
                        }
                    }
                }

                if dups > 9 {
                    stats.total_dups += dups;
                    stats.tile_overlaps += 1;
                }
            }

            yjump += y_tile;
        }
        xjump += x_tile;
    }

    (duplicated, stats)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check and record a vertical gridline partner (right side of x = xjump).
fn check_vertical_partner(
    orig_i: usize,
    rec_i: &TranscriptRecord,
    orig_n: usize,
    rec_n: &TranscriptRecord,
    xjump: f32,
    w: f32,
    common_genes: &HashSet<&str>,
    pair_xd: &mut Vec<f32>,
    pair_yd: &mut Vec<f32>,
    pair_zd: &mut Vec<f32>,
    partners_map: &mut HashMap<usize, Vec<usize>>,
) {
    if rec_n.gene != rec_i.gene || common_genes.contains(rec_n.gene.as_str()) {
        return;
    }
    if rec_n.x < xjump {
        return; // partner must be on the right side of the gridline
    }
    let dx = rec_n.x - rec_i.x;
    let dy = rec_n.y - rec_i.y;
    let dz = rec_n.z - rec_i.z;
    if dz.abs() >= 7.0 || dy.abs() >= 7.0 {
        return;
    }
    if dx <= 2.0 || dx >= w - 5.0 {
        return;
    }
    pair_xd.push(dx);
    pair_yd.push(dy);
    pair_zd.push(dz);
    partners_map.entry(orig_i).or_default().push(orig_n);
    partners_map.entry(orig_n).or_default().push(orig_i);
}

/// Check and record a horizontal gridline partner (above y = yjump).
fn check_horizontal_partner(
    orig_i: usize,
    rec_i: &TranscriptRecord,
    orig_n: usize,
    rec_n: &TranscriptRecord,
    w: f32,
    common_genes: &HashSet<&str>,
    pair_xd: &mut Vec<f32>,
    pair_yd: &mut Vec<f32>,
    pair_zd: &mut Vec<f32>,
    partners_map: &mut HashMap<usize, Vec<usize>>,
) {
    if rec_n.gene != rec_i.gene || common_genes.contains(rec_n.gene.as_str()) {
        return;
    }
    let dx = rec_n.x - rec_i.x;
    let dy = rec_n.y - rec_i.y;
    let dz = rec_n.z - rec_i.z;
    if dz.abs() >= 7.0 || dx.abs() >= 7.0 {
        return;
    }
    if dy <= 2.0 || dy >= w - 5.0 {
        return;
    }
    pair_xd.push(dx);
    pair_yd.push(dy);
    pair_zd.push(dz);
    partners_map.entry(orig_i).or_default().push(orig_n);
    partners_map.entry(orig_n).or_default().push(orig_i);
}

/// Count gene frequencies; return set of genes whose count exceeds `max_freq`.
fn count_common_genes<'a>(
    window: &[(usize, &'a TranscriptRecord)],
    max_freq: usize,
) -> HashSet<&'a str> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, rec) in window {
        *counts.entry(rec.gene.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter(|(_, c)| *c > max_freq)
        .map(|(g, _)| g)
        .collect()
}

/// Compute the mode (Δx, Δy, Δz) shift, summing counts of close secondary modes.
///
/// Matching Python's `mode3D`: converts to integer (truncation), counts exact triples,
/// sums the 2nd/3rd most common if |diff| < 3 from the top mode. Returns `[0,0,0]`
/// when no mode exceeds `min_mode`.
fn find_mode_shift(
    pair_xd: &[f32],
    pair_yd: &[f32],
    pair_zd: &[f32],
    min_mode: usize,
) -> [i32; 3] {
    if pair_xd.is_empty() {
        return [0, 0, 0];
    }

    let mut counts: HashMap<(i32, i32, i32), usize> = HashMap::new();
    for i in 0..pair_xd.len() {
        // Truncate to int (matching Python's `.astype(int)`)
        let key = (pair_xd[i] as i32, pair_yd[i] as i32, pair_zd[i] as i32);
        *counts.entry(key).or_insert(0) += 1;
    }

    let mut sorted: Vec<((i32, i32, i32), usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let check_rows = sorted.len().min(3);
    let best = sorted[0].0;
    let mut total_count = sorted[0].1;

    for i in 1..check_rows {
        let other = sorted[i].0;
        let diff = (best.0 - other.0).abs()
            + (best.1 - other.1).abs()
            + (best.2 - other.2).abs();
        if diff < 3 {
            total_count += sorted[i].1;
        }
    }

    if total_count > min_mode {
        [best.0, best.1, best.2]
    } else {
        [0, 0, 0]
    }
}

/// Find the best partner for `point` among `partners`, given the expected `shift`.
///
/// Weighted distance = |sig·Δx − shift_x|·6 + |sig·Δy − shift_y|·6 + |sig·Δz − shift_z|·1
/// where `sig` adjusts for which side of the gridline `point` is on.
///
/// Ties broken by smallest standard deviation of the three distance components.
fn find_best_partner(
    point: &TranscriptRecord,
    partners: &[usize],
    records: &[TranscriptRecord],
    shift: [i32; 3],
) -> Option<(usize, i32)> {
    if partners.is_empty() {
        return None;
    }

    let first = &records[partners[0]];
    let sig_raw = (point.x - first.x) + (point.y - first.y) + (point.z - first.z) + 0.0001;
    let sig: f32 = if sig_raw >= 0.0 { 1.0 } else { -1.0 };

    let mut best_idx = None;
    let mut best_multdist = i32::MAX;
    let mut best_dists: Option<[i32; 3]> = None;

    for &p_idx in partners {
        let p = &records[p_idx];
        // Match Python `.astype(int)` — truncation toward zero
        let dx = (sig * point.x - sig * p.x - shift[0] as f32).abs() as i32;
        let dy = (sig * point.y - sig * p.y - shift[1] as f32).abs() as i32;
        let dz = (sig * point.z - sig * p.z - shift[2] as f32).abs() as i32;
        let multdist = dx * 6 + dy * 6 + dz;

        if multdist < best_multdist {
            best_multdist = multdist;
            best_idx = Some(p_idx);
            best_dists = Some([dx, dy, dz]);
        } else if multdist == best_multdist {
            if let Some(curr) = best_dists {
                if std_dev_3([dx, dy, dz]) < std_dev_3(curr) {
                    best_idx = Some(p_idx);
                    best_dists = Some([dx, dy, dz]);
                }
            }
        }
    }

    best_idx.map(|idx| (idx, best_multdist))
}

fn std_dev_3(vals: [i32; 3]) -> f32 {
    let mean = (vals[0] + vals[1] + vals[2]) as f32 / 3.0;
    let var = ((vals[0] as f32 - mean).powi(2)
        + (vals[1] as f32 - mean).powi(2)
        + (vals[2] as f32 - mean).powi(2))
        / 3.0;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(x: f32, y: f32, z: f32, gene: &str) -> TranscriptRecord {
        TranscriptRecord {
            x,
            y,
            z,
            gene: gene.to_string(),
        }
    }

    #[test]
    fn test_find_mode_shift_basic() {
        // Dominant shift: (10, 2, 0)
        let xd = vec![10.0, 10.0, 10.0, 5.0];
        let yd = vec![2.0, 2.0, 2.0, 1.0];
        let zd = vec![0.0, 0.0, 0.0, 0.0];
        let shift = find_mode_shift(&xd, &yd, &zd, 2);
        assert_eq!(shift, [10, 2, 0]);
    }

    #[test]
    fn test_find_mode_shift_below_min_mode() {
        let xd = vec![10.0, 10.0];
        let yd = vec![2.0, 2.0];
        let zd = vec![0.0, 0.0];
        let shift = find_mode_shift(&xd, &yd, &zd, 5);
        assert_eq!(shift, [0, 0, 0]);
    }

    #[test]
    fn test_no_duplicates_no_transcripts() {
        let records: Vec<TranscriptRecord> = vec![];
        let config = DuplicateFinderConfig::default();
        let (dups, stats) = find_duplicates(&records, &config);
        assert!(dups.is_empty());
        assert_eq!(stats.tile_pairs, 0);
    }

    #[test]
    fn test_duplicate_detection_synthetic() {
        // Two transcripts of same gene on opposite sides of x=2144, close in y, z
        // Left: (2134, 100, 0), Right: (2154, 102, 0) — Δx=20, Δy=2, Δz=0
        // Repeated enough times to exceed min_mode=1
        let mut records = Vec::new();
        let min_mode = 1;
        for i in 0..5 {
            let y_offset = i as f32 * 20.0;
            // Left transcript
            records.push(make_record(2134.0, 100.0 + y_offset, 0.0, "GeneA"));
            // Right transcript (partner)
            records.push(make_record(2154.0, 102.0 + y_offset, 0.0, "GeneA"));
        }
        let config = DuplicateFinderConfig {
            x_tilesize: 2144,
            y_tilesize: 10000,
            window_size: 30,
            max_freq: 400,
            min_mode,
            show_progress: false,
        };
        let (dups, _stats) = find_duplicates(&records, &config);
        // Left transcripts (even indices 0,2,4,6,8) should be found
        for i in (0..10).step_by(2) {
            assert!(
                dups.contains(&i),
                "Expected record {} to be a duplicate",
                i
            );
        }
    }
}
