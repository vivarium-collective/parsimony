//! Genome annotation → transcription-site positions.
//!
//! The bacterial chromosome's supercoils and DNA-binding proteins aren't
//! placed at random along the genome — transcription happens at genes, and
//! RNA polymerase rides there. This module parses the committed *Mycoplasma
//! genitalium* G37 gene table (`examples/genome/mycoplasma_g37_genes.csv`,
//! from RefSeq GCF_000027325.1) and maps our protein ingredient names to the
//! genomic coordinate of their gene(s), so the chromosome stage can seat
//! RNAP/transcription at real loci instead of uniformly along the fiber.
//!
//! Ingredient → gene: an ingredient name carries its gene's old locus tag —
//! `MG_022_MONOMER` → `MG_022`; a complex lists each subunit —
//! `MG_098_099_100_TRIMER` → `MG_098`, `MG_099`, `MG_100`. ~99% of the
//! mycoplasma_full proteins map this way.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use rand::Rng;

/// One annotated gene (1-based inclusive `start`/`end`, like GFF).
#[derive(Debug, Clone)]
pub struct Gene {
    /// Old locus tag (e.g. `MG_022`) — what our ingredient names reference.
    pub locus: String,
    pub start: u32,
    pub end: u32,
    /// `+` or `-`.
    pub strand: char,
    pub biotype: String,
}

impl Gene {
    /// Genomic midpoint in base pairs.
    pub fn midpoint(&self) -> u32 {
        (self.start + self.end) / 2
    }
}

/// A parsed genome annotation: total length + genes, indexed by locus tag.
#[derive(Debug, Clone)]
pub struct Genome {
    pub length_bp: u32,
    pub genes: Vec<Gene>,
    by_locus: HashMap<String, usize>,
}

impl Genome {
    /// Parse the committed gene-table CSV. Format: a `#` comment line carrying
    /// `genome_length_bp=<N>`, a column header, then
    /// `old_locus_tag,locus_tag,start,end,strand,biotype` rows.
    pub fn from_csv(path: &Path) -> io::Result<Genome> {
        let text = std::fs::read_to_string(path)?;
        let mut length_bp = 0u32;
        let mut genes = Vec::new();
        let mut by_locus = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                if let Some(idx) = rest.find("genome_length_bp=") {
                    length_bp = rest[idx + "genome_length_bp=".len()..]
                        .split(|c: char| !c.is_ascii_digit())
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
                continue;
            }
            if line.starts_with("old_locus_tag") {
                continue; // column header
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 6 {
                continue;
            }
            let (start, end) = match (f[2].parse::<u32>(), f[3].parse::<u32>()) {
                (Ok(s), Ok(e)) => (s, e),
                _ => continue,
            };
            let locus = f[0].to_string();
            if !locus.is_empty() {
                by_locus.insert(locus.clone(), genes.len());
            }
            genes.push(Gene {
                locus,
                start,
                end,
                strand: f[4].chars().next().unwrap_or('+'),
                biotype: f[5].to_string(),
            });
        }
        if length_bp == 0 || genes.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("no genome_length / genes parsed from {}", path.display()),
            ));
        }
        Ok(Genome { length_bp, genes, by_locus })
    }

    /// Look up a gene by its old locus tag (e.g. `MG_022`).
    pub fn gene(&self, locus: &str) -> Option<&Gene> {
        self.by_locus.get(locus).map(|&i| &self.genes[i])
    }

    /// Genomic midpoints (bp) of the gene(s) named in an ingredient — e.g.
    /// `MG_022_MONOMER` → one site, `MG_098_099_100_TRIMER` → three. Empty
    /// when the name carries no recognizable `MG_<digits>` locus (rnap, tRNA…).
    pub fn ingredient_sites(&self, ingredient_name: &str) -> Vec<u32> {
        self.loci_in_name(ingredient_name)
            .into_iter()
            .filter_map(|locus| self.gene(&locus).map(Gene::midpoint))
            .collect()
    }

    /// Locus tags referenced by an ingredient name, validated against the
    /// genome's real locus set (organism-agnostic — no hardcoded prefix).
    /// Two candidate forms are checked: (a) each `_`-separated token verbatim
    /// (`EG10001`, `b0014`); (b) a letter-prefixed token joined with each
    /// following run of digit tokens (`MG`,`022` → `MG_022`,
    /// `MG_098_099_100` → `MG_098`, `MG_099`, `MG_100`). Only candidates present
    /// in `by_locus` are returned, preserving order and de-duplicating.
    fn loci_in_name(&self, name: &str) -> Vec<String> {
        let parts: Vec<&str> = name.split('_').collect();
        let mut out: Vec<String> = Vec::new();
        let mut push = |c: String, out: &mut Vec<String>| {
            if self.by_locus.contains_key(&c) && !out.contains(&c) {
                out.push(c);
            }
        };
        for (i, &p) in parts.iter().enumerate() {
            // (a) token verbatim
            push(p.to_string(), &mut out);
            // (b) <letters>_<digits>… reconstruction (Mycoplasma style)
            let is_alpha = !p.is_empty() && p.bytes().all(|b| b.is_ascii_alphabetic());
            if is_alpha {
                let mut j = i + 1;
                while j < parts.len()
                    && !parts[j].is_empty()
                    && parts[j].bytes().all(|b| b.is_ascii_digit())
                {
                    push(format!("{}_{}", p, parts[j]), &mut out);
                    j += 1;
                }
            }
        }
        out
    }

    /// Position around the circular genome as a fraction in `[0, 1)`.
    pub fn fraction(&self, bp: u32) -> f32 {
        if self.length_bp == 0 {
            0.0
        } else {
            (bp % self.length_bp) as f32 / self.length_bp as f32
        }
    }

    /// Split the circular genome into `domains` arcs at the **largest
    /// intergenic gaps** (≈ operon / transcription-unit boundaries) and return
    /// a bead count per arc, proportional to its bp span (summing to ~`total_
    /// beads`, each ≥ 2). Drives the rosette's plectoneme domains so they
    /// correspond to real gene clusters instead of being evenly spaced — while
    /// keeping bp/bead ~constant so the bp↔arc mapping (for RNAP) stays linear.
    pub fn domain_bead_allocation(&self, total_beads: usize, domains: usize) -> Vec<usize> {
        let domains = domains.max(1);
        if domains == 1 || self.genes.len() <= domains {
            return vec![(total_beads / domains).max(2); domains];
        }
        let mut mids: Vec<u32> = self
            .genes
            .iter()
            .map(|g| g.midpoint() % self.length_bp)
            .collect();
        mids.sort_unstable();
        let n = mids.len();
        // Circular gap after each gene midpoint, tagged with its index.
        let gap_after = |i: usize| -> u32 {
            let a = mids[i];
            let b = if i + 1 < n { mids[i + 1] } else { mids[0] + self.length_bp };
            b - a
        };
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&x, &y| gap_after(y).cmp(&gap_after(x)));
        // Boundary positions = midpoints of the `domains` largest gaps.
        let mut bounds: Vec<u32> = idx
            .iter()
            .take(domains)
            .map(|&i| {
                let a = mids[i];
                let b = if i + 1 < n { mids[i + 1] } else { mids[0] + self.length_bp };
                ((a + b) / 2) % self.length_bp
            })
            .collect();
        bounds.sort_unstable();
        let m = bounds.len();
        let spans: Vec<f64> = (0..m)
            .map(|i| {
                let a = bounds[i];
                let b = if i + 1 < m { bounds[i + 1] } else { bounds[0] + self.length_bp };
                (b - a) as f64
            })
            .collect();
        let total_span: f64 = spans.iter().sum::<f64>().max(1.0);
        spans
            .iter()
            .map(|&s| ((total_beads as f64 * s / total_span).round() as usize).max(2))
            .collect()
    }

    /// Genomic positions ([0,1) fractions) at which to seat each chromosome
    /// bound protein. `bound` is `(name, count)` of DNA-binding proteins
    /// (RNAP, DNAP); `abundances` is `(ingredient_name, count)` from the recipe
    /// composition. RNAP (the default) is sampled over genes weighted by the
    /// abundance of the protein each encodes (≈ transcription level), so it
    /// clusters at highly-expressed loci; DNAP-like proteins cluster near the
    /// replication origin (genome position 0, two diverging forks). Returns one
    /// `Vec` of `count` fractions per entry of `bound`, aligned with it.
    pub fn binding_sites<R: Rng>(
        &self,
        bound: &[(String, u32)],
        abundances: &[(String, u32)],
        rng: &mut R,
    ) -> Vec<Vec<f32>> {
        // Abundance-weighted gene fractions (cumulative weights for sampling).
        let mut frac: Vec<f32> = Vec::new();
        let mut cum: Vec<f32> = Vec::new();
        let mut acc = 0.0_f32;
        for (name, count) in abundances {
            if *count == 0 {
                continue;
            }
            for bp in self.ingredient_sites(name) {
                frac.push(self.fraction(bp));
                acc += *count as f32;
                cum.push(acc);
            }
        }
        let total = acc;
        bound
            .iter()
            .map(|(name, count)| {
                let lname = name.to_ascii_lowercase();
                let is_dnap = lname.contains("dnap")
                    || lname.contains("dna_pol")
                    || lname.contains("replisome");
                (0..*count)
                    .map(|_| {
                        if is_dnap {
                            // Replisome near oriC (origin), two diverging forks.
                            (rng.gen_range(-0.03_f32..0.03) + 1.0).fract()
                        } else if total > 0.0 && !cum.is_empty() {
                            let x = rng.gen_range(0.0..total);
                            let idx = cum.partition_point(|&c| c < x).min(cum.len() - 1);
                            frac[idx]
                        } else {
                            rng.gen_range(0.0..1.0)
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn table() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/genome/mycoplasma_g37_genes.csv")
    }

    #[test]
    fn parses_committed_gene_table() {
        let g = Genome::from_csv(&table()).expect("parse gene table");
        assert_eq!(g.length_bp, 580_076);
        assert!(g.genes.len() > 500, "expected >500 genes, got {}", g.genes.len());
        let mg1 = g.gene("MG_001").expect("MG_001 present");
        assert_eq!((mg1.start, mg1.end, mg1.strand), (686, 1828, '+'));
    }

    #[test]
    fn maps_ingredient_names_to_gene_sites() {
        let g = Genome::from_csv(&table()).unwrap();
        // Monomer → one site, at its gene midpoint.
        let s = g.ingredient_sites("MG_001_MONOMER");
        assert_eq!(s, vec![(686 + 1828) / 2]);
        // Complex → one site per named subunit (those present in the table).
        let trimer = g.ingredient_sites("MG_098_099_100_TRIMER");
        assert!(trimer.len() >= 2, "expected multiple subunit sites, got {trimer:?}");
        // Non-gene names map to nothing.
        assert!(g.ingredient_sites("rnap").is_empty());
        assert!(g.ingredient_sites("tRNA").is_empty());
    }

    #[test]
    fn loci_tokenizer_handles_complexes() {
        let g = Genome::from_csv(&table()).unwrap();
        assert_eq!(g.loci_in_name("MG_022_MONOMER"), vec!["MG_022"]);
        assert_eq!(
            g.loci_in_name("MG_098_099_100_TRIMER"),
            vec!["MG_098", "MG_099", "MG_100"]
        );
        assert!(g.loci_in_name("rnap").is_empty());
    }

    #[test]
    fn loci_lookup_is_organism_agnostic() {
        // A tiny in-memory E. coli-style genome (b-numbers + an EcoCyc id).
        let csv = "# genome_length_bp=4641652\n\
                   old_locus_tag,locus_tag,start,end,strand,biotype\n\
                   b0014,dnaK,12163,14079,+,protein_coding\n\
                   EG10001,thrA,337,2799,+,protein_coding\n";
        let tmp = std::env::temp_dir().join("ecoli_loci_test.csv");
        std::fs::write(&tmp, csv).unwrap();
        let g = Genome::from_csv(&tmp).unwrap();
        // b-number embedded in an ingredient name resolves to its gene midpoint.
        assert_eq!(g.ingredient_sites("b0014_dnaK_MONOMER"), vec![(12163 + 14079) / 2]);
        // EcoCyc single-token id resolves too.
        assert_eq!(g.ingredient_sites("EG10001"), vec![(337 + 2799) / 2]);
        // Unknown tokens map to nothing.
        assert!(g.ingredient_sites("ribosome_30S").is_empty());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn fraction_is_unit_interval() {
        let g = Genome::from_csv(&table()).unwrap();
        assert!((g.fraction(0)).abs() < 1e-6);
        let f = g.fraction(g.length_bp / 2);
        assert!((f - 0.5).abs() < 1e-3, "got {f}");
    }

    #[test]
    fn domain_allocation_is_nonuniform_and_sums() {
        let g = Genome::from_csv(&table()).unwrap();
        let alloc = g.domain_bead_allocation(90_000, 50);
        assert_eq!(alloc.len(), 50);
        let sum: usize = alloc.iter().sum();
        assert!((sum as i64 - 90_000).abs() < 3_000, "sum {sum} ~ 90000");
        assert!(alloc.iter().all(|&b| b >= 2));
        let (mn, mx) = (*alloc.iter().min().unwrap(), *alloc.iter().max().unwrap());
        assert!(mx > mn * 2, "transcription-coupled domains should vary: min {mn} max {mx}");
    }

    #[test]
    fn binding_sites_follow_abundance_and_origin() {
        use rand::SeedableRng;
        let g = Genome::from_csv(&table()).unwrap();
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
        let bound = vec![("rnap".to_string(), 200u32), ("dnap".to_string(), 20u32)];
        // One dominant gene → RNAP should cluster near its fraction.
        let abund = vec![("MG_001_MONOMER".to_string(), 10_000u32)];
        let sites = g.binding_sites(&bound, &abund, &mut rng);
        assert_eq!(sites.len(), 2);
        assert_eq!((sites[0].len(), sites[1].len()), (200, 20));
        let target = g.fraction(g.gene("MG_001").unwrap().midpoint());
        let near = sites[0].iter().filter(|&&f| (f - target).abs() < 0.01).count();
        assert!(near > 160, "RNAP should cluster at the abundant gene: {near}/200");
        assert!(
            sites[1].iter().all(|&f| f < 0.05 || f > 0.95),
            "DNAP should sit near oriC"
        );
    }
}
